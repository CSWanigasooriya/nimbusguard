"""Simplified reward system for DQN that only considers CPU and memory metrics.

Available Metrics:
- process_cpu_seconds_total_rate: CPU usage rate (seconds/second)
- process_resident_memory_bytes: Current memory usage (bytes)
"""

import logging
import math
from typing import Dict, Any, Optional

from config.settings import load_config

config = load_config()

logger = logging.getLogger(__name__)


class ProactiveRewardCalculator:
    """
    Calculates a reward based on a weighted combination of the current state's
    "goodness" and the predicted future state's "goodness".
    """

    def __init__(self):
        # --- Core State Tuning Parameters ---
        self.TARGET_CPU_UTIL: float = 0.70  # 70%
        self.TARGET_MEM_UTIL: float = 0.80  # 80%
        self.CPU_REWARD_WIDTH: float = 0.4
        self.MEM_REWARD_WIDTH: float = 0.4

        # --- Cost-Benefit Balance (CRITICAL TUNING PARAMETER) ---
        # This penalty should be small enough that good utilization can outweigh cost,
        # but large enough that unnecessary replicas are discouraged.
        # Based on observed utilization rewards of 0.4-0.8, cost should be ~0.03-0.05 per replica
        # so that 3-5 replicas create meaningful trade-offs with utilization benefits
        self.COST_PER_REPLICA_PENALTY: float = 0.035  # Increased based on observed utilization rewards

        self.CPU_WEIGHT: float = 0.5
        self.MEM_WEIGHT: float = 0.5

        # --- Weights for Combining Current and Forecast Rewards ---
        self.CURRENT_STATE_WEIGHT: float = 0.4  # Weight for immediate conditions
        self.FORECAST_STATE_WEIGHT: float = 0.6  # Weight for predicted conditions

        # --- Default Resource Requests ---
        self.DEFAULT_CPU_REQUEST_CORES: float = 0.5
        self.DEFAULT_MEM_REQUEST_MB: float = 1024.0

        logger.info(
            "Initialized ProactiveRewardCalculator with weights: current=%s, forecast=%s",
            self.CURRENT_STATE_WEIGHT, self.FORECAST_STATE_WEIGHT
        )

        # Log the cost-benefit balance for tuning reference
        max_util_reward = 1.0  # Maximum possible utilization reward
        example_costs = [self.COST_PER_REPLICA_PENALTY * replicas for replicas in [1, 3, 5, 10, 20]]
        logger.info(f"💰 Cost-benefit balance analysis:")
        logger.info(f"   Max utilization reward: {max_util_reward:.3f}")
        logger.info(f"   Cost per replica: {self.COST_PER_REPLICA_PENALTY:.3f}")
        logger.info(f"   Example costs: 1r={example_costs[0]:.3f}, 3r={example_costs[1]:.3f}, "
                    f"5r={example_costs[2]:.3f}, 10r={example_costs[3]:.3f}, 20r={example_costs[4]:.3f}")
        logger.info(f"   ➡️ Perfect utilization can justify up to {max_util_reward/self.COST_PER_REPLICA_PENALTY:.0f} replicas")
        logger.info(f"   🎯 Target cost ratio range: 0.1-2.0 (current warnings trigger outside this range)")

    @staticmethod
    def _validate_replicas(replicas: int) -> int:
        if replicas <= 0:
            logger.warning("⚠️ Invalid replica count (%s) detected. Defaulting to 1.", replicas)
            return 1
        return replicas

    def _get_resource_requests(self, deployment_info: Optional[Dict[str, Any]]) -> tuple[float, float]:
        logger.debug("🔍 Checking deployment resource requests...")

        # Log the entire deployment_info structure for debugging
        if deployment_info:
            logger.debug(f"📋 Deployment info available: {list(deployment_info.keys())}")
            if 'resource_requests' in deployment_info:
                logger.debug(f"📊 Resource requests found: {deployment_info['resource_requests']}")
            else:
                logger.warning("⚠️ No 'resource_requests' key in deployment_info")
                logger.debug(f"📋 Available keys: {list(deployment_info.keys())}")
        else:
            logger.warning("⚠️ No deployment_info provided - using defaults")

        if deployment_info and deployment_info.get('resource_requests'):
            requests = deployment_info['resource_requests']
            cpu_cores = requests.get('cpu', self.DEFAULT_CPU_REQUEST_CORES)
            memory_bytes = requests.get('memory', self.DEFAULT_MEM_REQUEST_MB * 1024 * 1024)
            memory_mb = memory_bytes / (1024 * 1024)

            logger.info(f"✅ Using deployment resource requests: CPU={cpu_cores:.3f} cores, Memory={memory_mb:.1f}MB")
            return cpu_cores, memory_mb
        else:
            logger.info(f"📐 Using default resource requests: CPU={self.DEFAULT_CPU_REQUEST_CORES:.3f} cores, Memory={self.DEFAULT_MEM_REQUEST_MB:.1f}MB")
            return self.DEFAULT_CPU_REQUEST_CORES, self.DEFAULT_MEM_REQUEST_MB

    def _calculate_state_goodness(self,
                                  metrics: Dict[str, float],
                                  replicas: int,
                                  deployment_info: Optional[Dict[str, Any]]) -> float:
        """Core logic to calculate the reward for a given system state (current or future)."""
        cpu_request_cores, mem_request_mb = self._get_resource_requests(deployment_info)

        if cpu_request_cores <= 0 or mem_request_mb <= 0:
            logger.error("🚨 CRITICAL: CPU or Memory request is zero. State is considered bad.")
            return -10.0

        cpu_rate = metrics.get("process_cpu_seconds_total_rate", 0.0)
        memory_bytes = metrics.get("process_resident_memory_bytes", 0.0)

        # Log raw metrics for debugging
        logger.debug(f"📊 Raw metrics: CPU_rate={cpu_rate:.4f}, Memory_bytes={memory_bytes:.0f}, Replicas={replicas}")

        util_cpu = (cpu_rate / replicas) / cpu_request_cores
        util_mem = (memory_bytes / (1024 * 1024)) / replicas / mem_request_mb

        # Log calculated utilizations
        logger.debug(f"📈 Calculated utilizations: CPU={util_cpu:.3f} (target={self.TARGET_CPU_UTIL}), Memory={util_mem:.3f} (target={self.TARGET_MEM_UTIL})")

        def get_gaussian_reward(current_util: float, target_util: float, width: float) -> float:
            return math.exp(-((current_util - target_util) ** 2) / (2 * width ** 2))

        cpu_reward = get_gaussian_reward(util_cpu, self.TARGET_CPU_UTIL, self.CPU_REWARD_WIDTH)
        mem_reward = get_gaussian_reward(util_mem, self.TARGET_MEM_UTIL, self.MEM_REWARD_WIDTH)

        utilization_reward = (self.CPU_WEIGHT * cpu_reward + self.MEM_WEIGHT * mem_reward)
        cost_penalty = self.COST_PER_REPLICA_PENALTY * replicas

        final_goodness = utilization_reward - cost_penalty

        # Log the reward breakdown with balance analysis
        balance_ratio = cost_penalty / max(utilization_reward, 0.001)  # Avoid division by zero
        logger.debug(f"🎯 State goodness breakdown: CPU_reward={cpu_reward:.3f}, MEM_reward={mem_reward:.3f}, "
                     f"utilization_reward={utilization_reward:.3f}, cost_penalty={cost_penalty:.3f}, "
                     f"final_goodness={final_goodness:.3f}, cost_ratio={balance_ratio:.2f}")

        # Log balance warnings
        if balance_ratio > 2.0:
            logger.warning(f"⚠️ Cost penalty ({cost_penalty:.3f}) >> utilization reward ({utilization_reward:.3f}). "
                           f"Agent may over-prefer scale_down. Consider reducing COST_PER_REPLICA_PENALTY.")
        elif balance_ratio < 0.1:
            logger.warning(f"⚠️ Cost penalty ({cost_penalty:.3f}) << utilization reward ({utilization_reward:.3f}). "
                           f"Agent may over-prefer scale_up. Consider increasing COST_PER_REPLICA_PENALTY.")

        return final_goodness

    def calculate(self,
                  action: str,
                  current_metrics: Dict[str, float],
                  current_replicas: int,
                  desired_replicas: int,  # The agent's choice of replicas for the next state
                  forecast_result: Optional[Dict[str, Any]] = None,
                  deployment_info: Optional[Dict[str, Any]] = None) -> float:
        """Calculates a combined reward from current and forecasted states."""
        try:
            validated_current_replicas = self._validate_replicas(current_replicas)
            validated_desired_replicas = self._validate_replicas(desired_replicas)

            # --- Handle Emergencies ---
            if deployment_info and deployment_info.get('ready_replicas', 1) == 0:
                logger.error("🚨 EMERGENCY: 0 ready replicas! Overriding normal reward logic.")
                return 5.0 if action == "scale_up" else -5.0

            # Log the scaling decision context
            logger.debug(f"🎯 Calculating reward for: action={action}, current_replicas={validated_current_replicas}, desired_replicas={validated_desired_replicas}")

            # --- Calculate Reward for Current State ---
            # We use `validated_current_replicas` as this is the state we are observing.
            logger.debug("📊 Calculating current state goodness...")
            current_reward = self._calculate_state_goodness(
                current_metrics, validated_current_replicas, deployment_info
            )
            logger.debug(f"📊 Current state reward: {current_reward:.3f}")

            forecast_reward = 0.0
            # --- Calculate Reward for Forecasted State ---
            if forecast_result and forecast_result.get("predicted_metrics"):
                predicted_metrics = forecast_result["predicted_metrics"]
                logger.debug(f"🔮 Forecast metrics available: {list(predicted_metrics.keys())}")
                logger.debug(f"🔮 Forecast values: CPU_rate={predicted_metrics.get('process_cpu_seconds_total_rate', 'N/A')}, Memory_bytes={predicted_metrics.get('process_resident_memory_bytes', 'N/A')}")

                # We use `validated_desired_replicas` here because that's the number of replicas
                # that will be handling the predicted load. This is key.
                logger.debug("🔮 Calculating forecast state goodness...")
                forecast_reward = self._calculate_state_goodness(
                    predicted_metrics, validated_desired_replicas, deployment_info
                )
                logger.debug(f"🔮 Forecast state reward: {forecast_reward:.3f}")

                final_reward = (self.CURRENT_STATE_WEIGHT * current_reward) + \
                               (self.FORECAST_STATE_WEIGHT * forecast_reward)
                logger.info(
                    f"Reward (Proactive): current_R={current_reward:.2f} (weight={self.CURRENT_STATE_WEIGHT}), "
                    f"forecast_R={forecast_reward:.2f} (weight={self.FORECAST_STATE_WEIGHT}) | "
                    f"Total={final_reward:.2f}"
                )
            else:
                # If no forecast is available, the reward is based solely on the current state.
                final_reward = current_reward
                logger.info(f"Reward (Current Only): Total={final_reward:.2f} (no forecast available)")

            # Optional: Add small bonus for correct proactive scaling
            is_very_low_load = current_metrics.get("process_cpu_seconds_total_rate", 0) < 0.1 * validated_current_replicas
            min_replicas = deployment_info.get('min_replicas', 1) if deployment_info else 1
            if action == "scale_down" and is_very_low_load and validated_current_replicas > min_replicas:
                final_reward += 0.1

            return final_reward

        except Exception as e:
            logger.error("Failed to calculate proactive reward: %s", e, exc_info=True)
            return 0.0

    def get_reward_explanation(self,
                               action: str,
                               current_metrics: Dict[str, float],
                               current_replicas: int,
                               desired_replicas: int,
                               forecast_result: Optional[Dict[str, Any]] = None) -> str:
        """Get human-readable explanation of reward calculation."""
        validated_current_replicas = self._validate_replicas(current_replicas)
        validated_desired_replicas = self._validate_replicas(desired_replicas)

        explanation = f"Action: {action}, Replicas: {validated_current_replicas} → {validated_desired_replicas}"

        if forecast_result:
            explanation += f", Forecast: available"
        else:
            explanation += f", Forecast: none"

        return explanation