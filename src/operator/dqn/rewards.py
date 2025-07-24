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
    "goodness" and the predicted future state's "goodness", with an added
    penalty for scaling actions to reduce oscillation.
    """

    def __init__(self):
        # --- Core State Tuning Parameters ---
        self.TARGET_CPU_UTIL: float = 0.70  # 70%
        self.TARGET_MEM_UTIL: float = 0.80  # 80%
        self.CPU_REWARD_WIDTH: float = 0.4
        self.MEM_REWARD_WIDTH: float = 0.4
        
        # --- Cost-Benefit Balance (CRITICAL TUNING PARAMETER) ---
        self.COST_PER_REPLICA_PENALTY: float = 0.035
        
        # --- Anti-Oscillation Parameters ---
        self.ACTION_CHANGE_PENALTY: float = 0.05  # Penalty for any scaling action
        
        self.CPU_WEIGHT: float = 0.5
        self.MEM_WEIGHT: float = 0.5

        # --- Weights for Combining Current and Forecast Rewards ---
        self.CURRENT_STATE_WEIGHT: float = 0.4 # Weight for immediate conditions
        self.FORECAST_STATE_WEIGHT: float = 0.6 # Weight for predicted conditions

        # --- Default Resource Requests ---
        self.DEFAULT_CPU_REQUEST_CORES: float = 0.5
        self.DEFAULT_MEM_REQUEST_MB: float = 1024.0

        logger.info(
            "Initialized ProactiveRewardCalculator with Action Change Penalty: %s",
            self.ACTION_CHANGE_PENALTY
        )
        logger.info(f"💰 Cost per replica: {self.COST_PER_REPLICA_PENALTY:.3f}")

    @staticmethod
    def _validate_replicas(replicas: int) -> int:
        if replicas <= 0:
            logger.warning("⚠️ Invalid replica count (%s) detected. Defaulting to 1.", replicas)
            return 1
        return replicas

    def _get_resource_requests(self, deployment_info: Optional[Dict[str, Any]]) -> tuple[float, float]:
        if deployment_info and deployment_info.get('resource_requests'):
            requests = deployment_info['resource_requests']
            cpu_cores = requests.get('cpu', self.DEFAULT_CPU_REQUEST_CORES)
            memory_bytes = requests.get('memory', self.DEFAULT_MEM_REQUEST_MB * 1024 * 1024)
            memory_mb = memory_bytes / (1024 * 1024)
            return cpu_cores, memory_mb
        else:
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

        util_cpu = (cpu_rate / replicas) / cpu_request_cores
        util_mem = (memory_bytes / (1024 * 1024)) / replicas / mem_request_mb

        def get_gaussian_reward(current_util: float, target_util: float, width: float) -> float:
            return math.exp(-((current_util - target_util) ** 2) / (2 * width ** 2))

        cpu_reward = get_gaussian_reward(util_cpu, self.TARGET_CPU_UTIL, self.CPU_REWARD_WIDTH)
        mem_reward = get_gaussian_reward(util_mem, self.TARGET_MEM_UTIL, self.MEM_REWARD_WIDTH)

        utilization_reward = (self.CPU_WEIGHT * cpu_reward + self.MEM_WEIGHT * mem_reward)
        cost_penalty = self.COST_PER_REPLICA_PENALTY * replicas
        
        final_goodness = utilization_reward - cost_penalty
        return final_goodness

    def calculate(self,
                  action: str,
                  current_metrics: Dict[str, float],
                  current_replicas: int,
                  desired_replicas: int, # The agent's choice of replicas for the next state
                  forecast_result: Optional[Dict[str, Any]] = None,
                  deployment_info: Optional[Dict[str, Any]] = None) -> float:
        """Calculates a combined reward from current and forecasted states."""
        try:
            validated_current_replicas = self._validate_replicas(current_replicas)
            validated_desired_replicas = self._validate_replicas(desired_replicas)

            if deployment_info and deployment_info.get('ready_replicas', 1) == 0:
                logger.error("🚨 EMERGENCY: 0 ready replicas! Overriding normal reward logic.")
                return 5.0 if action == "scale_up" else -5.0

            # Calculate Reward for Current State
            current_reward = self._calculate_state_goodness(
                current_metrics, validated_current_replicas, deployment_info
            )
            
            # Calculate Reward for Forecasted State
            if forecast_result and forecast_result.get("predicted_metrics"):
                predicted_metrics = forecast_result["predicted_metrics"]
                forecast_reward = self._calculate_state_goodness(
                    predicted_metrics, validated_desired_replicas, deployment_info
                )
                
                final_reward = (self.CURRENT_STATE_WEIGHT * current_reward) + \
                               (self.FORECAST_STATE_WEIGHT * forecast_reward)
            else:
                final_reward = current_reward
            
            # Apply penalty if a scaling action was taken to reduce oscillation
            if action != "keep_same":
                final_reward -= self.ACTION_CHANGE_PENALTY
                logger.info(f"⚖️ Applied action change penalty of {self.ACTION_CHANGE_PENALTY}. "
                           f"New reward: {final_reward:.3f}")

            # Optional: Add small bonus for scaling down on very low load
            is_very_low_load = current_metrics.get("process_cpu_seconds_total_rate", 0) < 0.1 * validated_current_replicas
            min_replicas = deployment_info.get('min_replicas', 1) if deployment_info else 1
            if action == "scale_down" and is_very_low_load and validated_current_replicas > min_replicas:
                final_reward += 0.1

            logger.info(
                f"Reward Calculated: action={action}, current_R={current_reward:.2f}, total_R={final_reward:.2f}"
            )
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