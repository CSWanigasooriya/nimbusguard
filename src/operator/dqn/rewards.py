"""Simplified reward system for DQN that only considers CPU and memory metrics.

Available Metrics:
- process_cpu_seconds_total_rate: CPU usage rate (seconds/second)
- process_resident_memory_bytes: Current memory usage (bytes)
"""
import logging
from typing import Dict, Any, Optional

from config.settings import load_config

config = load_config()

logger = logging.getLogger(__name__)


class ProactiveRewardCalculator:
    """Calculates rewards for DQN considering both current and predicted future state."""

    def __init__(self):
        # Get configuration values from config with safe fallbacks
        try:
            self.forecast_confidence_threshold = getattr(config.reward, 'forecast_confidence_threshold', 0.2)
            self.efficiency_targets = getattr(config.reward, 'efficiency_targets', {
                "cpu_utilization": 0.70,
                "memory_utilization": 0.80,
                "default_memory_mb": 1024,  # 1Gi limit
                "default_cpu_cores": 0.5   # 500m limit
            })
        except AttributeError as e:
            logger.warning(f"Config access issue, using defaults: {e}")
            self.forecast_confidence_threshold = 0.2
            self.efficiency_targets = {
                "cpu_utilization": 0.70,
                "memory_utilization": 0.80,
                "default_memory_mb": 1024,  # 1Gi limit
                "default_cpu_cores": 0.5   # 500m limit
            }
        
        # Initialize action tracking for stability analysis
        self.recent_actions = []
        self.recent_load_classes = []
        self.max_action_history = 5
        self.max_load_history = 3
        logger.info(f"Initialized ProactiveRewardCalculator with CPU and memory metrics only.")
        logger.info(f"Using forecast confidence threshold: {self.forecast_confidence_threshold}")
        logger.info(f"Using efficiency targets: {self.efficiency_targets}")

    @staticmethod
    def _validate_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
        """Validate that required metrics are present and handle missing ones."""
        required_metrics = [
            "process_cpu_seconds_total_rate",
            "process_resident_memory_bytes"
        ]

        validated_metrics = metrics.copy()
        missing_metrics = []

        for metric in required_metrics:
            if metric not in metrics:
                missing_metrics.append(metric)
                validated_metrics[metric] = 0.0  # Safe default

        if missing_metrics:
            logger.warning(f"⚠️ Missing metrics (using defaults): {missing_metrics}")

        return validated_metrics

    @staticmethod
    def _get_context_aware_weights(load_class: str, forecast_confidence: float) -> tuple:
        """Adjust reward weights based on system context."""
        # CRITICAL: Zero metrics (stuck pods) - prioritize immediate action over forecasting
        if load_class == "ZERO":
            return 0.9, 0.1  # Strong emphasis on current state reward to encourage scaling up

        # High load: prioritize immediate response over forecasting
        elif load_class == "HIGH":
            return 0.8, 0.2  # current_weight, forecast_weight

        # Low load with good forecast: prioritize proactive planning
        elif load_class == "LOW" and forecast_confidence > 0.5:
            return 0.4, 0.6

        # Medium load or uncertain forecast: balanced approach
        else:
            return 0.6, 0.4

    @staticmethod
    def _validate_replicas(replicas: int, context: str = "") -> int:
        """Validate replica count and handle edge cases explicitly."""
        if replicas < 0:
            logger.error(f"⚠️ CRITICAL: Negative replica count detected: {replicas} {context}")
            return 1  # Force to minimum viable state
        elif replicas == 0:
            logger.warning(f"⚠️ CRITICAL: Zero replicas detected - no pods running! {context}")
            return 1  # Prevent division by zero and alert to critical state
        else:
            return replicas

    def _track_action(self, action: str) -> None:
        """Track recent actions for stability analysis."""
        self.recent_actions.append(action)
        if len(self.recent_actions) > self.max_action_history:
            self.recent_actions.pop(0)
        logger.info(
            f"Action tracked: {action}, history: {self.recent_actions} ({len(self.recent_actions)}/{self.max_action_history})")

    def _track_load_class(self, load_class: str) -> None:
        """Track recent load classifications to detect load changes."""
        self.recent_load_classes.append(load_class)
        if len(self.recent_load_classes) > self.max_load_history:
            self.recent_load_classes.pop(0)
        logger.debug(f"Load tracked: {load_class}, history: {self.recent_load_classes}")

    def _detect_load_change(self) -> bool:
        """Detect if there was a significant load change recently."""
        if len(self.recent_load_classes) < 2:
            return False

        # Check if load changed from LOW/MEDIUM to HIGH or HIGH/MEDIUM to LOW
        current = self.recent_load_classes[-1]
        previous = self.recent_load_classes[-2]

        significant_change = (
                (previous in ["LOW", "MEDIUM"] and current == "HIGH") or
                (previous in ["HIGH", "MEDIUM"] and current == "LOW") or
                (previous == "LOW" and current == "MEDIUM") or
                (previous == "MEDIUM" and current == "LOW")
        )

        if significant_change:
            logger.info(f"🔄 Load change detected: {previous} → {current}")

        return significant_change

    def _calculate_stability_penalty(self, action: str, current_metrics: Dict[str, float] = None,
                                     current_replicas: int = 1, load_class: str = None) -> float:
        """Context-aware stability system that distinguishes between justified quick scaling and harmful oscillations."""
        logger.debug(
            f"Stability analysis: action={action}, history_size={len(self.recent_actions)}, recent_actions={self.recent_actions}")

        if len(self.recent_actions) < 2:
            logger.debug(f"Insufficient history for stability analysis: {len(self.recent_actions)}/2")
            return 0.1  # Small positive reward for early actions

        stability_score = 0.0

        # Get current load context if available (use passed load_class to avoid duplicate calculation)
        current_load_class = load_class
        load_change_detected = False
        if current_load_class:
            self._track_load_class(current_load_class)
            load_change_detected = self._detect_load_change()
            logger.debug(f"Current load context: {current_load_class}, load_change: {load_change_detected}")
        elif current_metrics and current_replicas:
            # Fallback to calculating if not provided
            current_load_class = self._classify_load(current_metrics, current_replicas)
            self._track_load_class(current_load_class)
            load_change_detected = self._detect_load_change()
            logger.debug(f"Fallback load context: {current_load_class}, load_change: {load_change_detected}")

        # 1. CONTEXT-AWARE OSCILLATION ANALYSIS
        if len(self.recent_actions) >= 2:
            last_action = self.recent_actions[-1]
            second_last = self.recent_actions[-2] if len(self.recent_actions) >= 2 else None

            # Check for immediate flip-flop
            is_flip_flop = ((last_action == "scale_up" and action == "scale_down") or
                            (last_action == "scale_down" and action == "scale_up"))

            if is_flip_flop:
                # HIGHLY JUSTIFIED: Quick scaling due to detected load change
                if load_change_detected:
                    logger.info(f"🚀 EXCELLENT quick scaling response to load change: {last_action} → {action}")
                    stability_score += 0.5  # High reward for responding to load changes

                # JUSTIFIED QUICK SCALING: If load is high and we're scaling up, or load is low and scaling down
                elif current_load_class:
                    if current_load_class == "HIGH" and action == "scale_up":
                        logger.info(f"✅ Justified quick scale-up for HIGH load: {last_action} → {action}")
                        stability_score += 0.3  # Reward quick response to high load
                    elif current_load_class == "LOW" and action == "scale_down":
                        logger.info(f"✅ Justified quick scale-down for LOW load: {last_action} → {action}")
                        stability_score += 0.3  # Reward quick response to low load
                    else:
                        # Unjustified flip-flop
                        logger.warning(
                            f"⚠️ Unjustified flip-flop ({current_load_class} load): {last_action} → {action}")
                        stability_score -= 0.3  # Moderate penalty (reduced from 0.5)
                else:
                    # No context available, apply moderate penalty
                    logger.warning(f"⚠️ Flip-flop without context: {last_action} → {action}")
                    stability_score -= 0.2  # Reduced penalty without context

            # Check for 3-step oscillation
            is_3_step_oscillation = (second_last and
                                     ((
                                                  second_last == "scale_up" and last_action == "scale_down" and action == "scale_up") or
                                      (
                                                  second_last == "scale_down" and last_action == "scale_up" and action == "scale_down")))

            if is_3_step_oscillation:
                # 3-step oscillations are generally bad regardless of load
                logger.warning(f"⚠️ 3-step oscillation: {second_last} → {last_action} → {action}")
                stability_score -= 0.5  # Reduced from 0.8

        # Track the action for future stability analysis
        self._track_action(action)

        # Clamp the stability score to reasonable bounds
        stability_score = max(-1.0, min(1.0, stability_score))

        logger.info(f"🎯 Stability analysis result: {stability_score:.3f} (action: {action})")
        return stability_score

    def _calculate_resource_efficiency(self, metrics: Dict[str, float], replicas: int,
                                       deployment_info: Optional[Dict[str, Any]] = None) -> float:
        """Calculate resource efficiency with proper validation and fallback handling."""
        try:
            # Validate replicas first
            validated_replicas = self._validate_replicas(replicas, "in resource efficiency calculation")

            # Get resource utilization metrics
            cpu_rate = metrics.get("process_cpu_seconds_total_rate", 0.0)
            memory_bytes = metrics.get("process_resident_memory_bytes", 0.0)

            # Calculate per-replica utilization
            cpu_per_replica = cpu_rate / validated_replicas
            memory_per_replica = memory_bytes / validated_replicas

            # Proper validation of deployment limits with explicit warnings
            cpu_limit_per_replica = None
            memory_limit_per_replica = None

            if deployment_info and deployment_info.get('resource_limits'):
                resource_limits = deployment_info['resource_limits']

                # Validate CPU limit
                cpu_limit_total = resource_limits.get('cpu', 0.0)
                if cpu_limit_total > 0:
                    cpu_limit_per_replica = cpu_limit_total / validated_replicas
                    logger.debug(f"Using deployment CPU limit: {cpu_limit_per_replica:.3f} per replica")
                else:
                    logger.warning("⚠️ Invalid or missing CPU limit in deployment_info, using default targets")

                # Validate memory limit
                memory_limit_total = resource_limits.get('memory', 0)
                if memory_limit_total > 0:
                    memory_limit_per_replica = memory_limit_total / validated_replicas
                    logger.debug(f"Using deployment memory limit: {memory_limit_per_replica/1024/1024:.1f}MB per replica")
                else:
                    logger.warning("⚠️ Invalid or missing memory limit in deployment_info, using default targets")
            else:
                logger.info("No deployment resource limits available, using configurable default targets")

            # Calculate targets (use deployment limits if available, otherwise configurable defaults)
            if cpu_limit_per_replica and cpu_limit_per_replica > 0:
                target_cpu = cpu_limit_per_replica * self.efficiency_targets["cpu_utilization"]
            else:
                target_cpu = self.efficiency_targets["default_cpu_cores"] * self.efficiency_targets["cpu_utilization"]  # 0.5 * 0.7 = 0.35

            if memory_limit_per_replica and memory_limit_per_replica > 0:
                target_memory = memory_limit_per_replica * self.efficiency_targets["memory_utilization"]
            else:
                target_memory = self.efficiency_targets["default_memory_mb"] * 1024 * 1024  # Convert MB to bytes

            logger.debug(f"Efficiency targets: CPU={target_cpu:.3f}, Memory={target_memory/1024/1024:.1f}MB")

            # Calculate efficiency scores
            if target_cpu > 0:
                cpu_efficiency = 1.0 - abs(cpu_per_replica - target_cpu) / target_cpu
            else:
                cpu_efficiency = 0.5  # Neutral if no target
                logger.warning("⚠️ Zero CPU target, using neutral efficiency score")

            if target_memory > 0:
                memory_efficiency = 1.0 - abs(memory_per_replica - target_memory) / target_memory
            else:
                memory_efficiency = 0.5  # Neutral if no target
                logger.warning("⚠️ Zero memory target, using neutral efficiency score")

            # Ensure scores are between 0 and 1
            cpu_efficiency = max(0, min(1, cpu_efficiency))
            memory_efficiency = max(0, min(1, memory_efficiency))

            efficiency_score = (cpu_efficiency * 0.6) + (memory_efficiency * 0.4)

            logger.debug(f"Resource efficiency: CPU={cpu_efficiency:.3f} ({cpu_per_replica:.3f}/{target_cpu:.3f}), "
                         f"Memory={memory_efficiency:.3f} ({memory_per_replica / 1024 / 1024:.1f}MB/{target_memory / 1024 / 1024:.1f}MB), "
                         f"Overall={efficiency_score:.3f}")

            return efficiency_score

        except Exception as e:
            logger.warning(f"Failed to calculate resource efficiency: {e}")
            return 0.5  # Neutral score

    def _calculate_cpu_memory_reward(self, cpu_rate, memory_bytes, replicas, target_cpu_cores=0.5, target_memory_mb=1024):
        """
        Calculate reward based on CPU and memory utilization efficiency.
        Rewards optimal utilization and penalizes under/over-utilization.
        
        Args:
            cpu_rate: Total CPU rate across all replicas
            memory_bytes: Total memory usage across all replicas  
            replicas: Number of replicas
            target_cpu_cores: Target CPU per replica (default: 0.5 cores = 500m)
            target_memory_mb: Target memory per replica (default: 1024MB = 1Gi)
        """
        validated_replicas = self._validate_replicas(replicas, "in CPU/memory reward calculation")
        
        # Calculate per-replica utilization
        cpu_per_replica = cpu_rate / validated_replicas
        memory_per_replica = memory_bytes / validated_replicas
        memory_mb_per_replica = memory_per_replica / (1024 * 1024)
        
        # Calculate utilization efficiency
        cpu_efficiency = min(cpu_per_replica / target_cpu_cores, 1.0) if target_cpu_cores > 0 else 0.0
        memory_efficiency = min(memory_mb_per_replica / target_memory_mb, 1.0) if target_memory_mb > 0 else 0.0
        
        # Calculate penalties for exceeding targets
        cpu_penalty = max(0, (cpu_per_replica - target_cpu_cores) / target_cpu_cores) * 2.0
        memory_penalty = max(0, (memory_mb_per_replica - target_memory_mb) / target_memory_mb) * 1.5
        
        # Calculate penalties for severe under-utilization
        cpu_waste_penalty = max(0, (target_cpu_cores * 0.3 - cpu_per_replica) / target_cpu_cores) * 0.5
        memory_waste_penalty = max(0, (target_memory_mb * 0.3 - memory_mb_per_replica) / target_memory_mb) * 0.5
        
        # Combined reward
        efficiency_reward = (cpu_efficiency * 0.6 + memory_efficiency * 0.4)
        total_penalty = cpu_penalty + memory_penalty + cpu_waste_penalty + memory_waste_penalty
        
        reward = efficiency_reward - total_penalty
        
        # Clamp reward to reasonable bounds
        reward = max(-2.0, min(2.0, reward))
        
        logger.debug(f"CPU/Memory reward: cpu_eff={cpu_efficiency:.3f}, mem_eff={memory_efficiency:.3f}, "
                    f"penalties={total_penalty:.3f}, final_reward={reward:.3f}")
        
        return reward

    def calculate_reward(self,
                         action: str,
                         current_metrics: Dict[str, float],
                         current_replicas: int,
                         desired_replicas: int,
                         forecast_result: Optional[Dict[str, Any]] = None,
                         deployment_info: Optional[Dict[str, Any]] = None) -> float:
        """Calculate reward considering current state and forecast."""
        try:
            # Validate and clean metrics first
            validated_metrics = self._validate_metrics(current_metrics)
            validated_current_replicas = self._validate_replicas(current_replicas, "in reward calculation")
            validated_desired_replicas = self._validate_replicas(desired_replicas, "for desired replicas")

            # Get load classification for emergency detection
            load_class = self._classify_load(validated_metrics, validated_current_replicas, deployment_info)

            # Base reward from current state
            current_reward = self._calculate_current_state_reward(
                action, validated_metrics, validated_current_replicas, validated_desired_replicas, deployment_info
            )

            # CPU and Memory efficiency reward
            cpu_rate = validated_metrics.get("process_cpu_seconds_total_rate", 0.0)
            memory_bytes = validated_metrics.get("process_resident_memory_bytes", 0.0)
            cpu_memory_reward = self._calculate_cpu_memory_reward(
                cpu_rate, memory_bytes, validated_current_replicas,
                self.efficiency_targets["default_cpu_cores"], 
                self.efficiency_targets["default_memory_mb"]
            )

            # Stability analysis
            stability_reward = self._calculate_stability_penalty(
                action, validated_metrics, validated_current_replicas, load_class
            )

            # Forecast-based reward adjustment with confidence threshold from config
            forecast_reward = 0.0
            forecast_confidence = 0.0
            logger.debug(f"Forecast result in reward calculation: {forecast_result}")
            if forecast_result:
                logger.debug(f"Forecast confidence: {forecast_result.get('confidence', None)} (threshold: {self.forecast_confidence_threshold})")
            if forecast_result and forecast_result.get("confidence", 0) > self.forecast_confidence_threshold:
                forecast_reward = self._calculate_forecast_reward(
                    action, validated_metrics, forecast_result, validated_current_replicas, validated_desired_replicas
                )
                forecast_confidence = forecast_result.get("confidence", 0)
                logger.debug(f"Using forecast reward: confidence={forecast_confidence:.3f} > threshold={self.forecast_confidence_threshold}")
            else:
                logger.debug(f"Skipping forecast reward: confidence={forecast_result.get('confidence', 0) if forecast_result else 0:.3f} <= threshold={self.forecast_confidence_threshold}")

            # Use configurable weights for current and forecast rewards with safe fallbacks
            try:
                current_weight = getattr(config.reward, 'current_weight', 0.5)
                forecast_weight = getattr(config.reward, 'forecast_weight', 0.5)
            except AttributeError:
                logger.warning("Config.reward not accessible, using default weights")
                current_weight = 0.5
                forecast_weight = 0.5
            
            logger.debug(f"Reward weights: current={current_weight}, forecast={forecast_weight}")

            # Combined reward 
            total_reward = (
                current_reward * current_weight + 
                forecast_reward * forecast_weight + 
                cpu_memory_reward * 0.4 + 
                stability_reward * 0.2
            )

            logger.info(f"Reward calculation: current={current_reward:.2f}, forecast={forecast_reward:.2f}, "
                       f"cpu_memory={cpu_memory_reward:.2f}, stability={stability_reward:.2f}, total={total_reward:.2f}")
            
            # Special logging for emergency situations
            if load_class == "EMERGENCY":
                logger.error(f"🚨 EMERGENCY REWARD BREAKDOWN: action={action}, load_class={load_class}, total_reward={total_reward:.2f}")
                if total_reward < 5.0:
                    logger.error(f"🚨 WARNING: Emergency reward {total_reward:.2f} may be too low! DQN might ignore emergency scaling!")
                else:
                    logger.error(f"🚨 GOOD: Emergency reward {total_reward:.2f} should force emergency scaling!")

            return total_reward

        except Exception as e:
            logger.error(f"Failed to calculate reward: {e}")
            return 0.0

    def _calculate_current_state_reward(self,
                                        action: str,
                                        current_metrics: Dict[str, float],
                                        current_replicas: int,
                                        desired_replicas: int,
                                        deployment_info: Optional[Dict[str, Any]] = None) -> float:
        """Calculate reward based on current system state."""
        # More aggressive reward matrix for better scaling responsiveness
        reward_matrix = {
            "EMERGENCY": {"scale_up": 10.0, "keep_same": -10.0, "scale_down": -20.0},  # EMERGENCY: MUST scale up immediately!
            "ZERO": {"scale_up": -1.0, "keep_same": -0.5, "scale_down": 2.0},       # Strong scale_down incentive for zero load
            "LOW": {"scale_up": -0.5, "keep_same": 0.3, "scale_down": 1.2},        # Prefer scale_down for low load
            "MEDIUM": {"scale_up": 1.2, "keep_same": -0.2, "scale_down": -1.0},    # More aggressive scale_up for medium
            "HIGH": {"scale_up": 2.0, "keep_same": -1.0, "scale_down": -2.0}       # Strong scale_up for high load
        }
        
        # Use a simple load classification for reward matrix selection
        load_class = self._classify_load(current_metrics, current_replicas, deployment_info)
        base_reward = reward_matrix.get(load_class, {}).get(action, 0.0)
        
        # Log emergency detection clearly
        if load_class == "EMERGENCY":
            logger.error(f"🚨🚨🚨 EMERGENCY DETECTED: Unhealthy pods with zero metrics - ACTION REQUIRED! 🚨🚨🚨")
            logger.error(f"🚨 Current action: {action}, Base reward: {base_reward}")
        
        # Initialize bonus variables
        scale_down_bonus = 0.0
        emergency_bonus = 0.0
        
        # Penalties for invalid replica counts
        if desired_replicas < 1:
            base_reward -= 5.0
        if current_replicas >= 1 and desired_replicas >= 1:
            base_reward += 0.2
            
        # Resource efficiency considerations
        efficiency_penalty = self._calculate_efficiency_penalty(
            current_metrics, current_replicas, desired_replicas, deployment_info
        )
        waste_penalty = self._calculate_waste_penalty(
            current_metrics, current_replicas, desired_replicas
        )
        
        # Resource efficiency bonus
        resource_efficiency = self._calculate_resource_efficiency(current_metrics, current_replicas, deployment_info)
        resource_bonus = resource_efficiency * 0.3
        
        # SCALE-DOWN BONUS: Extra reward for scaling down during low load
        scale_down_bonus = 0.0
        if action == "scale_down" and load_class in ["ZERO", "LOW"]:
            # Get CPU and memory utilization for bonus calculation
            cpu_rate = current_metrics.get("process_cpu_seconds_total_rate", 0.0)
            memory_bytes = current_metrics.get("process_resident_memory_bytes", 0.0)
            cpu_per_replica = cpu_rate / current_replicas
            memory_per_replica = memory_bytes / current_replicas
            
            # Calculate how underutilized the system is
            cpu_underutil = max(0, 0.3 - cpu_per_replica) / 0.3  # 0.3 cores = 60% of 500m limit
            memory_underutil = max(0, (400 * 1024 * 1024) - memory_per_replica) / (400 * 1024 * 1024)  # 400MB baseline
            
            underutil_score = (cpu_underutil + memory_underutil) / 2
            scale_down_bonus = underutil_score * 0.8  # Up to +0.8 bonus for scaling down when very underutilized
            
            if scale_down_bonus > 0.1:
                logger.info(f"Scale-down bonus: +{scale_down_bonus:.2f} for scaling down during {load_class} load")
        
        # EMERGENCY OVERRIDE: If this is an emergency, heavily override all other considerations
        emergency_bonus = 0.0
        if load_class == "EMERGENCY":
            if action == "scale_up":
                emergency_bonus = 20.0  # Massive bonus for scaling up during emergency
                logger.warning(f"🚨 EMERGENCY SCALE-UP BONUS: +{emergency_bonus} for scaling up during pod health emergency!")
            elif action == "keep_same":
                emergency_bonus = -15.0  # Heavy penalty for not acting during emergency
                logger.warning(f"🚨 EMERGENCY PENALTY: {emergency_bonus} for not scaling up during pod health emergency!")
            elif action == "scale_down":
                emergency_bonus = -25.0  # Severe penalty for scaling down during emergency
                logger.warning(f"🚨 EMERGENCY SEVERE PENALTY: {emergency_bonus} for scaling down during pod health emergency!")
        
        total_reward = base_reward + efficiency_penalty + waste_penalty + resource_bonus + scale_down_bonus + emergency_bonus
        
        logger.debug(f"Current state reward: load={load_class}, action={action}, "
                     f"base={base_reward}, efficiency={efficiency_penalty}, "
                     f"waste={waste_penalty}, resource_bonus={resource_bonus:.2f}, "
                     f"scale_down_bonus={scale_down_bonus:.2f}, emergency_bonus={emergency_bonus:.2f}, total={total_reward}")
        
        return total_reward

    def _calculate_forecast_reward(self,
                                   action: str,
                                   current_metrics: Dict[str, float],
                                   forecast_result: Dict[str, Any],
                                   current_replicas: int,
                                   desired_replicas: int) -> float:
        """Calculate reward based on forecast predictions."""
        forecast_summary = forecast_result.get("forecast_summary", {})
        forecast_recommendation = forecast_result.get("recommendation", "keep_same")
        forecast_confidence = forecast_result.get("confidence", 0.0)

        # Reward for following forecast recommendation
        alignment_reward = 0.0
        if action == forecast_recommendation:
            alignment_reward = 1.0 * forecast_confidence
        else:
            alignment_reward = -0.5 * forecast_confidence

        # Proactive scaling rewards
        proactive_reward = self._calculate_proactive_reward(
            action, forecast_summary, current_replicas, desired_replicas
        )

        # Confidence-weighted total
        total_forecast_reward = (alignment_reward + proactive_reward) * forecast_confidence

        logger.debug(f"Forecast reward details: action={action}, forecast_recommendation={forecast_recommendation}, "
                     f"alignment_reward={alignment_reward}, proactive_reward={proactive_reward}, "
                     f"forecast_confidence={forecast_confidence}, total_forecast_reward={total_forecast_reward}")

        return total_forecast_reward

    @staticmethod
    def _calculate_proactive_reward(action: str,
                                    forecast_summary: Dict[str, float],
                                    current_replicas: int,
                                    desired_replicas: int) -> float:
        """Calculate reward for proactive scaling decisions based on CPU and memory trends."""
        cpu_change = forecast_summary.get("cpu_change_percent", 0.0)
        memory_change = forecast_summary.get("memory_change_percent", 0.0)

        # Reward for scaling up before predicted resource increase
        if action == "scale_up" and (cpu_change > 20 or memory_change > 15):
            return 1.5  # High reward for proactive scale-up

        # Reward for scaling down before predicted resource decrease
        elif action == "scale_down" and (cpu_change < -20 or memory_change < -15):
            return 1.5  # High reward for proactive scale-down

        # Penalty for scaling against predicted trend
        elif action == "scale_up" and (cpu_change < -10 and memory_change < -10):
            return -1.0  # Penalty for scaling up when resources will decrease
        elif action == "scale_down" and (cpu_change > 10 and memory_change > 10):
            return -1.0  # Penalty for scaling down when resources will increase

        return 0.0

    def _classify_load(self, metrics: Dict[str, float], replicas: int, deployment_info: Optional[Dict[str, Any]] = None) -> str:
        """Classify current load using CPU and memory metrics only."""
        try:
            validated_replicas = self._validate_replicas(replicas, "in load classification")
            cpu_rate = metrics.get("process_cpu_seconds_total_rate", 0.0)
            memory_bytes = metrics.get("process_resident_memory_bytes", 0.0)

            cpu_per_replica = cpu_rate / validated_replicas
            memory_per_replica = memory_bytes / validated_replicas

            # CRITICAL: Check if we have healthy pods by looking at pod readiness
            if deployment_info and deployment_info.get('ready_replicas', 0) == 0:
                logger.warning("🚨 CRITICAL: No healthy pods available - emergency scale up needed!")
                return "EMERGENCY"  # New classification for pod health issues
            
            # CRITICAL: Check if all metrics are truly zero which indicates unhealthy pods
            if cpu_rate == 0.0 and memory_bytes == 0.0:
                logger.warning("🚨 CRITICAL: All metrics are zero - unhealthy/crashed pods detected!")
                return "EMERGENCY"  # This is a pod health emergency, not low load
            
            # Check for very low load (near zero) - More realistic thresholds for healthy pods
            if cpu_rate < 0.01 and memory_bytes < (50 * 1024 * 1024):  # < 0.01 CPU cores and < 50MB
                logger.info(f"💤 VERY LOW WORKLOAD: CPU={cpu_rate:.3f}, Memory={memory_bytes/1024/1024:.1f}MB - minimal processing demand")
                return "ZERO"

            if not (deployment_info and deployment_info.get('resource_limits')):
                # Fallback to absolute thresholds when no deployment limits available
                logger.warning("No deployment resource limits available; using absolute thresholds for load classification.")
                
                # CRITICAL: Check for unhealthy pods first
                if cpu_per_replica == 0.0 and memory_per_replica == 0.0:
                    logger.warning("🚨 CRITICAL: Zero metrics per replica - unhealthy pods detected!")
                    return "EMERGENCY"
                
                # Use thresholds based on default Kubernetes resource configuration
                # CPU: 500m limit, Memory: 1Gi limit - More aggressive scaling
                if cpu_per_replica > 0.3 or memory_per_replica > (600 * 1024 * 1024):  # 60% of 1Gi
                    return "HIGH"
                elif cpu_per_replica > 0.15 or memory_per_replica > (300 * 1024 * 1024):  # 30% of 1Gi  
                    return "MEDIUM"
                else:
                    return "LOW"

            resource_limits = deployment_info['resource_limits']
            cpu_limit_total = resource_limits.get('cpu', 0.0)
            memory_limit_total = resource_limits.get('memory', 0)
            
            if cpu_limit_total <= 0 or memory_limit_total <= 0:
                logger.warning("Deployment resource limits missing or zero; using absolute thresholds.")
                
                # CRITICAL: Check for unhealthy pods first
                if cpu_per_replica == 0.0 and memory_per_replica == 0.0:
                    logger.warning("🚨 CRITICAL: Zero metrics per replica - unhealthy pods detected!")
                    return "EMERGENCY"
                
                # Fallback to absolute thresholds based on 500m CPU / 1Gi memory - More aggressive
                if cpu_per_replica > 0.3 or memory_per_replica > (600 * 1024 * 1024):  # 60% of limits
                    return "HIGH"
                elif cpu_per_replica > 0.15 or memory_per_replica > (300 * 1024 * 1024):  # 30% of limits
                    return "MEDIUM"
                else:
                    return "LOW"

            cpu_limit_per_replica = cpu_limit_total / validated_replicas
            memory_limit_per_replica = memory_limit_total / validated_replicas

            # Utilizations (clamped to [0,1])
            cpu_util = min(cpu_per_replica / cpu_limit_per_replica, 1.0) if cpu_limit_per_replica > 0 else 0.0
            memory_util = min(memory_per_replica / memory_limit_per_replica, 1.0) if memory_limit_per_replica > 0 else 0.0

            # CPU-first load classification for better scaling sensitivity
            # If CPU is high, immediately classify as high regardless of memory
            if cpu_util > 0.7:  # 70% CPU utilization
                logger.info(f"Classified as HIGH load: cpu_util={cpu_util:.2f} > 0.7 (CPU-first rule)")
                return "HIGH"
            elif cpu_util > 0.45:  # 45% CPU utilization
                logger.info(f"Classified as MEDIUM load: cpu_util={cpu_util:.2f} > 0.45 (CPU-first rule)")
                return "MEDIUM"
            
            # For lower CPU, use composite score but with adjusted thresholds
            composite_score = (cpu_util * 0.8) + (memory_util * 0.2)  # Increased CPU weight
            
            logger.info(f"Load analysis: cpu_util={cpu_util:.2f}, memory_util={memory_util:.2f}, composite_score={composite_score:.2f}")

            if composite_score > 0.6:  # Lowered from 0.7
                logger.info(f"Classified as HIGH load: composite_score={composite_score:.2f}")
                return "HIGH"
            elif composite_score > 0.3:  # Lowered from 0.4
                logger.info(f"Classified as MEDIUM load: composite_score={composite_score:.2f}")
                return "MEDIUM"
            else:
                logger.info(f"Classified as LOW load: composite_score={composite_score:.2f}")
                return "LOW"
                
        except Exception as e:
            logger.warning(f"Failed to classify load: {e}")
            return "LOW"  # Default to low load on error

    def _calculate_efficiency_penalty(self, metrics: Dict[str, float],
                                      current_replicas: int,
                                      desired_replicas: int,
                                      deployment_info: Optional[Dict[str, Any]] = None) -> float:
        """Calculate penalty with proper resource limit validation."""
        # Validate replicas
        validated_current_replicas = self._validate_replicas(current_replicas, "in efficiency penalty calculation")

        # Penalty for over-scaling
        if desired_replicas > validated_current_replicas + 2:
            return -0.5  # Penalty for scaling up too aggressively

        # Penalty for under-scaling during high load
        cpu_rate = metrics.get("process_cpu_seconds_total_rate", 0.0)
        cpu_per_replica = cpu_rate / validated_current_replicas

        # Proper validation of deployment CPU limits
        cpu_limit_per_replica = None
        if deployment_info and deployment_info.get('resource_limits'):
            resource_limits = deployment_info['resource_limits']
            cpu_limit_total = resource_limits.get('cpu', 0.0)

            if cpu_limit_total > 0:
                cpu_limit_per_replica = cpu_limit_total / validated_current_replicas
                cpu_utilization = cpu_per_replica / cpu_limit_per_replica
                high_cpu_threshold = 0.6  # 60% of limit (more aggressive)
                logger.debug(
                    f"Using deployment CPU limit: {cpu_utilization:.3f} utilization ({cpu_per_replica:.3f}/{cpu_limit_per_replica:.3f})")
            else:
                logger.warning("⚠️ Invalid CPU limit in deployment_info, using absolute threshold")
                cpu_utilization = cpu_per_replica
                high_cpu_threshold = 0.3  # 60% of 500m default limit (more aggressive)
        else:
            logger.debug("No deployment limits available, using absolute CPU threshold")
            cpu_utilization = cpu_per_replica
            high_cpu_threshold = 0.3  # 60% of 500m default limit (more aggressive)

        if cpu_utilization > high_cpu_threshold and desired_replicas <= validated_current_replicas:
            logger.info(
                f"Efficiency penalty: High CPU utilization {cpu_utilization:.3f} > {high_cpu_threshold}, not scaling up")
            return -1.0  # Penalty for not scaling during very high CPU

        return 0.0

    def _calculate_waste_penalty(self, metrics: Dict[str, float],
                                 current_replicas: int,
                                 desired_replicas: int) -> float:
        """Calculate waste penalty with proper replica validation."""
        # Validate replicas
        validated_current_replicas = self._validate_replicas(current_replicas, "in waste penalty calculation")

        # Penalty for maintaining too many replicas during low load
        cpu_rate = metrics.get("process_cpu_seconds_total_rate", 0.0)
        memory_bytes = metrics.get("process_resident_memory_bytes", 0.0)

        cpu_per_replica = cpu_rate / validated_current_replicas
        memory_per_replica = memory_bytes / validated_current_replicas

        # More aggressive waste penalty that applies to all replica counts
        if (cpu_per_replica < 0.05 and memory_per_replica < (200 * 1024 * 1024)):  # Very low usage
            # Scale penalty based on replica count - more replicas = bigger waste
            if validated_current_replicas > 3:
                waste_multiplier = 1.5  # Heavy penalty for >3 replicas
            elif validated_current_replicas > 2:
                waste_multiplier = 1.2  # Moderate penalty for 3 replicas
            elif validated_current_replicas > 1:
                waste_multiplier = 0.8  # Light penalty even for 2 replicas
            else:
                waste_multiplier = 0.0  # No penalty for single replica
            
            if desired_replicas >= validated_current_replicas and waste_multiplier > 0:
                penalty = -0.6 * waste_multiplier
                logger.info(f"Waste penalty: Very low usage with {validated_current_replicas} replicas → penalty {penalty:.2f}")
                return penalty

        return 0.0

    def get_reward_explanation(self,
                               action: str,
                               current_metrics: Dict[str, float],
                               current_replicas: int,
                               desired_replicas: int,
                               forecast_result: Optional[Dict[str, Any]] = None) -> str:
        """Get human-readable explanation of reward calculation."""
        # Validate metrics and replicas in explanation
        validated_metrics = self._validate_metrics(current_metrics)
        validated_current_replicas = self._validate_replicas(current_replicas, "in reward explanation")
        validated_desired_replicas = self._validate_replicas(desired_replicas, "for desired replicas in explanation")

        load_class = self._classify_load(validated_metrics, validated_current_replicas)

        explanation = f"Action: {action}, Replicas: {validated_current_replicas} → {validated_desired_replicas}, Load: {load_class}"

        if forecast_result:
            forecast_rec = forecast_result.get("recommendation", "unknown")
            forecast_conf = forecast_result.get("confidence", 0.0)
            explanation += f", Forecast: {forecast_rec} (conf: {forecast_conf:.2f})"

        return explanation