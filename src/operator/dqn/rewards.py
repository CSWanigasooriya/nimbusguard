"""Reward system for DQN that considers current state and forecast predictions.

Available Metrics:
- http_requests_total_process_rate: Total HTTP request rate (req/s)
- process_cpu_seconds_total_rate: CPU usage rate (seconds/second)
- process_resident_memory_bytes: Current memory usage (bytes)
- http_response_size_bytes_sum_rate: Response size throughput rate
- process_open_fds: Open file descriptors count
- kube_pod_container_resource_limits_cpu: CPU resource limits
- http_server_active_connections: Active connections (using request rate)
"""
import logging
from typing import Dict, Any, Optional

from config.settings import load_config

config = load_config()

logger = logging.getLogger(__name__)


class ProactiveRewardCalculator:
    """Calculates rewards for DQN considering both current and predicted future state."""

    def __init__(self):
        # Use attribute access for config, with sensible defaults
        self.cpu_thresholds = getattr(config, 'reward_thresholds', {}).get('cpu', {
            "low": 0.01, "medium": 0.1, "high": 0.5
        })
        self.memory_thresholds = getattr(config, 'reward_thresholds', {}).get('memory', {
            "low": 80, "medium": 150, "high": 300
        })  # MB
        self.request_thresholds = getattr(config, 'reward_thresholds', {}).get('request', {
            "low": 0.2, "medium": 1.0, "high": 5.0
        })  # req/s total

        reward_config = getattr(config, 'reward_config', {})
        self.forecast_confidence_threshold = reward_config.get(
            "forecast_confidence_threshold", 0.3
        )

        self.efficiency_targets = reward_config.get('efficiency_targets', {
            "cpu_utilization": 0.70,
            "memory_utilization": 0.80,
            "default_memory_mb": 400
        })

        self.recent_actions = []
        self.max_action_history = reward_config.get('max_action_history', 10)
        self.recent_load_classes = []
        self.max_load_history = reward_config.get('max_load_history', 5)

        logger.info(f"Initialized ProactiveRewardCalculator with configurable thresholds: "
                    f"CPU={self.cpu_thresholds}, Memory={self.memory_thresholds}, "
                    f"Request={self.request_thresholds}, ForecastConf={self.forecast_confidence_threshold}")

    @staticmethod
    def _validate_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
        """Validate that required metrics are present and handle missing ones."""
        required_metrics = [
            "http_requests_total_process_rate",
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

        # Optional metrics with defaults
        optional_defaults = {
            "http_response_size_bytes_sum_rate": 0.0,
            "process_open_fds": 0.0,
            "kube_pod_container_resource_limits_cpu": 0.0,
            "http_server_active_connections": 0.0
        }

        for metric, default_value in optional_defaults.items():
            if metric not in validated_metrics:
                validated_metrics[metric] = default_value

        return validated_metrics

    @staticmethod
    def _get_context_aware_weights(load_class: str, forecast_confidence: float) -> tuple:
        """IoT-inspired: Adjust reward weights based on system context."""
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
        """FIXED: Validate replica count and handle edge cases explicitly."""
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

        # 2. SMART EXCESSIVE SCALING ANALYSIS
        if len(self.recent_actions) >= 3:
            recent_3 = self.recent_actions[-3:]

            # Check for excessive scaling in same direction
            if all(a == "scale_up" for a in recent_3) and action == "scale_up":
                if load_change_detected or current_load_class == "HIGH":
                    logger.info(
                        f"✅ Justified sustained scale-up ({'load change' if load_change_detected else 'HIGH load'}): {recent_3} → {action}")
                    stability_score += 0.15  # Reward for sustained response to high load or load changes
                else:
                    logger.warning(
                        f"⚠️ Excessive scale-up ({current_load_class or 'unknown'} load): {recent_3} → {action}")
                    stability_score -= 0.15  # Reduced penalty (was 0.2)

            elif all(a == "scale_down" for a in recent_3) and action == "scale_down":
                if load_change_detected or current_load_class == "LOW":
                    logger.info(
                        f"✅ Justified sustained scale-down ({'load change' if load_change_detected else 'LOW load'}): {recent_3} → {action}")
                    stability_score += 0.15  # Reward for sustained response to low load or load changes
                else:
                    logger.warning(
                        f"⚠️ Excessive scale-down ({current_load_class or 'unknown'} load): {recent_3} → {action}")
                    stability_score -= 0.15  # Reduced penalty (was 0.2)

        # 3. CONSISTENCY REWARDS (Positive rewards for stable behavior)
        if len(self.recent_actions) >= 2:
            last_action = self.recent_actions[-1]

            # Reward for maintaining stability with keep_same
            if last_action == "keep_same" and action == "keep_same":
                logger.info(f"✅ Stability maintained: {last_action} → {action}")
                stability_score += 0.2

            # Reward for gradual scaling (scale → keep_same)
            elif ((last_action == "scale_up" and action == "keep_same") or
                  (last_action == "scale_down" and action == "keep_same")):
                logger.info(f"✅ Gradual scaling: {last_action} → {action}")
                stability_score += 0.15

            # Small reward for not oscillating
            elif not ((last_action == "scale_up" and action == "scale_down") or
                      (last_action == "scale_down" and action == "scale_up")):
                stability_score += 0.05  # Small reward for non-oscillating behavior

        # 4. PATTERN ANALYSIS REWARDS (Long-term stability)
        if len(self.recent_actions) >= 4:
            recent_4 = self.recent_actions[-4:]

            # Reward for stable patterns
            if recent_4.count("keep_same") >= 3:
                logger.info(f"✅ Long-term stability: {recent_4}")
                stability_score += 0.25

            # Reward for controlled scaling patterns (scale → keep → scale → keep)
            controlled_patterns = [
                ["scale_up", "keep_same", "scale_up", "keep_same"],
                ["scale_down", "keep_same", "scale_down", "keep_same"]
            ]
            for pattern in controlled_patterns:
                if recent_4 == pattern[:3] and action == pattern[3]:
                    logger.info(f"✅ Controlled scaling pattern: {recent_4} → {action}")
                    stability_score += 0.2

        # 5. ADAPTIVE SCALING REWARDS (Context-aware stability)
        if len(self.recent_actions) >= 3:
            recent_3 = self.recent_actions[-3:]

            # Reward for stopping excessive scaling
            if all(a == "scale_up" for a in recent_3) and action in ["keep_same", "scale_down"]:
                logger.info(f"✅ Stopped excessive scale-up: {recent_3} → {action}")
                stability_score += 0.3
            elif all(a == "scale_down" for a in recent_3) and action in ["keep_same", "scale_up"]:
                logger.info(f"✅ Stopped excessive scale-down: {recent_3} → {action}")
                stability_score += 0.3

        # Clamp the stability score to reasonable bounds
        stability_score = max(-1.0, min(1.0, stability_score))

        logger.info(f"🎯 Stability analysis result: {stability_score:.3f} (action: {action})")
        return stability_score

    @staticmethod
    def _calculate_performance_score(metrics: Dict[str, float]) -> float:
        """Calculate performance score using available throughput metrics."""
        try:
            # Use available request rate metric (all your queries return the same request rate)
            request_rate = metrics.get("http_requests_total_process_rate", 0.0)  # Request throughput (req/s)

            # Since duration sum isn't actually available (your query returns request rate),
            # use throughput as the primary performance indicator
            throughput_score = min(1.0, request_rate / 10.0)  # Normalize to 0-1, 10 req/s = max score

            # Optional: Use response size rate as additional performance indicator if available
            response_size_rate = metrics.get("http_response_size_bytes_sum_rate", 0.0)
            if response_size_rate > 0:
                # Higher response size rate indicates more data throughput
                size_score = min(1.0, response_size_rate / 1000000.0)  # Normalize to 1MB/s = max score
                performance_score = (throughput_score * 0.8) + (size_score * 0.2)
                logger.debug(f"Performance: throughput={throughput_score:.3f}, size_rate={size_score:.3f}, combined={performance_score:.3f}")
            else:
                performance_score = throughput_score
                logger.debug(f"Performance (throughput only): {performance_score:.3f}")

            return performance_score

        except Exception as e:
            logger.warning(f"Failed to calculate performance score: {e}")
            return 0.5  # Neutral score

    def _calculate_resource_efficiency(self, metrics: Dict[str, float], replicas: int,
                                       deployment_info: Optional[Dict[str, Any]] = None) -> float:
        """FIXED: Calculate resource efficiency with proper validation and fallback handling."""
        try:
            # FIXED: Validate replicas first
            validated_replicas = self._validate_replicas(replicas, "in resource efficiency calculation")

            # Get resource utilization metrics
            cpu_rate = metrics.get("process_cpu_seconds_total_rate", 0.0)
            memory_bytes = metrics.get("process_resident_memory_bytes", 0.0)

            # Calculate per-replica utilization
            cpu_per_replica = cpu_rate / validated_replicas
            memory_per_replica = memory_bytes / validated_replicas

            # FIXED: Proper validation of deployment limits with explicit warnings
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
                target_cpu = self.efficiency_targets["cpu_utilization"]  # Absolute target

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

            # Validate replicas early to prevent issues downstream
            validated_current_replicas = self._validate_replicas(current_replicas, "in reward calculation")
            validated_desired_replicas = self._validate_replicas(desired_replicas, "for desired replicas")

            # Calculate load classification first to avoid variable reference error
            load_class = self._classify_load(validated_metrics, validated_current_replicas)

            # Base reward from current state (pass cached load_class)
            current_reward = self._calculate_current_state_reward(
                action, validated_metrics, validated_current_replicas, validated_desired_replicas, deployment_info, load_class
            )

            # Forecast-based reward adjustment with configurable confidence threshold
            forecast_reward = 0.0
            forecast_confidence = 0.0
            if forecast_result and forecast_result.get("confidence", 0) > self.forecast_confidence_threshold:
                forecast_reward = self._calculate_forecast_reward(
                    action, validated_metrics, forecast_result, validated_current_replicas, validated_desired_replicas
                )
                forecast_confidence = forecast_result.get("confidence", 0)
                logger.debug(f"Using forecast reward: confidence={forecast_confidence:.3f} > threshold={self.forecast_confidence_threshold}")
            else:
                logger.debug(f"Skipping forecast reward: confidence={forecast_result.get('confidence', 0) if forecast_result else 0:.3f} <= threshold={self.forecast_confidence_threshold}")

            # IoT-inspired: Context-aware weight adjustment
            current_weight, forecast_weight = self._get_context_aware_weights(load_class, forecast_confidence)

            # Track this action for stability analysis BEFORE calculating penalty
            self._track_action(action)

            # Context-aware stability analysis that considers current load (pass cached load_class)
            stability_penalty = self._calculate_stability_penalty(action, validated_metrics, validated_current_replicas, load_class)

            # Combined reward with adaptive weights and stability consideration
            total_reward = (current_reward * current_weight) + (forecast_reward * forecast_weight) + stability_penalty

            logger.info(f"Reward calculation: current={current_reward:.2f}, "
                        f"forecast={forecast_reward:.2f}, stability={stability_penalty:.2f}, "
                        f"total={total_reward:.2f}, weights=({current_weight:.1f},{forecast_weight:.1f}), load={load_class}")

            return total_reward

        except Exception as e:
            logger.error(f"Failed to calculate reward: {e}")
            return 0.0

    def _calculate_current_state_reward(self,
                                        action: str,
                                        current_metrics: Dict[str, float],
                                        current_replicas: int,
                                        desired_replicas: int,
                                        deployment_info: Optional[Dict[str, Any]] = None,
                                        load_class: str = None) -> float:
        """Calculate reward based on current system state."""
        # Use cached load classification or calculate if not provided
        if load_class is None:
            load_class = self._classify_load(current_metrics, current_replicas)

        # BALANCED reward matrix that gives fair consideration to all three actions
        reward_matrix = {
            "ZERO": {"scale_up": 2.0, "keep_same": -1.0, "scale_down": -2.0},
            # Strong preference for scale_up when no workload (need capacity)
            "LOW": {"scale_up": -0.3, "keep_same": 0.8, "scale_down": 0.3},
            # Slight preference for stability, but allow scale_down
            "MEDIUM": {"scale_up": 0.5, "keep_same": 0.2, "scale_down": -0.5},  # Moderate preference for scale_up
            "HIGH": {"scale_up": 1.5, "keep_same": -0.5, "scale_down": -1.5}
            # Strong preference for scale_up for high load
        }

        base_reward = reward_matrix.get(load_class, {}).get(action, 0.0)

        # Additional penalty for scaling below minimum viable replicas
        if desired_replicas < 1:
            base_reward -= 5.0  # Strong penalty for going below 1 replica

        # Bonus for maintaining baseline availability (at least 1 replica)
        if current_replicas >= 1 and desired_replicas >= 1:
            base_reward += 0.2

        # Penalties for inefficient scaling
        efficiency_penalty = self._calculate_efficiency_penalty(
            current_metrics, current_replicas, desired_replicas, deployment_info
        )

        # Resource waste penalty
        waste_penalty = self._calculate_waste_penalty(
            current_metrics, current_replicas, desired_replicas
        )

        # IoT-inspired: Performance-resource balance adjustment
        performance_score = self._calculate_performance_score(current_metrics)
        resource_efficiency = self._calculate_resource_efficiency(current_metrics, current_replicas, deployment_info)

        # Dynamic balance: prioritize performance when it's poor, efficiency when performance is good
        if performance_score < 0.4:  # Poor performance
            perf_resource_bonus = performance_score * 0.5  # Reward improving performance
        else:  # Good performance
            perf_resource_bonus = resource_efficiency * 0.3  # Reward efficiency

        total_reward = base_reward + efficiency_penalty + waste_penalty + perf_resource_bonus

        logger.debug(f"Current state reward: load={load_class}, action={action}, "
                     f"base={base_reward}, efficiency={efficiency_penalty}, "
                     f"waste={waste_penalty}, perf_resource={perf_resource_bonus:.2f}, total={total_reward}")

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

        logger.debug(f"Forecast reward: alignment={alignment_reward}, "
                     f"proactive={proactive_reward}, confidence={forecast_confidence}, "
                     f"total={total_forecast_reward}")

        return total_forecast_reward

    @staticmethod
    def _calculate_proactive_reward(action: str,
                                    forecast_summary: Dict[str, float],
                                    current_replicas: int,
                                    desired_replicas: int) -> float:
        """Calculate reward for proactive scaling decisions."""
        load_trend = forecast_summary.get("overall_load_trend", 0.0)
        cpu_change = forecast_summary.get("cpu_change_percent", 0.0)
        memory_change = forecast_summary.get("memory_change_percent", 0.0)
        request_change = forecast_summary.get("request_change_percent", 0.0)

        # Reward for scaling up before predicted load increase
        if action == "scale_up" and (load_trend > 0.1 or cpu_change > 20 or
                                     memory_change > 15 or request_change > 30):
            return 1.5  # High reward for proactive scale-up

        # Reward for scaling down before predicted load decrease
        elif action == "scale_down" and (load_trend < -0.1 or cpu_change < -20 or
                                         memory_change < -15 or request_change < -30):
            return 1.5  # High reward for proactive scale-down

        # Penalty for scaling against predicted trend
        elif action == "scale_up" and load_trend < -0.1:
            return -1.0  # Penalty for scaling up when load will decrease
        elif action == "scale_down" and load_trend > 0.1:
            return -1.0  # Penalty for scaling down when load will increase

        return 0.0

    def _classify_load(self, metrics: Dict[str, float], replicas: int) -> str:
        """Classify current load using available metrics with replica validation."""
        try:
            # Validate replicas first
            validated_replicas = self._validate_replicas(replicas, "in load classification")

            # Extract available metrics
            cpu_rate = metrics.get("process_cpu_seconds_total_rate", 0.0)
            memory_bytes = metrics.get("process_resident_memory_bytes", 0.0)
            request_rate = metrics.get("http_requests_total_process_rate", 0.0)  # Primary workload indicator

            # Convert to per-replica metrics for analysis
            cpu_per_replica = cpu_rate / validated_replicas
            memory_mb_per_replica = (memory_bytes / validated_replicas) / (1024 * 1024)
            requests_per_replica = request_rate / validated_replicas

            # Log the actual values for debugging
            logger.info(f"Load classification: cpu_per_replica={cpu_per_replica:.4f}, "
                        f"memory_mb_per_replica={memory_mb_per_replica:.1f}, "
                        f"requests_per_replica={requests_per_replica:.4f}, "
                        f"total_request_rate={request_rate:.4f}, replicas={validated_replicas}")

            # CRITICAL: Check for zero workload (no actual requests)
            if request_rate == 0.0:
                logger.info(f"💤 NO WORKLOAD: Request rate is 0.0 - no actual processing demand")
                return "ZERO"

            # Classification logic using configurable thresholds
            # HIGH load conditions - REQUEST RATE IS PRIMARY INDICATOR
            if (request_rate > self.request_thresholds["high"] or
                    cpu_rate > self.cpu_thresholds["high"] or
                    memory_bytes > self.memory_thresholds["high"] * 1024 * 1024):  # Convert MB to bytes
                logger.info(
                    f"Classified as HIGH load: request_rate>{self.request_thresholds['high']}? {request_rate > self.request_thresholds['high']}, "
                    f"cpu_rate>{self.cpu_thresholds['high']}? {cpu_rate > self.cpu_thresholds['high']}, "
                    f"memory>{self.memory_thresholds['high']}MB? {memory_bytes > self.memory_thresholds['high'] * 1024 * 1024}")
                return "HIGH"

            # MEDIUM load conditions
            elif (request_rate > self.request_thresholds["medium"] or
                  cpu_rate > self.cpu_thresholds["medium"] or
                  memory_bytes > self.memory_thresholds["medium"] * 1024 * 1024):  # Convert MB to bytes
                logger.info(
                    f"Classified as MEDIUM load: request_rate>{self.request_thresholds['medium']}? {request_rate > self.request_thresholds['medium']}, "
                    f"cpu_rate>{self.cpu_thresholds['medium']}? {cpu_rate > self.cpu_thresholds['medium']}, "
                    f"memory>{self.memory_thresholds['medium']}MB? {memory_bytes > self.memory_thresholds['medium'] * 1024 * 1024}")
                return "MEDIUM"

            # Low load (default) - has some workload but below medium thresholds
            else:
                logger.info(
                    f"Classified as LOW load: workload={request_rate:.4f} req/s, all metrics below medium thresholds")
                return "LOW"

        except Exception as e:
            logger.warning(f"Failed to classify load: {e}")
            return "LOW"  # Default to low load on error

    def _calculate_efficiency_penalty(self, metrics: Dict[str, float],
                                      current_replicas: int,
                                      desired_replicas: int,
                                      deployment_info: Optional[Dict[str, Any]] = None) -> float:
        """FIXED: Calculate penalty with proper resource limit validation."""
        # FIXED: Validate replicas
        validated_current_replicas = self._validate_replicas(current_replicas, "in efficiency penalty calculation")

        # Penalty for over-scaling
        if desired_replicas > validated_current_replicas + 2:
            return -0.5  # Penalty for scaling up too aggressively

        # Penalty for under-scaling during high load
        cpu_rate = metrics.get("process_cpu_seconds_total_rate", 0.0)
        cpu_per_replica = cpu_rate / validated_current_replicas

        # FIXED: Proper validation of deployment CPU limits
        cpu_limit_per_replica = None
        if deployment_info and deployment_info.get('resource_limits'):
            resource_limits = deployment_info['resource_limits']
            cpu_limit_total = resource_limits.get('cpu', 0.0)

            if cpu_limit_total > 0:
                cpu_limit_per_replica = cpu_limit_total / validated_current_replicas
                cpu_utilization = cpu_per_replica / cpu_limit_per_replica
                high_cpu_threshold = self.efficiency_targets["cpu_utilization"]  # Use configurable threshold
                logger.debug(
                    f"Using deployment CPU limit: {cpu_utilization:.3f} utilization ({cpu_per_replica:.3f}/{cpu_limit_per_replica:.3f})")
            else:
                logger.warning("⚠️ Invalid CPU limit in deployment_info, using absolute threshold")
                cpu_utilization = cpu_per_replica
                high_cpu_threshold = self.efficiency_targets["cpu_utilization"]  # Use configurable threshold
        else:
            logger.debug("No deployment limits available, using absolute CPU threshold")
            cpu_utilization = cpu_per_replica
            high_cpu_threshold = self.efficiency_targets["cpu_utilization"]  # Use configurable threshold

        if cpu_utilization > high_cpu_threshold and desired_replicas <= validated_current_replicas:
            logger.info(
                f"Efficiency penalty: High CPU utilization {cpu_utilization:.3f} > {high_cpu_threshold}, not scaling up")
            return -1.0  # Penalty for not scaling during very high CPU

        return 0.0

    def _calculate_waste_penalty(self, metrics: Dict[str, float],
                                 current_replicas: int,
                                 desired_replicas: int) -> float:
        """FIXED: Calculate waste penalty with proper replica validation."""
        # FIXED: Validate replicas
        validated_current_replicas = self._validate_replicas(current_replicas, "in waste penalty calculation")

        # Penalty for maintaining too many replicas during low load
        cpu_rate = metrics.get("process_cpu_seconds_total_rate", 0.0)
        request_rate = metrics.get("http_requests_total_process_rate", 0.0)  # FIXED: Use correct metric name

        cpu_per_replica = cpu_rate / validated_current_replicas
        requests_per_replica = request_rate / validated_current_replicas

        # If load is very low but maintaining high replica count
        if (cpu_per_replica < 0.05 and requests_per_replica < 1.0 and
                desired_replicas >= validated_current_replicas > 2):
            return -0.8  # Penalty for resource waste

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

        explanation = f"Action: {action}, Load: {load_class}, "
        explanation += f"Replicas: {validated_current_replicas} → {validated_desired_replicas}"

        if forecast_result:
            forecast_rec = forecast_result.get("recommendation", "unknown")
            forecast_conf = forecast_result.get("confidence", 0.0)
            explanation += f", Forecast: {forecast_rec} (conf: {forecast_conf:.2f})"

        return explanation