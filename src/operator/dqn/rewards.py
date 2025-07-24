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
        # Efficiency targets for resource utilization
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

    def calculate(self,
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

            # Base reward from current state (includes CPU/memory efficiency)
            current_reward = self._calculate_current_state_reward(
                action, validated_metrics, validated_current_replicas, validated_desired_replicas, deployment_info
            )

            # Forecast-based reward adjustment (always use forecast if available)
            forecast_reward = 0.0
            if forecast_result and forecast_result.get("predicted_metrics"):
                forecast_reward = self._calculate_forecast_reward(
                    action, validated_metrics, forecast_result, validated_current_replicas, validated_desired_replicas
                )
                logger.debug(f"Using forecast reward (no confidence threshold)")
            else:
                logger.debug(f"No forecast available for reward calculation")

            # Use configurable weights for current and forecast rewards with safe fallbacks
            try:
                current_weight = getattr(config.reward, 'current_weight', 0.7)  # Increased since it includes more factors
                forecast_weight = getattr(config.reward, 'forecast_weight', 0.3)
            except AttributeError:
                logger.warning("Config.reward not accessible, using default weights")
                current_weight = 0.7
                forecast_weight = 0.3
            
            logger.debug(f"Reward weights: current={current_weight}, forecast={forecast_weight}")

            # Combined reward (simplified to just current and forecast)
            total_reward = (
                current_reward * current_weight + 
                forecast_reward * forecast_weight
            )

            logger.info(f"Reward calculation: current={current_reward:.2f}, forecast={forecast_reward:.2f}, "
                       f"total={total_reward:.2f}")
            
            # Special logging for emergency situations
            load_class = self._classify_load(validated_metrics, validated_current_replicas, deployment_info)
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
        
        # Validate replicas first
        validated_current_replicas = self._validate_replicas(current_replicas, "in current state reward calculation")
        validated_desired_replicas = self._validate_replicas(desired_replicas, "for desired replicas in current state reward")
        
        # Normalized reward matrix for stable training
        reward_matrix = {
            "EMERGENCY": {"scale_up": 5.0, "keep_same": -3.0, "scale_down": -5.0},   # Emergency: strong scale up
            "ZERO": {"scale_up": -1.0, "keep_same": -0.5, "scale_down": 2.0},        # Zero load: prefer scale down
            "LOW": {"scale_up": -0.5, "keep_same": 0.0, "scale_down": 1.0},          # Low load: mild scale down
            "MEDIUM": {"scale_up": 0.5, "keep_same": 0.0, "scale_down": -0.5},       # Medium load: slight scale up
            "HIGH": {"scale_up": 2.0, "keep_same": -1.0, "scale_down": -2.0}         # High load: strong scale up
        }
        
        # Use load classification for reward matrix selection
        load_class = self._classify_load(current_metrics, validated_current_replicas, deployment_info)
        base_reward = reward_matrix.get(load_class, {}).get(action, 0.0)
        
        # Log emergency detection clearly
        if load_class == "EMERGENCY":
            logger.error(f"🚨🚨🚨 EMERGENCY DETECTED: Unhealthy pods with zero metrics - ACTION REQUIRED! 🚨🚨🚨")
            logger.error(f"🚨 Current action: {action}, Base reward: {base_reward}")
        
        # Initialize bonus variables
        scale_down_bonus = 0.0
        emergency_bonus = 0.0
        excessive_replica_penalty = 0.0
        
        # Get deployment-specific replica constraints
        min_replicas = deployment_info.get('min_replicas', 1) if deployment_info else 1
        max_replicas = deployment_info.get('max_replicas', 50) if deployment_info else 50
        
        # Log deployment constraints for debugging
        if deployment_info:
            deployment_name = deployment_info.get('name', 'unknown')
            deployment_namespace = deployment_info.get('namespace', 'unknown')
            current_deployment_replicas = deployment_info.get('replicas', 'unknown')
            logger.info(f"📋 Deployment constraints: {deployment_namespace}/{deployment_name} - "
                       f"min={min_replicas}, max={max_replicas}, current_spec={current_deployment_replicas}, "
                       f"current_actual={validated_current_replicas}, desired={validated_desired_replicas}")
        else:
            logger.warning("⚠️ No deployment_info available, using default constraints: min=1, max=50")

        # Deployment-aware penalties for invalid replica counts
        if desired_replicas < min_replicas:
            penalty = (min_replicas - desired_replicas) * -2.0  # -2.0 per replica below minimum
            base_reward += penalty
            logger.warning(f"🚨 BELOW MIN REPLICAS: desired={desired_replicas} < min={min_replicas}, penalty={penalty:.1f}")
        elif desired_replicas > max_replicas:
            penalty = (desired_replicas - max_replicas) * -1.0  # -1.0 per replica above maximum
            base_reward += penalty
            logger.warning(f"🚨 ABOVE MAX REPLICAS: desired={desired_replicas} > max={max_replicas}, penalty={penalty:.1f}")
        else:
            base_reward += 0.2  # Small bonus for staying within deployment constraints
            
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
        
        # Deployment-aware scale-down bonus
        scale_down_bonus = 0.0
        if action == "scale_down" and load_class in ["ZERO", "LOW"] and validated_current_replicas > min_replicas:
            # Only give scale-down bonus if we're above minimum replicas
            cpu_rate = current_metrics.get("process_cpu_seconds_total_rate", 0.0)
            memory_bytes = current_metrics.get("process_resident_memory_bytes", 0.0)
            cpu_per_replica = cpu_rate / current_replicas
            memory_per_replica = memory_bytes / current_replicas
            
            # Calculate how underutilized the system is
            cpu_underutil = max(0, 0.3 - cpu_per_replica) / 0.3  # 0.3 cores = 60% of 500m limit
            memory_underutil = max(0, (400 * 1024 * 1024) - memory_per_replica) / (400 * 1024 * 1024)  # 400MB baseline
            
            underutil_score = (cpu_underutil + memory_underutil) / 2
            # Scale bonus based on how far above minimum we are
            distance_from_min = validated_current_replicas - min_replicas
            scale_down_bonus = underutil_score * 0.8 * min(1.0, distance_from_min / 2.0)  # Reduce bonus as we approach minimum
            
            if scale_down_bonus > 0.1:
                logger.info(f"Scale-down bonus: +{scale_down_bonus:.2f} for scaling down during {load_class} load (current={validated_current_replicas}, min={min_replicas})")
        
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
        
        # Deployment-aware replica waste penalty
        excessive_replica_penalty = 0.0
        # Calculate optimal replica count: somewhere between min and a reasonable upper bound
        optimal_replicas = min_replicas + 2  # Allow 2 replicas above minimum as reasonable
        
        logger.debug(f"🎯 Optimal replica calculation: min={min_replicas} + 2 = {optimal_replicas} (current={validated_current_replicas})")
        
        if validated_current_replicas > optimal_replicas:  # More than optimal
            cpu_rate = current_metrics.get("process_cpu_seconds_total_rate", 0.0)
            cpu_per_replica = cpu_rate / validated_current_replicas
            
            if cpu_per_replica < 0.3:  # Low CPU per replica indicates waste
                excess_replicas = min(validated_current_replicas - optimal_replicas, max_replicas - optimal_replicas)
                if action in ["keep_same", "scale_up"]:
                    excessive_replica_penalty = -0.5 * excess_replicas
                    logger.warning(f"🚨 EXCESSIVE REPLICA PENALTY: {validated_current_replicas} replicas > optimal {optimal_replicas} with low CPU → penalty {excessive_replica_penalty:.1f}")
                elif action == "scale_down":
                    excessive_replica_penalty = 0.5 * excess_replicas
                    logger.info(f"🎯 SCALE-DOWN REWARD: +{excessive_replica_penalty:.1f} for reducing {validated_current_replicas} excessive replicas (optimal: {optimal_replicas})")
        
        # CPU and Memory efficiency reward (now part of current state assessment)
        cpu_rate = current_metrics.get("process_cpu_seconds_total_rate", 0.0)
        memory_bytes = current_metrics.get("process_resident_memory_bytes", 0.0)
        cpu_memory_reward = self._calculate_cpu_memory_reward(
            cpu_rate, memory_bytes, validated_current_replicas,
            self.efficiency_targets["default_cpu_cores"], 
            self.efficiency_targets["default_memory_mb"]
        )
        
        total_reward = base_reward + efficiency_penalty + waste_penalty + resource_bonus + scale_down_bonus + emergency_bonus + excessive_replica_penalty + cpu_memory_reward
        
        # Clip total reward to prevent training instability
        total_reward = max(-10.0, min(10.0, total_reward))
        
        logger.debug(f"Current state reward: load={load_class}, action={action}, "
                     f"base={base_reward}, efficiency={efficiency_penalty}, "
                     f"waste={waste_penalty}, resource_bonus={resource_bonus:.2f}, "
                     f"scale_down_bonus={scale_down_bonus:.2f}, emergency_bonus={emergency_bonus:.2f}, "
                     f"excessive_replica_penalty={excessive_replica_penalty:.2f}, cpu_memory={cpu_memory_reward:.2f}, total={total_reward}")
        
        # Summary log of deployment-aware decisions
        if deployment_info:
            logger.info(f"🎯 Deployment-aware reward summary: total={total_reward:.2f}, "
                       f"within_constraints={'✅' if min_replicas <= validated_desired_replicas <= max_replicas else '❌'}, "
                       f"optimal_range={'✅' if validated_current_replicas <= optimal_replicas else f'❌(+{validated_current_replicas - optimal_replicas})'}")

        return total_reward

    def _calculate_forecast_reward(self,
                                   action: str,
                                   current_metrics: Dict[str, float],
                                   forecast_result: Dict[str, Any],
                                   current_replicas: int,
                                   desired_replicas: int) -> float:
        """Calculate reward based on forecast predictions (no confidence weighting)."""
        forecast_summary = forecast_result.get("predicted_metrics", {})
        
        # Proactive reward based on predicted vs current values
        proactive_reward = self._calculate_proactive_reward(
            action, current_metrics, forecast_summary, current_replicas, desired_replicas
        )

        logger.debug(f"Forecast reward details: action={action}, proactive_reward={proactive_reward}")

        return proactive_reward

    @staticmethod
    def _calculate_proactive_reward(action: str,
                                          current_metrics: Dict[str, float],
                                          forecast_metrics: Dict[str, float],
                                          current_replicas: int,
                                          desired_replicas: int) -> float:
        """Calculate proactive reward by analyzing forecast trends against current metrics."""
        try:
            current_cpu = current_metrics.get("process_cpu_seconds_total_rate", 0.0)
            current_memory = current_metrics.get("process_resident_memory_bytes", 0.0)
            
            forecast_cpu = forecast_metrics.get("process_cpu_seconds_total_rate", current_cpu)
            forecast_memory = forecast_metrics.get("process_resident_memory_bytes", current_memory)
            
            # Calculate predicted changes
            cpu_change = forecast_cpu - current_cpu
            memory_change = forecast_memory - current_memory
            
            # Reward for scaling up before predicted resource increase
            if action == "scale_up" and (cpu_change > 0.1 or memory_change > 50*1024*1024):  # >50MB
                return 1.0  # Good proactive scale-up
            
            # Reward for scaling down before predicted resource decrease  
            elif action == "scale_down" and (cpu_change < -0.05 or memory_change < -25*1024*1024):  # <-25MB
                return 1.0  # Good proactive scale-down
            
            # Penalty for scaling against predicted trend
            elif action == "scale_up" and (cpu_change < -0.05 and memory_change < -25*1024*1024):
                return -0.5  # Bad scale-up when resources will decrease
            elif action == "scale_down" and (cpu_change > 0.05 and memory_change > 25*1024*1024):
                return -0.5  # Bad scale-down when resources will increase
            
            return 0.0  # Neutral for other cases
            
        except Exception as e:
            logger.warning(f"Failed to calculate proactive reward: {e}")
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
                
                # MEMORY LEAK DETECTION: High memory + Low CPU = Memory leak, not real load
                if memory_per_replica > (800 * 1024 * 1024) and cpu_per_replica < 0.15:  # >800MB memory but <0.15 CPU
                    logger.warning(f"🚨 MEMORY LEAK DETECTED: memory={memory_per_replica/1024/1024:.0f}MB, cpu={cpu_per_replica:.3f} - treating as LOW load")
                    return "LOW"
                
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
                
                # MEMORY LEAK DETECTION: High memory + Low CPU = Memory leak, not real load
                if memory_per_replica > (800 * 1024 * 1024) and cpu_per_replica < 0.15:  # >800MB memory but <0.15 CPU
                    logger.warning(f"🚨 MEMORY LEAK DETECTED: memory={memory_per_replica/1024/1024:.0f}MB, cpu={cpu_per_replica:.3f} - treating as LOW load")
                    return "LOW"
                
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
            
            # MEMORY LEAK DETECTION: High memory + Low CPU = Memory leak, not real load
            if memory_util > 0.8 and cpu_util < 0.3:  # Memory >80% but CPU <30%
                logger.warning(f"🚨 MEMORY LEAK DETECTED: memory_util={memory_util:.2f}, cpu_util={cpu_util:.2f} - treating as LOW load for scaling")
                return "LOW"  # Memory leak should trigger scale-down, not keep high replicas
            
            # For lower CPU, use composite score but prioritize CPU over memory
            # When CPU is low, memory utilization shouldn't prevent scale-down
            if cpu_util < 0.25:  # Very low CPU
                composite_score = cpu_util * 0.9 + memory_util * 0.1  # Heavily prioritize CPU
                logger.info(f"Low CPU detected: cpu_util={cpu_util:.2f}, using CPU-weighted composite_score={composite_score:.2f}")
            else:
                composite_score = (cpu_util * 0.8) + (memory_util * 0.2)  # Standard weighting
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

        # If high CPU usage and we're not scaling up appropriately
        if cpu_per_replica > 0.4 and desired_replicas <= validated_current_replicas:
            return -0.3  # Penalty for not scaling up during high load

        return 0.0

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

            # Calculate targets (use default targets for simplicity)
            target_cpu = self.efficiency_targets["default_cpu_cores"] * self.efficiency_targets["cpu_utilization"]
            target_memory = self.efficiency_targets["default_memory_mb"] * 1024 * 1024  # Convert MB to bytes

            # Calculate efficiency scores
            if target_cpu > 0:
                cpu_efficiency = 1.0 - abs(cpu_per_replica - target_cpu) / target_cpu
            else:
                cpu_efficiency = 0.5  # Neutral if no target

            if target_memory > 0:
                memory_efficiency = 1.0 - abs(memory_per_replica - target_memory) / target_memory
            else:
                memory_efficiency = 0.5  # Neutral if no target

            # Ensure scores are between 0 and 1
            cpu_efficiency = max(0, min(1, cpu_efficiency))
            memory_efficiency = max(0, min(1, memory_efficiency))

            efficiency_score = (cpu_efficiency * 0.6) + (memory_efficiency * 0.4)

            return efficiency_score

        except Exception as e:
            logger.warning(f"Failed to calculate resource efficiency: {e}")
            return 0.5  # Neutral score

    def _calculate_cpu_memory_reward(self, cpu_rate, memory_bytes, replicas, target_cpu_cores=0.5, target_memory_mb=1024):
        """
        Calculate reward based on CPU and memory utilization efficiency.
        Rewards optimal utilization and penalizes under/over-utilization.
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
        
        return reward

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
            explanation += f", Forecast: {forecast_rec}"

        return explanation