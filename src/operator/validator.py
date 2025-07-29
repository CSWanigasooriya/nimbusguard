import logging
import os
import json
import requests
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from validation_prompt import VALIDATION_PROMPT

class ScalingValidator:
    """
    Validates scaling actions and deployment states for safety and compliance.
    Provides comprehensive checks before scaling operations are executed.
    Supports both regular validation and optional LLM-based validation.
    """
    
    def __init__(self, config=None):
        """
        Initialize validator with configuration.
        
        Args:
            config: Dictionary with validation parameters
        """
        self.config = config or {}
        
        # Import and use centralized AI configuration
        from config import ai_config
        
        # LLM validation settings from centralized config
        self.enable_llm_validation = ai_config.enable_llm_validation
        self.openai_api_key = ai_config.openai_api_key
        self.ai_model = ai_config.model_name
        self.ai_temperature = ai_config.temperature
        self.mcp_server_url = ai_config.mcp_server_url
        self.enable_exploration_leniency = ai_config.enable_exploration_leniency
        
        # Scaling rate limits
        self.max_scale_up_per_minute = self.config.get('max_scale_up_per_minute', 4)  # Increased for responsiveness
        self.max_scale_down_per_minute = self.config.get('max_scale_down_per_minute', 2)
        self.min_time_between_scales = self.config.get('min_time_between_scales', 15)  # Reduced to 15s for faster reaction
        
        # Resource thresholds
        self.max_cpu_util_for_scale_down = self.config.get('max_cpu_util_for_scale_down', 80.0)
        self.min_cpu_util_for_scale_up = self.config.get('min_cpu_util_for_scale_up', 20.0)  # Set to 20%
        self.min_memory_util_for_scale_up = self.config.get('min_memory_util_for_scale_up', 20.0)  # Reduced to 20%
        self.critical_memory_threshold = self.config.get('critical_memory_threshold', 95.0)
        self.critical_cpu_threshold = self.config.get('critical_cpu_threshold', 90.0)
        
        # Stability requirements
        self.min_stable_period_seconds = self.config.get('min_stable_period_seconds', 60)
        self.max_utilization_variance = self.config.get('max_utilization_variance', 20.0)
        
        # Safety limits
        self.emergency_scale_up_threshold = self.config.get('emergency_scale_up_threshold', 90.0)
        self.force_scale_down_threshold = self.config.get('force_scale_down_threshold', 10.0)
        
        # Log validation mode
        if self.enable_llm_validation:
            if self.openai_api_key:
                logging.info("LLM validation enabled with OpenAI API")
            else:
                logging.warning("LLM validation enabled but OpenAI API key not configured - falling back to regular validation")
                self.enable_llm_validation = False
        else:
            logging.info("Using regular validation (LLM validation disabled)")
        
        # Log exploration leniency configuration
        leniency_status = "enabled" if self.enable_exploration_leniency else "disabled"
        logging.info(f"Exploration leniency: {leniency_status}")
    
    def validate_scaling_action(self, action: int, current_replicas: int, target_replicas: int,
                               min_replicas: int, max_replicas: int, deployment_info: Dict = None, 
                               exploration_mode: bool = False) -> Tuple[bool, str, int]:
        """
        Comprehensive validation of a scaling action for multi-pod environments.
        
        Key validation principles:
        - Trust memory metrics as reported (high usage indicates real problems)
        - Prevent dangerous scale-downs during memory pressure
        - Allow emergency scale-ups when critically needed
        - Rate limit scaling actions to prevent oscillation
        - Validate system stability before scaling
        
        Args:
            action: Scaling action (0=none, 1=scale_up, 2=scale_down)
            current_replicas: Current replica count
            target_replicas: Desired replica count
            min_replicas: Minimum allowed replicas
            max_replicas: Maximum allowed replicas
            deployment_info: Deployment metrics including:
                - current_cpu_util: CPU utilization %
                - current_mem_util: Memory utilization % (trusted as accurate)
                - predicted_mem_util: Predicted memory utilization %
            
        Returns:
            Tuple of (is_valid, reason, adjusted_target)
        """
        # First, always run regular validation for safety
        regular_valid, regular_reason, regular_target = self._validate_regular(
            action, current_replicas, target_replicas, min_replicas, max_replicas, deployment_info, exploration_mode
        )
        
        # If regular validation fails, return immediately
        if not regular_valid:
            return regular_valid, regular_reason, regular_target
        
        # If LLM validation is enabled, run additional LLM check
        if self.enable_llm_validation and self.openai_api_key:
            try:
                llm_valid, llm_reason, llm_target = self._validate_with_llm(
                    action, current_replicas, target_replicas, min_replicas, max_replicas, deployment_info
                )
                
                # LLM validation can override regular validation decision
                if not llm_valid:
                    logging.info(f"LLM validation override: {llm_reason}")
                    return llm_valid, f"LLM Override: {llm_reason}", llm_target
                else:
                    logging.info(f"LLM validation confirmed: {llm_reason}")
                    
            except Exception as e:
                logging.error(f"LLM validation failed, falling back to regular validation: {e}")
                # Continue with regular validation result
        
        return regular_valid, regular_reason, regular_target
    
    def _validate_replica_boundaries(self, target_replicas: int, min_replicas: int, max_replicas: int) -> Tuple[bool, str, int]:
        """Validate replica count against boundaries."""
        if target_replicas < min_replicas:
            return False, f"Target replicas {target_replicas} below minimum {min_replicas}", min_replicas
        
        if target_replicas > max_replicas:
            return False, f"Target replicas {target_replicas} above maximum {max_replicas}", max_replicas
        
        return True, "Boundary validation passed", target_replicas
    
    def _validate_scaling_rate(self, action: int, current_replicas: int, target_replicas: int, 
                              deployment_info: Dict = None) -> Tuple[bool, str]:
        """
        Validate scaling rate limits with intelligent emergency handling.
        
        Allows bypassing rate limits for emergency conditions to ensure responsiveness.
        """
        from state_manager import state
        
        # Check if we have scaling history
        if not hasattr(state, 'scaling_history') or not state.scaling_history:
            return True, "No scaling history to check"
        
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        
        # Count recent scaling actions
        recent_scale_ups = 0
        recent_scale_downs = 0
        last_scaling_time = None
        
        for record in state.scaling_history:
            # Convert ISO string timestamp back to datetime for comparison
            record_time = datetime.fromisoformat(record['timestamp']) if isinstance(record['timestamp'], str) else record['timestamp']
            if record_time > one_minute_ago:
                if record['action'] == 'scale_up':
                    recent_scale_ups += 1
                elif record['action'] == 'scale_down':
                    recent_scale_downs += 1
                
                if last_scaling_time is None or record_time > last_scaling_time:
                    last_scaling_time = record_time
        
        # Check for emergency conditions that should bypass rate limits
        emergency_bypass = False
        if deployment_info and action == 1:  # Scale up only
            current_cpu_util = deployment_info.get('current_cpu_util', 0)
            current_mem_util = deployment_info.get('current_mem_util', 0)
            predicted_mem_util = deployment_info.get('predicted_mem_util', current_mem_util)
            
            # Allow bypassing rate limits for critical resource pressure
            if (current_cpu_util > self.critical_cpu_threshold or 
                current_mem_util > self.critical_memory_threshold or
                predicted_mem_util > self.critical_memory_threshold):
                emergency_bypass = True
                logging.info(f"[VALIDATOR] Emergency bypass: CPU={current_cpu_util:.1f}%, Mem={current_mem_util:.1f}%, PredMem={predicted_mem_util:.1f}%")
        
        # Check rate limits (with emergency bypass for scale-ups)
        if action == 1:  # Scale up
            if recent_scale_ups >= self.max_scale_up_per_minute and not emergency_bypass:
                return False, f"Scale up rate limit exceeded: {recent_scale_ups}/{self.max_scale_up_per_minute} in last minute"
        elif action == 2:  # Scale down
            if recent_scale_downs >= self.max_scale_down_per_minute:
                return False, f"Scale down rate limit exceeded: {recent_scale_downs}/{self.max_scale_down_per_minute} in last minute"
        
        # Check minimum time between scaling actions (with emergency bypass for scale-ups)
        if last_scaling_time:
            time_since_last = (now - last_scaling_time).total_seconds()
            min_time = self.min_time_between_scales
            
            # For scale-ups, be more lenient with timing
            if action == 1:
                # Reduce minimum time for scale-ups to prioritize responsiveness
                min_time = max(10, self.min_time_between_scales * 0.7)  # At least 10s, but 30% less than configured
                
                # Emergency bypass for critical conditions
                if emergency_bypass:
                    min_time = 5  # Allow very quick scale-ups in emergencies
            
            if time_since_last < min_time:
                if emergency_bypass and action == 1:
                    logging.info(f"[VALIDATOR] Emergency override: Scaling up despite {time_since_last:.0f}s < {min_time}s")
                    return True, f"Emergency scale-up allowed (time: {time_since_last:.0f}s)"
                else:
                    return False, f"Minimum time between scales not met: {time_since_last:.0f}s < {min_time}s"
        
        reason = "Rate limit validation passed"
        if emergency_bypass and action == 1:
            reason = "Rate limit validation passed (emergency conditions detected)"
        
        return True, reason
    
    def _validate_resource_constraints(self, action: int, current_replicas: int, target_replicas: int, 
                                     deployment_info: Dict, exploration_mode: bool = False, log_details: bool = True) -> Tuple[bool, str]:
        """Validate against resource utilization constraints."""
        # Handle case where deployment_info is None
        if deployment_info is None:
            logging.warning("[VALIDATOR] No deployment info provided for resource validation - allowing action")
            return True, "No deployment info available for resource validation"
        
        # Get current utilization (would come from metrics)
        current_cpu_util = deployment_info.get('current_cpu_util', 0)
        current_mem_util = deployment_info.get('current_mem_util', 0)
        predicted_mem_util = deployment_info.get('predicted_mem_util', current_mem_util)
        
        # Calculate effective memory utilization (accounting for multiple pods)
        effective_mem_util = self._calculate_effective_memory_utilization(current_mem_util, current_replicas)
        
        max_mem_util = max(effective_mem_util, predicted_mem_util)
        
        # Emergency scale-up logic - trust both memory and CPU metrics and act decisively
        if max_mem_util > self.emergency_scale_up_threshold:
            if action == 1:  # Already scaling up - approve it
                return True, f"Scale-up approved for emergency memory pressure (memory: {max_mem_util:.1f}% > {self.emergency_scale_up_threshold}%)"
            elif action == 2:  # Scale down - absolutely prevent it
                return False, f"Scale-down BLOCKED: Emergency memory pressure detected (memory: {max_mem_util:.1f}% > {self.emergency_scale_up_threshold}%)"
            else:  # No action - allow validation to continue, emergency logic in should_force_action()
                # High memory usage is a real problem that needs addressing
                # Don't reject here, but should_force_action() will recommend emergency scaling
                logging.warning(f"High memory pressure detected but no scaling action proposed (memory: {max_mem_util:.1f}%)")
        
        # Emergency scale-up logic for CPU
        if current_cpu_util > self.critical_cpu_threshold:
            if action == 1:  # Already scaling up - approve it
                return True, f"Scale-up approved for emergency CPU pressure (CPU: {current_cpu_util:.1f}% > {self.critical_cpu_threshold}%)"
            elif action == 2:  # Scale down - absolutely prevent it
                return False, f"Scale-down BLOCKED: Emergency CPU pressure detected (CPU: {current_cpu_util:.1f}% > {self.critical_cpu_threshold}%)"
            else:  # No action - allow validation to continue, emergency logic in should_force_action()
                # High CPU usage is a real problem that needs addressing
                # Don't reject here, but should_force_action() will recommend emergency scaling
                logging.warning(f"High CPU pressure detected but no scaling action proposed (CPU: {current_cpu_util:.1f}%)")
        
        # Prevent scale-down under high load
        if action == 2:  # Scale down
            if current_cpu_util > self.max_cpu_util_for_scale_down:
                return False, f"Cannot scale down: CPU utilization too high ({current_cpu_util:.1f}% > {self.max_cpu_util_for_scale_down}%)"
            
            if current_mem_util > self.max_cpu_util_for_scale_down:
                return False, f"Cannot scale down: Memory utilization too high ({current_mem_util:.1f}% > {self.max_cpu_util_for_scale_down}%)"
        
        # Require minimum utilization for scale-up (more lenient during exploration)
        if action == 1:  # Scale up
            # Check both CPU and memory utilization
            cpu_justified = current_cpu_util >= self.min_cpu_util_for_scale_up
            memory_justified = (current_mem_util >= self.min_memory_util_for_scale_up or 
                               predicted_mem_util >= self.min_memory_util_for_scale_up)
            
            # Scale-up is justified if either CPU OR memory is high enough
            if not cpu_justified and not memory_justified:
                # During exploration, be more lenient to allow learning (if enabled in config)
                if exploration_mode and self.enable_exploration_leniency:
                    logging.info(f"[VALIDATOR] Exploration mode: Allowing scale-up despite low utilization (CPU: {current_cpu_util:.1f}%, Memory: {current_mem_util:.1f}%)")
                    return True, f"Exploration mode: Scale-up allowed for learning (CPU: {current_cpu_util:.1f}%, Memory: {current_mem_util:.1f}%)"
                elif exploration_mode and not self.enable_exploration_leniency:
                    logging.info(f"[VALIDATOR] Exploration detected but leniency disabled - applying strict validation")
                    return False, f"Scale-up not justified: Both CPU ({current_cpu_util:.1f}% < {self.min_cpu_util_for_scale_up}%) and Memory ({current_mem_util:.1f}% < {self.min_memory_util_for_scale_up}%) utilization too low"
                else:
                    return False, f"Scale-up not justified: Both CPU ({current_cpu_util:.1f}% < {self.min_cpu_util_for_scale_up}%) and Memory ({current_mem_util:.1f}% < {self.min_memory_util_for_scale_up}%) utilization too low"
            else:
                # Log which resource justified the scale-up
                if cpu_justified and memory_justified:
                    reason = f"Scale-up justified by both CPU ({current_cpu_util:.1f}% >= {self.min_cpu_util_for_scale_up}%) and Memory ({current_mem_util:.1f}% >= {self.min_memory_util_for_scale_up}%)"
                elif cpu_justified:
                    reason = f"Scale-up justified by CPU utilization ({current_cpu_util:.1f}% >= {self.min_cpu_util_for_scale_up}%)"
                else:  # memory_justified
                    reason = f"Scale-up justified by Memory utilization ({current_mem_util:.1f}% >= {self.min_memory_util_for_scale_up}%)"
                if log_details:
                    logging.info(f"[VALIDATOR] {reason}")
        
        return True, "Resource constraint validation passed"
    
    def _validate_system_stability(self, action: int) -> Tuple[bool, str]:
        """
        Validate system stability before scaling.
        
        Improved logic:
        - Scale-ups are generally safer and should be less restricted
        - Focus on detecting rapid oscillation rather than just mixed actions
        - Allow emergency scale-ups even during instability
        - Consider time gaps between actions
        """
        from state_manager import state
        
        # Check for recent deployment instability
        if hasattr(state, 'scaling_history') and state.scaling_history:
            now = datetime.now()
            two_minutes_ago = now - timedelta(minutes=2)  # Reduced from 5 minutes
            one_minute_ago = now - timedelta(minutes=1)   # Reduced from 2 minutes
            
            recent_actions = []
            very_recent_actions = []
            
            for record in state.scaling_history:
                # Convert ISO string timestamp back to datetime for comparison
                record_time = datetime.fromisoformat(record['timestamp']) if isinstance(record['timestamp'], str) else record['timestamp']
                
                if record_time > two_minutes_ago:  # Changed from 5 minutes
                    recent_actions.append({
                        'action': record['action'],
                        'timestamp': record_time
                    })
                
                if record_time > one_minute_ago:  # Changed from 2 minutes
                    very_recent_actions.append({
                        'action': record['action'],
                        'timestamp': record_time
                    })
            
            # For scale-ups (action == 1), be very permissive
            if action == 1:
                # Only block scale-ups if there's true rapid oscillation in the last 1 minute
                if len(very_recent_actions) >= 3:
                    very_recent_action_types = [a['action'] for a in very_recent_actions]
                    scale_ups_recent = very_recent_action_types.count('scale_up')
                    scale_downs_recent = very_recent_action_types.count('scale_down')
                    
                    # Block only if we have extreme back-and-forth in last 1 minute
                    if scale_ups_recent >= 2 and scale_downs_recent >= 2:
                        return False, f"Rapid oscillation detected: {scale_ups_recent} scale-ups and {scale_downs_recent} scale-downs in last 1 minute"
                
                # Allow more scale-ups in a shorter window (increased from 4 to 6)
                recent_scale_ups = sum(1 for a in recent_actions if a['action'] == 'scale_up')
                if recent_scale_ups >= 6:
                    return False, f"Too many scale-ups recently: {recent_scale_ups} scale-ups in last 2 minutes"
                
                # Otherwise allow scale-up (prioritize responsiveness)
                return True, "Scale-up allowed for responsiveness"
            
            # For scale-downs (action == 2), be more restrictive but with shorter window
            elif action == 2:
                # Check for any oscillation in the last 2 minutes (reduced from 5)
                if len(recent_actions) >= 3:  # Reduced threshold from 4 to 3
                    recent_action_types = [a['action'] for a in recent_actions]
                    scale_ups = recent_action_types.count('scale_up')
                    scale_downs = recent_action_types.count('scale_down')
                    
                    # Only block if there's been recent scale-up activity
                    if scale_ups > 0 and scale_downs >= 2:
                        return False, f"System instability detected: {scale_ups} scale-ups and {scale_downs} scale-downs in last 2 minutes - blocking scale-down"
                
                # Check for too many recent scale-downs (reduced window)
                recent_scale_downs = sum(1 for a in recent_actions if a['action'] == 'scale_down')
                if recent_scale_downs >= 2:  # Reduced from 3 to 2
                    return False, f"Too many scale-downs recently: {recent_scale_downs} scale-downs in last 2 minutes"
            
            # For no action (action == 0), always allow
            else:
                return True, "No action - stability check passed"
        
        return True, "Stability validation passed"
    
    def validate_deployment_health(self, deployment_info: Dict) -> Tuple[bool, str]:
        """
        Validate deployment health before allowing scaling.
        
        Args:
            deployment_info: Deployment information dictionary
            
        Returns:
            Tuple of (is_healthy, reason)
        """
        if not deployment_info:
            return False, "No deployment information provided"
        
        current = deployment_info.get('current_replicas', 0)
        available = deployment_info.get('available_replicas', 0)
        ready = deployment_info.get('ready_replicas', 0)
        
        if available < current:
            return False, f"Not all replicas available: {available}/{current}"
        
        if ready < current:
            return False, f"Not all replicas ready: {ready}/{current}"
        
        # Check for recent failed scaling attempts
        from state_manager import state
        if hasattr(state, 'scaling_history'):
            recent_failures = []
            ten_minutes_ago = datetime.now() - timedelta(minutes=10)
            
            for record in state.scaling_history:
                # Convert ISO string timestamp back to datetime for comparison
                record_time = datetime.fromisoformat(record['timestamp']) if isinstance(record['timestamp'], str) else record['timestamp']
                if (record_time > ten_minutes_ago and 
                    record.get('success') is False):
                    recent_failures.append(record)
            
            if len(recent_failures) >= 3:
                return False, f"Too many recent scaling failures: {len(recent_failures)} in last 10 minutes"
        
        return True, "Deployment is healthy"
    
    def should_force_action(self, current_cpu_util: float, current_mem_util: float, 
                           predicted_mem_util: float, current_replicas: int, 
                           min_replicas: int, max_replicas: int) -> Tuple[Optional[int], str]:
        """
        Determine if a forced scaling action is required for safety.
        
        Args:
            current_cpu_util: Current CPU utilization percentage
            current_mem_util: Current memory utilization percentage
            predicted_mem_util: Predicted memory utilization percentage
            current_replicas: Current replica count
            min_replicas: Minimum allowed replicas
            max_replicas: Maximum allowed replicas
            
        Returns:
            Tuple of (forced_action, reason) where forced_action is None if no force needed
        """
        # Apply multi-pod memory adjustment for force action decisions
        effective_mem_util = self._calculate_effective_memory_utilization(current_mem_util, current_replicas)
        
        max_mem_util = max(effective_mem_util, predicted_mem_util)
        
        # Force scale-up for critical resource usage (memory or CPU)
        if max_mem_util > self.critical_memory_threshold:
            if current_replicas < max_replicas:
                return 1, f"CRITICAL: Force scale-up due to memory pressure ({max_mem_util:.1f}% > {self.critical_memory_threshold}%)"
        
        if current_cpu_util > self.critical_cpu_threshold:
            if current_replicas < max_replicas:
                return 1, f"CRITICAL: Force scale-up due to CPU pressure ({current_cpu_util:.1f}% > {self.critical_cpu_threshold}%)"
        
        # Force scale-up for emergency threshold (less critical but still urgent)
        if max_mem_util > self.emergency_scale_up_threshold:
            if current_replicas < max_replicas:
                return 1, f"EMERGENCY: Force scale-up due to high memory ({max_mem_util:.1f}% > {self.emergency_scale_up_threshold}%)"
        
        # Force scale-down for very low utilization
        if (current_cpu_util < self.force_scale_down_threshold and 
            effective_mem_util < self.force_scale_down_threshold):
            if current_replicas > min_replicas:
                return 2, f"Force scale-down due to very low utilization (CPU: {current_cpu_util:.1f}%, Mem: {effective_mem_util:.1f}% < {self.force_scale_down_threshold}%)"
        
        return None, "No forced action required"
    
    def get_validation_summary(self, action: int, current_replicas: int, target_replicas: int,
                              min_replicas: int, max_replicas: int, deployment_info: Dict = None) -> Dict:
        """
        Get a comprehensive validation summary for analysis.
        
        Returns:
            Dictionary with detailed validation results
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'current_replicas': current_replicas,
            'target_replicas': target_replicas,
            'constraints': {
                'min_replicas': min_replicas,
                'max_replicas': max_replicas
            },
            'validations': {}
        }
        
        # Run all validations
        boundary_valid, boundary_reason, boundary_target = self._validate_replica_boundaries(
            target_replicas, min_replicas, max_replicas
        )
        summary['validations']['boundary'] = {
            'valid': boundary_valid,
            'reason': boundary_reason,
            'adjusted_target': boundary_target
        }
        
        rate_valid, rate_reason = self._validate_scaling_rate(action, current_replicas, target_replicas, deployment_info)
        summary['validations']['rate_limit'] = {
            'valid': rate_valid,
            'reason': rate_reason
        }
        
        if deployment_info:
            resource_valid, resource_reason = self._validate_resource_constraints(
                action, current_replicas, target_replicas, deployment_info, exploration_mode=False, log_details=False
            )
            summary['validations']['resource_constraints'] = {
                'valid': resource_valid,
                'reason': resource_reason
            }
            
            health_valid, health_reason = self.validate_deployment_health(deployment_info)
            summary['validations']['deployment_health'] = {
                'valid': health_valid,
                'reason': health_reason
            }
        
        stability_valid, stability_reason = self._validate_system_stability(action)
        summary['validations']['stability'] = {
            'valid': stability_valid,
            'reason': stability_reason
        }
        
        # Overall validation result
        all_valid = all(v.get('valid', True) for v in summary['validations'].values())
        summary['overall_valid'] = all_valid
        summary['final_target'] = target_replicas if all_valid else current_replicas
        
        return summary
    
    def update_config(self, new_config: Dict):
        """Update validator configuration."""
        self.config.update(new_config)
        
        # Update individual parameters
        for key, value in new_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        logging.info(f"Updated validator config: {new_config}")
    
    def _calculate_effective_memory_utilization(self, current_mem_util: float, current_replicas: int) -> float:
        """
        Calculate effective memory utilization for validation purposes.
        
        Memory utilization is now correctly calculated as:
        (total_memory_usage) / (per_pod_limit × current_replicas) × 100
        
        This accounts for the distributed nature of multi-pod deployments.
        
        Args:
            current_mem_util: Memory utilization percentage (already accounts for multiple pods)
            current_replicas: Current number of replicas
            
        Returns:
            Memory utilization percentage (already correctly calculated)
        """
        # Memory utilization is now correctly calculated upstream to account for multiple pods
        # No adjustment needed here
        return current_mem_util
    
    def get_config(self) -> Dict:
        """Get current validator configuration."""
        return {
            'max_scale_up_per_minute': self.max_scale_up_per_minute,
            'max_scale_down_per_minute': self.max_scale_down_per_minute,
            'min_time_between_scales': self.min_time_between_scales,
            'max_cpu_util_for_scale_down': self.max_cpu_util_for_scale_down,
            'min_cpu_util_for_scale_up': self.min_cpu_util_for_scale_up,
            'min_memory_util_for_scale_up': self.min_memory_util_for_scale_up,
            'critical_memory_threshold': self.critical_memory_threshold,
            'critical_cpu_threshold': self.critical_cpu_threshold,
            'min_stable_period_seconds': self.min_stable_period_seconds,
            'max_utilization_variance': self.max_utilization_variance,
            'emergency_scale_up_threshold': self.emergency_scale_up_threshold,
            'force_scale_down_threshold': self.force_scale_down_threshold
        } 

    def _validate_regular(self, action: int, current_replicas: int, target_replicas: int,
                         min_replicas: int, max_replicas: int, deployment_info: Dict = None, 
                         exploration_mode: bool = False) -> Tuple[bool, str, int]:
        """
        Regular validation logic (the original validation).
        """        
        # Basic boundary validation
        boundary_valid, boundary_reason, boundary_target = self._validate_replica_boundaries(
            target_replicas, min_replicas, max_replicas
        )
        if not boundary_valid:
            return False, boundary_reason, boundary_target
        
        # No-change validation
        if target_replicas == current_replicas:
            return True, "No scaling required", current_replicas
        
        # Rate limiting validation
        rate_valid, rate_reason = self._validate_scaling_rate(action, current_replicas, target_replicas, deployment_info)
        if not rate_valid:
            return False, rate_reason, current_replicas
        
        # Resource-based validation (more lenient during exploration)
        if deployment_info:
            resource_valid, resource_reason = self._validate_resource_constraints(
                action, current_replicas, target_replicas, deployment_info, exploration_mode
            )
            if not resource_valid:
                return False, resource_reason, current_replicas
        
        # Stability validation
        stability_valid, stability_reason = self._validate_system_stability(action)
        if not stability_valid:
            return False, stability_reason, current_replicas
        
        return True, "Regular validation passed", target_replicas
    
    def _validate_with_llm(self, action: int, current_replicas: int, target_replicas: int,
                          min_replicas: int, max_replicas: int, deployment_info: Dict = None) -> Tuple[bool, str, int]:
        """
        LLM-based validation using OpenAI API with MCP tools.
        
        Returns:
            Tuple of (is_valid, reason, adjusted_target)
        """
        try:
            # Run async validation
            return asyncio.run(self._validate_with_llm_async(
                action, current_replicas, target_replicas, min_replicas, max_replicas, deployment_info
            ))
        except Exception as e:
            logging.error(f"LLM validation failed: {e}")
            return False, f"LLM validation error: {str(e)}", current_replicas
    
    async def _validate_with_llm_async(self, action: int, current_replicas: int, target_replicas: int,
                                      min_replicas: int, max_replicas: int, deployment_info: Dict = None) -> Tuple[bool, str, int]:
        """
        Async LLM-based validation using OpenAI API with MCP tools.
        """
        try:
                         # Import LLM dependencies
            from langchain_openai import ChatOpenAI
            from langchain_mcp_adapters.client import MultiServerMCPClient
            from langgraph.prebuilt import create_react_agent
            
            # Initialize LLM
            llm = ChatOpenAI(
                model=self.ai_model,
                temperature=self.ai_temperature,
                api_key=self.openai_api_key
            )
            
            # Initialize MCP client
            mcp_client = MultiServerMCPClient(
                connections={
                    "kubernetes": {
                        "url": f"{self.mcp_server_url}/sse",
                        "transport": "sse"
                    }
                }
            )
            
            # Get tools from MCP client
            tools = await mcp_client.get_tools()
            
            # Create react agent with LLM and tools
            agent = create_react_agent(llm, tools)
            
            # Prepare context for LLM
            context = {
                "scaling_action": {
                    "action": action,
                    "action_name": ["no_action", "scale_up", "scale_down"][action] if 0 <= action <= 2 else "unknown",
                    "current_replicas": current_replicas,
                    "target_replicas": target_replicas,
                    "replica_change": target_replicas - current_replicas
                },
                "constraints": {
                    "min_replicas": min_replicas,
                    "max_replicas": max_replicas
                },
                "deployment_info": deployment_info or {},
                "timestamp": datetime.now().isoformat(),
                "system_thresholds": {
                    "emergency_scale_up_threshold": self.emergency_scale_up_threshold,
                    "critical_memory_threshold": self.critical_memory_threshold,
                    "critical_cpu_threshold": self.critical_cpu_threshold,
                    "max_cpu_util_for_scale_down": self.max_cpu_util_for_scale_down,
                    "min_cpu_util_for_scale_up": self.min_cpu_util_for_scale_up,
                    "min_memory_util_for_scale_up": self.min_memory_util_for_scale_up
                }
            }
            
                         # Create validation prompt with MCP tools
            validation_prompt = self._create_validation_prompt(context, tools)
            
            # Get agent response with access to Kubernetes tools
            response = await agent.ainvoke({
                "messages": [("human", validation_prompt)]
            })
            
            # Extract the final message content
            if response and "messages" in response:
                final_message = response["messages"][-1]
                response_content = final_message.content if hasattr(final_message, 'content') else str(final_message)
            else:
                response_content = str(response)
            
            # Parse LLM response
            return self._parse_llm_response_content(response_content, target_replicas, current_replicas)
            
        except Exception as e:
            logging.error(f"LLM validation with MCP tools failed: {e}")
            return False, f"LLM validation error: {str(e)}", current_replicas
    

    
    def _create_validation_prompt(self, context: Dict, mcp_tools) -> str:
        """Create a structured prompt for LLM validation using MCP tools."""
        action_name = context["scaling_action"]["action_name"]
        current_replicas = context["scaling_action"]["current_replicas"]
        target_replicas = context["scaling_action"]["target_replicas"]
        replica_change = context["scaling_action"]["replica_change"]
        
        deployment_info = context["deployment_info"]
        if deployment_info:
            current_cpu = deployment_info.get("current_cpu_util", "unknown")
            current_mem = deployment_info.get("current_mem_util", "unknown")
            predicted_mem = deployment_info.get("predicted_mem_util", "unknown")
        else:
            current_cpu = "unknown"
            current_mem = "unknown"
            predicted_mem = "unknown"
        
        # Extract tool names and descriptions from MCP tools
        tool_descriptions = []
        for tool in mcp_tools:
            tool_name = getattr(tool, 'name', str(tool))
            tool_desc = getattr(tool, 'description', '')
            if tool_desc:
                tool_descriptions.append(f"{tool_name}: {tool_desc}")
            else:
                tool_descriptions.append(tool_name)
        tools_text = ", ".join(tool_descriptions)
        
        # Format the prompt with all the context
        prompt = VALIDATION_PROMPT.format(
            tools=tools_text,
            action_name=action_name,
            current_replicas=current_replicas,
            target_replicas=target_replicas,
            replica_change=replica_change,
            min_replicas=context["constraints"]["min_replicas"],
            max_replicas=context["constraints"]["max_replicas"],
            current_cpu=current_cpu,
            current_mem=current_mem,
            predicted_mem=predicted_mem,
            emergency_scale_up_threshold=context["system_thresholds"]["emergency_scale_up_threshold"],
            critical_memory_threshold=context["system_thresholds"]["critical_memory_threshold"],
            critical_cpu_threshold=context["system_thresholds"]["critical_cpu_threshold"],
            max_cpu_util_for_scale_down=context["system_thresholds"]["max_cpu_util_for_scale_down"],
            min_cpu_util_for_scale_up=context["system_thresholds"]["min_cpu_util_for_scale_up"],
            min_memory_util_for_scale_up=context["system_thresholds"]["min_memory_util_for_scale_up"],
            deployment_info=json.dumps(deployment_info, indent=2) if deployment_info else "No additional deployment information"
        )
        
        return prompt.strip()
    
    def _parse_llm_response_content(self, content: str, target_replicas: int, current_replicas: int) -> Tuple[bool, str, int]:
        """Parse LLM response content and extract validation decision."""
        try:
            # Clean up the content - sometimes LLM responses have extra text
            content = content.strip()
            
            # Try to extract JSON from the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_content = content[json_start:json_end]
            else:
                json_content = content
            
            # Try to parse JSON response
            llm_decision = json.loads(json_content)
            
            approved = llm_decision.get("approved", False)
            reason = llm_decision.get("reason", "LLM validation completed")
            confidence = llm_decision.get("confidence", 0.0)
            recommended_target = llm_decision.get("recommended_target", target_replicas)
            risk_assessment = llm_decision.get("risk_assessment", "UNKNOWN")
            
            # Log detailed LLM response
            logging.info(f"LLM Validation Decision: approved={approved}, confidence={confidence:.2f}, risk={risk_assessment}")
            
            if not approved:
                return False, f"{reason} (confidence: {confidence:.2f}, risk: {risk_assessment})", recommended_target
            else:
                return True, f"{reason} (confidence: {confidence:.2f}, risk: {risk_assessment})", recommended_target
                
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM response as JSON: {e}")
            logging.error(f"Raw response: {content}")
            return False, "LLM response parsing failed", current_replicas
        except Exception as e:
            logging.error(f"Unexpected error parsing LLM response: {e}")
            return False, f"LLM validation error: {str(e)}", current_replicas 