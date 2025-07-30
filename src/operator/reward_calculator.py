import logging
import numpy as np

class RewardCalculator:
    """
    Simple reward calculator for fast DQN learning.
    Focuses on core objectives with clear, consistent signals.
    """
    
    def __init__(self, config=None):
        """
        Initialize simple reward calculator.
        
        Args:
            config: Dictionary with reward calculation parameters
        """
        self.config = config or {}
        
        # More realistic targets: keep resources in optimal range
        self.cpu_target = self.config.get('cpu_target', 60.0)  # Lower, more realistic target
        self.memory_target = self.config.get('memory_target', 60.0)  # Lower, more realistic target
        self.resource_emergency = self.config.get('resource_emergency', 90.0)
        
        # Weights for current vs predicted memory
        self.current_weight = self.config.get('current_weight', 0.6)  # Favor current state slightly
        self.predicted_weight = self.config.get('predicted_weight', 0.4)  # But consider forecast
        
        # Resource importance weights
        self.cpu_weight = self.config.get('cpu_weight', 0.4)  # CPU importance
        self.memory_weight = self.config.get('memory_weight', 0.6)  # Memory slightly more important
        
        # Simple penalties and rewards
        self.action_penalty = self.config.get('action_penalty', 2.0)  # Encourage stability
        self.emergency_penalty = self.config.get('emergency_penalty', 50.0)  # Avoid high memory
        self.boundary_penalty = self.config.get('boundary_penalty', 20.0)  # Avoid limits
    
    def calculate_reward(self, current_cpu_util, current_mem_util, predicted_mem_util, 
                        action, replicas, min_replicas, max_replicas):
        """
        Simple reward calculation for fast DQN learning.
        
        Reward = Memory Distance Reward - Action Penalty - Emergency Penalty - Boundary Penalty
        
        Args:
            current_cpu_util: Current CPU utilization percentage
            current_mem_util: Current memory utilization percentage  
            predicted_mem_util: Predicted memory utilization percentage
            action: Action taken (0=none, 1=scale_up, 2=scale_down)
            replicas: Current number of replicas
            min_replicas: Minimum allowed replicas
            max_replicas: Maximum allowed replicas
            
        Returns:
            Float reward value
        """
        # Calculate weighted memory utilization (current + predicted)
        memory_util = (current_mem_util * self.current_weight) + (predicted_mem_util * self.predicted_weight)
        
        # Calculate combined resource utilization (CPU + memory)
        combined_util = (current_cpu_util * self.cpu_weight) + (memory_util * self.memory_weight)
        
        logging.info(f"--- Simple Reward Calculation ---")
        logging.info(f"CPU: {current_cpu_util:.1f}% (current), Target: {self.cpu_target:.1f}%")
        logging.info(f"Memory: {current_mem_util:.1f}% (current), {predicted_mem_util:.1f}% (predicted)")
        logging.info(f"Weighted Memory: {memory_util:.1f}%")
        logging.info(f"Combined Resource: {combined_util:.1f}% (CPU:{self.cpu_weight:.1f}, Mem:{self.memory_weight:.1f})")
        logging.info(f"Action: {action}, Replicas: {replicas} (range: {min_replicas}-{max_replicas})")
        
        # 1. Resource distance reward: closer to target = better (considers both CPU and memory)
        resource_distance = abs(combined_util - self.cpu_target)  # Use same target for combined
        # Use a more gradual reward curve that doesn't drop to 0 so quickly
        if resource_distance <= 10.0:
            resource_reward = 25.0 - resource_distance  # Full range when close to target
        else:
            # Gradual decay for larger distances, but never fully zero
            resource_reward = max(1.0, 15.0 - (resource_distance - 10.0) * 0.2)
        
        # 2. Special handling for very low utilization - reward scale down decisions
        if combined_util < 20.0 and action == 2 and replicas > min_replicas:
            resource_reward += 10.0  # Bonus for scaling down when under-utilized
            logging.info(f"      [+10.0] Under-utilization scale-down bonus")
        elif combined_util < 20.0 and action == 0:
            resource_reward += 2.0  # Small bonus for not scaling up when under-utilized
            logging.info(f"      [+2.0] Under-utilization stability bonus")
        
        # 3. Action penalty: encourage stability (doing nothing is often best)
        action_cost = self.action_penalty if action != 0 else 0
        
        # 4. Emergency penalty: heavily penalize high resource usage
        emergency_cost = self.emergency_penalty if combined_util > self.resource_emergency else 0
        
        # 5. Boundary penalties
        boundary_cost = 0
        if action == 1 and replicas >= max_replicas:
            boundary_cost = self.boundary_penalty
            logging.info(f"      [-{boundary_cost:.1f}] Boundary penalty: scale up at max")
        elif action == 2 and replicas <= min_replicas:
            boundary_cost = self.boundary_penalty  
            logging.info(f"      [-{boundary_cost:.1f}] Boundary penalty: scale down at min")
        
        # Calculate final reward
        total_reward = resource_reward - action_cost - emergency_cost - boundary_cost
        
        logging.info(f"      [+{resource_reward:.1f}] Resource distance reward (CPU+Memory)")
        logging.info(f"      [-{action_cost:.1f}] Action penalty") 
        logging.info(f"      [-{emergency_cost:.1f}] Emergency penalty")
        logging.info(f"Final Reward: {total_reward:.2f}")
        
        return total_reward
    
    def get_reward_breakdown(self, current_cpu_util, current_mem_util, predicted_mem_util, 
                            action, replicas, min_replicas, max_replicas):
        """
        Get detailed breakdown of simple reward components for analysis.
        
        Returns:
            Dictionary with reward component breakdown
        """
        memory_util = (current_mem_util * self.current_weight) + (predicted_mem_util * self.predicted_weight)
        combined_util = (current_cpu_util * self.cpu_weight) + (memory_util * self.memory_weight)
        
        # Calculate components
        resource_distance = abs(combined_util - self.cpu_target)
        # Use the same improved reward curve
        if resource_distance <= 10.0:
            resource_reward = 25.0 - resource_distance
        else:
            resource_reward = max(1.0, 15.0 - (resource_distance - 10.0) * 0.2)
        
        # Add under-utilization bonuses
        if combined_util < 20.0 and action == 2 and replicas > min_replicas:
            resource_reward += 10.0  # Under-utilization scale-down bonus
        elif combined_util < 20.0 and action == 0:
            resource_reward += 2.0  # Under-utilization stability bonus
            
        action_cost = self.action_penalty if action != 0 else 0
        emergency_cost = self.emergency_penalty if combined_util > self.resource_emergency else 0
        
        boundary_cost = 0
        if action == 1 and replicas >= max_replicas:
            boundary_cost = self.boundary_penalty
        elif action == 2 and replicas <= min_replicas:
            boundary_cost = self.boundary_penalty
        
        total_reward = resource_reward - action_cost - emergency_cost - boundary_cost
        
        return {
            'cpu_util': current_cpu_util,
            'memory_util': memory_util,
            'combined_util': combined_util,
            'resource_distance': resource_distance,
            'resource_reward': resource_reward,
            'action_cost': action_cost,
            'emergency_cost': emergency_cost,
            'boundary_cost': boundary_cost,
            'total_reward': total_reward,
            'is_emergency': combined_util > self.resource_emergency,
            'at_boundary': (action == 1 and replicas >= max_replicas) or (action == 2 and replicas <= min_replicas)
        }
    

    
    def update_config(self, new_config):
        """Update simple reward calculation configuration."""
        self.config.update(new_config)
        # Update individual parameters
        for key, value in new_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
        logging.info(f"Updated simple reward calculator config: {new_config}")
    
    def get_config(self):
        """Get current simple configuration."""
        return {
            'cpu_target': self.cpu_target,
            'memory_target': self.memory_target,
            'resource_emergency': self.resource_emergency,
            'current_weight': self.current_weight,
            'predicted_weight': self.predicted_weight,
            'cpu_weight': self.cpu_weight,
            'memory_weight': self.memory_weight,
            'action_penalty': self.action_penalty,
            'emergency_penalty': self.emergency_penalty,
            'boundary_penalty': self.boundary_penalty
        } 

    def calculate_rejection_penalty(self, dqn_action, validation_reason, current_cpu_util, 
                                   current_mem_util, predicted_mem_util):
        """
        Calculate negative reward for actions rejected by the validator.
        This teaches the DQN to avoid making decisions that will be blocked.
        
        Args:
            dqn_action: Action the DQN wanted to take (0=none, 1=scale_up, 2=scale_down)
            validation_reason: Reason why validator rejected the action
            current_cpu_util: Current CPU utilization percentage
            current_mem_util: Current memory utilization percentage
            predicted_mem_util: Predicted memory utilization percentage
            
        Returns:
            Negative reward value
        """
        base_penalty = -5.0  # Base penalty for any rejection
        severity_penalty = 0.0
        
        logging.info(f"--- Rejection Penalty Calculation ---")
        logging.info(f"DQN Action: {dqn_action}, CPU: {current_cpu_util:.1f}%, Memory: {current_mem_util:.1f}%")
        logging.info(f"Validation Reason: {validation_reason}")
        
        # Categorize rejection reasons and assign appropriate penalties
        reason_lower = validation_reason.lower()
        
        # Check if this is an LLM validation rejection
        is_llm_rejection = "llm override:" in reason_lower
        
        # High-severity penalties for dangerous decisions
        if ("cpu utilization too high" in reason_lower or 
            ("high cpu" in reason_lower and "risky" in reason_lower) or
            ("cpu" in reason_lower and "dangerous" in reason_lower)) and dqn_action == 2:  # Scale down during high CPU
            severity_penalty = -15.0
            rejection_type = "LLM" if is_llm_rejection else "Rule-based"
            logging.info(f"      [-15.0] CRITICAL ({rejection_type}): Attempted scale-down during high CPU ({current_cpu_util:.1f}%)")
            
        elif ("memory utilization too high" in reason_lower or
              ("high memory" in reason_lower and "risky" in reason_lower) or
              ("memory" in reason_lower and "dangerous" in reason_lower)) and dqn_action == 2:  # Scale down during high memory
            severity_penalty = -12.0
            rejection_type = "LLM" if is_llm_rejection else "Rule-based"
            logging.info(f"      [-12.0] CRITICAL ({rejection_type}): Attempted scale-down during high memory ({current_mem_util:.1f}%)")
            
        elif ("emergency" in reason_lower or 
              ("risk: high" in reason_lower and dqn_action == 2)) and dqn_action == 2:  # Scale down during emergency
            severity_penalty = -20.0
            rejection_type = "LLM" if is_llm_rejection else "Rule-based"
            logging.info(f"      [-20.0] CRITICAL ({rejection_type}): Attempted scale-down during emergency conditions")
            
        elif ("cpu utilization justifies scale-up" in reason_lower or
              ("should scale up" in reason_lower and "cpu" in reason_lower)) and dqn_action == 2:  # Scale down when should scale up
            severity_penalty = -10.0
            rejection_type = "LLM" if is_llm_rejection else "Rule-based"
            logging.info(f"      [-10.0] POOR DECISION ({rejection_type}): Scale-down when CPU justifies scale-up")
            
        elif ("memory utilization justifies scale-up" in reason_lower or
              ("should scale up" in reason_lower and "memory" in reason_lower)) and dqn_action == 2:  # Scale down when should scale up
            severity_penalty = -10.0
            rejection_type = "LLM" if is_llm_rejection else "Rule-based"
            logging.info(f"      [-10.0] POOR DECISION ({rejection_type}): Scale-down when memory justifies scale-up")
            
        # Medium-severity penalties for premature or boundary violations
        elif "above maximum" in reason_lower and dqn_action == 1:  # Scale up beyond max
            severity_penalty = -8.0
            logging.info(f"      [-8.0] BOUNDARY: Attempted scale-up beyond maximum replicas")
            
        elif "below minimum" in reason_lower and dqn_action == 2:  # Scale down below min
            severity_penalty = -8.0
            logging.info(f"      [-8.0] BOUNDARY: Attempted scale-down below minimum replicas")
            
        # Low-severity penalties for rate limiting and stability
        elif "rate limit" in reason_lower:
            severity_penalty = -3.0
            logging.info(f"      [-3.0] RATE LIMIT: Too frequent scaling actions")
            
        elif "system instability" in reason_lower or "oscillation" in reason_lower:
            severity_penalty = -4.0
            logging.info(f"      [-4.0] STABILITY: Action would cause system instability")
            
        elif "minimum time between scales" in reason_lower:
            severity_penalty = -2.0
            logging.info(f"      [-2.0] TIMING: Scaling too soon after previous action")
            
        elif is_llm_rejection:
            # LLM-specific rejection patterns
            if "risk: high" in reason_lower:
                severity_penalty = -12.0
                logging.info(f"      [-12.0] LLM HIGH RISK: Action assessed as high risk by AI validator")
            elif "risk: medium" in reason_lower:
                severity_penalty = -6.0
                logging.info(f"      [-6.0] LLM MEDIUM RISK: Action assessed as medium risk by AI validator")
            elif "inappropriate" in reason_lower or "not recommended" in reason_lower:
                severity_penalty = -8.0
                logging.info(f"      [-8.0] LLM REJECTION: Action deemed inappropriate by AI validator")
            elif "unsafe" in reason_lower or "risky" in reason_lower:
                severity_penalty = -10.0
                logging.info(f"      [-10.0] LLM SAFETY: Action flagged as unsafe by AI validator")
            elif "confidence:" in reason_lower:
                # Extract confidence level for penalty scaling
                try:
                    import re
                    confidence_match = re.search(r'confidence: ([\d.]+)', reason_lower)
                    if confidence_match:
                        confidence = float(confidence_match.group(1))
                        # Higher confidence in rejection = higher penalty (3.0 to 9.0)
                        severity_penalty = -3.0 - (confidence * 6.0)
                        logging.info(f"      [{severity_penalty:.1f}] LLM CONFIDENT REJECTION: AI confidence {confidence:.2f}")
                    else:
                        severity_penalty = -5.0
                        logging.info(f"      [-5.0] LLM REJECTION: Generic AI validator rejection")
                except:
                    severity_penalty = -5.0
                    logging.info(f"      [-5.0] LLM REJECTION: Generic AI validator rejection")
            else:
                severity_penalty = -4.0
                logging.info(f"      [-4.0] LLM GENERIC: Unclassified AI validator rejection")
        else:
            # Generic penalty for unclassified rule-based rejections
            severity_penalty = -3.0
            logging.info(f"      [-3.0] GENERIC: Unclassified rule-based validation failure")
        
        # Additional context-based penalties
        context_penalty = 0.0
        
        # Extra penalty for very bad timing
        if dqn_action == 2:  # Scale down
            if current_cpu_util > 90.0:
                context_penalty -= 5.0
                logging.info(f"      [-5.0] CONTEXT: Scale-down during extreme CPU pressure")
            elif current_mem_util > 90.0:
                context_penalty -= 5.0
                logging.info(f"      [-5.0] CONTEXT: Scale-down during extreme memory pressure")
        
        if dqn_action == 1:  # Scale up
            if current_cpu_util < 10.0 and current_mem_util < 10.0:
                context_penalty -= 3.0
                logging.info(f"      [-3.0] CONTEXT: Scale-up during very low utilization")
        
        # Calculate total penalty
        total_penalty = base_penalty + severity_penalty + context_penalty
        
        # Clamp to reasonable bounds
        total_penalty = max(-25.0, min(-1.0, total_penalty))
        
        logging.info(f"      [{base_penalty:.1f}] Base rejection penalty")
        logging.info(f"      [{severity_penalty:.1f}] Severity penalty")
        logging.info(f"      [{context_penalty:.1f}] Context penalty")
        logging.info(f"      [={total_penalty:.1f}] TOTAL REJECTION PENALTY")
        
        return total_penalty 