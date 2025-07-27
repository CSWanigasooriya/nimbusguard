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