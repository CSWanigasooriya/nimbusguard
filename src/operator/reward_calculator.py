import logging
import numpy as np

class RewardCalculator:
    """
    Calculates rewards for DQN agent based on resource utilization and scaling actions.
    Uses pure functions without dependency on shared state.
    """
    
    def __init__(self, config=None):
        """
        Initialize reward calculator with configuration.
        
        Args:
            config: Dictionary with reward calculation parameters
        """
        # Default configuration
        self.config = config or {}
        
        # Target utilization settings
        self.cpu_target_util = self.config.get('cpu_target_util', 70.0)
        self.mem_target_band_low = self.config.get('mem_target_band_low', 70.0)
        self.mem_target_band_high = self.config.get('mem_target_band_high', 85.0)
        self.low_util_threshold = self.config.get('low_util_threshold', 30.0)
        
        # Weights for prediction vs current
        self.current_weight = self.config.get('current_weight', 0.5)
        self.predicted_weight = self.config.get('predicted_weight', 0.5)
        
        # Penalty weights
        self.cpu_penalty_weight = self.config.get('cpu_penalty_weight', 20.0)
        self.memory_penalty_weight = self.config.get('memory_penalty_weight', 15.0)
        self.action_penalty = self.config.get('action_penalty', 1.0)
        self.boundary_penalty_scale_up = self.config.get('boundary_penalty_scale_up', 20.0)
        self.boundary_penalty_scale_down = self.config.get('boundary_penalty_scale_down', 10.0)
        
        # Rewards
        self.base_reward = self.config.get('base_reward', 10.0)
        self.low_util_scale_down_reward = self.config.get('low_util_scale_down_reward', 15.0)
        self.low_util_inaction_penalty = self.config.get('low_util_inaction_penalty', 5.0)
        self.low_util_scale_up_penalty = self.config.get('low_util_scale_up_penalty', 20.0)
        self.min_replicas_efficiency_bonus = self.config.get('min_replicas_efficiency_bonus', 5.0)
    
    def calculate_reward(self, current_cpu_util, current_mem_util, predicted_mem_util, 
                        action, replicas, min_replicas, max_replicas):
        """
        Calculate reward based on current state and action taken.
        
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
        # Calculate weighted memory utilization
        weighted_mem_util = (current_mem_util * self.current_weight) + (predicted_mem_util * self.predicted_weight)
        
        logging.info(f"--- Reward Calculation ---")
        logging.info(f"Current CPU: {current_cpu_util:.2f}%, Current Mem: {current_mem_util:.2f}%")
        logging.info(f"Predicted Mem: {predicted_mem_util:.2f}%, Weighted Mem: {weighted_mem_util:.2f}%")
        logging.info(f"Action: {action}, Replicas: {replicas} (range: {min_replicas}-{max_replicas})")
        
        # Check utilization zone
        if self._is_low_utilization_zone(current_cpu_util, weighted_mem_util):
            reward = self._calculate_low_utilization_reward(action, replicas, min_replicas)
        else:
            reward = self._calculate_normal_utilization_reward(
                current_cpu_util, weighted_mem_util, action
            )
        
        # Apply boundary penalties
        boundary_penalty = self._calculate_boundary_penalties(action, replicas, min_replicas, max_replicas)
        reward += boundary_penalty
        
        logging.info(f"Final Calculated Reward: {reward:.2f}")
        return reward
    
    def _is_low_utilization_zone(self, cpu_util, mem_util):
        """Check if we're in low utilization zone."""
        return cpu_util < self.low_util_threshold and mem_util < self.low_util_threshold
    
    def _calculate_low_utilization_reward(self, action, replicas, min_replicas):
        """Calculate reward for low utilization zone."""
        logging.info("--> Operating in LOW UTILIZATION ZONE")
        
        if action == 2:  # SCALE DOWN
            reward = self.low_util_scale_down_reward
            logging.info(f"      [+{reward:.1f}] Rewarding correct action (Scale Down).")
        elif action == 0:  # DO NOTHING
            reward = -self.low_util_inaction_penalty
            logging.info(f"      [{reward:.1f}] Penalizing inaction.")
        elif action == 1:  # SCALE UP
            reward = -self.low_util_scale_up_penalty
            logging.info(f"      [{reward:.1f}] Penalizing incorrect action (Scale Up).")
        else:
            reward = 0
            logging.warning(f"      [0.0] Unknown action {action}")
        
        # Efficiency bonus for being at minimum replicas
        if replicas == min_replicas:
            efficiency_bonus = self.min_replicas_efficiency_bonus
            reward += efficiency_bonus
            logging.info(f"      [+{efficiency_bonus:.1f}] Efficiency bonus for being at min_replicas.")
        
        return reward
    
    def _calculate_normal_utilization_reward(self, cpu_util, mem_util, action):
        """Calculate reward for normal/high utilization zone."""
        logging.info("--> Operating in NORMAL/HIGH UTILIZATION ZONE")
        
        # Start with base reward
        reward = self.base_reward
        
        # Calculate CPU penalty
        cpu_error = (cpu_util - self.cpu_target_util) / 100.0
        cpu_penalty = self.cpu_penalty_weight * (cpu_error ** 2)
        reward -= cpu_penalty
        
        # Calculate memory penalty based on target band
        mem_error = self._calculate_memory_error(mem_util)
        mem_penalty = self.memory_penalty_weight * (mem_error ** 2)
        reward -= mem_penalty
        
        logging.info(f"      [-{cpu_penalty:.2f}] CPU penalty. [-{mem_penalty:.2f}] Memory penalty.")
        
        # Action penalty for scaling
        if action in [1, 2]:
            reward -= self.action_penalty
            logging.info(f"      [-{self.action_penalty:.1f}] Penalty for scaling action.")
        
        return reward
    
    def _calculate_memory_error(self, mem_util):
        """Calculate memory error based on target band."""
        if mem_util > self.mem_target_band_high:
            error = (mem_util - self.mem_target_band_high) / 100.0
            logging.info(f"      Memory is ABOVE target band by {error*100:.2f}%.")
        elif mem_util < self.mem_target_band_low:
            error = (self.mem_target_band_low - mem_util) / 100.0
            logging.info(f"      Memory is BELOW target band by {error*100:.2f}%.")
        else:
            error = 0.0
            logging.info("      Memory is WITHIN target band. No penalty.")
        
        return error
    
    def _calculate_boundary_penalties(self, action, replicas, min_replicas, max_replicas):
        """Calculate penalties for hitting replica boundaries."""
        penalty = 0.0
        
        if action == 1 and replicas >= max_replicas:
            penalty = -self.boundary_penalty_scale_up
            logging.info(f"      [{penalty:.1f}] Penalty for attempting to scale up at max_replicas.")
        
        if action == 2 and replicas <= min_replicas:
            penalty = -self.boundary_penalty_scale_down
            logging.info(f"      [{penalty:.1f}] Penalty for attempting to scale down at min_replicas.")
        
        return penalty
    
    def get_reward_components(self, current_cpu_util, current_mem_util, predicted_mem_util, 
                             action, replicas, min_replicas, max_replicas):
        """
        Get detailed breakdown of reward components for analysis.
        
        Returns:
            Dictionary with reward component breakdown
        """
        weighted_mem_util = (current_mem_util * self.current_weight) + (predicted_mem_util * self.predicted_weight)
        
        components = {
            'weighted_memory_util': weighted_mem_util,
            'is_low_utilization': self._is_low_utilization_zone(current_cpu_util, weighted_mem_util),
            'base_reward': 0.0,
            'cpu_penalty': 0.0,
            'memory_penalty': 0.0,
            'action_penalty': 0.0,
            'boundary_penalty': 0.0,
            'efficiency_bonus': 0.0,
            'zone_reward': 0.0
        }
        
        if components['is_low_utilization']:
            if action == 2:
                components['zone_reward'] = self.low_util_scale_down_reward
            elif action == 0:
                components['zone_reward'] = -self.low_util_inaction_penalty
            elif action == 1:
                components['zone_reward'] = -self.low_util_scale_up_penalty
            
            if replicas == min_replicas:
                components['efficiency_bonus'] = self.min_replicas_efficiency_bonus
        else:
            components['base_reward'] = self.base_reward
            
            cpu_error = (current_cpu_util - self.cpu_target_util) / 100.0
            components['cpu_penalty'] = -self.cpu_penalty_weight * (cpu_error ** 2)
            
            mem_error = self._calculate_memory_error(weighted_mem_util)
            components['memory_penalty'] = -self.memory_penalty_weight * (mem_error ** 2)
            
            if action in [1, 2]:
                components['action_penalty'] = -self.action_penalty
        
        # Boundary penalties
        if action == 1 and replicas >= max_replicas:
            components['boundary_penalty'] = -self.boundary_penalty_scale_up
        elif action == 2 and replicas <= min_replicas:
            components['boundary_penalty'] = -self.boundary_penalty_scale_down
        
        # Calculate total
        components['total_reward'] = sum([
            components['base_reward'],
            components['cpu_penalty'],
            components['memory_penalty'],
            components['action_penalty'],
            components['boundary_penalty'],
            components['efficiency_bonus'],
            components['zone_reward']
        ])
        
        return components
    
    def update_config(self, new_config):
        """Update reward calculation configuration."""
        self.config.update(new_config)
        # Update individual parameters
        for key, value in new_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
        logging.info(f"Updated reward calculator config: {new_config}")
    
    def get_config(self):
        """Get current configuration."""
        return {
            'cpu_target_util': self.cpu_target_util,
            'mem_target_band_low': self.mem_target_band_low,
            'mem_target_band_high': self.mem_target_band_high,
            'low_util_threshold': self.low_util_threshold,
            'current_weight': self.current_weight,
            'predicted_weight': self.predicted_weight,
            'cpu_penalty_weight': self.cpu_penalty_weight,
            'memory_penalty_weight': self.memory_penalty_weight,
            'action_penalty': self.action_penalty,
            'boundary_penalty_scale_up': self.boundary_penalty_scale_up,
            'boundary_penalty_scale_down': self.boundary_penalty_scale_down,
            'base_reward': self.base_reward,
            'low_util_scale_down_reward': self.low_util_scale_down_reward,
            'low_util_inaction_penalty': self.low_util_inaction_penalty,
            'low_util_scale_up_penalty': self.low_util_scale_up_penalty,
            'min_replicas_efficiency_bonus': self.min_replicas_efficiency_bonus
        } 