"""
Experience Replay Buffer for DQN Agent.
"""

import logging
import random
from collections import deque, namedtuple
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Define experience tuple
Experience = namedtuple('Experience', [
    'state',          # Current state (metrics + context)
    'action',         # Action taken (0: scale_down, 1: no_action, 2: scale_up)
    'reward',         # Reward received
    'next_state',     # Next state after action
    'done',           # Whether episode is complete
    'deployment_context'  # Deployment context for reward calculation
])


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 10000, min_size: int = 100):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            min_size: Minimum number of experiences before sampling
        """
        self.capacity = capacity
        self.min_size = min_size
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        
        logger.info(f"Initialized replay buffer with capacity={capacity}, min_size={min_size}")
    
    def add(self, experience: Experience):
        """Add an experience to the buffer."""
        self.buffer.append(experience)
        self.position = (self.position + 1) % self.capacity
        
        if len(self.buffer) % 100 == 0:  # Log every 100 experiences
            logger.debug(f"Replay buffer size: {len(self.buffer)}/{self.capacity}")
    
    def sample(self, batch_size: int) -> Optional[List[Experience]]:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of sampled experiences or None if not enough data
        """
        if len(self.buffer) < self.min_size:
            logger.debug(f"Buffer too small for sampling: {len(self.buffer)} < {self.min_size}")
            return None
        
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        logger.debug(f"Sampled batch of {len(batch)} experiences")
        return batch
    
    def can_sample(self) -> bool:
        """Check if buffer has enough experiences for sampling."""
        return len(self.buffer) >= self.min_size
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.position = 0
        logger.info("Replay buffer cleared")
    
    def get_stats(self) -> dict:
        """Get buffer statistics."""
        if not self.buffer:
            return {
                'size': 0,
                'capacity': self.capacity,
                'min_size': self.min_size,
                'can_sample': False,
                'fill_percentage': 0.0
            }
        
        # Analyze recent experiences
        recent_rewards = [exp.reward for exp in list(self.buffer)[-100:]]
        recent_actions = [exp.action for exp in list(self.buffer)[-100:]]
        
        action_counts = {
            'scale_down': recent_actions.count(0),
            'no_action': recent_actions.count(1),
            'scale_up': recent_actions.count(2)
        }
        
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'min_size': self.min_size,
            'can_sample': self.can_sample(),
            'fill_percentage': (len(self.buffer) / self.capacity) * 100,
            'avg_recent_reward': np.mean(recent_rewards) if recent_rewards else 0.0,
            'action_distribution': action_counts
        } 