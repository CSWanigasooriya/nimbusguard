# engine/ml/dqn_model.py
# ============================================================================
# Deep Q-Network Model for Kubernetes Autoscaling
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import logging

LOG = logging.getLogger(__name__)

class DQNModel(nn.Module):
    """
    Deep Q-Network for Kubernetes autoscaling decisions.
    
    Architecture designed for the 43-dimensional state space from your observability stack.
    """
    
    def __init__(self, 
                 state_dim: int = 43, 
                 action_dim: int = 5,
                 hidden_dims: Tuple[int, ...] = (256, 512, 256, 128),
                 dropout_rate: float = 0.1):
        """
        Initialize the DQN model.
        
        Args:
            state_dim: Dimension of state space (43 for your setup)
            action_dim: Number of actions (5 scaling actions)
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super(DQNModel, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Input normalization layer
        self.input_norm = nn.LayerNorm(state_dim)
        
        # Build the network layers
        layers = []
        prev_dim = state_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_rate) if i < len(hidden_dims) - 1 else nn.Dropout(0.05)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        LOG.info(f"Initialized DQN model: {state_dim} -> {hidden_dims} -> {action_dim}")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Q-values for each action of shape (batch_size, action_dim)
        """
        # Normalize input
        x = self.input_norm(state)
        
        # Forward through network
        q_values = self.network(x)
        
        return q_values
    
    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: Single state tensor of shape (state_dim,)
            epsilon: Exploration probability
            
        Returns:
            Selected action index
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)
        
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)  # Add batch dimension
            q_values = self.forward(state)
            return q_values.argmax(dim=1).item()
    
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Get Q-values for a state without exploration"""
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            return self.forward(state)
    
    def save_model(self, filepath: str):
        """Save model weights and configuration"""
        torch.save({
            'state_dict': self.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'model_class': self.__class__.__name__
        }, filepath)
        LOG.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, device: Optional[torch.device] = None) -> 'DQNModel':
        """Load model from saved weights"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        checkpoint = torch.load(filepath, map_location=device)
        
        model = cls(
            state_dim=checkpoint['state_dim'],
            action_dim=checkpoint['action_dim']
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        
        LOG.info(f"Model loaded from {filepath}")
        return model


class DoubleDQNModel(DQNModel):
    """
    Double DQN variant with separate target network for more stable learning.
    This is recommended for production use.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Create target network (copy of main network)
        self.target_network = DQNModel(*args, **kwargs)
        self.update_target_network()
        
        LOG.info("Initialized Double DQN with target network")
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.state_dict())
    
    def get_target_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Get Q-values from target network"""
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            return self.target_network.forward(state)


class PrioritizedReplayBuffer:
    """
    Experience replay buffer with prioritized sampling for better learning efficiency.
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Priority exponent (0 = uniform sampling, 1 = full prioritization)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool, priority: Optional[float] = None):
        """Add experience to buffer"""
        
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        """Sample batch with prioritized sampling"""
        if len(self.buffer) < batch_size:
            return None
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities[:len(self.buffer)])
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        # Get experiences
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            np.array(weights)
        )
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)
