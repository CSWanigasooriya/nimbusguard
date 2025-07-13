"""DQN neural network model for the agent."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


logger = logging.getLogger(__name__)


class DQNNetwork(nn.Module):
    """Deep Q-Network for scaling decisions."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(DQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # Feature processing layers
        self.feature_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Value stream (estimates state value)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Advantage stream (estimates advantage of each action)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, action_size)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"DQNNetwork created: state_size={state_size}, action_size={action_size}, "
                   f"hidden_size={hidden_size}, total_params={self._count_parameters()}")
    
    def forward(self, x):
        """Forward pass through the network."""
        # Feature extraction
        features = self.feature_layers(x)
        
        # Dueling DQN: separate value and advantage streams
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_feature_importance(self, state: torch.Tensor) -> dict:
        """Get feature importance using gradient-based analysis."""
        self.eval()
        state.requires_grad_(True)
        
        # Forward pass
        q_values = self.forward(state)
        
        # Get gradients for the best action
        best_action = q_values.argmax(dim=1)
        best_q_value = q_values.gather(1, best_action.unsqueeze(1))
        
        # Backward pass to get gradients
        best_q_value.backward()
        
        # Feature importance is the absolute gradient
        importance = torch.abs(state.grad).squeeze().detach().cpu().numpy()
        
        self.train()
        return importance 