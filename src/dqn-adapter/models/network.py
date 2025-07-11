import torch.nn as nn
import torch


# --- DQN Model Definition ---
class EnhancedQNetwork(nn.Module):
    """Enhanced Q-Network matching the learner's architecture."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=None):
        super().__init__()

        # Use appropriately sized hidden dimensions for the feature space
        if hidden_dims is None:
            hidden_dims = [64, 32]  # Simplified architecture for 11 input features

        layers = []
        prev_dim = state_dim

        # Build hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            # Only add dropout to hidden layers (not output)
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(0.1))  # Reduced dropout rate
            prev_dim = hidden_dim

        # Output layer (no dropout, no batchnorm)
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.network(x)
