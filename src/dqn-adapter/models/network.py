import torch.nn as nn
import torch
import torch.nn.functional as F


class EnhancedQNetwork(nn.Module):
    """
    Enhanced Q-Network with modern deep learning improvements for autoscaling.
    
    Features:
    - Dueling DQN architecture for better value estimation
    - Layer normalization for training stability
    - Residual connections for better gradient flow
    - Improved weight initialization
    - Adaptive dropout
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=None, dueling=True, use_layer_norm=True):
        super().__init__()

        # Use appropriately sized hidden dimensions for the feature space
        if hidden_dims is None:
            hidden_dims = [64, 32]  # Optimized architecture for 9 consumer performance features

        self.dueling = dueling
        self.use_layer_norm = use_layer_norm
        self.hidden_dims = hidden_dims
        
        # Shared feature extraction layers
        self.feature_layers = nn.ModuleList()
        prev_dim = state_dim

        # Build shared feature extraction network
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            linear = nn.Linear(prev_dim, hidden_dim)
            self.feature_layers.append(linear)
            
            # Layer normalization for training stability
            if self.use_layer_norm:
                self.feature_layers.append(nn.LayerNorm(hidden_dim))
            
            # Activation
            self.feature_layers.append(nn.ReLU())
            
            # Adaptive dropout (higher for earlier layers)
            dropout_rate = 0.1 if i == 0 else 0.05
            self.feature_layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim

        self.feature_dim = prev_dim

        if self.dueling:
            # Dueling DQN: Separate value and advantage streams
            
            # Value stream (estimates state value)
            self.value_stream = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim // 2),
                nn.LayerNorm(self.feature_dim // 2) if self.use_layer_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.feature_dim // 2, 1)  # Single value output
            )
            
            # Advantage stream (estimates action advantages)
            self.advantage_stream = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim // 2),
                nn.LayerNorm(self.feature_dim // 2) if self.use_layer_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.feature_dim // 2, action_dim)  # One advantage per action
            )
        else:
            # Standard DQN: Direct Q-value output
            self.q_head = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim // 2),
                nn.LayerNorm(self.feature_dim // 2) if self.use_layer_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.feature_dim // 2, action_dim)
            )

        # Apply improved weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Improved weight initialization for stable training."""
        if isinstance(module, nn.Linear):
            # He initialization for ReLU activations
            torch.nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        """
        Forward pass with optional residual connections and dueling architecture.
        
        Args:
            x: Input state tensor [batch_size, state_dim]
            
        Returns:
            Q-values tensor [batch_size, action_dim]
        """
        # Ensure input is 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Feature extraction with residual connections
        features = self._extract_features(x)
        
        if self.dueling:
            # Dueling DQN: Combine value and advantage streams
            value = self.value_stream(features)  # [batch_size, 1]
            advantages = self.advantage_stream(features)  # [batch_size, action_dim]
            
            # Combine using the dueling formula: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
            # This ensures that the value function represents the expected return
            q_values = value + advantages - advantages.mean(dim=1, keepdim=True)
            
        else:
            # Standard DQN
            q_values = self.q_head(features)
            
        return q_values

    def _extract_features(self, x):
        """
        Extract features with optional residual connections for better gradient flow.
        
        Args:
            x: Input tensor
            
        Returns:
            Extracted features tensor
        """
        current = x
        residual = None
        layer_idx = 0
        
        for i, layer in enumerate(self.feature_layers):
            if isinstance(layer, nn.Linear):
                # Store input for potential residual connection
                if layer_idx == 0:
                    residual = current
                
                current = layer(current)
                layer_idx += 1
                
                # Add residual connection for deeper networks (skip first layer)
                if layer_idx > 1 and residual is not None and current.shape == residual.shape:
                    current = current + residual
                    residual = current  # Update residual for next connection
                    
            elif isinstance(layer, (nn.LayerNorm, nn.ReLU, nn.Dropout)):
                current = layer(current)
                
        return current

    def get_feature_importance(self, x):
        """
        Analyze feature importance using gradient-based attribution.
        Useful for understanding which metrics drive scaling decisions.
        
        Args:
            x: Input state tensor [batch_size, state_dim]
            
        Returns:
            Feature importance scores [batch_size, state_dim]
        """
        x.requires_grad_(True)
        q_values = self.forward(x)
        
        # Use max Q-value for attribution
        max_q = q_values.max(dim=1)[0].sum()
        
        # Compute gradients
        max_q.backward(retain_graph=True)
        
        # Feature importance is the absolute gradient
        importance = torch.abs(x.grad)
        
        return importance

    def get_action_preferences(self, x):
        """
        Get detailed action preferences for interpretability.
        
        Args:
            x: Input state tensor [batch_size, state_dim]
            
        Returns:
            Dictionary with action analysis
        """
        with torch.no_grad():
            q_values = self.forward(x)
            
            # Softmax for action probabilities
            action_probs = F.softmax(q_values, dim=1)
            
            # Action preferences
            action_names = ['Scale Down', 'Keep Same', 'Scale Up']
            
            results = {
                'q_values': q_values,
                'action_probabilities': action_probs,
                'preferred_action': q_values.argmax(dim=1),
                'confidence': action_probs.max(dim=1)[0],
                'action_names': action_names
            }
            
            return results


class LegacyQNetwork(nn.Module):
    """
    Legacy Q-Network for backward compatibility.
    Simpler architecture without advanced features.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

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
                layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        # Output layer (no dropout, no normalization)
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.network(x)


def create_qnetwork(state_dim: int, action_dim: int, hidden_dims=None, 
                   architecture='enhanced', **kwargs):
    """
    Factory function to create Q-networks with different architectures.
    
    Args:
        state_dim: Input state dimension
        action_dim: Number of actions
        hidden_dims: Hidden layer dimensions
        architecture: 'enhanced' or 'legacy'
        **kwargs: Additional arguments for enhanced network
        
    Returns:
        Q-network instance
    """
    if architecture == 'enhanced':
        return EnhancedQNetwork(state_dim, action_dim, hidden_dims, **kwargs)
    elif architecture == 'legacy':
        return LegacyQNetwork(state_dim, action_dim, hidden_dims)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
