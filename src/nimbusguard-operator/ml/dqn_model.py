# engine/ml/dqn_model.py - Fixed DoubleDQNModel structure
# ============================================================================
# Deep Q-Network Model for Kubernetes Autoscaling
# ============================================================================

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

LOG = logging.getLogger(__name__)


class DQNModel(nn.Module):
    """
    Deep Q-Network for Kubernetes autoscaling decisions.
    This is the base neural network architecture.
    """

    def __init__(self,
                 state_dim: int = 11,
                 action_dim: int = 5,
                 hidden_dims: Tuple[int, ...] = (128, 256, 128),
                 dropout_rate: float = 0.1):
        """
        Initialize the DQN model.

        Args:
            state_dim: Dimension of the state space.
            action_dim: Number of possible actions.
            hidden_dims: Tuple defining the size of each hidden layer.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.input_norm = nn.LayerNorm(state_dim)

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network to get Q-values."""
        # Check dimensions before applying layer norm to prevent cryptic errors
        if state.size(-1) != self.state_dim:
            raise ValueError(f"Input state dimension mismatch: expected {self.state_dim}, got {state.size(-1)}")

        x = self.input_norm(state)
        return self.network(x)

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Get Q-values for a state without exploration (used for action selection)."""
        with torch.no_grad():
            if state.dim() == 1:
                # Ensure the state has the correct dimension before unsqueeze
                if state.size(0) != self.state_dim:
                    raise ValueError(f"Input state dimension mismatch: expected {self.state_dim}, got {state.size(0)}")
                state = state.unsqueeze(0)
            elif state.dim() == 2 and state.size(1) != self.state_dim:
                raise ValueError(f"Input state dimension mismatch: expected {self.state_dim}, got {state.size(1)}")

            return self.forward(state)


class DoubleDQNModel(nn.Module):
    """
    Double DQN variant with separate policy and target networks for stable learning.
    This class now *contains* two DQNModel instances (composition) instead of inheriting.
    """

    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        super().__init__()

        # Store dimensions for validation
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Contains two separate DQNModel instances
        self.policy_net = DQNModel(state_dim, action_dim, **kwargs)
        self.target_net = DQNModel(state_dim, action_dim, **kwargs)

        # Initialize target network with the same weights as the policy network
        self.update_target_network()
        # Target network is only for inference, so it's always in evaluation mode
        self.target_net.eval()

        LOG.info(f"Initialized DoubleDQNModel with state_dim={state_dim}, action_dim={action_dim}")

    def update_target_network(self):
        """Copies the weights from the policy network to the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # --- Pass-through methods to allow DQNAgent to interact with the policy_net ---

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the policy network."""
        return self.policy_net(state)

    def parameters(self, recurse: bool = True):
        """Returns the parameters of the policy network for the optimizer."""
        return self.policy_net.parameters(recurse)

    def train(self, mode: bool = True):
        """Sets the policy network to training mode. Target network always stays in eval mode."""
        self.policy_net.train(mode)
        self.target_net.eval()
        return self

    def eval(self):
        """Sets the policy network to evaluation mode."""
        self.policy_net.eval()
        self.target_net.eval()
        return self

    # --- Methods for DQN logic ---

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Get Q-values from the policy network for action selection."""
        # Validate dimensions before passing to policy_net
        if state.dim() == 1 and state.size(0) != self.state_dim:
            raise ValueError(
                f"Input state dimension mismatch in DoubleDQNModel: expected {self.state_dim}, got {state.size(0)}")
        elif state.dim() == 2 and state.size(1) != self.state_dim:
            raise ValueError(
                f"Input state dimension mismatch in DoubleDQNModel: expected {self.state_dim}, got {state.size(1)}")

        return self.policy_net.get_q_values(state)

    def get_target_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Get Q-values from the stable target network for loss calculation."""
        # Validate dimensions before passing to target_net
        if state.dim() == 1 and state.size(0) != self.state_dim:
            raise ValueError(
                f"Input state dimension mismatch in DoubleDQNModel: expected {self.state_dim}, got {state.size(0)}")
        elif state.dim() == 2 and state.size(1) != self.state_dim:
            raise ValueError(
                f"Input state dimension mismatch in DoubleDQNModel: expected {self.state_dim}, got {state.size(1)}")

        # The target_net is always in eval mode, so no_grad is implicitly handled
        return self.target_net.get_q_values(state)


# The PrioritizedReplayBuffer class does not need any changes.
# It is omitted here for brevity but should remain in your file.
class PrioritizedReplayBuffer:
    """
    Experience replay buffer with prioritized sampling for better learning efficiency.
    """

    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool,
             priority: Optional[float] = None):
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

    def sample(self, batch_size: int, beta: float = 0.4) -> Optional[Tuple]:
        if len(self.buffer) < batch_size:
            return None

        priorities = np.array(self.priorities[:len(self.buffer)])
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones), indices,
                np.array(weights))

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)
