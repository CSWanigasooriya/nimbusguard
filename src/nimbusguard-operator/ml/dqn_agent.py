# engine/ml/dqn_agent.py - Optimized with model compatibility checks
# ============================================================================
# DQN Agent for Kubernetes Autoscaling
# ============================================================================

import logging
import os
import random
import time
from collections import deque
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

from .dqn_model import DoubleDQNModel, PrioritizedReplayBuffer
from .reward_system import RewardSystem, RewardComponents
from .state_representation import EnvironmentState, ScalingActions

LOG = logging.getLogger(__name__)


class DQNAgent:
    """
    DQN Agent for intelligent Kubernetes autoscaling, optimized for a 11-dimensional state.
    """

    def __init__(self,
                 state_dim: int = 11,
                 action_dim: int = 5,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.95,
                 epsilon_start: float = 0.1,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 32,
                 memory_size: int = 100000,
                 target_update_freq: int = 1000,
                 device: Optional[torch.device] = None,
                 model_path: Optional[str] = None):
        """
        Initialize DQN Agent.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        LOG.info(f"DQN Agent using device: {self.device}")

        # Initialize networks and dependencies
        self.policy_net = DoubleDQNModel(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = PrioritizedReplayBuffer(memory_size)
        self.reward_system = RewardSystem()

        # Internal trackers
        self.training_step_count = 0
        self.total_decisions = 0
        self.loss_history = deque(maxlen=1000)

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            LOG.info("No pre-trained model found. Initializing DQN Agent with random weights.")

    def select_action(self,
                      state: EnvironmentState,
                      training: bool = False,
                      force_valid: bool = True) -> Tuple[ScalingActions, Dict[str, Any]]:
        """Selects a scaling action using an epsilon-greedy policy."""
        state_vector = state.to_dqn_input()

        # Validate state dimensions before processing
        if len(state_vector) != self.state_dim:
            LOG.error(f"State dimension mismatch: Expected {self.state_dim}, got {len(state_vector)}")
            LOG.warning("Falling back to NO_ACTION due to dimension mismatch")
            return ScalingActions.NO_ACTION, {"error": "state_dimension_mismatch"}

        state_tensor = torch.FloatTensor(state_vector).to(self.device)
        valid_actions = state.get_valid_actions() if force_valid else list(ScalingActions)

        # Initialize action to silence the "might be referenced before assignment" warning
        # It will always be overwritten in the logic below
        action = valid_actions[0] if valid_actions else ScalingActions.NO_ACTION

        decision_metadata = {
            "timestamp": time.time(),
            "epsilon": self.epsilon,
            "training_mode": training,
            "valid_actions": [action.name for action in valid_actions],
        }

        if training and random.random() < self.epsilon:
            action = random.choice(valid_actions)
            decision_metadata["decision_type"] = False  # False for exploration
        else:
            with torch.no_grad():
                q_values = self.policy_net.get_q_values(state_tensor)
                decision_metadata["q_values"] = q_values.cpu().numpy().tolist()

                if force_valid:
                    valid_indices = [action.value for action in valid_actions]
                    masked_q_values = torch.full((1, self.action_dim), float('-inf'), device=self.device)
                    masked_q_values[0, valid_indices] = q_values[0, valid_indices]
                    action_idx = masked_q_values.argmax().item()
                    action = ScalingActions(action_idx)
                else:
                    action_idx = q_values.argmax().item()
                    action = ScalingActions(action_idx)

                decision_metadata["decision_type"] = True  # True for exploitation
                decision_metadata["max_q_value"] = float(q_values.max())

        if training and self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        self.total_decisions += 1
        return action, decision_metadata

    def store_experience(self,
                         state: EnvironmentState,
                         action: ScalingActions,
                         reward_components: RewardComponents,
                         next_state: EnvironmentState,
                         done: bool = False):
        """Calculates TD error and stores the experience in the replay buffer."""
        state_vec = state.to_dqn_input()
        next_state_vec = next_state.to_dqn_input()

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state_vec).unsqueeze(0).to(self.device)
            current_q = self.policy_net(state_tensor)[0][action.value]
            next_q = self.policy_net.get_target_q_values(next_state_tensor).max(1)[0]
            target_q = reward_components.total_reward + (self.gamma * next_q * (1 - int(done)))
            td_error = abs(current_q - target_q).item()

        self.memory.push(state_vec, action.value, reward_components.total_reward, next_state_vec, done,
                         priority=td_error + 1e-6)

    def train_step(self) -> Optional[Dict[str, float]]:
        """Performs one training step by sampling from the replay buffer."""
        if len(self.memory) < self.batch_size: return None
        batch = self.memory.sample(self.batch_size, beta=0.4)
        if batch is None: return None

        states, actions, rewards, next_states, dones, indices, weights = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.policy_net.get_target_q_values(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        td_errors = target_q_values - current_q_values.squeeze()
        loss = (weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.memory.update_priorities(indices, np.abs(td_errors.detach().cpu().numpy()) + 1e-6)

        self.training_step_count += 1
        if self.training_step_count % self.target_update_freq == 0:
            self.policy_net.update_target_network()
            LOG.info(f"Updated target network at step {self.training_step_count}")

        self.loss_history.append(loss.item())
        return {"loss": loss.item(), "epsilon": self.epsilon}

    def evaluate_decision(self, *args, **kwargs) -> RewardComponents:
        """Evaluates a scaling decision by calculating its reward."""
        return self.reward_system.calculate_reward(*args, **kwargs)

    def save_model(self, filepath: str):
        """Saves the model and agent's state, including config for compatibility checks."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step_count': self.training_step_count,
            'total_decisions': self.total_decisions,
            'agent_config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
            }
        }
        torch.save(checkpoint, filepath)
        LOG.info(f"DQN Agent saved to {filepath}")

    def load_model(self, filepath: str):
        """Loads a model and agent's state, now with a compatibility check."""
        LOG.info(f"Attempting to load model from {filepath}...")
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
        except Exception as e:
            LOG.error(f"Failed to load model from {filepath}: {e}")
            LOG.warning("Agent will start with random weights.")
            return

        # --- Enhanced Compatibility Check ---
        agent_config = checkpoint.get('agent_config', {})
        checkpoint_state_dim = agent_config.get('state_dim')

        if checkpoint_state_dim is None:
            LOG.warning(f"Model checkpoint doesn't contain state_dim. This may be an older model format.")
            LOG.info(f"Current agent is configured for state_dim={self.state_dim}. Will attempt to load anyway.")

        elif checkpoint_state_dim != self.state_dim:
            LOG.error(f"MODEL INCOMPATIBLE! Checkpoint has state_dim={checkpoint_state_dim}, "
                      f"but agent is configured for state_dim={self.state_dim}.")
            LOG.warning(
                f"Please delete the old model file ('{filepath}') and restart the operator to train a new model.")
            LOG.warning("Ignoring incompatible model file. Agent will start with random weights.")
            return  # IMPORTANT: Abort loading the incompatible model

        # If compatible, proceed with loading the state
        try:
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
            self.training_step_count = checkpoint.get('training_step_count', 0)
            self.total_decisions = checkpoint.get('total_decisions', 0)
            self.policy_net.update_target_network()
            LOG.info(f"Successfully loaded and validated compatible model from {filepath}")
        except Exception as e:
            LOG.error(f"Error while loading model components: {e}")
            LOG.warning("Agent will start with random weights.")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Returns a dictionary of the agent's current performance metrics."""
        return {
            "total_decisions": self.total_decisions,
            "current_epsilon": self.epsilon,
            "training_steps": self.training_step_count,
            "avg_recent_loss": np.mean(self.loss_history) if self.loss_history else 0.0,
            "memory_utilization_percent": len(self.memory) / self.memory.capacity * 100,
        }

    def set_training_mode(self, training: bool):
        """Sets the agent to training or evaluation mode."""
        self.policy_net.train(training)
        if not training:
            self.epsilon = self.epsilon_end
        LOG.info(f"DQN Agent set to {'TRAINING' if training else 'EVALUATION'} mode.")
