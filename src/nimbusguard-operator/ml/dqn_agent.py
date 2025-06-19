# engine/ml/dqn_agent.py
# ============================================================================
# DQN Agent for Kubernetes Autoscaling
# ============================================================================

import os
import time
import random
import logging
from typing import Dict, Any, Optional, Tuple, List
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .dqn_model import DQNModel, DoubleDQNModel, PrioritizedReplayBuffer
from .state_representation import EnvironmentState, ScalingActions
from .reward_system import RewardSystem, RewardComponents

LOG = logging.getLogger(__name__)

class DQNAgent:
    """
    DQN Agent for intelligent Kubernetes autoscaling.
    
    Integrates with your existing observability stack to make scaling decisions.
    """
    
    def __init__(self, 
                 state_dim: int = 43,
                 action_dim: int = 5,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.95,
                 epsilon_start: float = 0.1,  # Start with low exploration in production
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 32,
                 memory_size: int = 100000,
                 target_update_freq: int = 1000,
                 device: Optional[torch.device] = None,
                 model_path: Optional[str] = None):
        """
        Initialize DQN Agent.
        
        Args:
            state_dim: Dimension of state space (43 for your observability features)
            action_dim: Number of possible actions (5 scaling actions)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            batch_size: Training batch size
            memory_size: Replay buffer size
            target_update_freq: Frequency to update target network
            device: PyTorch device (auto-detected if None)
            model_path: Path to pre-trained model (optional)
        """
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        LOG.info(f"DQN Agent using device: {self.device}")
        
        # Initialize networks
        self.policy_net = DoubleDQNModel(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()
        
        # Experience replay
        self.memory = PrioritizedReplayBuffer(memory_size)
        
        # Reward system
        self.reward_system = RewardSystem()
        
        # Training metrics
        self.training_step = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.loss_history = deque(maxlen=1000)
        
        # Production metrics
        self.total_decisions = 0
        self.successful_decisions = 0
        self.last_action_time = 0
        self.recent_states = deque(maxlen=10)  # For experience collection
        
        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            LOG.info(f"Loaded pre-trained model from {model_path}")
        else:
            LOG.info("Initialized DQN Agent with random weights")
    
    def select_action(self, 
                     state: EnvironmentState, 
                     training: bool = False,
                     force_valid: bool = True) -> Tuple[ScalingActions, Dict[str, Any]]:
        """
        Select scaling action using epsilon-greedy policy.
        
        Args:
            state: Current environment state
            training: Whether in training mode (affects exploration)
            force_valid: Whether to ensure action is valid given constraints
            
        Returns:
            Tuple of (selected_action, decision_metadata)
        """
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state.to_dqn_input()).to(self.device)
        
        # Get valid actions if constraint enforcement is enabled
        valid_actions = state.get_valid_actions() if force_valid else list(ScalingActions)
        
        decision_metadata = {
            "timestamp": time.time(),
            "epsilon": self.epsilon,
            "training_mode": training,
            "valid_actions": [action.name for action in valid_actions],
            "state_health_score": state.health_score,
            "state_confidence": state.confidence_score
        }
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            # Exploration: random valid action
            action = random.choice(valid_actions)
            decision_metadata["decision_type"] = "exploration"
            LOG.debug(f"Exploration action selected: {action.name}")
        else:
            # Exploitation: best Q-value action
            with torch.no_grad():
                q_values = self.policy_net.get_q_values(state_tensor)
                decision_metadata["q_values"] = q_values.cpu().numpy().tolist()
                
                if force_valid:
                    # Mask invalid actions
                    valid_indices = [action.value for action in valid_actions]
                    masked_q_values = torch.full((self.action_dim,), float('-inf'))
                    masked_q_values[valid_indices] = q_values[0][valid_indices]
                    action_idx = masked_q_values.argmax().item()
                else:
                    action_idx = q_values.argmax().item()
                
                action = ScalingActions(action_idx)
                decision_metadata["decision_type"] = "exploitation"
                decision_metadata["max_q_value"] = float(q_values.max())
                LOG.debug(f"Exploitation action selected: {action.name} (Q-value: {q_values.max():.3f})")
        
        # Update exploration rate
        if training and self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        # Update metrics
        self.total_decisions += 1
        self.last_action_time = time.time()
        
        return action, decision_metadata
    
    def store_experience(self, 
                        state: EnvironmentState,
                        action: ScalingActions,
                        reward_components: RewardComponents,
                        next_state: EnvironmentState,
                        done: bool = False):
        """Store experience in replay buffer"""
        
        state_vector = state.to_dqn_input()
        next_state_vector = next_state.to_dqn_input()
        
        # Calculate TD error for prioritized replay
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state_vector).unsqueeze(0).to(self.device)
            
            current_q = self.policy_net(state_tensor)[0][action.value]
            next_q = self.policy_net.get_target_q_values(next_state_tensor).max(1)[0]
            target_q = reward_components.total_reward + (self.gamma * next_q * (1 - done))
            
            td_error = abs(current_q - target_q).item()
        
        self.memory.push(
            state_vector,
            action.value,
            reward_components.total_reward,
            next_state_vector,
            done,
            priority=td_error + 1e-6  # Small epsilon to avoid zero priority
        )
        
        LOG.debug(f"Stored experience: action={action.name}, reward={reward_components.total_reward:.2f}")
    
    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Perform one training step if enough experiences are available.
        
        Returns:
            Training metrics or None if not enough data
        """
        
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        batch = self.memory.sample(self.batch_size, beta=0.4)
        if batch is None:
            return None
        
        states, actions, rewards, next_states, dones, indices, weights = batch
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.policy_net.get_target_q_values(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Calculate loss with importance sampling weights
        td_errors = current_q_values.squeeze() - target_q_values
        loss = (weights * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update priorities
        new_priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
        self.memory.update_priorities(indices, new_priorities)
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.policy_net.update_target_network()
            LOG.info(f"Updated target network at step {self.training_step}")
        
        # Track training metrics
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        return {
            "loss": loss_value,
            "training_step": self.training_step,
            "epsilon": self.epsilon,
            "q_value_mean": current_q_values.mean().item(),
            "td_error_mean": abs(td_errors).mean().item(),
            "memory_size": len(self.memory)
        }
    
    def evaluate_decision(self, 
                         previous_state: EnvironmentState,
                         action: ScalingActions,
                         current_state: EnvironmentState,
                         execution_success: bool = True) -> RewardComponents:
        """
        Evaluate a scaling decision and calculate reward.
        
        Args:
            previous_state: State before action
            action: Action that was taken
            current_state: State after action
            execution_success: Whether the action executed successfully
            
        Returns:
            Reward components for the decision
        """
        
        reward_components = self.reward_system.calculate_reward(
            previous_state, action, current_state, execution_success
        )
        
        if execution_success:
            self.successful_decisions += 1
        
        LOG.info(f"Decision evaluation - Action: {action.name}, "
                f"Total Reward: {reward_components.total_reward:.2f}")
        LOG.debug(self.reward_system.get_reward_explanation(reward_components))
        
        return reward_components
    
    def save_model(self, filepath: str):
        """Save model and agent state"""
        
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'total_decisions': self.total_decisions,
            'successful_decisions': self.successful_decisions,
            'agent_config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq
            }
        }
        
        torch.save(checkpoint, filepath)
        LOG.info(f"DQN Agent saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model and agent state"""
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.training_step = checkpoint.get('training_step', 0)
        self.total_decisions = checkpoint.get('total_decisions', 0)
        self.successful_decisions = checkpoint.get('successful_decisions', 0)
        
        # Update target network
        self.policy_net.update_target_network()
        
        LOG.info(f"DQN Agent loaded from {filepath}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current agent performance metrics"""
        
        success_rate = (self.successful_decisions / max(self.total_decisions, 1)) * 100
        avg_loss = np.mean(self.loss_history) if self.loss_history else 0.0
        
        return {
            "total_decisions": self.total_decisions,
            "successful_decisions": self.successful_decisions,
            "success_rate_percent": success_rate,
            "current_epsilon": self.epsilon,
            "training_step": self.training_step,
            "avg_recent_loss": avg_loss,
            "memory_utilization": len(self.memory) / self.memory.capacity * 100,
            "last_action_time": self.last_action_time
        }
    
    def set_training_mode(self, training: bool):
        """Set agent to training or evaluation mode"""
        
        if training:
            self.policy_net.train()
            LOG.info("DQN Agent set to training mode")
        else:
            self.policy_net.eval()
            self.epsilon = self.epsilon_end  # Minimal exploration in production
            LOG.info("DQN Agent set to evaluation mode")
