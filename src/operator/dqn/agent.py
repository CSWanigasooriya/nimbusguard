"""
DQN Agent for intelligent autoscaling decisions.
"""

import logging
import random
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .model import DQNModel
from .replay_buffer import ReplayBuffer, Experience
from .rewards import ContextAwareRewardCalculator
from metrics.collector import metrics

logger = logging.getLogger(__name__)


class DQNAgent:
    """
    Deep Q-Network agent for making scaling decisions with epsilon-greedy exploration
    and experience replay.
    """
    
    def __init__(self, 
                 state_size: int = 10,
                 action_size: int = 3,
                 hidden_dims: List[int] = [64, 32],
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon_start: float = 0.9,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 replay_buffer_size: int = 10000,
                 min_replay_size: int = 100,
                 batch_size: int = 32,
                 target_update_frequency: int = 100):
        """
        Initialize DQN Agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of actions (3: scale_down, no_action, scale_up)
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for neural network
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
            replay_buffer_size: Size of experience replay buffer
            min_replay_size: Minimum replay buffer size before training
            batch_size: Training batch size
            target_update_frequency: How often to update target network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        
        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Training counters
        self.training_steps = 0
        self.episodes = 0
        self.total_reward = 0.0
        
        # Initialize components
        self.model = DQNModel(
            state_size=state_size,
            action_size=action_size,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate
        )
        
        # Model persistence paths
        self.model_save_path = "/tmp/dqn_model.keras"
        
        self.replay_buffer = ReplayBuffer(
            capacity=replay_buffer_size,
            min_size=min_replay_size
        )
        
        self.reward_calculator = ContextAwareRewardCalculator()
        
        # Action mapping
        self.action_map = {
            0: 'scale_down',
            1: 'keep_same',  # Fixed: changed from 'no_action' to 'keep_same'
            2: 'scale_up'
        }
        
        self.reverse_action_map = {v: k for k, v in self.action_map.items()}
        
        # Store last state for experience creation
        self.last_state = None
        self.last_action = None
        self.last_deployment_context = None
        
        # Scaling history for context
        self.scaling_history = []
        self.max_history_length = 10
        
        logger.info(f"Initialized DQN Agent: state_size={state_size}, "
                   f"epsilon={epsilon_start}, buffer_size={replay_buffer_size}")
        
        # Try to load existing model from MinIO
        self._load_existing_model()
    
    def _load_existing_model(self):
        """Try to load existing DQN model from MinIO."""
        try:
            model_loaded = self.model.load_model(self.model_save_path, load_from_minio=True)
            if model_loaded:
                logger.info("🎯 Resumed DQN training from existing model")
                # TODO: Could also load training state (epsilon, steps, etc.) if stored
            else:
                logger.info("🚀 Starting DQN training with fresh model")
        except Exception as e:
            logger.warning(f"Failed to load existing model: {e}")
            logger.info("🚀 Starting DQN training with fresh model")
    
    def _create_state(self, 
                     metrics: Dict[str, float], 
                     current_replicas: int,
                     ready_replicas: int,
                     forecast: Optional[Dict[str, float]] = None,
                     deployment_context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Create context-aware state vector from metrics and deployment context.
        
        Args:
            metrics: Current metrics
            current_replicas: Current number of replicas
            ready_replicas: Number of ready replicas
            forecast: Forecast metrics (optional)
            deployment_context: Deployment context with resource constraints (optional)
            
        Returns:
            Context-aware state vector as numpy array
        """
        state_features = []
        
        # Current metrics (normalized)
        cpu_rate = metrics.get('process_cpu_seconds_total_rate', 0.0)
        memory_bytes = metrics.get('process_resident_memory_bytes', 0.0)
        replica_count = metrics.get('kube_deployment_status_replicas', current_replicas)
        
        # Get deployment context for context-aware features
        if deployment_context:
            resource_requests = deployment_context.get('resource_requests', {})
            cpu_request = resource_requests.get('cpu', 0.5)  # Default 0.5 cores
            memory_request = resource_requests.get('memory', 512 * 1024 * 1024)  # Default 512MB
            min_replicas = deployment_context.get('min_replicas', 1)
            max_replicas = deployment_context.get('max_replicas', 10)
        else:
            # Fallback defaults
            cpu_request = 0.5
            memory_request = 512 * 1024 * 1024
            min_replicas = 1
            max_replicas = 10
        
        # Basic metrics
        cpu_per_replica = cpu_rate / max(current_replicas, 1)
        memory_per_replica = memory_bytes / max(current_replicas, 1) / (1024 * 1024)  # MB
        
        # Context-aware resource utilization ratios
        total_cpu_capacity = cpu_request * current_replicas
        total_memory_capacity = memory_request * current_replicas / (1024 * 1024)  # MB
        
        cpu_utilization = cpu_rate / max(total_cpu_capacity, 0.001)
        memory_utilization = (memory_bytes / (1024 * 1024)) / max(total_memory_capacity, 1)
        
        # Scaling position within boundaries (normalized)
        replica_position = (current_replicas - min_replicas) / max(max_replicas - min_replicas, 1)
        
        state_features.extend([
            cpu_utilization,             # CPU utilization vs requests [0-1+]
            memory_utilization,          # Memory utilization vs requests [0-1+]
            cpu_per_replica,             # CPU per replica (absolute)
            memory_per_replica,          # Memory per replica (absolute, MB)
            current_replicas / max_replicas,  # Replica ratio vs max [0-1]
            ready_replicas / max(current_replicas, 1),  # Stability ratio [0-1]
            replica_position,            # Position within scaling bounds [0-1]
        ])
        
        # Add forecast features if available
        if forecast:
            forecast_cpu = forecast.get('process_cpu_seconds_total_rate', cpu_rate)
            forecast_memory = forecast.get('process_resident_memory_bytes', memory_bytes)
            
            state_features.extend([
                forecast_cpu,                           # Forecast CPU
                forecast_memory / (1024 * 1024),       # Forecast memory (MB)
                forecast_cpu - cpu_rate,                # CPU trend
            ])
        else:
            # Fill with zeros if no forecast
            state_features.extend([0.0, 0.0, 0.0])
        
        # Pad or truncate to state_size
        while len(state_features) < self.state_size:
            state_features.append(0.0)
        
        state_features = state_features[:self.state_size]
        
        return np.array(state_features, dtype=np.float32)
    
    def select_action(self, 
                     state: np.ndarray, 
                     deployment_context: Dict[str, Any],
                     use_epsilon: bool = True) -> Tuple[int, str, float, Dict[str, float]]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state vector
            deployment_context: Deployment context for constraints
            use_epsilon: Whether to use epsilon-greedy exploration
            
        Returns:
            Tuple of (action_index, action_name, confidence, q_values)
        """
        current_replicas = deployment_context.get('current_replicas', 1)
        
        # Get Q-values from the model
        q_values_dict = self.model.get_action_values(state)
        q_values = np.array([
            q_values_dict['scale_down'],
            q_values_dict['keep_same'],  # Fixed: changed from 'no_action' to 'keep_same'
            q_values_dict['scale_up']
        ])
        
        # Apply constraints
        valid_actions = [0, 1, 2]  # All actions initially valid
        
        # Can't scale down below 1 replica
        if current_replicas <= 1:
            q_values[0] = -np.inf  # Make scale_down invalid
            valid_actions.remove(0)
        
        # Can't scale up beyond reasonable limit
        if current_replicas >= 20:
            q_values[2] = -np.inf  # Make scale_up invalid
            valid_actions.remove(2)
        
        # Epsilon-greedy action selection
        is_exploration = False
        if use_epsilon and random.random() < self.epsilon:
            # Exploration: choose random valid action
            action_idx = random.choice(valid_actions)
            is_exploration = True
            logger.info(f"🎲 DQN Exploration: {self.action_map[action_idx]} (ε={self.epsilon:.3f})")
        else:
            # Exploitation: choose best action
            action_idx = np.argmax(q_values)
            logger.info(f"🎯 DQN Exploitation: {self.action_map[action_idx]} (Q={q_values[action_idx]:.3f})")
        
        action_name = self.action_map[action_idx]
        confidence = 1.0 - self.epsilon if not is_exploration else self.epsilon
        
        # Update metrics
        metrics.record_dqn_action(is_exploration)
        metrics.update_dqn_training(
            loss=0.0,  # Will be updated during training
            epsilon=self.epsilon,
            buffer_size=self.replay_buffer.size()
        )
        
        logger.info(f"🤖 DQN Action: {action_name} (conf={confidence:.3f}, exploration={is_exploration})")
        
        return action_idx, action_name, confidence, q_values_dict
    
    def store_experience(self,
                        current_metrics: Dict[str, float],
                        current_replicas: int,
                        ready_replicas: int,
                        action: str,
                        reward: float,
                        next_metrics: Dict[str, float],
                        next_replicas: int,
                        next_ready_replicas: int,
                        deployment_context: Dict[str, Any],
                        done: bool = False,
                        forecast: Optional[Dict[str, float]] = None,
                        next_forecast: Optional[Dict[str, float]] = None):
        """
        Store experience in replay buffer.
        """
        # Create states
        current_state = self._create_state(current_metrics, current_replicas, ready_replicas, forecast, deployment_context)
        next_state = self._create_state(next_metrics, next_replicas, next_ready_replicas, next_forecast, deployment_context)
        
        # Convert action to index
        action_idx = self.reverse_action_map[action]
        
        # Create experience
        experience = Experience(
            state=current_state,
            action=action_idx,
            reward=reward,
            next_state=next_state,
            done=done,
            deployment_context=deployment_context
        )
        
        self.replay_buffer.add(experience)
        metrics.add_experience()
        
        logger.debug(f"Stored experience: action={action}, reward={reward:.3f}, "
                    f"buffer_size={self.replay_buffer.size()}")
    
    def train(self) -> Optional[float]:
        """
        Train the DQN model using experience replay.
        
        Returns:
            Training loss or None if not enough experiences
        """
        if not self.replay_buffer.can_sample():
            logger.debug("Not enough experiences for training")
            return None
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return None
        
        # Prepare training data
        states = np.array([exp.state for exp in batch])
        actions = np.array([exp.action for exp in batch])
        rewards = np.array([exp.reward for exp in batch])
        next_states = np.array([exp.next_state for exp in batch])
        dones = np.array([exp.done for exp in batch])
        
        # Train the model
        loss = self.model.train_step(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            gamma=self.gamma
        )
        
        self.training_steps += 1
        
        # Update target network periodically
        if self.training_steps % self.target_update_frequency == 0:
            self.model.update_target_network()
            logger.info(f"Target network updated (step {self.training_steps})")
            
            # Save model to MinIO periodically (same frequency as target updates)
            try:
                self.model.save_model(self.model_save_path, save_to_minio=True)
                logger.debug(f"Model saved to MinIO (step {self.training_steps})")
            except Exception as e:
                logger.warning(f"Failed to save model to MinIO: {e}")
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update metrics
        metrics.update_dqn_training(
            loss=loss,
            epsilon=self.epsilon,
            buffer_size=self.replay_buffer.size()
        )
        
        logger.debug(f"Training step {self.training_steps}: loss={loss:.4f}, ε={self.epsilon:.3f}")
        
        return loss
    
    def calculate_reward(self,
                        action: str,
                        current_metrics: Dict[str, float],
                        deployment_context: Dict[str, Any],
                        forecast_metrics: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, float]]:
        """
        Calculate reward for the given action and state.
        
        Args:
            action: Action taken
            current_metrics: Current system metrics
            deployment_context: Deployment context
            forecast_metrics: Forecasted metrics (optional)
            
        Returns:
            Tuple of (reward, reward_components)
        """
        reward, components = self.reward_calculator.calculate_reward(
            action=action,
            current_metrics=current_metrics,
            deployment_context=deployment_context,
            forecast_metrics=forecast_metrics
        )
        
        self.total_reward += reward
        metrics.update_reward(reward)
        
        return reward, components
    
    def make_scaling_decision(self,
                            current_metrics: Dict[str, float],
                            current_replicas: int,
                            ready_replicas: int,
                            forecast: Optional[Dict[str, float]] = None,
                            resource_requests: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Make a scaling decision based on current state.
        
        Args:
            current_metrics: Current system metrics
            current_replicas: Current number of replicas
            ready_replicas: Number of ready replicas
            forecast: Forecasted metrics (optional)
            resource_requests: Resource requests per pod
            
        Returns:
            Decision dictionary with action, target_replicas, confidence, etc.
        """
        start_time = time.time()
        
        # Create deployment context first
        deployment_context = self.reward_calculator.get_deployment_context(
            current_replicas=current_replicas,
            ready_replicas=ready_replicas,
            target_replicas=current_replicas,  # Will be updated based on action
            resource_requests=resource_requests,
            scaling_history=self.scaling_history.copy()
        )
        
        # Create context-aware state vector
        state = self._create_state(current_metrics, current_replicas, ready_replicas, forecast, deployment_context)
        
        # Select action
        action_idx, action_name, confidence, q_values = self.select_action(
            state, deployment_context, use_epsilon=True
        )
        
        # Calculate target replicas
        if action_name == 'scale_up':
            target_replicas = min(current_replicas + 1, 20)
        elif action_name == 'scale_down':
            target_replicas = max(current_replicas - 1, 1)
        else:
            target_replicas = current_replicas
        
        # Update deployment context with target replicas
        deployment_context['target_replicas'] = target_replicas
        
        # Calculate reward (for monitoring/logging purposes)
        reward, reward_components = self.calculate_reward(
            action=action_name,
            current_metrics=current_metrics,
            deployment_context=deployment_context,
            forecast_metrics=forecast
        )
        
        # Update scaling history
        if action_name != 'keep_same':  # Fixed: changed from 'no_action' to 'keep_same'
            self.scaling_history.append(action_name)
            if len(self.scaling_history) > self.max_history_length:
                self.scaling_history.pop(0)
        
        # Update metrics
        metrics.update_dqn_decision(
            action=action_name,
            desired_replicas=target_replicas,
            current_replicas=current_replicas,
            confidence=confidence,
            q_values=q_values
        )
        
        decision_time = time.time() - start_time
        
        decision = {
            'action': action_name,
            'target_replicas': target_replicas,
            'current_replicas': current_replicas,
            'confidence': confidence,
            'q_values': q_values,
            'reward': reward,
            'reward_components': reward_components,
            'decision_time': decision_time,
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }
        
        logger.info(f"DQN decision: {action_name} ({current_replicas} → {target_replicas}), "
                   f"confidence={confidence:.3f}, reward={reward:.3f}, time={decision_time:.3f}s")
        
        return decision
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        buffer_stats = self.replay_buffer.get_stats()
        
        return {
            'training_steps': self.training_steps,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
            'total_reward': self.total_reward,
            'buffer_stats': buffer_stats,
            'scaling_history': self.scaling_history.copy()
        }
    
    def save_model(self, filepath: str):
        """Save the DQN model."""
        self.model.save_model(filepath)
        logger.info(f"DQN model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved DQN model."""
        self.model.load_model(filepath)
        logger.info(f"DQN model loaded from {filepath}") 