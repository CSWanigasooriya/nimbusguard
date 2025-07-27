import logging
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from collections import deque
import random

# METRICS: Import the metric objects
import metrics

# Import configuration and storage
from config import dqn_config

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class DQNAgent:
    """
    An upgraded Deep Q-Network Agent with integrated Prometheus metrics, configurable hyperparameters,
    and persistent model storage.
    """
    def __init__(self, state_size, action_size, config=None):
        """
        Initializes the DQN agent and its components.
        
        Args:
            state_size: Size of the state vector
            action_size: Number of possible actions
            config: DQNConfig object (optional, uses global config if None)
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Use provided config or global config
        self.config = config if config is not None else dqn_config
        
        # Initialize replay memory with configurable size
        self.memory = deque(maxlen=self.config.memory_size)

        # --- Hyperparameters from configuration ---
        self.gamma = self.config.gamma
        self.epsilon = self.config.epsilon
        self.epsilon_min = self.config.epsilon_min
        self.epsilon_decay = self.config.epsilon_decay
        self.learning_rate = self.config.learning_rate
        self.batch_size = self.config.batch_size
        
        # --- Target Network Configuration ---
        self.update_target_freq = self.config.update_target_freq
        self.update_target_counter = 0
        
        # --- Network Architecture ---
        self.hidden_units = self.config.hidden_units
        self.hidden_layers = self.config.hidden_layers
        
        # --- Model Persistence ---
        self.save_frequency = self.config.save_frequency
        self.save_on_improvement = self.config.save_on_improvement
        self.auto_load_model = self.config.auto_load_model
        self.training_steps = 0
        self.best_avg_reward = float('-inf')
        self.recent_rewards = deque(maxlen=100)  # Track recent rewards for improvement detection
        
        # Initialize storage (lazy loading to avoid startup dependencies)
        self._storage = None
        
        # Build models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Auto-load existing model if enabled
        if self.auto_load_model:
            self._try_load_model()
        
        # Log initialization
        logging.info(f"DQN Agent initialized with state_size={state_size}, action_size={action_size}")
        logging.info(f"Using configuration: memory_size={self.config.memory_size}, "
                    f"learning_rate={self.learning_rate}, batch_size={self.batch_size}")
        logging.info(f"Model persistence: save_freq={self.save_frequency}, "
                    f"save_on_improvement={self.save_on_improvement}, auto_load={self.auto_load_model}")

    @property
    def storage(self):
        """Lazy initialization of storage client."""
        if self._storage is None:
            try:
                from storage import DQNModelStorage
                self._storage = DQNModelStorage()
            except Exception as e:
                logging.warning(f"Failed to initialize DQN storage: {e}")
                self._storage = None
        return self._storage

    def _build_model(self):
        """
        Builds the neural network for approximating Q-values using configurable architecture.
        """
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        
        # Add configurable number of hidden layers
        for i in range(self.hidden_layers):
            model.add(Dense(self.hidden_units, activation='relu', name=f'hidden_{i+1}'))
        
        # Output layer
        model.add(Dense(self.action_size, activation='linear', name='output'))
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        # Log model architecture
        if hasattr(self, 'model') and self.model is None:  # Only log for the first model
            logging.info(f"DQN model architecture: {self.hidden_layers} hidden layers "
                        f"with {self.hidden_units} units each")
        
        return model

    def _try_load_model(self):
        """Try to load existing model from storage."""
        if not self.storage:
            logging.warning("Storage not available - skipping model loading")
            return
        
        try:
            success, metadata = self.storage.load_model(self)
            if success:
                logging.info("🔄 DQN model loaded from persistent storage")
                if metadata:
                    self.training_steps = metadata.get('training_steps', 0)
                    if 'best_avg_reward' in metadata:
                        self.best_avg_reward = metadata['best_avg_reward']
            else:
                logging.info("No existing DQN model found - starting with fresh model")
        except Exception as e:
            logging.warning(f"Failed to load existing model: {e}")

    def _should_save_model(self, current_reward=None):
        """
        Determine if the model should be saved based on training steps and improvement.
        
        Args:
            current_reward: Current reward value (optional)
            
        Returns:
            Tuple of (should_save, reason)
        """
        # Always save based on frequency
        if self.training_steps > 0 and self.training_steps % self.save_frequency == 0:
            return True, f"periodic save (every {self.save_frequency} steps)"
        
        # Save on improvement if enabled and we have a reward
        if self.save_on_improvement and current_reward is not None:
            self.recent_rewards.append(current_reward)
            
            # Only check for improvement if we have enough recent rewards
            if len(self.recent_rewards) >= 50:
                current_avg_reward = np.mean(list(self.recent_rewards))
                
                if current_avg_reward > self.best_avg_reward:
                    improvement = current_avg_reward - self.best_avg_reward
                    self.best_avg_reward = current_avg_reward
                    return True, f"performance improvement (+{improvement:.3f} avg reward)"
        
        return False, "no save criteria met"

    def _save_model(self, reason="manual", additional_metadata=None):
        """
        Save the current model to storage.
        
        Args:
            reason: Reason for saving
            additional_metadata: Additional metadata to include
        """
        if not self.storage:
            logging.warning("Storage not available - skipping model save")
            return False
        
        try:
            metadata = {
                'save_reason': reason,
                'best_avg_reward': self.best_avg_reward,
                'recent_rewards_count': len(self.recent_rewards),
                'recent_avg_reward': np.mean(list(self.recent_rewards)) if self.recent_rewards else 0.0
            }
            
            if additional_metadata:
                metadata.update(additional_metadata)
            
            success = self.storage.save_model(self, metadata)
            if success:
                logging.info(f"💾 DQN model saved - {reason}")
            return success
            
        except Exception as e:
            logging.error(f"Failed to save model: {e}")
            return False

    def update_target_model(self):
        """
        Copies the weights from the main model to the target model.
        """
        logging.debug("Updating target model weights.")
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience tuple and updates relevant metrics.
        """
        self.memory.append((state, action, reward, next_state, done))
        # METRICS: Update buffer size and experiences added (agent-level metrics)
        metrics.DQN_REPLAY_BUFFER_SIZE.set(len(self.memory))
        metrics.DQN_EXPERIENCES_ADDED_TOTAL.inc()

    def act(self, state, return_exploration_status=False):
        """
        Selects an action using an epsilon-greedy policy and updates metrics.
        """
        # METRICS: Update epsilon gauge (agent-level metric)
        metrics.DQN_EPSILON_VALUE.set(self.epsilon)
        
        is_exploring = False
        if np.random.rand() <= self.epsilon:
            is_exploring = True
            action = random.randrange(self.action_size)
        else:
            act_values = self.model.predict(state, verbose=0)
            action = np.argmax(act_values[0])

        # METRICS: Update exploration/exploitation counters (agent-level metrics)
        if is_exploring:
            metrics.DQN_EXPLORATION_ACTIONS_TOTAL.inc()
        else:
            metrics.DQN_EXPLOITATION_ACTIONS_TOTAL.inc()

        if return_exploration_status:
            return action, is_exploring
        return action

    def replay(self, current_reward=None):
        """
        Trains the neural network using a random sample from the replay memory.
        This version is vectorized for efficiency and includes model persistence.
        
        Args:
            current_reward: Current reward value for save-on-improvement logic
            
        Returns:
            Training loss or None if not enough samples
        """
        if len(self.memory) < self.batch_size:
            return None  # Not enough samples to train

        minibatch = random.sample(self.memory, self.batch_size)

        # Vectorized implementation for efficiency
        states = np.array([transition[0] for transition in minibatch]).reshape(-1, self.state_size)
        next_states = np.array([transition[3] for transition in minibatch]).reshape(-1, self.state_size)

        q_values_current = self.model.predict(states, verbose=0)
        q_values_next_main = self.model.predict(next_states, verbose=0)
        q_values_next_target = self.target_model.predict(next_states, verbose=0)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                target = reward
            else:
                action_prime = np.argmax(q_values_next_main[i])
                q_future = q_values_next_target[i][action_prime]
                target = reward + self.gamma * q_future
            
            q_values_current[i][action] = target

        history = self.model.fit(states, q_values_current, epochs=1, verbose=0)
        loss = history.history['loss'][0]

        # Increment training steps
        self.training_steps += 1

        # METRICS: Update training loss and steps (agent-level metrics)
        metrics.DQN_TRAINING_LOSS.set(loss)
        metrics.DQN_TRAINING_STEPS_TOTAL.inc()

        # Decay epsilon and update epsilon metric
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            # Update epsilon metric after decay
            metrics.DQN_EPSILON_VALUE.set(self.epsilon)
            
        # Periodically update the target network
        self.update_target_counter += 1
        if self.update_target_counter >= self.update_target_freq:
            self.update_target_model()
            self.update_target_counter = 0
        
        # Check if we should save the model
        should_save, save_reason = self._should_save_model(current_reward)
        if should_save:
            self._save_model(save_reason)
            
        return loss

    def get_config(self):
        """
        Returns the current configuration as a dictionary.
        """
        return {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'memory_size': len(self.memory),
            'memory_maxlen': self.memory.maxlen,
            'current_epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'best_avg_reward': self.best_avg_reward,
            **self.config.to_dict()
        }

    def reset_epsilon(self, new_epsilon=None):
        """
        Reset epsilon to initial value or specified value.
        
        Args:
            new_epsilon: New epsilon value (optional, uses config default if None)
        """
        old_epsilon = self.epsilon
        self.epsilon = new_epsilon if new_epsilon is not None else self.config.epsilon
        logging.info(f"Epsilon reset from {old_epsilon:.3f} to {self.epsilon:.3f}")
        metrics.DQN_EPSILON_VALUE.set(self.epsilon)

    def save_weights(self, filepath):
        """
        Save the model weights to a file.
        
        Args:
            filepath: Path to save the weights
        """
        self.model.save_weights(filepath)
        logging.info(f"DQN model weights saved to {filepath}")

    def load_weights(self, filepath):
        """
        Load model weights from a file.
        
        Args:
            filepath: Path to load the weights from
        """
        self.model.load_weights(filepath)
        self.update_target_model()  # Update target model with loaded weights
        logging.info(f"DQN model weights loaded from {filepath}")

    def save_to_storage(self, reason="manual"):
        """
        Manually save the model to persistent storage.
        
        Args:
            reason: Reason for saving
            
        Returns:
            True if save was successful
        """
        return self._save_model(reason)

    def get_storage_status(self):
        """
        Get the status of the model storage system.
        
        Returns:
            Dictionary with storage status information
        """
        if not self.storage:
            return {'available': False, 'error': 'Storage not initialized'}
        
        try:
            status = self.storage.get_storage_status()
            status.update({
                'available': True,
                'training_steps': self.training_steps,
                'best_avg_reward': self.best_avg_reward,
                'auto_save_enabled': self.save_frequency > 0 or self.save_on_improvement
            })
            return status
        except Exception as e:
            return {'available': False, 'error': str(e)}
