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

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class DQNAgent:
    """
    An upgraded Deep Q-Network Agent with integrated Prometheus metrics.
    """
    def __init__(self, state_size, action_size):
        """
        Initializes the DQN agent and its components.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)

        # --- Hyperparameters ---
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        
        # --- Target Network ---
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.update_target_freq = 10
        self.update_target_counter = 0

    def _build_model(self):
        """
        Builds the neural network for approximating Q-values.
        """
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(24, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """
        Copies the weights from the main model to the target model.
        """
        logging.info("Updating target model weights.")
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience tuple and updates relevant metrics.
        """
        self.memory.append((state, action, reward, next_state, done))
        # METRICS: Update buffer size and experiences added
        metrics.DQN_REPLAY_BUFFER_SIZE.set(len(self.memory))
        metrics.DQN_EXPERIENCES_ADDED_TOTAL.inc()

    def act(self, state, return_exploration_status=False):
        """
        Selects an action using an epsilon-greedy policy and updates metrics.
        """
        # METRICS: Update epsilon gauge
        metrics.DQN_EPSILON_VALUE.set(self.epsilon)
        
        is_exploring = False
        if np.random.rand() <= self.epsilon:
            is_exploring = True
            action = random.randrange(self.action_size)
        else:
            act_values = self.model.predict(state, verbose=0)
            action = np.argmax(act_values[0])
        
        # METRICS: Update Q-values if they were calculated
        if not is_exploring and 'act_values' in locals():
            metrics.DQN_Q_VALUE_KEEP_SAME.set(act_values[0][0])
            metrics.DQN_Q_VALUE_SCALE_UP.set(act_values[0][1])
            metrics.DQN_Q_VALUE_SCALE_DOWN.set(act_values[0][2])

        # METRICS: Update exploration/exploitation counters
        if is_exploring:
            metrics.DQN_EXPLORATION_ACTIONS_TOTAL.inc()
        else:
            metrics.DQN_EXPLOITATION_ACTIONS_TOTAL.inc()

        if return_exploration_status:
            return action, is_exploring
        return action

    def replay(self):
        """
        Trains the neural network using a random sample from the replay memory.
        This version is vectorized for efficiency and returns the training loss.
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

        # METRICS: Update training loss and steps
        metrics.DQN_TRAINING_LOSS.set(loss)
        metrics.DQN_TRAINING_STEPS_TOTAL.inc()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Periodically update the target network
        self.update_target_counter += 1
        if self.update_target_counter >= self.update_target_freq:
            self.update_target_model()
            self.update_target_counter = 0
            
        return loss
