import logging
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from collections import deque
import random

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class DQNAgent:
    """
    An upgraded Deep Q-Network Agent.

    This version includes a Target Network and the Double DQN (DDQN) algorithm
    for improved stability and performance.
    """
    def __init__(self, state_size, action_size):
        """
        Initializes the DQN agent and its components.

        Args:
            state_size (int): The number of features in the state vector.
            action_size (int): The number of possible actions.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)

        # --- Hyperparameters ---
        self.gamma = 0.95    # Discount factor
        self.epsilon = 1.0   # Initial exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        
        # ✨ --- Target Network --- ✨
        # The 'model' is the main network we train every step.
        # The 'target_model' is a clone that we only update periodically.
        # This provides a stable target for the main model to learn towards.
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model() # Initialize target model with same weights
        self.update_target_freq = 10 # How often to update the target network
        self.update_target_counter = 0


    def _build_model(self):
        """
        Builds the neural network for approximating Q-values.
        """
        model = Sequential([
            # ✨ FIX: Use a dedicated Input layer to define the model's input shape.
            # This is the modern Keras practice and removes the UserWarning.
            Input(shape=(self.state_size,)),
            
            # The rest of the layers no longer need to specify the input size.
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
        Stores an experience tuple in the replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Selects an action using an epsilon-greedy policy.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        """
        Trains the neural network using a random sample from the replay memory.
        This method now implements the Double DQN logic.
        """
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # ✨ --- Double DQN Logic --- ✨
                # 1. Use the main model to pick the best action for the next state.
                action_prime = np.argmax(self.model.predict(next_state, verbose=0)[0])
                
                # 2. Use the target model to get the Q-value of taking that action.
                # This decouples action selection from value estimation.
                q_future = self.target_model.predict(next_state, verbose=0)[0][action_prime]
                
                target = reward + self.gamma * q_future

            # Get current Q-values for the starting state from the main model
            target_f = self.model.predict(state, verbose=0)
            
            # Update the Q-value for the action that was actually taken
            target_f[0][action] = target
            
            # Train the main model
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Periodically update the target network
        self.update_target_counter += 1
        if self.update_target_counter > self.update_target_freq:
            self.update_target_model()
            self.update_target_counter = 0
