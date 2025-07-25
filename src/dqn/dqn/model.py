"""
Deep Q-Network models for scaling decisions.
"""

import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from typing import Tuple, List

logger = logging.getLogger(__name__)


class DQNModel:
    """Deep Q-Network model with main and target networks."""
    
    def __init__(self, 
                 state_size: int, 
                 action_size: int = 3,  # scale_down, no_action, scale_up
                 hidden_dims: List[int] = [64, 32],
                 learning_rate: float = 0.001):
        """
        Initialize DQN model.
        
        Args:
            state_size: Dimension of the state space
            action_size: Number of possible actions
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate for optimizer
        """
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        
        # Create main and target networks
        self.main_network = self._build_network("main")
        self.target_network = self._build_network("target")
        
        # Initialize target network with same weights as main network
        self.update_target_network()
        
        logger.info(f"Initialized DQN model: state_size={state_size}, action_size={action_size}, "
                   f"hidden_dims={hidden_dims}, lr={learning_rate}")
    
    def _build_network(self, name: str) -> keras.Model:
        """Build a neural network."""
        inputs = layers.Input(shape=(self.state_size,), name=f"{name}_input")
        
        # Add batch normalization to input
        x = layers.BatchNormalization(name=f"{name}_input_bn")(inputs)
        
        # Hidden layers with residual connections for deeper networks
        for i, hidden_dim in enumerate(self.hidden_dims):
            # Dense layer
            dense = layers.Dense(
                hidden_dim, 
                activation='relu',
                kernel_initializer='he_normal',
                name=f"{name}_dense_{i}"
            )(x)
            
            # Batch normalization
            bn = layers.BatchNormalization(name=f"{name}_bn_{i}")(dense)
            
            # Dropout for regularization
            dropout = layers.Dropout(0.2, name=f"{name}_dropout_{i}")(bn)
            
            # Residual connection if dimensions match
            if i > 0 and x.shape[-1] == dropout.shape[-1]:
                x = layers.Add(name=f"{name}_residual_{i}")([x, dropout])
            else:
                x = dropout
        
        # Output layer - Q-values for each action
        q_values = layers.Dense(
            self.action_size,
            activation='linear',  # Linear activation for Q-values
            kernel_initializer='glorot_uniform',
            name=f"{name}_q_values"
        )(x)
        
        model = keras.Model(inputs=inputs, outputs=q_values, name=f"{name}_network")
        
        # Compile with optimizer
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def predict(self, state: np.ndarray, use_target: bool = False) -> np.ndarray:
        """
        Predict Q-values for given state.
        
        Args:
            state: Input state
            use_target: Whether to use target network
            
        Returns:
            Q-values for each action
        """
        # Ensure state is properly shaped
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        network = self.target_network if use_target else self.main_network
        q_values = network.predict(state, verbose=0)
        
        return q_values
    
    def train_step(self, 
                   states: np.ndarray, 
                   actions: np.ndarray, 
                   rewards: np.ndarray, 
                   next_states: np.ndarray, 
                   dones: np.ndarray, 
                   gamma: float = 0.95) -> float:
        """
        Perform one training step using Double DQN.
        
        Args:
            states: Batch of current states
            actions: Batch of actions taken
            rewards: Batch of rewards received
            next_states: Batch of next states
            dones: Batch of done flags
            gamma: Discount factor
            
        Returns:
            Training loss
        """
        batch_size = states.shape[0]
        
        # Get current Q-values for all actions
        current_q_values = self.main_network.predict(states, verbose=0)
        
        # Get next Q-values from main network (for action selection)
        next_q_values_main = self.main_network.predict(next_states, verbose=0)
        
        # Get next Q-values from target network (for value estimation)
        next_q_values_target = self.target_network.predict(next_states, verbose=0)
        
        # Double DQN: use main network to select actions, target network to evaluate
        next_actions = np.argmax(next_q_values_main, axis=1)
        next_q_values = next_q_values_target[np.arange(batch_size), next_actions]
        
        # Calculate target Q-values
        target_q_values = current_q_values.copy()
        
        for i in range(batch_size):
            if dones[i]:
                target_q_values[i, actions[i]] = rewards[i]
            else:
                target_q_values[i, actions[i]] = rewards[i] + gamma * next_q_values[i]
        
        # Train the main network
        history = self.main_network.fit(
            states, 
            target_q_values, 
            batch_size=batch_size,
            epochs=1, 
            verbose=0
        )
        
        loss = history.history['loss'][0]
        logger.debug(f"Training step completed with loss: {loss:.4f}")
        
        return loss
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.set_weights(self.main_network.get_weights())
        logger.debug("Target network updated")
    
    def get_action_values(self, state: np.ndarray) -> dict:
        """
        Get Q-values for each action with readable names.
        
        Args:
            state: Input state
            
        Returns:
            Dictionary with action names and Q-values
        """
        q_values = self.predict(state, use_target=False)[0]
        
        return {
            'scale_down': float(q_values[0]),
            'keep_same': float(q_values[1]),  # Fixed: changed from 'no_action' to 'keep_same'
            'scale_up': float(q_values[2])
        }
    
    def save_model(self, filepath: str, save_to_minio: bool = True):
        """Save the main network model locally and optionally to MinIO."""
        try:
            # Save locally first
            self.main_network.save(filepath)
            logger.info(f"Model saved locally to {filepath}")
            
            # Save to MinIO if requested and available
            if save_to_minio:
                try:
                    from storage.minio_client import minio_storage
                    if minio_storage.client:
                        minio_storage.save_model(filepath)
                except ImportError:
                    logger.warning("MinIO storage not available")
                except Exception as e:
                    logger.warning(f"Failed to save to MinIO: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self, filepath: str, load_from_minio: bool = True):
        """Load a saved model from MinIO (if available) or local filesystem."""
        model_loaded = False
        
        # Try loading from MinIO first
        if load_from_minio:
            try:
                from storage.minio_client import minio_storage
                if minio_storage.client and minio_storage.model_exists():
                    if minio_storage.load_model(filepath):
                        model_loaded = True
                        logger.info("Model loaded from MinIO")
            except ImportError:
                logger.warning("MinIO storage not available")
            except Exception as e:
                logger.warning(f"Failed to load from MinIO: {e}")
        
        # Try loading from local filesystem
        if not model_loaded:
            try:
                import os
                if os.path.exists(filepath):
                    self.main_network = keras.models.load_model(filepath)
                    model_loaded = True
                    logger.info(f"Model loaded from local filesystem: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to load from local filesystem: {e}")
        
        # If model was loaded, rebuild target network
        if model_loaded:
            try:
                self.main_network = keras.models.load_model(filepath)
                
                # Rebuild target network and update it
                self.target_network = self._build_network("target")
                self.update_target_network()
                
                logger.info("Model and target network initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize loaded model: {e}")
                model_loaded = False
        else:
            logger.info("No existing model found, will start with fresh model")
        
        return model_loaded
    
    def get_model_summary(self) -> str:
        """Get a summary of the model architecture."""
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        try:
            self.main_network.summary()
            summary = buffer.getvalue()
        finally:
            sys.stdout = old_stdout
        
        return summary 