"""DQN agent that uses both current metrics and forecast data for proactive scaling decisions."""
import asyncio
import logging
import os
import random
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Dict, List, Optional, Any, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from config.settings import load_config, ScalingConfig
from dqn.model import DQNNetwork
from dqn.rewards import ProactiveRewardCalculator

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Experience replay buffer for DQN training."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self._lock = threading.Lock()

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer (thread-safe)."""
        experience = (state, action, reward, next_state, done)
        with self._lock:
            self.buffer.append(experience)

    def sample(self, batch_size: int):
        """Sample a batch of experiences (thread-safe)."""
        with self._lock:
            return random.sample(self.buffer, batch_size)

    def __len__(self):
        with self._lock:
            return len(self.buffer)


class ProactiveDQNAgent:
    """
    Enhanced DQN agent that uses forecast data for proactive scaling decisions.
    Includes model persistence to MinIO storage and feature scaling with async training.
    """

    def __init__(self, config: ScalingConfig, device: Optional[str] = None, minio_client=None):
        self.config = config
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.minio_client = minio_client

        # Feature scaler for preprocessing
        self.scaler = None
        self.scaler_path = "/tmp/feature_scaler.pkl"  # Must match DataPreprocessor
        self._load_feature_scaler()

        # State and action dimensions
        # 9 current + 9 forecast + 4 meta = 22 total features
        self.state_dim = len(config.selected_features) * 2 + 4  # current + forecast + meta
        self.action_dim = 3  # scale_down, keep_same, scale_up

        # Initialize networks
        self.q_network = DQNNetwork(
            state_size=self.state_dim,
            action_size=self.action_dim,
            hidden_size=config.dqn_hidden_dims[0]  # Use first hidden dimension
        ).to(self.device)

        self.target_network = DQNNetwork(
            state_size=self.state_dim,
            action_size=self.action_dim,
            hidden_size=config.dqn_hidden_dims[0]  # Use first hidden dimension
        ).to(self.device)

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Training components
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config.dqn_learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=config.dqn_memory_capacity)

        # Exploration parameters
        self.epsilon = config.dqn_epsilon_start
        self.epsilon_decay = config.dqn_epsilon_decay
        self.epsilon_min = config.dqn_epsilon_end

        # Training state
        self.training_steps = 0
        self.last_save_time = time.time()
        self.last_loss = None

        # Async training setup
        self.training_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="DQN-Training")
        self.training_lock = asyncio.Lock()
        self.is_training = False

        # Training frequency control
        self.min_training_interval = 120  # 2 minutes minimum between training sessions
        self.last_training_time = 0

        # Target network update frequency
        self.target_update_frequency = 1000  # Update target network every 1000 steps

        # Reward calculator
        self.reward_calculator = ProactiveRewardCalculator()

        logger.info(f"ProactiveDQNAgent initialized with async training: "
                    f"state_dim={self.state_dim}, action_dim={self.action_dim}, device={self.device}")

    def _load_feature_scaler(self):
        if os.path.exists(self.scaler_path):
            try:
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Loaded feature scaler from {self.scaler_path}")
            except Exception as e:
                logger.warning(f"Failed to load feature scaler: {e}")
                self.scaler = None
        else:
            logger.warning(f"Feature scaler file not found at {self.scaler_path}")
            self.scaler = None

    def _scale_features(self, raw_features: List[float]) -> List[float]:
        """
        Scale raw features using the loaded scaler.
        """
        if not self.scaler:
            logger.warning("No feature scaler loaded, returning raw features")
            logger.debug(f"Raw features (no scaler): {raw_features}")
            return raw_features

        try:
            # Create DataFrame with feature names to match how scaler was fitted
            raw_features_df = pd.DataFrame([raw_features], columns=self.config.selected_features)
            scaled_features = self.scaler.transform(raw_features_df)
            logger.debug(f"Raw features: {raw_features}")
            logger.debug(f"Scaled features: {scaled_features[0].tolist()}")
            return scaled_features[0].tolist()
        except Exception as e:
            logger.warning(f"Feature scaling failed: {e}, returning raw features")
            logger.debug(f"Raw features (scaling failed): {raw_features}")
            return raw_features

    def create_state_vector(self, current_metrics: Dict[str, float],
                            forecast_metrics: Optional[Dict[str, float]] = None,
                            current_replicas: int = 1) -> np.ndarray:
        """
        Create 22-dimensional state vector from current metrics, forecast, and metadata.
        """
        state = []

        # 1. Current metrics (9 features) - scaled
        current_raw_features = [current_metrics.get(feature, 0.0) for feature in self.config.selected_features]
        current_scaled_features = self._scale_features(current_raw_features)
        state.extend(current_scaled_features)

        # 2. Forecast metrics (9 features) - scaled
        if forecast_metrics:
            forecast_raw_features = [forecast_metrics.get(feature, 0.0) for feature in self.config.selected_features]
            forecast_scaled_features = self._scale_features(forecast_raw_features)
            state.extend(forecast_scaled_features)
        else:
            # Use current metrics as fallback forecast (already scaled)
            state.extend(current_scaled_features)

        # 3. Meta features (4 features) - not scaled
        state.extend([
            current_replicas,
            current_replicas / self.config.max_replicas,  # Normalized replica ratio
            self.epsilon,  # Current exploration rate
            len(self.replay_buffer) / self.config.dqn_memory_capacity  # Buffer fullness
        ])

        return np.array(state, dtype=np.float32)

    def _update_epsilon_by_experience(self):
        """Update epsilon based on experience count for more consistent exploration decay."""
        experience_count = len(self.replay_buffer)

        # Decay epsilon based on experiences collected (0 to 1000 experiences)
        decay_experiences = 1000
        if experience_count < decay_experiences:
            # Linear decay from start to end over 1000 experiences
            progress = experience_count / decay_experiences
            self.epsilon = self.config.dqn_epsilon_start * (1 - progress) + self.config.dqn_epsilon_end * progress
        else:
            # Keep at minimum after decay period
            self.epsilon = self.config.dqn_epsilon_end

        # Log epsilon changes occasionally
        if experience_count % 50 == 0 or experience_count < 10:
            logger.info(f"🎯 Epsilon updated by experience: {self.epsilon:.4f} (experiences: {experience_count}/1000)")

    def select_action(self, state: np.ndarray, use_forecast_guidance: bool = True,
                      forecast_confidence: float = 0.5) -> Tuple[int, float, bool]:
        """
        Select action using epsilon-greedy with forecast-guided exploration.
        Returns: (action_index, decision_confidence, is_exploration)
        """
        # Update epsilon based on experience count (not just training steps)
        self._update_epsilon_by_experience()

        # Adjust epsilon based on forecast confidence if enabled
        if use_forecast_guidance:
            adjusted_epsilon = self.epsilon * (1.0 - forecast_confidence * 0.3)
        else:
            adjusted_epsilon = self.epsilon

        # Get Q-values for confidence calculation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            q_values_np = q_values.cpu().numpy().flatten()

        # Calculate decision confidence based on Q-value spread
        q_max = float(np.max(q_values_np))  # Ensure Python float
        q_min = float(np.min(q_values_np))  # Ensure Python float
        q_range = q_max - q_min
        confidence = min(1.0, max(0.0, q_range / 5.0))  # Normalize to 0-1

        # Exploration vs exploitation decision
        if random.random() < adjusted_epsilon:
            # Exploration
            action = random.randint(0, 2)
            is_exploration = True
            # Lower confidence for exploration
            confidence *= 0.5
        else:
            # Exploitation - use Q-network
            action = int(np.argmax(q_values_np))  # Use numpy array instead of tensor
            is_exploration = False

        return action, confidence, is_exploration

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions given a state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy().flatten()

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool = False):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    async def train_async(self) -> Optional[float]:
        """
        Perform asynchronous training step using experience replay.
        Returns the training loss if training occurred, None otherwise.
        """
        if len(self.replay_buffer) < self.config.dqn_batch_size:
            return None

        # Check if already training
        if self.is_training:
            logger.debug("Training already in progress, skipping")
            return None

        # Check training frequency limit
        current_time = time.time()
        if current_time - self.last_training_time < self.min_training_interval:
            logger.debug(
                f"Training skipped: {current_time - self.last_training_time:.1f}s < {self.min_training_interval}s interval")
            return None

        async with self.training_lock:
            if self.is_training:  # Double-check after acquiring lock
                return None

            self.is_training = True
            self.last_training_time = current_time

        try:
            # Run training in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            loss = await loop.run_in_executor(self.training_executor, self._train_step_sync)
            return loss
        except Exception as e:
            logger.error(f"Async training failed: {e}")
            return None
        finally:
            self.is_training = False

    def _train_step_sync(self) -> Optional[float]:
        """
        Synchronous training step (runs in thread pool).
        """
        if len(self.replay_buffer) < self.config.dqn_batch_size:
            return None

        try:
            # Sample batch from replay buffer
            batch = self.replay_buffer.sample(self.config.dqn_batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert to tensors
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones = torch.BoolTensor(dones).to(self.device)

            # Current Q values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

            # Target Q values using target network
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + (self.config.dqn_gamma * next_q_values * ~dones)

            # Compute loss
            loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update training state
            self.training_steps += 1

            # Update target network periodically
            if self.training_steps % self.target_update_frequency == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                logger.info(f"Target network updated at step {self.training_steps}")

            # Note: Epsilon is now primarily updated by experience count in select_action()
            # This ensures consistent decay regardless of training frequency
            logger.info(
                f"🔍 Training completed: loss={loss.item():.6f}, step={self.training_steps}, epsilon={self.epsilon:.4f}")

            # Store last loss for metrics
            self.last_loss = loss.item()

            return loss.item()

        except Exception as e:
            logger.error(f"Training step failed: {e}")
            return None

    def train(self) -> Optional[float]:
        """
        Legacy synchronous training method for compatibility.
        """
        return self._train_step_sync()

    def train_step(self) -> Optional[float]:
        """
        Legacy synchronous training step for compatibility.
        """
        return self._train_step_sync()

    async def save_model(self, model_name: str = "dqn_model.pt") -> bool:
        """
        Save model checkpoint to MinIO storage.
        """
        if not self.minio_client:
            logger.warning("No MinIO client available for model saving")
            return False

        try:
            # Create comprehensive checkpoint
            checkpoint = {
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'training_steps': self.training_steps,
                'config': {
                    'state_dim': self.state_dim,
                    'action_dim': self.action_dim,
                    'hidden_dims': self.config.dqn_hidden_dims,
                    'selected_features': self.config.selected_features
                },
                'saved_at': time.time()
            }

            # Serialize to buffer
            buffer = BytesIO()
            torch.save(checkpoint, buffer)
            buffer.seek(0)

            # Upload to MinIO
            self.minio_client.put_object(
                bucket_name="models",
                object_name=model_name,
                data=buffer,
                length=buffer.getbuffer().nbytes,
                content_type='application/octet-stream'
            )

            self.last_save_time = time.time()
            logger.info(f"Model saved to MinIO: {model_name} ({buffer.getbuffer().nbytes} bytes)")
            return True

        except Exception as e:
            logger.error(f"Failed to save model to MinIO: {e}")
            return False

    async def load_model(self, model_name: str = "dqn_model.pt") -> bool:
        """
        Load model checkpoint from MinIO storage.
        """
        if not self.minio_client:
            logger.warning("No MinIO client available for model loading")
            return False

        try:
            # Download from MinIO
            response = self.minio_client.get_object("models", model_name)
            buffer = BytesIO(response.read())

            # Load checkpoint
            checkpoint = torch.load(buffer, map_location=self.device)

            # Verify compatibility
            if checkpoint['config']['state_dim'] != self.state_dim:
                logger.error(
                    f"State dimension mismatch: expected {self.state_dim}, got {checkpoint['config']['state_dim']}")
                return False

            # Load state dicts
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load training state
            self.epsilon = checkpoint.get('epsilon', self.config.dqn_epsilon_start)
            self.training_steps = checkpoint.get('training_steps', 0)

            logger.info(
                f"Model loaded from MinIO: {model_name} (steps: {self.training_steps}, epsilon: {self.epsilon:.3f})")
            return True

        except Exception as e:
            logger.error(f"Failed to load model from MinIO: {e}")
            return False

    def should_save_model(self, save_interval: int = 300) -> bool:
        """Check if model should be saved based on time interval."""
        return time.time() - self.last_save_time > save_interval

    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            'training_steps': self.training_steps,
            'epsilon': self.epsilon,
            'replay_buffer_size': len(self.replay_buffer),
            'replay_buffer_capacity': self.replay_buffer.capacity,
            'last_loss': self.last_loss,
            'is_training': self.is_training,
            'last_training_time': self.last_training_time,
            'min_training_interval': self.min_training_interval
        }

    async def cleanup(self):
        """Cleanup resources including thread pool executor."""
        try:
            logger.info("🧹 Cleaning up DQN agent resources...")

            # Wait for any ongoing training to complete
            if self.is_training:
                logger.info("⏳ Waiting for ongoing training to complete...")
                async with self.training_lock:
                    pass  # Just acquire and release the lock

            # Shutdown thread pool executor
            self.training_executor.shutdown(wait=True)
            logger.info("✅ DQN agent cleanup completed")

        except Exception as e:
            logger.error(f"❌ Error during DQN agent cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, 'training_executor'):
                self.training_executor.shutdown(wait=False)
        except:
            pass  # Ignore errors during destruction
