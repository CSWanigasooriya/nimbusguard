import os
import logging
import json
import time
from collections import deque
import random
from io import BytesIO
from urllib.parse import urlparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import redis
from minio import Minio

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DQN_Learner")

# --- Environment & Configuration ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
# The Minio client needs just 'host:port', so we parse the full URL.
MINIO_URL = urlparse(MINIO_ENDPOINT).netloc
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MODEL_NAME = os.getenv("MODEL_NAME", "dqn_model.pt")
SCALER_NAME = os.getenv("SCALER_NAME", "feature_scaler.joblib")
BUCKET_NAME = os.getenv("BUCKET_NAME", "models")
REPLAY_BUFFER_KEY = "replay_buffer"
MEMORY_CAPACITY = int(os.getenv("MEMORY_CAPACITY", 10000))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 128))
GAMMA = float(os.getenv("GAMMA", 0.99))
LR = float(os.getenv("LR", 1e-4))
TARGET_UPDATE_INTERVAL = int(os.getenv("TARGET_UPDATE_INTERVAL", 10)) # in number of batches trained
SAVE_INTERVAL_SECONDS = int(os.getenv("SAVE_INTERVAL_SECONDS", 300))
# These must match the dqn-adapter
STATE_DIM = 5 
ACTION_DIM = 3 

ACTION_MAP = {"Scale Down": 0, "Keep Same": 1, "Scale Up": 2}
FEATURE_ORDER = [
    'current_replicas', 'request_rate', 'avg_latency', 
    'cpu_usage', 'memory_usage_mb'
]

# --- Clients ---
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    logger.info("Successfully connected to Redis.")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}", exc_info=True)
    exit(1)

try:
    minio_client = Minio(
        MINIO_URL,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
    # Ensure bucket exists
    if not minio_client.bucket_exists(BUCKET_NAME):
        minio_client.make_bucket(BUCKET_NAME)
        logger.info(f"Created MinIO bucket: {BUCKET_NAME}")
    logger.info("Successfully connected to MinIO.")
except Exception as e:
    logger.error(f"Failed to connect to MinIO: {e}", exc_info=True)
    exit(1)


# --- DQN Components (from train_dqn.py) ---
class ReplayMemory:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)

# --- Core Learner Logic ---
class DQNTrainer:
    def __init__(self, device):
        self.device = device
        self.policy_net = QNetwork(STATE_DIM, ACTION_DIM).to(device)
        self.target_net = QNetwork(STATE_DIM, ACTION_DIM).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MEMORY_CAPACITY)
        self.batches_trained = 0
        self.load_model() # Load existing model from MinIO if available
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
    def _dict_to_feature_vector(self, state_dict):
        """Converts a state dictionary to a numpy array based on FEATURE_ORDER."""
        return np.array([state_dict.get(feat, 0.0) for feat in FEATURE_ORDER], dtype=np.float32)

    def load_model(self):
        try:
            response = minio_client.get_object(BUCKET_NAME, MODEL_NAME)
            buffer = BytesIO(response.read())
            self.policy_net.load_state_dict(torch.load(buffer, map_location=self.device))
            logger.info(f"Successfully loaded existing model '{MODEL_NAME}' from MinIO.")
        except Exception as e:
            logger.warning(f"Could not load model from MinIO (may not exist yet): {e}")

    def save_model(self):
        try:
            buffer = BytesIO()
            torch.save(self.policy_net.state_dict(), buffer)
            buffer.seek(0)
            minio_client.put_object(
                BUCKET_NAME,
                MODEL_NAME,
                data=buffer,
                length=buffer.getbuffer().nbytes,
                content_type='application/octet-stream'
            )
            logger.info(f"Successfully saved model '{MODEL_NAME}' to MinIO bucket '{BUCKET_NAME}'.")
        except Exception as e:
            logger.error(f"Failed to save model to MinIO: {e}", exc_info=True)

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)
        
        # Convert to numpy arrays first for efficiency
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)

        # Convert to tensors
        b_state = torch.tensor(states, dtype=torch.float32, device=self.device)
        b_action = torch.tensor(actions, dtype=torch.long, device=self.device)
        b_reward = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        b_next_state = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        
        # Note: In this online scenario, 'done' is always false as the process is continuous.
        # We don't terminate episodes in the same way as in offline/gym environments.

        # Get current Q values for chosen actions: Q(s_t, a_t)
        current_q_values = self.policy_net(b_state).gather(1, b_action.unsqueeze(1)).squeeze(1)

        # Get next Q values from target network: max_a' Q_target(s_t+1, a')
        with torch.no_grad():
            next_q_values = self.target_net(b_next_state).max(1)[0]
        
        # Compute the expected Q values: r_t + gamma * max_a' Q_target(s_t+1, a')
        expected_q_values = b_reward + (GAMMA * next_q_values)

        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1) # Gradient clipping
        self.optimizer.step()

        self.batches_trained += 1
        if self.batches_trained % TARGET_UPDATE_INTERVAL == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logger.info(f"Updated target network. Batches trained: {self.batches_trained}. Loss: {loss.item():.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    trainer = DQNTrainer(device)
    
    last_save_time = time.time()

    logger.info("Starting learner loop...")
    while True:
        try:
            # Blocking pop from the right of the list (FIFO)
            _, experience_json = redis_client.brpop(REPLAY_BUFFER_KEY)
            exp = json.loads(experience_json)
            
            # Extract and process data
            state_dict = exp.get("state")
            action_str = exp.get("action")
            reward = exp.get("reward")
            next_state_dict = exp.get("next_state")
            
            if not all([state_dict, action_str, isinstance(reward, (int, float)), next_state_dict]):
                logger.warning(f"Skipping malformed experience: {experience_json}")
                continue

            state_vec = trainer._dict_to_feature_vector(state_dict)
            action_idx = ACTION_MAP.get(action_str)
            next_state_vec = trainer._dict_to_feature_vector(next_state_dict)

            if action_idx is None:
                logger.warning(f"Skipping experience with unknown action: {action_str}")
                continue

            # Push to replay memory
            trainer.memory.push(state_vec, action_idx, reward, next_state_vec)
            
            # Perform a training step
            if len(trainer.memory) >= BATCH_SIZE:
                trainer.train_step()
            
            # Periodically save the model
            if time.time() - last_save_time > SAVE_INTERVAL_SECONDS:
                trainer.save_model()
                last_save_time = time.time()

        except redis.exceptions.RedisError as e:
            logger.error(f"Redis error: {e}. Reconnecting...")
            time.sleep(5)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode experience JSON: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred in the learner loop: {e}", exc_info=True)
            time.sleep(5)

if __name__ == "__main__":
    main()
