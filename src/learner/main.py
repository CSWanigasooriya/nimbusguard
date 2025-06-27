import os
import logging
import json
import time
from collections import deque, namedtuple
import random
from io import BytesIO
from urllib.parse import urlparse
from datetime import datetime
import pickle
import asyncio

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import redis
from minio import Minio
import requests
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Kopf for Kubernetes operator functionality
import kopf
import kubernetes.client as k8s_client
from kubernetes.client.rest import ApiException

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DQN_Learner_Operator")

# --- Environment & Configuration ---
REDIS_URL = os.getenv("REDIS_URL", "redis://redis.nimbusguard.svc:6379")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio.nimbusguard.svc:9000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus.nimbusguard.svc:9090")
MINIO_URL = urlparse(MINIO_ENDPOINT).netloc
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MODEL_NAME = os.getenv("MODEL_NAME", "dqn_model.pt")
SCALER_NAME = os.getenv("SCALER_NAME", "feature_scaler.pkl")
BUCKET_NAME = os.getenv("BUCKET_NAME", "models")
REPLAY_BUFFER_KEY = "replay_buffer"
EVALUATION_RESULTS_KEY = "evaluation_results"
TARGET_NAMESPACE = os.getenv("TARGET_NAMESPACE", "nimbusguard")
SCALEDOBJECT_NAME = os.getenv("SCALEDOBJECT_NAME", "consumer-scaler-dqn")

# Enhanced DQN Parameters
MEMORY_CAPACITY = int(os.getenv("MEMORY_CAPACITY", 50000))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
GAMMA = float(os.getenv("GAMMA", 0.99))
LR = float(os.getenv("LR", 1e-4))
TARGET_UPDATE_INTERVAL = int(os.getenv("TARGET_UPDATE_INTERVAL", 1000))
SAVE_INTERVAL_SECONDS = int(os.getenv("SAVE_INTERVAL_SECONDS", 300))
EVALUATION_INTERVAL = int(os.getenv("EVALUATION_INTERVAL", 100))  # Every 100 training steps

# Prioritized Experience Replay Parameters
ALPHA = float(os.getenv("ALPHA", 0.6))  # Prioritization exponent
BETA = float(os.getenv("BETA", 0.4))    # Importance sampling exponent
BETA_INCREMENT = float(os.getenv("BETA_INCREMENT", 0.001))

ACTION_MAP = {"Scale Down": 0, "Keep Same": 1, "Scale Up": 2}
ACTION_NAMES = ["Scale Down", "Keep Same", "Scale Up"]

# Experience tuple for prioritized replay
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'priority'])

# Global trainer instance
trainer = None
last_save_time = time.time()
last_evaluation_step = 0

# --- Enhanced Feature Engineering ---
class PrometheusFeatureExtractor:
    """Advanced feature extraction from Prometheus metrics."""
    
    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url
        self.feature_definitions = self._define_features()
        self.scaler = RobustScaler()
        self.is_fitted = False
        
    def _define_features(self):
        """Define comprehensive feature set for DQN training."""
        return {
            # === WORKLOAD FEATURES ===
            'request_rate': {
                'query': 'sum(rate(http_requests_total{job="prometheus.scrape.annotated_pods"}[1m]))',
                'description': 'HTTP requests per second'
            },
            'request_latency_p95': {
                'query': 'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job="prometheus.scrape.annotated_pods"}[2m])) by (le))',
                'description': '95th percentile request latency'
            },
            'error_rate': {
                'query': 'sum(rate(http_requests_total{job="prometheus.scrape.annotated_pods",code!~"2.."}[1m])) / sum(rate(http_requests_total{job="prometheus.scrape.annotated_pods"}[1m]))',
                'description': 'HTTP error rate'
            },
            
            # === RESOURCE FEATURES ===
            'current_replicas': {
                'query': 'kube_deployment_status_replicas{deployment="consumer",namespace="nimbusguard"}',
                'description': 'Current number of replicas'
            },
            'cpu_usage_avg': {
                'query': 'avg(rate(container_cpu_usage_seconds_total{container="consumer",namespace="nimbusguard"}[2m]))',
                'description': 'Average CPU usage'
            },
            'memory_usage_avg': {
                'query': 'avg(container_memory_usage_bytes{container="consumer",namespace="nimbusguard"}) / 1024 / 1024',
                'description': 'Average memory usage in MB'
            },
            'memory_usage_max': {
                'query': 'max(container_memory_usage_bytes{container="consumer",namespace="nimbusguard"}) / 1024 / 1024',
                'description': 'Maximum memory usage in MB'
            },
            
            # === SYSTEM HEALTH ===
            'gc_pressure': {
                'query': 'sum(rate(go_gc_duration_seconds_sum{job="prometheus.scrape.annotated_pods"}[2m])) / sum(rate(go_gc_duration_seconds_count{job="prometheus.scrape.annotated_pods"}[2m]))',
                'description': 'GC pressure'
            },
            
            # === TREND FEATURES ===
            'request_rate_trend': {
                'query': 'rate(sum(rate(http_requests_total{job="prometheus.scrape.annotated_pods"}[1m]))[5m:])',
                'description': 'Request rate trend'
            },
            
            # === TEMPORAL FEATURES ===
            'hour_of_day': {
                'query': 'hour()',
                'description': 'Hour of day (0-23)'
            },
            'day_of_week': {
                'query': 'day_of_week()',
                'description': 'Day of week (0-6)'
            }
        }
    
    def query_prometheus(self, query: str) -> float:
        """Query Prometheus and return a single value."""
        try:
            url = f"{self.prometheus_url}/api/v1/query"
            response = requests.get(url, params={'query': query}, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data['status'] == 'success' and data['data']['result']:
                return float(data['data']['result'][0]['value'][1])
            return 0.0
        except Exception as e:
            logger.warning(f"Error querying Prometheus: {e}")
            return 0.0
    
    def extract_features(self) -> np.ndarray:
        """Extract all features and return as numpy array."""
        features = {}
        current_time = datetime.now()
        
        for feature_name, feature_config in self.feature_definitions.items():
            try:
                if feature_name == 'hour_of_day':
                    features[feature_name] = current_time.hour
                elif feature_name == 'day_of_week':
                    features[feature_name] = current_time.weekday()
                else:
                    value = self.query_prometheus(feature_config['query'])
                    features[feature_name] = value
            except Exception as e:
                logger.warning(f"Error extracting feature {feature_name}: {e}")
                features[feature_name] = 0.0
        
        # Convert to ordered array
        feature_vector = np.array([features[name] for name in self.feature_definitions.keys()])
        
        # Apply scaling if fitted
        if self.is_fitted:
            feature_vector = self.scaler.transform(feature_vector.reshape(1, -1)).flatten()
        
        return feature_vector
    
    def fit_scaler(self, feature_data: np.ndarray):
        """Fit the scaler on historical data."""
        if feature_data.shape[0] > 0:
            self.scaler.fit(feature_data)
            self.is_fitted = True
            logger.info("Feature scaler fitted successfully")
    
    def get_feature_names(self):
        return list(self.feature_definitions.keys())

# --- Prioritized Experience Replay ---
class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer with Sum Tree."""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def push(self, experience: Experience):
        """Add experience to buffer with maximum priority."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample batch with prioritized sampling."""
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=True)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize
        
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)

# --- Enhanced DQN Network ---
class EnhancedQNetwork(nn.Module):
    """Enhanced Q-Network with improved architecture."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims=[512, 256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Remove last batch norm and dropout
        layers = layers[:-2]
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.network(x)

# --- Enhanced DQN Trainer ---
class AdvancedDQNTrainer:
    def __init__(self, device, feature_extractor):
        self.device = device
        self.feature_extractor = feature_extractor
        
        # Determine state dimension from feature extractor
        sample_features = self.feature_extractor.extract_features()
        self.state_dim = len(sample_features)
        self.action_dim = 3
        
        logger.info(f"State dimension: {self.state_dim}, Action dimension: {self.action_dim}")
        
        # Networks
        self.policy_net = EnhancedQNetwork(self.state_dim, self.action_dim).to(device)
        self.target_net = EnhancedQNetwork(self.state_dim, self.action_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        
        # Replay buffer
        self.memory = PrioritizedReplayBuffer(MEMORY_CAPACITY, ALPHA)
        
        # Training state
        self.batches_trained = 0
        self.training_losses = []
        self.evaluation_results = []
        self.beta = BETA
        
        # Load existing model and scaler
        self.load_model()
        self.load_scaler()
        
        # Initialize target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
    def _dict_to_feature_vector(self, state_dict):
        """Convert state dictionary to feature vector using Prometheus features."""
        # Always use the enhanced feature extraction
        return self.feature_extractor.extract_features()

    def load_model(self):
        """Load model from MinIO."""
        try:
            response = minio_client.get_object(BUCKET_NAME, MODEL_NAME)
            buffer = BytesIO(response.read())
            checkpoint = torch.load(buffer, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'policy_net_state_dict' in checkpoint:
                # New format with full checkpoint
                self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.batches_trained = checkpoint.get('batches_trained', 0)
                self.training_losses = checkpoint.get('training_losses', [])
                self.beta = checkpoint.get('beta', BETA)
            else:
                # Legacy format - just state dict
                self.policy_net.load_state_dict(checkpoint)
                
            logger.info(f"Successfully loaded model from MinIO. Batches trained: {self.batches_trained}")
        except Exception as e:
            logger.warning(f"Could not load model from MinIO: {e}")

    def save_model(self):
        """Save complete model checkpoint to MinIO."""
        try:
            checkpoint = {
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'batches_trained': self.batches_trained,
                'training_losses': self.training_losses,
                'beta': self.beta,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'timestamp': datetime.now().isoformat()
            }
            
            buffer = BytesIO()
            torch.save(checkpoint, buffer)
            buffer.seek(0)
            
            minio_client.put_object(
                BUCKET_NAME,
                MODEL_NAME,
                data=buffer,
                length=buffer.getbuffer().nbytes,
                content_type='application/octet-stream'
            )
            logger.info(f"Model checkpoint saved to MinIO. Batches: {self.batches_trained}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_scaler(self):
        """Load feature scaler from MinIO."""
        try:
            response = minio_client.get_object(BUCKET_NAME, SCALER_NAME)
            buffer = BytesIO(response.read())
            self.feature_extractor.scaler = pickle.load(buffer)
            self.feature_extractor.is_fitted = True
            logger.info("Feature scaler loaded from MinIO")
        except Exception as e:
            logger.warning(f"Could not load scaler from MinIO: {e}")

    def save_scaler(self):
        """Save feature scaler to MinIO."""
        try:
            buffer = BytesIO()
            pickle.dump(self.feature_extractor.scaler, buffer)
            buffer.seek(0)
            
            minio_client.put_object(
                BUCKET_NAME,
                SCALER_NAME,
                data=buffer,
                length=buffer.getbuffer().nbytes,
                content_type='application/octet-stream'
            )
            logger.info("Feature scaler saved to MinIO")
        except Exception as e:
            logger.error(f"Failed to save scaler: {e}")

    def train_step(self):
        """Enhanced training step with Double DQN and prioritized replay."""
        if len(self.memory) < BATCH_SIZE:
            return 0.0

        # Sample from prioritized replay buffer
        experiences, indices, weights = self.memory.sample(BATCH_SIZE, self.beta)
        
        if not experiences:
            return 0.0

        # Prepare batch
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: Use policy network to select actions, target network to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1]
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (GAMMA * next_q_values * ~dones)
        
        # Compute TD errors for priority updates
        td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()

        # Compute weighted loss
        loss = (weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities
        self.memory.update_priorities(indices, td_errors + 1e-6)

        # Update target network
        self.batches_trained += 1
        if self.batches_trained % TARGET_UPDATE_INTERVAL == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logger.info(f"Target network updated. Batches: {self.batches_trained}, Loss: {loss.item():.4f}")

        # Update beta (importance sampling)
        self.beta = min(1.0, self.beta + BETA_INCREMENT)
        
        self.training_losses.append(loss.item())
        return loss.item()

    def evaluate_model(self):
        """Evaluate model performance and save results."""
        logger.info("Evaluating model performance...")
        
        try:
            # Collect current metrics
            current_state = self.feature_extractor.extract_features()
            
            # Get Q-values for current state
            with torch.no_grad():
                state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor).squeeze().cpu().numpy()
            
            # Calculate action probabilities (softmax)
            action_probs = torch.softmax(torch.FloatTensor(q_values), dim=0).numpy()
            
            evaluation_result = {
                'timestamp': datetime.now().isoformat(),
                'batches_trained': self.batches_trained,
                'current_state': current_state.tolist(),
                'q_values': q_values.tolist(),
                'action_probabilities': action_probs.tolist(),
                'predicted_action': int(np.argmax(q_values)),
                'predicted_action_name': ACTION_NAMES[np.argmax(q_values)],
                'replay_buffer_size': len(self.memory),
                'recent_loss': np.mean(self.training_losses[-10:]) if self.training_losses else 0.0,
                'feature_names': self.feature_extractor.get_feature_names()
            }
            
            self.evaluation_results.append(evaluation_result)
            
            # Save to Redis for monitoring
            redis_client.lpush(EVALUATION_RESULTS_KEY, json.dumps(evaluation_result))
            redis_client.ltrim(EVALUATION_RESULTS_KEY, 0, 99)  # Keep last 100 evaluations
            
            logger.info(f"Model evaluation completed. Predicted action: {evaluation_result['predicted_action_name']}")
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")

    def create_visualizations(self):
        """Create and save model performance visualizations."""
        try:
            if len(self.training_losses) < 10:
                return
                
            # Training progress plot
            plt.figure(figsize=(15, 10))
            
            # Loss plot
            plt.subplot(2, 3, 1)
            plt.plot(self.training_losses)
            plt.title('Training Loss')
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.grid(True)
            
            # Recent evaluations
            if len(self.evaluation_results) > 5:
                recent_evals = self.evaluation_results[-20:]
                
                # Q-values over time
                plt.subplot(2, 3, 2)
                q_vals_array = np.array([eval_result['q_values'] for eval_result in recent_evals])
                for i, action_name in enumerate(ACTION_NAMES):
                    plt.plot(q_vals_array[:, i], label=action_name)
                plt.title('Q-Values Over Time')
                plt.xlabel('Evaluation Step')
                plt.ylabel('Q-Value')
                plt.legend()
                plt.grid(True)
                
                # Action distribution
                plt.subplot(2, 3, 3)
                predicted_actions = [eval_result['predicted_action'] for eval_result in recent_evals]
                action_counts = [predicted_actions.count(i) for i in range(3)]
                plt.pie(action_counts, labels=ACTION_NAMES, autopct='%1.1f%%')
                plt.title('Predicted Action Distribution')
                
                # Feature importance (current state)
                plt.subplot(2, 3, 4)
                if recent_evals:
                    current_features = recent_evals[-1]['current_state']
                    feature_names = recent_evals[-1]['feature_names']
                    plt.barh(range(len(current_features)), current_features)
                    plt.yticks(range(len(feature_names)), feature_names)
                    plt.title('Current State Features')
                    plt.xlabel('Feature Value')
                
                # Replay buffer size
                plt.subplot(2, 3, 5)
                buffer_sizes = [eval_result['replay_buffer_size'] for eval_result in recent_evals]
                plt.plot(buffer_sizes)
                plt.title('Replay Buffer Size')
                plt.xlabel('Evaluation Step')
                plt.ylabel('Buffer Size')
                plt.grid(True)
                
                # Recent loss trend
                plt.subplot(2, 3, 6)
                recent_losses = [eval_result['recent_loss'] for eval_result in recent_evals]
                plt.plot(recent_losses)
                plt.title('Recent Loss Trend')
                plt.xlabel('Evaluation Step')
                plt.ylabel('Average Loss (last 10 steps)')
                plt.grid(True)
            
            plt.tight_layout()
            
            # Save plot to buffer and upload to MinIO
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            minio_client.put_object(
                BUCKET_NAME,
                f"training_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                data=buffer,
                length=buffer.getbuffer().nbytes,
                content_type='image/png'
            )
            
            plt.close()
            logger.info("Training visualizations saved to MinIO")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")

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
    if not minio_client.bucket_exists(BUCKET_NAME):
        minio_client.make_bucket(BUCKET_NAME)
        logger.info(f"Created MinIO bucket: {BUCKET_NAME}")
    logger.info("Successfully connected to MinIO.")
except Exception as e:
    logger.error(f"Failed to connect to MinIO: {e}", exc_info=True)
    exit(1)

# --- Kopf Operator Event Handlers ---

@kopf.on.startup()
async def startup_handler(**kwargs):
    """Initialize the DQN trainer when the operator starts."""
    global trainer, last_save_time, last_evaluation_step
    
    logger.info("🚀 DQN Learner Operator starting up...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize feature extractor
    feature_extractor = PrometheusFeatureExtractor(PROMETHEUS_URL)
    
    # Collect initial data to fit scaler if not already fitted
    if not feature_extractor.is_fitted:
        logger.info("Collecting initial data to fit feature scaler...")
        initial_features = []
        for i in range(15):  # Reduced from 50 to 15 samples
            try:
                features = feature_extractor.extract_features()
                initial_features.append(features)
                logger.info(f"Collected sample {i+1}/15")
                await asyncio.sleep(1)  # Reduced from 2 to 1 second
            except Exception as e:
                logger.warning(f"Failed to collect sample {i+1}: {e}")
                await asyncio.sleep(0.5)  # Shorter sleep on error
        
        if initial_features:
            feature_extractor.fit_scaler(np.array(initial_features))
        else:
            logger.warning("No initial features collected, using default scaler")
    
    # Initialize trainer
    trainer = AdvancedDQNTrainer(device, feature_extractor)
    
    # Save scaler if fitted
    if feature_extractor.is_fitted:
        trainer.save_scaler()
    
    last_save_time = time.time()
    last_evaluation_step = 0

    logger.info("✅ DQN Learner Operator ready - listening for ScaledObject events!")

@kopf.on.create('keda.sh', 'v1alpha1', 'scaledobjects', labels={'component': 'keda-dqn'})
@kopf.on.update('keda.sh', 'v1alpha1', 'scaledobjects', labels={'component': 'keda-dqn'})
async def on_scaledobject_event(spec, status, meta, **kwargs):
    """React to ScaledObject events and trigger learning."""
    global trainer, last_save_time, last_evaluation_step
    
    if not trainer:
        logger.warning("Trainer not initialized yet, skipping event")
        return
    
    logger.info(f"🎯 ScaledObject event: {meta['name']} - {kwargs.get('reason', 'unknown')}")
    
    # Check if this is our target ScaledObject
    if meta['name'] != SCALEDOBJECT_NAME:
        return
    
    # Extract scaling information
    min_replicas = spec.get('minReplicaCount', 1)
    max_replicas = spec.get('maxReplicaCount', 50)
    current_replicas = status.get('originalReplicaCount', 1)
    
    # Check if ScaledObject is active
    is_active = False
    conditions = status.get('conditions', [])
    for condition in conditions:
        if condition.get('type') == 'Active' and condition.get('status') == 'True':
            is_active = True
            break
    
    logger.info(f"📊 ScaledObject state: {current_replicas} replicas, Active: {is_active}, Range: {min_replicas}-{max_replicas}")
    
    # Process any pending experiences from Redis
    await process_redis_experiences()
    
    # Trigger model evaluation if enough training has occurred
    if trainer.batches_trained - last_evaluation_step >= EVALUATION_INTERVAL:
        trainer.evaluate_model()
        last_evaluation_step = trainer.batches_trained
    
    # Periodic saves
    current_time = time.time()
    if current_time - last_save_time > SAVE_INTERVAL_SECONDS:
        trainer.save_model()
        trainer.create_visualizations()
        last_save_time = current_time
        
        logger.info(f"🎓 Training progress - Batches: {trainer.batches_trained}, "
                  f"Buffer size: {len(trainer.memory)}, "
                  f"Recent loss: {np.mean(trainer.training_losses[-10:]):.4f}")

async def process_redis_experiences():
    """Process experiences from Redis replay buffer."""
    global trainer
    
    if not trainer:
        return
    
    try:
        # Non-blocking check for experiences
        while True:
            result = redis_client.lpop(REPLAY_BUFFER_KEY)
            if not result:
                break
                
            exp = json.loads(result)
            
            # Extract and validate experience
            state_dict = exp.get("state")
            action_str = exp.get("action")
            reward = exp.get("reward")
            next_state_dict = exp.get("next_state")
            done = exp.get("done", False)
            
            if not all([state_dict is not None, action_str, isinstance(reward, (int, float)), next_state_dict is not None]):
                logger.warning(f"Skipping malformed experience: {result}")
                continue

            # Convert to feature vectors
            state_vec = trainer._dict_to_feature_vector(state_dict)
            next_state_vec = trainer._dict_to_feature_vector(next_state_dict)
            action_idx = ACTION_MAP.get(action_str)

            if action_idx is None:
                logger.warning(f"Unknown action: {action_str}")
                continue

            # Store in prioritized replay buffer
            experience = Experience(state_vec, action_idx, reward, next_state_vec, done, priority=1.0)
            trainer.memory.push(experience)
            
            # Perform training step
            if len(trainer.memory) >= BATCH_SIZE:
                loss = trainer.train_step()
                logger.info(f"🎯 Experience processed: Action={action_str}, Reward={reward:.2f}, Loss={loss:.4f}")
                
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Redis connection error: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode experience JSON: {e}")
    except Exception as e:
        logger.error(f"Error processing experiences: {e}", exc_info=True)

if __name__ == "__main__":
    # The Kopf CLI will handle running the operator
    # No need to call kopf.run() manually
    pass
