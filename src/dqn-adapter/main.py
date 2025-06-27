import os
import logging
import asyncio
import time
import random
from typing import TypedDict, List, Dict, Any
import json
from datetime import datetime

import requests
import numpy as np
import joblib
import redis
import kopf
import pickle
from io import BytesIO
from urllib.parse import urlparse
from minio import Minio
import torch
import torch.nn as nn
from dotenv import load_dotenv
from prometheus_client import Gauge, generate_latest
from aiohttp import web

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langchain_mcp_adapters.client import MultiServerMCPClient

# Import evaluator
from evaluator import DQNEvaluator

# --- Basic Setup ---
load_dotenv()
# Note: kopf handles its own logging setup, we'll configure it in the startup handler
logger = logging.getLogger("DQN_Adapter")

# --- Kopf Health Check Probes ---
@kopf.on.probe(id='status')
def health_status(**kwargs):
    """Health status probe for Kopf liveness check"""
    return {"status": "healthy", "service": "dqn-adapter"}

@kopf.on.probe(id='redis_connection')
def redis_health(**kwargs):
    """Check Redis connection health"""
    try:
        if redis_client and redis_client.ping():
            return {"redis": "connected"}
        else:
            return {"redis": "disconnected"}
    except Exception as e:
        return {"redis": f"error: {str(e)}"}

@kopf.on.probe(id='scaler_loaded')
def scaler_health(**kwargs):
    """Check if the feature scaler is loaded"""
    return {"scaler": "loaded" if scaler else "not_loaded"}

# --- KEDA Metrics Server with HTTP/2 Error Handling ---
async def metrics_handler(request):
    """Provide Prometheus metrics for KEDA"""
    metrics_data = generate_latest()
    return web.Response(text=metrics_data.decode('utf-8'), content_type='text/plain; version=0.0.4; charset=utf-8')

async def metrics_api_handler(request):
    """KEDA Metrics API endpoint - returns current DQN desired replicas (updated by timer)"""
    try:
        # Simply return the current gauge value - no decision making here
        current_value = float(DESIRED_REPLICAS_GAUGE._value._value)
        
        response = {
            "dqn": {
                "desired_replicas": current_value,
                "status": "active",
                "timestamp": int(time.time())
            }
        }
        return web.json_response(response)
    except Exception as e:
        logger.error(f"Error in metrics API handler: {e}")
        # Return default value on error
        response = {
            "dqn": {
                "desired_replicas": 1.0,
                "status": "error",
                "error": str(e),
                "timestamp": int(time.time())
            }
        }
        return web.json_response(response)

class HTTP2ErrorFilter(logging.Filter):
    """Filter to suppress HTTP/2 protocol error logs"""
    def filter(self, record):
        if record.name == "aiohttp.server":
            message = record.getMessage()
            if "PRI/Upgrade" in message or "BadHttpMessage" in message:
                return False  # Suppress this log
        return True  # Allow other logs

# --- Environment & Configuration ---
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus.nimbusguard.svc:9090")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")
TARGET_DEPLOYMENT = os.getenv("TARGET_DEPLOYMENT", "consumer")
TARGET_NAMESPACE = os.getenv("TARGET_NAMESPACE", "nimbusguard")
SCALER_PATH = os.getenv("SCALER_PATH", "/app/feature_scaler.gz")
STABILIZATION_PERIOD_SECONDS = int(os.getenv("STABILIZATION_PERIOD_SECONDS", 20)) # Time to wait after action for system stabilization
REWARD_LATENCY_WEIGHT = float(os.getenv("REWARD_LATENCY_WEIGHT", 10.0)) # Higher = more penalty for latency
REWARD_REPLICA_COST = float(os.getenv("REWARD_REPLICA_COST", 0.1)) # Cost per replica
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
REPLAY_BUFFER_KEY = "replay_buffer"

# MinIO Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio.nimbusguard.svc:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET_NAME = os.getenv("BUCKET_NAME", "models")
SCALER_NAME = os.getenv("SCALER_NAME", "feature_scaler.gz")

# Scientifically selected features from advanced feature selection pipeline
FEATURE_ORDER = [
    'avg_response_time',
    'http_request_duration_highr_seconds_sum',
    'http_request_duration_seconds_sum',
    'process_resident_memory_bytes',
    'node_network_iface_link',
    'node_network_transmit_queue_length',
    'http_request_duration_seconds_sum_dev_10',
    'http_request_duration_seconds_sum_ma_5',
    'http_request_duration_highr_seconds_sum_ma_10',
    'node_network_flags',
    'http_request_duration_highr_seconds_sum_ma_5'
]

# Exploration parameters for epsilon-greedy strategy
EPSILON_START = float(os.getenv("EPSILON_START", 0.3))
EPSILON_END = float(os.getenv("EPSILON_END", 0.05))
EPSILON_DECAY = float(os.getenv("EPSILON_DECAY", 0.995))

# Global exploration state
current_epsilon = EPSILON_START
decision_count = 0

# Evaluation settings
EVALUATION_INTERVAL = int(os.getenv("EVALUATION_INTERVAL", 300))  # 5 minutes
ENABLE_EVALUATION_OUTPUTS = os.getenv("ENABLE_EVALUATION_OUTPUTS", "true").lower() == "true"

# Training parameters for combined approach
MEMORY_CAPACITY = int(os.getenv("MEMORY_CAPACITY", 50000))
MIN_BATCH_SIZE = int(os.getenv("MIN_BATCH_SIZE", 8))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
TARGET_BATCH_SIZE = int(os.getenv("TARGET_BATCH_SIZE", 64))
GAMMA = float(os.getenv("GAMMA", 0.99))
LR = float(os.getenv("LR", 1e-4))
TARGET_UPDATE_INTERVAL = int(os.getenv("TARGET_UPDATE_INTERVAL", 1000))
SAVE_INTERVAL_SECONDS = int(os.getenv("SAVE_INTERVAL_SECONDS", 300))

# --- Prometheus Metrics ---
# These are defined globally and will be exposed by kopf's built-in metrics server.
DESIRED_REPLICAS_GAUGE = Gauge('nimbusguard_dqn_desired_replicas', 'Desired replicas calculated by the DQN adapter')
CURRENT_REPLICAS_GAUGE = Gauge('nimbusguard_current_replicas', 'Current replicas of the target deployment as seen by the adapter')

# --- DQN Model Definition ---
class EnhancedQNetwork(nn.Module):
    """Enhanced Q-Network matching the learner's architecture."""
    
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

# --- Training Components for Combined Approach ---
from collections import deque, namedtuple
import torch.optim as optim
import torch.nn.functional as F

# Experience tuple for replay buffer
TrainingExperience = namedtuple('TrainingExperience', ['state', 'action', 'reward', 'next_state', 'done'])

class SimpleReplayBuffer:
    """Simple replay buffer for combined DQN training."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, experience: TrainingExperience):
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int):
        """Sample random batch of experiences."""
        import random
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return batch
    
    def __len__(self):
        return len(self.buffer)

class CombinedDQNTrainer:
    """Combined DQN trainer that handles both inference and learning."""
    
    def __init__(self, policy_net, device, feature_order):
        self.device = device
        self.feature_order = feature_order
        
        # Networks
        self.policy_net = policy_net
        self.target_net = EnhancedQNetwork(len(feature_order), 3).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Training components
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = SimpleReplayBuffer(MEMORY_CAPACITY)
        self.training_queue = asyncio.Queue()
        self.model_lock = asyncio.Lock()
        
        # Training state
        self.batches_trained = 0
        self.last_save_time = time.time()
        self.last_research_time = time.time()
        self.training_losses = []
        
        logger.info(f"🎓 Combined DQN Trainer initialized (device: {device})")
        
        # Try to initialize replay buffer with historical data
        asyncio.create_task(self._load_historical_data())
    
    def _dict_to_feature_vector(self, state_dict):
        """Convert state dictionary to feature vector."""
        if not scaler:
            return np.zeros(len(self.feature_order))
        
        feature_vector = [state_dict.get(feat, 0.0) for feat in self.feature_order]
        try:
            scaled_features = scaler.transform([feature_vector])
            return scaled_features[0]
        except Exception as e:
            logger.warning(f"Feature scaling failed: {e}")
            return np.array(feature_vector)
    
    async def add_experience_for_training(self, experience_dict):
        """Add experience to training queue (non-blocking)."""
        try:
            await self.training_queue.put(experience_dict)
        except Exception as e:
            logger.error(f"Failed to queue experience: {e}")
    
    async def continuous_training_loop(self):
        """Background training loop that doesn't block decision making."""
        logger.info("🔄 Starting continuous training loop...")
        
        while True:
            try:
                # Wait for new experience (with timeout to prevent hanging)
                try:
                    experience_dict = await asyncio.wait_for(
                        self.training_queue.get(), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Validate and convert experience
                if not self._validate_experience(experience_dict):
                    continue
                
                # Convert to training format
                state_vec = self._dict_to_feature_vector(experience_dict['state'])
                next_state_vec = self._dict_to_feature_vector(experience_dict['next_state'])
                
                action_map = {"Scale Down": 0, "Keep Same": 1, "Scale Up": 2}
                action_idx = action_map.get(experience_dict['action'], 1)
                
                training_exp = TrainingExperience(
                    state=state_vec,
                    action=action_idx,
                    reward=experience_dict['reward'],
                    next_state=next_state_vec,
                    done=False
                )
                
                # Add to replay buffer
                self.memory.push(training_exp)
                
                # Train if we have enough experiences
                if len(self.memory) >= MIN_BATCH_SIZE:
                    await self._async_train_step()
                
                # Periodic saves
                if time.time() - self.last_save_time > SAVE_INTERVAL_SECONDS:
                    await self._save_model()
                    self.last_save_time = time.time()
                
                # Periodic evaluation
                if (ENABLE_EVALUATION_OUTPUTS and evaluator and 
                    time.time() - self.last_research_time > EVALUATION_INTERVAL):
                    await self._generate_evaluation_outputs()
                    self.last_research_time = time.time()
                
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                await asyncio.sleep(5)
    
    def _validate_experience(self, exp_dict):
        """Validate experience structure."""
        required_keys = ['state', 'action', 'reward', 'next_state']
        return all(key in exp_dict for key in required_keys)
    
    async def _async_train_step(self):
        """Async training step that doesn't block inference."""
        try:
            # Run training in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            loss = await loop.run_in_executor(None, self._sync_train_step)
            
            if self.batches_trained % 10 == 0:  # Log every 10 steps
                logger.info(f"🎯 Training: Batch {self.batches_trained}, Loss={loss:.4f}, Buffer={len(self.memory)}")
                
        except Exception as e:
            logger.error(f"Training step error: {e}")
    
    def _sync_train_step(self):
        """Synchronous training step (runs in thread pool)."""
        try:
            # Progressive batch sizing
            current_batch_size = min(TARGET_BATCH_SIZE, max(MIN_BATCH_SIZE, len(self.memory) // 4))
            batch = self.memory.sample(current_batch_size)
            
            if not batch:
                return 0.0
            
            # Prepare batch tensors (optimized to avoid slow list-to-tensor conversion)
            states_array = np.array([e.state for e in batch], dtype=np.float32)
            actions_array = np.array([e.action for e in batch], dtype=np.int64)
            rewards_array = np.array([e.reward for e in batch], dtype=np.float32)
            next_states_array = np.array([e.next_state for e in batch], dtype=np.float32)
            dones_array = np.array([e.done for e in batch], dtype=bool)
            
            states = torch.from_numpy(states_array).to(self.device)
            actions = torch.from_numpy(actions_array).to(self.device)
            rewards = torch.from_numpy(rewards_array).to(self.device)
            next_states = torch.from_numpy(next_states_array).to(self.device)
            dones = torch.from_numpy(dones_array).to(self.device)
            
            # Current Q values
            current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Double DQN: use policy net to select actions, target net to evaluate
            with torch.no_grad():
                next_actions = self.policy_net(next_states).max(1)[1]
                next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target_q_values = rewards + (GAMMA * next_q_values * ~dones)
            
            # Compute loss
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
            
            # Update target network periodically
            self.batches_trained += 1
            if self.batches_trained % TARGET_UPDATE_INTERVAL == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                logger.info(f"🎯 Target network updated at batch {self.batches_trained}")
            
            self.training_losses.append(loss.item())
            
            # Add training metrics to evaluator
            if evaluator:
                evaluator.add_training_metrics(
                    loss=loss.item(),
                    epsilon=current_epsilon,
                    batch_size=current_batch_size,
                    buffer_size=len(self.memory)
                )
            
            return loss.item()
            
        except Exception as e:
            logger.error(f"Sync training step error: {e}")
            return 0.0
    
    async def _save_model(self):
        """Save model to MinIO."""
        try:
            if not minio_client:
                return
                
            # Create checkpoint
            checkpoint = {
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'batches_trained': self.batches_trained,
                'epsilon': current_epsilon
            }
            
            # Save to buffer
            buffer = BytesIO()
            torch.save(checkpoint, buffer)
            buffer.seek(0)
            
            # Upload to MinIO
            minio_client.put_object(
                BUCKET_NAME,
                "dqn_model.pt",
                data=buffer,
                length=buffer.getbuffer().nbytes,
                content_type='application/octet-stream'
            )
            
            logger.info(f"📁 Model saved to MinIO (batch {self.batches_trained})")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    async def _load_historical_data(self):
        """Load historical training data to initialize replay buffer."""
        try:
            if not minio_client:
                return
            
            logger.info("📚 Attempting to load historical training data...")
            
            # Try to load the dataset from MinIO
            response = minio_client.get_object(BUCKET_NAME, "dqn_features.parquet")
            dataset_bytes = response.read()
            
            # Load dataset using pandas
            import pandas as pd
            from io import BytesIO
            df = pd.read_parquet(BytesIO(dataset_bytes))
            
            # Convert to training experiences (sample a subset to avoid overwhelming the buffer)
            max_historical = min(1000, len(df))  # Limit to 1000 historical experiences
            sampled_df = df.sample(n=max_historical, random_state=42) if len(df) > max_historical else df
            
            action_map = {0: "Scale Down", 1: "Keep Same", 2: "Scale Up"}
            loaded_count = 0
            
            for _, row in sampled_df.iterrows():
                try:
                    # Create state from features
                    state_dict = {feat: row[feat] for feat in self.feature_order if feat in row}
                    
                    # Create a simple next state (assume minimal change)
                    next_state_dict = state_dict.copy()
                    
                    experience_dict = {
                        'state': state_dict,
                        'action': action_map.get(row['scaling_action'], "Keep Same"),
                        'reward': 0.0,  # Historical reward unknown, use neutral
                        'next_state': next_state_dict
                    }
                    
                    # Convert and add to replay buffer
                    state_vec = self._dict_to_feature_vector(state_dict)
                    next_state_vec = self._dict_to_feature_vector(next_state_dict)
                    
                    training_exp = TrainingExperience(
                        state=state_vec,
                        action=row['scaling_action'],
                        reward=0.0,
                        next_state=next_state_vec,
                        done=False
                    )
                    
                    self.memory.push(training_exp)
                    loaded_count += 1
                    
                except Exception as exp_error:
                    continue  # Skip malformed experiences
            
            logger.info(f"📚 Loaded {loaded_count} historical experiences into replay buffer")
            
        except Exception as e:
            logger.info(f"📚 Could not load historical data (continuing without it): {e}")

    async def _generate_evaluation_outputs(self):
        """Generate evaluation outputs."""
        try:
            if not evaluator:
                return
            
            logger.info("🔬 Generating evaluation outputs...")
            
            # Get model state information
            total_params = sum(p.numel() for p in self.policy_net.parameters())
            model_state = {
                'total_params': total_params,
                'batches_trained': self.batches_trained,
                'device': str(self.device),
                'architecture': 'Enhanced DQN (512-256-128)',
                'training_losses': self.training_losses[-100:],  # Last 100 losses
            }
            
            # Generate all evaluation outputs
            loop = asyncio.get_event_loop()
            saved_files = await loop.run_in_executor(
                None, 
                evaluator.generate_all_outputs,
                model_state
            )
            
            logger.info(f"✅ Evaluation complete. Generated {len(saved_files)} files")
            
        except Exception as e:
            logger.error(f"Failed to generate evaluation outputs: {e}")

# --- Global Clients and Models ---
# These will be initialized in the kopf startup handler
prometheus_client = None
llm = None
scaler = None
dqn_model = None
validator_agent = None
redis_client = None
minio_client = None
metrics_server = None
dqn_trainer = None  # Combined trainer for real-time learning
evaluator = None  # Evaluation system

# --- LangGraph State ---
class Experience(TypedDict):
    state: Dict[str, Any]
    action: str
    reward: float
    next_state: Dict[str, Any]
    
class ScalingState(TypedDict):
    current_metrics: Dict[str, float]
    current_replicas: int
    dqn_prediction: Dict[str, Any]
    llm_validation_response: Dict[str, Any]
    final_decision: int
    experience: Experience
    error: str

# --- LangGraph Nodes ---
async def get_live_metrics(state: ScalingState, is_next_state: bool = False) -> Dict[str, Any]:
    node_name = "observe_next_state" if is_next_state else "get_live_metrics"
    logger.info(f"==> Node: {node_name}")
    
    from datetime import datetime
    current_time = datetime.now()
    
    # Scientifically selected feature queries (matching feature selector output)
    queries = {
        # Core response time metric (engineered feature)
        "avg_response_time": 'avg(http_request_duration_seconds_sum{job="prometheus.scrape.annotated_pods"} / http_request_duration_seconds_count{job="prometheus.scrape.annotated_pods"}) * 1000',
        
        # HTTP request duration metrics (raw Prometheus metrics)
        "http_request_duration_highr_seconds_sum": 'sum(http_request_duration_highr_seconds_sum{job="prometheus.scrape.annotated_pods"})',
        "http_request_duration_seconds_sum": 'sum(http_request_duration_seconds_sum{job="prometheus.scrape.annotated_pods"})',
        
        # Memory metric
        "process_resident_memory_bytes": 'sum(alloy_resources_process_resident_memory_bytes{job="prometheus.scrape.annotated_pods"})',
        
        # Network metrics
        "node_network_iface_link": 'sum(node_network_iface_link)',
        "node_network_transmit_queue_length": 'sum(node_network_transmit_queue_length)',
        "node_network_flags": 'sum(node_network_flags)',
        
        # Moving averages and deviations (calculated from current values as proxies)
        "http_request_duration_seconds_sum_dev_10": 'stddev_over_time(http_request_duration_seconds_sum{job="prometheus.scrape.annotated_pods"}[10m])',
        "http_request_duration_seconds_sum_ma_5": 'avg_over_time(http_request_duration_seconds_sum{job="prometheus.scrape.annotated_pods"}[5m])',
        "http_request_duration_highr_seconds_sum_ma_10": 'avg_over_time(http_request_duration_highr_seconds_sum{job="prometheus.scrape.annotated_pods"}[10m])',
        "http_request_duration_highr_seconds_sum_ma_5": 'avg_over_time(http_request_duration_highr_seconds_sum{job="prometheus.scrape.annotated_pods"}[5m])'
    }
    
    tasks = {name: prometheus_client.query(query) for name, query in queries.items()}
    
    # Also get current replicas separately
    current_replicas_query = f'kube_deployment_status_replicas{{deployment="{TARGET_DEPLOYMENT}",namespace="{TARGET_NAMESPACE}"}}'
    tasks['current_replicas'] = prometheus_client.query(current_replicas_query)
    
    results = await asyncio.gather(*tasks.values())
    
    metrics = dict(zip(tasks.keys(), results))
    current_replicas = int(metrics.pop('current_replicas', 1))
    
    CURRENT_REPLICAS_GAUGE.set(current_replicas)
    logger.info(f"  - Selected Features: avg_response_time={metrics.get('avg_response_time', 0):.2f}ms, "
                f"http_request_duration_seconds_sum={metrics.get('http_request_duration_seconds_sum', 0):.3f}, "
                f"process_memory={metrics.get('process_resident_memory_bytes', 0)/1024/1024:.1f}MB, "
                f"replicas={current_replicas}")
    return {"current_metrics": metrics, "current_replicas": current_replicas}

async def get_dqn_recommendation(state: ScalingState) -> Dict[str, Any]:
    logger.info("==> Node: get_dqn_recommendation")
    if not scaler:
        logger.error("Scaler not loaded. Skipping DQN prediction.")
        return {"error": "SCALER_NOT_LOADED"}

    metrics = state['current_metrics'].copy()
    
    # Feature vector using scientifically selected features (no current_replicas included)
    feature_vector = [metrics.get(feat, 0.0) for feat in FEATURE_ORDER]
    scaled_features = scaler.transform([feature_vector])
    logger.info(f"  - Using scientifically selected features ({len(feature_vector)} dimensions)")
    
    try:
        if dqn_model is not None:
            # Use local PyTorch model with epsilon-greedy exploration
            global current_epsilon, decision_count
            
            device = next(dqn_model.parameters()).device
            input_tensor = torch.FloatTensor(scaled_features).to(device)
            
            # Set model to evaluation mode to disable BatchNorm training behavior
            dqn_model.eval()
            
            with torch.no_grad():
                # Ensure proper batch dimension for BatchNorm layers
                if input_tensor.dim() == 1:
                    input_tensor = input_tensor.unsqueeze(0)
                elif input_tensor.shape[0] == 1 and len(input_tensor.shape) == 2:
                    # Already has batch dimension, keep as is
                    pass
                
                q_values = dqn_model(input_tensor).cpu().numpy().flatten()
            
            # Epsilon-greedy exploration
            if random.random() < current_epsilon:
                action_index = random.randint(0, 2)  # Random action
                exploration_type = "exploration"
                logger.info(f"  - Exploration: Random action selected (ε={current_epsilon:.3f})")
            else:
                action_index = np.argmax(q_values)  # Greedy action
                exploration_type = "exploitation"  
                logger.info(f"  - Exploitation: Best action selected (ε={current_epsilon:.3f})")
            
            # Decay epsilon
            current_epsilon = max(EPSILON_END, current_epsilon * EPSILON_DECAY)
            decision_count += 1
            
            action_map = {0: "Scale Down", 1: "Keep Same", 2: "Scale Up"}
            action_name = action_map.get(action_index, "Unknown")
            
            logger.info(f"  - DQN recommends: '{action_name}' ({exploration_type}, Q-values: {[f'{q:.3f}' for q in q_values]})")
            experience_update = {"state": metrics, "action": action_name}
            return {"dqn_prediction": {"action_name": action_name, "q_values": q_values.tolist(), "epsilon": current_epsilon, "exploration_type": exploration_type}, "experience": experience_update}
        else:
            # Fallback: Simple rule-based logic using selected features
            logger.info("  - Using fallback rule-based logic (DQN model not available)")
            avg_response_time = metrics.get('avg_response_time', 100)  # ms
            memory_usage = metrics.get('process_resident_memory_bytes', 0) / (1024 * 1024 * 1024)  # GB
            current_replicas = state['current_replicas']
            
            # Simple rules based on the most important features
            if avg_response_time > 500 or memory_usage > 2.0:  # High latency or memory pressure
                action_name = "Scale Up"
            elif avg_response_time < 100 and memory_usage < 0.5 and current_replicas > 1:  # Low load
                action_name = "Scale Down"
            else:
                action_name = "Keep Same"
                
            logger.info(f"  - Fallback recommends: '{action_name}' (latency={avg_response_time:.1f}ms, memory={memory_usage:.2f}GB)")
            experience_update = {"state": metrics, "action": action_name}
            return {"dqn_prediction": {"action_name": action_name, "q_values": [0.0, 1.0, 0.0]}, "experience": experience_update}
            
    except Exception as e:
        logger.error(f"DQN inference failed: {e}", exc_info=True)
        return {"error": f"DQN_INFERENCE_FAILED: {e}"}

async def validate_with_llm(state: ScalingState) -> Dict[str, Any]:
    logger.info("==> Node: validate_with_llm")
    if not validator_agent:
        logger.warning("  - Validator agent not initialized, skipping.")
        return {"llm_validation_response": {"approved": True, "reason": "Agent not available."}}

    # Handle missing dqn_prediction gracefully
    dqn_prediction = state.get('dqn_prediction', {'action_name': 'Keep Same'})
    action_name = dqn_prediction.get('action_name', 'Keep Same')

    prompt = f"""An automated DQN model suggests the scaling action: '{action_name}'.
    The current state of the deployment '{TARGET_DEPLOYMENT}' is:
    - Current Replicas: {state['current_replicas']}
    - Current Metrics: {state['current_metrics']}

    Validate if this is a safe action. Use your tools to check the live cluster state if necessary.
    Respond with a JSON object: {{"approved": boolean, "reason": "your reasoning"}}.
    """
    try:
        response = await validator_agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
        last_message = response['messages'][-1].content
        # TODO: Robust JSON parsing from last_message
        return {"llm_validation_response": {"approved": True, "reason": last_message}}
    except Exception as e:
        logger.error(f"LLM validation failed: {e}")
        return {"llm_validation_response": {"approved": True, "reason": f"Validation failed: {e}"}}

def plan_final_action(state: ScalingState) -> Dict[str, Any]:
    logger.info("==> Node: plan_final_action")
    current_replicas = state['current_replicas']
    # Handle missing dqn_prediction gracefully
    dqn_prediction = state.get('dqn_prediction', {'action_name': 'Keep Same'})
    action_name = dqn_prediction.get('action_name', 'Keep Same')
    
    new_replicas = current_replicas
    if action_name == 'Scale Up': new_replicas += 1
    elif action_name == 'Scale Down': new_replicas -= 1
    
    final_decision = max(1, new_replicas)
    DESIRED_REPLICAS_GAUGE.set(final_decision)
    logger.info(f"  - Final Decision: Scale from {current_replicas} to {final_decision}. Gauge updated.")
    return {"final_decision": final_decision}

async def wait_for_system_to_stabilize(state: ScalingState) -> None:
    logger.info(f"==> Node: wait_for_system_to_stabilize (waiting {STABILIZATION_PERIOD_SECONDS}s)")
    await asyncio.sleep(STABILIZATION_PERIOD_SECONDS)
    return {}

async def observe_next_state_and_calculate_reward(state: ScalingState) -> Dict[str, Any]:
    logger.info("==> Node: observe_next_state_and_calculate_reward")
    
    # Get next state data
    next_state_data = await get_live_metrics(state, is_next_state=True)
    next_state_metrics = next_state_data.get("current_metrics", {})
    current_replicas = next_state_data.get("current_replicas", 1)
    
    # Get current state for comparison
    current_state_metrics = state.get("current_metrics", {})
    
    # Extract action for context
    experience = state['experience']
    action = experience['action']
    
    # Calculate stable reward using the new reward function
    reward = calculate_stable_reward(current_state_metrics, next_state_metrics, action, current_replicas)
    
    experience['reward'] = reward
    experience['next_state'] = {**next_state_metrics, 'current_replicas': current_replicas}
    return {"experience": experience}

def calculate_stable_reward(current_state, next_state, action, current_replicas):
    """Calculate normalized, stable reward function."""
    
    # Get metrics with safe defaults (using the new feature name)
    current_latency = current_state.get('avg_response_time', 100) / 1000  # Convert ms to seconds
    next_latency = next_state.get('avg_response_time', 100) / 1000       # Convert ms to seconds
    
    # Ensure latency values are reasonable
    current_latency = max(0.01, current_latency)  # Min 10ms
    next_latency = max(0.01, next_latency)        # Min 10ms
    
    # 1. Latency improvement (normalized)
    latency_improvement = max(0, (current_latency - next_latency) / current_latency)
    latency_reward = latency_improvement * 10.0  # Scale to reasonable range
    
    # 2. Cost efficiency (non-linear to discourage over-scaling)
    base_cost = 1.0
    replica_cost = base_cost * (current_replicas ** 1.2)  # Slight exponential cost
    cost_penalty = -replica_cost * 0.1
    
    # 3. Stability bonus (discourage frequent changes)
    action_stability = 1.0 if action == "Keep Same" else 0.0
    stability_bonus = action_stability * 2.0
    
    # 4. Performance penalty (if latency is too high)
    performance_penalty = 0.0
    if next_latency > 0.2:  # 200ms threshold
        performance_penalty = -5.0 * (next_latency - 0.2)
    
    # 5. Efficiency bonus (if we reduced replicas but maintained good latency)
    efficiency_bonus = 0.0
    if action == "Scale Down" and next_latency < 0.15:  # Good latency with fewer replicas
        efficiency_bonus = 3.0
    
    # Combine components
    total_reward = latency_reward + cost_penalty + stability_bonus + performance_penalty + efficiency_bonus
    
    # Normalize to [-10, 10] range
    total_reward = np.clip(total_reward, -10.0, 10.0)
    
    logger.info(f"  - Reward components: latency={latency_reward:.2f}, cost={cost_penalty:.2f}, "
               f"stability={stability_bonus:.2f}, performance={performance_penalty:.2f}, "
               f"efficiency={efficiency_bonus:.2f}, total={total_reward:.2f}")
    
    return total_reward
    
async def log_experience(state: ScalingState) -> Dict[str, Any]:
    logger.info("==> Node: log_experience")
    exp = state['experience']
    
    # Combined approach: immediate training + Redis backup + research logging
    try:
        # 1. Trigger immediate training (primary)
        if dqn_trainer:
            await dqn_trainer.add_experience_for_training(exp)
            logger.info("🎯 Experience queued for immediate training")
        
        # 2. Also log to Redis as backup (for monitoring/debugging)
        if redis_client:
            experience_json = json.dumps(exp)
            redis_client.lpush(REPLAY_BUFFER_KEY, experience_json)
            redis_client.ltrim(REPLAY_BUFFER_KEY, 0, 99)  # Keep last 100 for monitoring
            logger.info("📝 Experience also logged to Redis for monitoring")
        
        # 3. Add to evaluator for analysis
        if evaluator and ENABLE_EVALUATION_OUTPUTS:
            evaluator.add_experience(exp)
        
    except Exception as e:
        logger.error(f"Error in experience logging/training: {e}")
    
    return {}

def create_graph():
    workflow = StateGraph(ScalingState)
    workflow.add_node("get_live_metrics", get_live_metrics)
    workflow.add_node("get_dqn_recommendation", get_dqn_recommendation)
    workflow.add_node("validate_with_llm", validate_with_llm)
    workflow.add_node("plan_final_action", plan_final_action)
    workflow.add_node("wait_for_system_to_stabilize", wait_for_system_to_stabilize)
    workflow.add_node("observe_next_state_and_calculate_reward", observe_next_state_and_calculate_reward)
    workflow.add_node("log_experience", log_experience)
    
    workflow.set_entry_point("get_live_metrics")
    workflow.add_edge("get_live_metrics", "get_dqn_recommendation")
    workflow.add_edge("get_dqn_recommendation", "validate_with_llm")
    workflow.add_edge("validate_with_llm", "plan_final_action")
    workflow.add_edge("plan_final_action", "wait_for_system_to_stabilize")
    workflow.add_edge("wait_for_system_to_stabilize", "observe_next_state_and_calculate_reward")
    workflow.add_edge("observe_next_state_and_calculate_reward", "log_experience")
    workflow.add_edge("log_experience", END)
    
    return workflow.compile()

# --- kopf Operator Setup ---
@kopf.on.startup()
async def configure(settings: kopf.OperatorSettings, **kwargs):
    # kopf's default logging format is slightly different, so we align it
    logging.getLogger().handlers.clear()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("🚀 NimbusGuard DQN Operator starting up...")
    
    # Initialize all global clients
    global prometheus_client, llm, scaler, dqn_model, validator_agent, redis_client, minio_client, metrics_server, dqn_trainer, evaluator
    
    prometheus_client = PrometheusClient() # Use our custom PrometheusClient
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Initialize MinIO client
    try:
        minio_url = urlparse(MINIO_ENDPOINT).netloc
        minio_client = Minio(
            minio_url,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        if not minio_client.bucket_exists(BUCKET_NAME):
            minio_client.make_bucket(BUCKET_NAME)
            logger.info(f"Created MinIO bucket: {BUCKET_NAME}")
        logger.info("Successfully connected to MinIO.")
    except Exception as e:
        logger.error(f"Failed to connect to MinIO: {e}")
        raise kopf.PermanentError(f"MinIO connection failed: {e}")

    # Load scaler from MinIO (upload from local file if not exists)
    try:
        logger.info(f"Loading feature scaler from MinIO: {SCALER_NAME}")
        response = minio_client.get_object(BUCKET_NAME, SCALER_NAME)
        buffer = BytesIO(response.read())
        scaler = joblib.load(buffer)
        logger.info(f"✅ Successfully loaded feature scaler from MinIO (RobustScaler with {len(FEATURE_ORDER)} features)")
    except Exception as e:
        logger.warning(f"Feature scaler not found in MinIO: {e}")
        logger.info("🔄 Attempting to upload local feature scaler to MinIO...")
        
        try:
            # Upload local feature scaler to MinIO
            local_scaler_path = "/app/feature_scaler.gz"
            if os.path.exists(local_scaler_path):
                with open(local_scaler_path, 'rb') as f:
                    scaler_data = f.read()
                
                # Upload to MinIO
                minio_client.put_object(
                    BUCKET_NAME, 
                    SCALER_NAME, 
                    BytesIO(scaler_data), 
                    len(scaler_data),
                    content_type='application/gzip'
                )
                logger.info(f"📤 Successfully uploaded local feature scaler to MinIO: {SCALER_NAME}")
                
                # Now load it
                scaler = joblib.load(BytesIO(scaler_data))
                logger.info(f"✅ Successfully loaded uploaded feature scaler (RobustScaler with {len(FEATURE_ORDER)} features)")
            else:
                logger.error(f"❌ FATAL: Local feature scaler not found at {local_scaler_path}")
                raise kopf.PermanentError(f"Feature scaler not available locally or in MinIO")
                
        except Exception as upload_error:
            logger.error(f"❌ FATAL: Could not upload/load feature scaler: {upload_error}")
            raise kopf.PermanentError(f"Feature scaler loading failed: {upload_error}")

    # Upload training dataset to MinIO if not exists (for research and retraining)
    try:
        # Check if dataset exists in MinIO
        minio_client.stat_object(BUCKET_NAME, "dqn_features.parquet")
        logger.info("✅ Training dataset already exists in MinIO")
    except Exception:
        logger.info("🔄 Uploading training dataset to MinIO...")
        try:
            local_dataset_path = "/app/dqn_features.parquet"
            if os.path.exists(local_dataset_path):
                with open(local_dataset_path, 'rb') as f:
                    dataset_data = f.read()
                
                minio_client.put_object(
                    BUCKET_NAME,
                    "dqn_features.parquet",
                    BytesIO(dataset_data),
                    len(dataset_data),
                    content_type='application/octet-stream'
                )
                logger.info(f"📤 Successfully uploaded training dataset to MinIO: dqn_features.parquet ({len(dataset_data)} bytes)")
            else:
                logger.warning(f"⚠️ Local training dataset not found at {local_dataset_path}")
        except Exception as dataset_upload_error:
            logger.warning(f"⚠️ Could not upload training dataset: {dataset_upload_error}")
            # Not a fatal error - the system can still operate without historical data

    # Load DQN model from MinIO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        logger.info(f"Loading DQN model from MinIO: dqn_model.pt")
        response = minio_client.get_object(BUCKET_NAME, "dqn_model.pt")
        buffer = BytesIO(response.read())
        checkpoint = torch.load(buffer, map_location=device)
        
        # Initialize model with correct dimensions
        state_dim = len(FEATURE_ORDER)
        action_dim = 3
        dqn_model = EnhancedQNetwork(state_dim, action_dim).to(device)
        
        if isinstance(checkpoint, dict) and 'policy_net_state_dict' in checkpoint:
            dqn_model.load_state_dict(checkpoint['policy_net_state_dict'])
        else:
            dqn_model.load_state_dict(checkpoint)
            
        dqn_model.eval()  # Set to evaluation mode
        logger.info(f"Successfully loaded DQN model from MinIO (device: {device})")
        
        # Initialize combined trainer for real-time learning
        dqn_trainer = CombinedDQNTrainer(dqn_model, device, FEATURE_ORDER)
        
        # Start background training loop
        asyncio.create_task(dqn_trainer.continuous_training_loop())
        logger.info("🚀 Combined DQN training loop started")
        
    except Exception as e:
        logger.warning(f"Could not load DQN model from MinIO: {e}")
        logger.info("DQN adapter will use fallback scaling logic")
        
        # Even without a pre-trained model, we can start with a fresh model
        try:
            state_dim = len(FEATURE_ORDER)
            action_dim = 3
            dqn_model = EnhancedQNetwork(state_dim, action_dim).to(device)
            logger.info("🎯 Starting with fresh DQN model")
            
            # Initialize trainer with fresh model
            dqn_trainer = CombinedDQNTrainer(dqn_model, device, FEATURE_ORDER)
            asyncio.create_task(dqn_trainer.continuous_training_loop())
            logger.info("🚀 Combined DQN training loop started with fresh model")
            
        except Exception as fresh_model_error:
            logger.error(f"Failed to create fresh DQN model: {fresh_model_error}")
            dqn_trainer = None

    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        logger.info("Successfully connected to Redis for experience logging.")
    except Exception as e:
        logger.error(f"Failed to connect to Redis during startup: {e}")
        # Not raising a permanent error, as the operator might still function for decision-making
        
    # Initialize MCP validation with LLM supervisor
    if MCP_SERVER_URL:
        try:
            logger.info(f"🔗 Connecting to MCP server at {MCP_SERVER_URL}")
            mcp_client = MultiServerMCPClient({
                "kubernetes": {
                    "url": f"{MCP_SERVER_URL}/sse/",
                    "transport": "sse"  # Use SSE transport instead of streamable_http
                }
            })
            tools = await mcp_client.get_tools()
            validator_agent = create_react_agent(llm, tools=tools)
            logger.info(f"🤖 LLM validator agent initialized with {len(tools)} MCP tools")
            logger.info(f"🛡️ Available tools: {[tool.name for tool in tools]}")
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            logger.info("🤖 Falling back to LLM-only validation")
            validator_agent = create_react_agent(llm, tools=[])
    else:
        logger.info("🤖 MCP_SERVER_URL not set. Using LLM-only validation")
        validator_agent = create_react_agent(llm, tools=[])
    
    # Initialize evaluator
    if ENABLE_EVALUATION_OUTPUTS:
        try:
            evaluator = DQNEvaluator(minio_client, bucket_name="evaluation-outputs")
            logger.info("🔬 Evaluator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize evaluator: {e}")
            evaluator = None
    else:
        logger.info("🔬 Evaluation outputs disabled")
        evaluator = None

    # Initialize gauges to a default value
    CURRENT_REPLICAS_GAUGE.set(1)
    DESIRED_REPLICAS_GAUGE.set(1)
    
    # Run initial DQN decision to get a baseline
    try:
        logger.info("🎯 Running initial DQN decision during startup...")
        await run_intelligent_scaling_decision()
        logger.info("✅ Initial DQN decision completed")
    except Exception as e:
        logger.warning(f"Initial DQN decision failed (will retry with timer): {e}")
    
    # Add HTTP/2 error log suppression
    aiohttp_server_logger = logging.getLogger("aiohttp.server")
    aiohttp_server_logger.addFilter(HTTP2ErrorFilter())
    
    # Start metrics server for KEDA
    app = web.Application()
    app.router.add_get('/metrics', metrics_handler)
    app.router.add_get('/api/v1/dqn-metrics', metrics_api_handler)
    
    # Add a root handler
    async def root_handler(request):
        return web.json_response({"message": "NimbusGuard DQN Adapter Metrics Server", "status": "running"})
    
    async def evaluation_trigger_handler(request):
        """Manual trigger for evaluation."""
        try:
            if not evaluator or not ENABLE_EVALUATION_OUTPUTS:
                return web.json_response({"error": "Evaluation not enabled"}, status=400)
            
            if not dqn_trainer:
                return web.json_response({"error": "DQN trainer not initialized"}, status=400)
            
            # Trigger evaluation
            await dqn_trainer._generate_evaluation_outputs()
            
            return web.json_response({
                "message": "Evaluation triggered successfully",
                "timestamp": datetime.now().isoformat(),
                "experiences": len(evaluator.experiences),
                "training_metrics": len(evaluator.training_metrics)
            })
            
        except Exception as e:
            logger.error(f"Evaluation trigger failed: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    app.router.add_get('/', root_handler)
    app.router.add_post('/evaluate', evaluation_trigger_handler)
    
    metrics_server = web.AppRunner(app)
    await metrics_server.setup()
    site = web.TCPSite(metrics_server, '0.0.0.0', 8080)
    await site.start()
    logger.info("🌐 KEDA Metrics API server with HTTP/2 error suppression started on port 8080")
        
    logger.info(f"✅ Startup complete. Watching for ScaledObject events.")

@kopf.on.cleanup()
async def cleanup(**kwargs):
    logger.info("Operator shutting down.")
    if redis_client:
        try:
            redis_client.close()
        except Exception as e:
            logger.error(f"Error closing Redis client: {e}")
    
    if metrics_server:
        try:
            await metrics_server.cleanup()
            logger.info("Metrics server stopped.")
        except Exception as e:
            logger.error(f"Error stopping metrics server: {e}")

# --- Timer-based DQN Decision Making ---
@kopf.timer('keda.sh', 'v1alpha1', 'scaledobjects', labels={'component': 'keda-dqn'}, interval=30)
async def periodic_dqn_decision(spec, status, meta, **kwargs):
    """Periodically run DQN scaling decisions for DQN-enabled ScaledObjects (every 60 seconds)"""
    logger.info(f"🕐 Timer triggered for ScaledObject '{meta['name']}': Running periodic DQN decision")
    
    # Only process our specific DQN ScaledObject
    if meta['name'] != 'consumer-scaler-dqn':
        return
    
    # Get current scaling status from KEDA
    current_replicas = status.get('originalReplicaCount', 1)
    is_active = any(condition.get('type') == 'Active' and condition.get('status') == 'True' 
                   for condition in status.get('conditions', []))
    
    logger.info(f"📊 ScaledObject state: {current_replicas} replicas, Active: {is_active}")
    
    try:
        # Run DQN decision making process
        await run_intelligent_scaling_decision()
        logger.info("✅ Periodic DQN decision completed successfully")
    except Exception as e:
        logger.error(f"❌ Error in periodic DQN decision: {e}", exc_info=True)

async def run_intelligent_scaling_decision():
    """Execute the intelligent scaling decision using DQN"""
    logger.info("🧠 Starting intelligent scaling decision process...")
    graph = create_graph()
    try:
        final_state = await graph.ainvoke({}, {"recursion_limit": 15})
        logger.info(f"✅ Intelligent scaling decision complete. Final decision: {final_state.get('final_decision', 'N/A')} replicas")
    except Exception as e:
        logger.error(f"❌ Critical error in intelligent scaling process: {e}", exc_info=True)

class PrometheusClient:
    """ A simple async-wrapper for the requests library for Prometheus. """
    async def query(self, promql: str) -> float:
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': promql}))
            response.raise_for_status()
            result = response.json()['data']['result']
            if result: return float(result[0]['value'][1])
            return 0.0
        except Exception as e:
            logger.error(f"Prometheus query failed for '{promql}': {e}")
            return 0.0