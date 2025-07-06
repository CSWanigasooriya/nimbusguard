import os
import logging
import asyncio
import time
import random
from typing import TypedDict, List, Dict, Any
import json
from datetime import datetime
import pandas as pd

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

# Import LSTM forecaster for temporal predictions
from lstm_forecaster import create_lstm_predictor, LSTMWorkloadPredictor

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
    return web.Response(text=metrics_data.decode('utf-8'), content_type='text/plain; version=0.0.4')

# OLD HTTP-based KEDA endpoints removed - now using gRPC External Scaler
# These fake HTTP endpoints never worked with KEDA external scaler anyway!
# gRPC External Scaler provides truly dynamic DQN control without artificial baselines.

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
STABILIZATION_PERIOD_SECONDS = int(os.getenv("STABILIZATION_PERIOD_SECONDS", 30)) # Increased from 5 to 30 seconds for proper Kubernetes stabilization
REWARD_LATENCY_WEIGHT = float(os.getenv("REWARD_LATENCY_WEIGHT", 10.0)) # Higher = more penalty for latency
REWARD_REPLICA_COST = float(os.getenv("REWARD_REPLICA_COST", 0.1)) # Cost per replica
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
REPLAY_BUFFER_KEY = "replay_buffer"

# AI Explainability Configuration
AI_MODEL = os.getenv("AI_MODEL", "gpt-3.5-turbo")  # More cost-effective than gpt-4o-mini
AI_TEMPERATURE = float(os.getenv("AI_TEMPERATURE", 0.1))  # Low temperature for consistent reasoning
ENABLE_DETAILED_REASONING = os.getenv("ENABLE_DETAILED_REASONING", "true").lower() == "true"
REASONING_LOG_LEVEL = os.getenv("REASONING_LOG_LEVEL", "INFO")  # INFO, DEBUG for more detail
ENABLE_LLM_VALIDATION = os.getenv("ENABLE_LLM_VALIDATION", "true").lower() == "true"  # Allow disabling LLM validation
ENABLE_IMPROVED_REWARDS = os.getenv("ENABLE_IMPROVED_REWARDS", "true").lower() == "true"  # Use improved multi-objective reward system

# Neural Network Architecture Configuration
DQN_HIDDEN_DIMS = [int(x) for x in os.getenv("DQN_HIDDEN_DIMS", "512,256,128").split(",")]
LSTM_HIDDEN_DIM = int(os.getenv("LSTM_HIDDEN_DIM", 64))
LSTM_NUM_LAYERS = int(os.getenv("LSTM_NUM_LAYERS", 2))
LSTM_DROPOUT = float(os.getenv("LSTM_DROPOUT", 0.2))
LSTM_SEQUENCE_LENGTH = int(os.getenv("LSTM_SEQUENCE_LENGTH", 24))

# Multi-Objective Reward Component Weights
REWARD_PERFORMANCE_WEIGHT = float(os.getenv("REWARD_PERFORMANCE_WEIGHT", 0.40))
REWARD_RESOURCE_WEIGHT = float(os.getenv("REWARD_RESOURCE_WEIGHT", 0.30))
REWARD_HEALTH_WEIGHT = float(os.getenv("REWARD_HEALTH_WEIGHT", 0.20))
REWARD_COST_WEIGHT = float(os.getenv("REWARD_COST_WEIGHT", 0.10))

# MinIO Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio.nimbusguard.svc:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET_NAME = os.getenv("BUCKET_NAME", "models")
SCALER_NAME = os.getenv("SCALER_NAME", "feature_scaler.gz")

# --- IMPROVED NIMBUSGUARD ARCHITECTURE ---
# This service is the intelligent core of the NimbusGuard autoscaling system. It combines the predictive power of a Deep Q-Network (DQN) model with LSTM temporal intelligence, LLM validation, and an improved multi-objective reward system. It runs a continuous reconciliation loop to determine the optimal number of replicas for a target service.

# IMPROVEMENTS IN THIS VERSION:
# 1. Increased stabilization period from 5s to 30s for proper Kubernetes stabilization
# 2. Multi-objective reward system that avoids circular dependencies  
# 3. Outcome-based metrics separate from DQN input features
# 4. Fixed LSTM integration to use predictions from decision time, not future state
# 5. Weighted reward components: Performance(40%) + Resource(30%) + Health(20%) + Cost(10%)

# The adapter uses **9 scientifically selected raw features** identified through advanced statistical analysis using 6 rigorous methods (Mutual Information, Random Forest, Correlation Analysis, RFECV, Statistical Significance, and VIF Analysis) from 894 real per-minute decision points. Multi-dimensional metrics like resource_limits have been properly separated into CPU and memory components with correct aggregation across multiple pods, containers, and scraping sources. These are combined with **6 LSTM temporal features** for predictive intelligence, creating a total of **15 features** with zero overlap and no redundancy.

# ## 🏗️ Clean Feature Architecture - Multi-Dimensional Fixed ✅

# The system uses a **clean separation** approach with **zero redundancy**:

# ### 📊 RAW BASE FEATURES (9) - Multi-Dimensional Properly Handled
# Current system state observations without any temporal processing or derived computations.

# ### 🧠 LSTM TEMPORAL FEATURES (6)
# Predictive intelligence and pattern-based features:
# 1. **30-sec Load Pressure** - PROACTIVE: Load pressure forecast in 30 seconds
# 2. **60-sec Load Pressure** - PROACTIVE: Load pressure forecast in 60 seconds  
# 3. **Trend Velocity** - How fast the trend is changing (-1: decreasing, 0: stable, 1: increasing)
# 4. **Pattern Detection** - Spike/gradual/cyclical pattern probabilities
# 5. **Temporal Intelligence** - Pattern confidence and anomaly detection
# 6. **Predictive Scaling** - LSTM's optimal replica recommendation

# **Total: 15 features (9 multi-dimensional base + 6 LSTM) with ZERO overlap and proper aggregation!**

# CLEAN FEATURE ARCHITECTURE: Raw base features + LSTM temporal features (no computed temporal features)

# Raw base features (current state observations) - exactly as selected by balanced feature_selector.py
# FIXED: Separate CPU and memory resource limits into distinct features
BASE_FEATURE_ORDER = [
    'kube_deployment_status_replicas_unavailable',       # 1. Current unavailable replicas
    'kube_pod_container_status_ready',                   # 2. Current pod readiness
    'kube_deployment_spec_replicas',                     # 3. Desired replica count
    'kube_pod_container_resource_limits_cpu',            # 4. Current CPU limits in cores (separated)
    'kube_pod_container_resource_limits_memory',         # 5. Current memory limits in bytes (separated)
    'kube_pod_container_status_running',                 # 6. Current running containers
    'kube_deployment_status_observed_generation',        # 7. Current deployment generation
    'node_network_up',                                   # 8. Current network status
    'kube_pod_container_status_last_terminated_exitcode' # 9. Container termination status (statistically selected)
]

# LSTM temporal features (PURE temporal intelligence - no current state redundancy)
LSTM_FEATURE_ORDER = [
    'next_30sec_pressure',                          # LSTM: PROACTIVE - Load pressure in 30 seconds
    'next_60sec_pressure',                          # LSTM: PROACTIVE - Load pressure in 60 seconds
    'trend_velocity',                               # LSTM: How fast the trend is changing (-1 to 1)
    'pattern_type_spike',                           # LSTM: Probability of spike pattern (0-1)
    'pattern_type_gradual',                         # LSTM: Probability of gradual pattern (0-1)
    'pattern_type_cyclical',                        # LSTM: Probability of cyclical pattern (0-1)
]

# Clean feature architecture: 9 raw base + 6 LSTM temporal = 15 total features (no redundancy)
FEATURE_ORDER = BASE_FEATURE_ORDER + LSTM_FEATURE_ORDER

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

# DQN Training Metrics
DQN_TRAINING_LOSS_GAUGE = Gauge('dqn_training_loss', 'Current DQN training loss')
DQN_EPSILON_GAUGE = Gauge('dqn_epsilon_value', 'Current exploration epsilon value')
DQN_BUFFER_SIZE_GAUGE = Gauge('dqn_replay_buffer_size', 'Current replay buffer size')
DQN_TRAINING_STEPS_GAUGE = Gauge('dqn_training_steps_total', 'Total training steps completed')

# DQN Decision Metrics
DQN_DECISION_CONFIDENCE_GAUGE = Gauge('dqn_decision_confidence_avg', 'Average decision confidence')
DQN_Q_VALUE_SCALE_UP_GAUGE = Gauge('dqn_q_value_scale_up', 'Q-value for scale up action')
DQN_Q_VALUE_SCALE_DOWN_GAUGE = Gauge('dqn_q_value_scale_down', 'Q-value for scale down action')
DQN_Q_VALUE_KEEP_SAME_GAUGE = Gauge('dqn_q_value_keep_same', 'Q-value for keep same action')

# DQN Action Counters
from prometheus_client import Counter
DQN_ACTION_SCALE_UP_COUNTER = Counter('dqn_action_scale_up_total', 'Total scale up actions taken')
DQN_ACTION_SCALE_DOWN_COUNTER = Counter('dqn_action_scale_down_total', 'Total scale down actions taken')
DQN_ACTION_KEEP_SAME_COUNTER = Counter('dqn_action_keep_same_total', 'Total keep same actions taken')
DQN_EXPLORATION_COUNTER = Counter('dqn_exploration_actions_total', 'Total exploration actions taken')
DQN_EXPLOITATION_COUNTER = Counter('dqn_exploitation_actions_total', 'Total exploitation actions taken')
DQN_EXPERIENCES_COUNTER = Counter('dqn_experiences_added_total', 'Total experiences added to replay buffer')
DQN_DECISIONS_COUNTER = Counter('dqn_decisions_total', 'Total decisions made by DQN')

# LSTM Feature Metrics
DQN_LSTM_NEXT_30SEC_GAUGE = Gauge('dqn_lstm_next_30sec_pressure', 'LSTM forecast for next 30 seconds')
DQN_LSTM_NEXT_60SEC_GAUGE = Gauge('dqn_lstm_next_60sec_pressure', 'LSTM forecast for next 60 seconds')
DQN_LSTM_TREND_VELOCITY_GAUGE = Gauge('dqn_lstm_trend_velocity', 'LSTM trend velocity')
DQN_LSTM_PATTERN_CONFIDENCE_GAUGE = Gauge('dqn_lstm_pattern_confidence', 'LSTM pattern confidence')

# Reward Component Metrics
DQN_REWARD_TOTAL_GAUGE = Gauge('dqn_reward_total', 'Total reward received')
DQN_REWARD_PERFORMANCE_GAUGE = Gauge('dqn_reward_performance_component', 'Performance component of reward')
DQN_REWARD_RESOURCE_GAUGE = Gauge('dqn_reward_resource_component', 'Resource component of reward')
DQN_REWARD_HEALTH_GAUGE = Gauge('dqn_reward_health_component', 'Health component of reward')
DQN_REWARD_COST_GAUGE = Gauge('dqn_reward_cost_component', 'Cost component of reward')

# --- DQN Model Definition ---
class EnhancedQNetwork(nn.Module):
    """Enhanced Q-Network matching the learner's architecture."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims=None):
        super().__init__()
        
        # Use configurable hidden dimensions
        if hidden_dims is None:
            hidden_dims = DQN_HIDDEN_DIMS
        
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
    """Combines real-time training with historical data loading."""
    
    def __init__(self, policy_net, device, feature_order):
        self.device = device
        self.feature_order = feature_order
        
        # Networks
        self.policy_net = policy_net
        self.target_net = EnhancedQNetwork(len(feature_order), 3, DQN_HIDDEN_DIMS).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Training components
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = SimpleReplayBuffer(MEMORY_CAPACITY)
        self.training_queue = asyncio.Queue(maxsize=1000)
        
        # Training state
        self.batches_trained = 0
        self.last_save_time = time.time()
        self.last_research_time = time.time()
        self.training_losses = []
        
        logger.info(f"TRAINER: initialized device={device}")
        
        # Try to initialize replay buffer with historical data
        asyncio.create_task(self._load_historical_data())
    
    def _dict_to_feature_vector(self, state_dict):
        """Convert state dictionary to feature vector with enhanced LSTM feature importance."""
        if not scaler:
            return np.zeros(len(self.feature_order))
        
        try:
            # Scale the 9 raw base features from Prometheus with proper feature names
            raw_features = [state_dict.get(feat, 0.0) for feat in BASE_FEATURE_ORDER]
            
            # Transform numpy array to maintain consistency with how scaler was fitted (no feature names)
            raw_features_array = np.array([raw_features])
            scaled_raw_features = scaler.transform(raw_features_array)
            
            # Get LSTM features with enhanced importance for temporal predictions
            lstm_features = []
            for feat in LSTM_FEATURE_ORDER:
                value = state_dict.get(feat, 0.0)
                
                # Boost importance of key predictive features
                if feat in ['next_30sec_pressure', 'next_60sec_pressure']:
                    # Amplify pressure predictions (these are most critical for proactive scaling)
                    value = value * 1.5  # 50% boost
                elif feat == 'trend_velocity':
                    # Amplify trend velocity (indicates acceleration/deceleration)
                    value = value * 1.3  # 30% boost
                elif feat == 'optimal_replicas_forecast':
                    # Normalize optimal replicas to 0-1 range for better DQN learning
                    value = min(value / 10.0, 1.0)  # Scale down from 1-10 to 0.1-1.0
                
                lstm_features.append(value)
            
            # Combine scaled raw features + enhanced LSTM features
            final_feature_vector = scaled_raw_features[0].tolist() + lstm_features
            
            return np.array(final_feature_vector)
            
        except Exception as e:
            logger.warning(f"Feature scaling failed: {e}")
            # Fallback: Return unscaled features in correct order
            raw_features = [state_dict.get(feat, 0.0) for feat in BASE_FEATURE_ORDER]
            lstm_features = [state_dict.get(feat, 0.0) for feat in LSTM_FEATURE_ORDER]
            return np.array(raw_features + lstm_features)
    
    async def add_experience_for_training(self, experience_dict):
        """Add experience to training queue (non-blocking)."""
        try:
            await self.training_queue.put(experience_dict)
        except Exception as e:
            logger.error(f"Failed to queue experience: {e}")
    
    async def continuous_training_loop(self):
        """Background training loop that doesn't block decision making."""
        logger.info("TRAINING: continuous_loop_started")
        
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
                logger.info(f"TRAINING: batch={self.batches_trained} loss={loss:.4f} buffer_size={len(self.memory)}")
                
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
            
            # Calculate loss and backpropagate
            loss = F.mse_loss(current_q_values, target_q_values)
            
            # Validate loss before proceeding
            if not torch.isfinite(loss) or loss.item() < 0:
                logger.warning(f"TRAINER: invalid_loss_detected loss={loss.item()} - skipping training step")
                return 0.0
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.batches_trained += 1
            
            # Update target network periodically
            if self.batches_trained % TARGET_UPDATE_INTERVAL == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                logger.info(f"TRAINING: target_network_updated batch={self.batches_trained}")
            
            # Track training metrics
            self.training_losses.append(loss.item())
            
            # Add training metrics to evaluator with validation
            if evaluator:
                try:
                    # Validate all metrics before sending to evaluator
                    loss_value = float(loss.item())
                    epsilon_value = float(current_epsilon) if current_epsilon is not None else 0.0
                    batch_size_value = int(current_batch_size) if current_batch_size is not None else 32
                    buffer_size_value = int(len(self.memory))
                    
                    # Additional validation
                    if np.isfinite(loss_value) and 0 <= loss_value <= 1000:
                        evaluator.add_training_metrics(
                            loss=loss_value,
                            epsilon=epsilon_value,
                            batch_size=batch_size_value,
                            buffer_size=buffer_size_value
                        )
                        logger.debug(f"TRAINER: metrics_sent_to_evaluator loss={loss_value:.4f} buffer_size={buffer_size_value}")
                        
                        # Update Prometheus training metrics
                        DQN_TRAINING_LOSS_GAUGE.set(loss_value)
                        DQN_BUFFER_SIZE_GAUGE.set(buffer_size_value)
                        DQN_TRAINING_STEPS_GAUGE.inc()
                        
                    else:
                        logger.warning(f"TRAINER: invalid_metrics_not_sent loss={loss_value} finite={np.isfinite(loss_value)}")
                        
                except Exception as eval_error:
                    logger.warning(f"TRAINER: evaluator_metrics_failed error={eval_error}")
            
            return loss.item()
            
        except Exception as e:
            logger.error(f"Sync training step error: {e}")
            return 0.0
    
    async def _save_model(self):
        """Save model to MinIO."""
        logger.info("MODEL_SAVE: attempting_minio_upload")
        try:
            global minio_client, BUCKET_NAME
            if not minio_client:
                logger.warning("MinIO client not available, skipping model save")
                return
                
            # Create comprehensive checkpoint
            checkpoint = {
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'batches_trained': self.batches_trained,
                'epsilon': current_epsilon,
                'feature_order': self.feature_order,
                'model_architecture': 'Enhanced DQN (512-256-128)',
                'state_dim': len(self.feature_order),
                'action_dim': 3,
                'saved_at': time.time()
            }
            
            # Save to buffer
            buffer = BytesIO()
            torch.save(checkpoint, buffer)
            buffer.seek(0)
            
            minio_client.put_object(
                BUCKET_NAME, "dqn_model.pt", data=buffer, length=buffer.getbuffer().nbytes,
                content_type='application/octet-stream'
            )
            logger.info(f"MODEL_SAVE: success batch={self.batches_trained} size={buffer.getbuffer().nbytes}")
            
        except Exception as e:
            logger.error(f"MODEL_SAVE: failed error={e}")
    
    async def _load_historical_data(self):
        """Load historical training data to initialize replay buffer with validation."""
        try:
            if not minio_client:
                logger.info("HISTORICAL_DATA: minio_client_not_available")
                return
            
            logger.info("HISTORICAL_DATA: loading_from_minio")
            
            # Try to load the dataset from MinIO
            response = minio_client.get_object(BUCKET_NAME, "dqn_features.parquet")
            dataset_bytes = response.read()
            
            # Load dataset using pandas
            import pandas as pd
            from io import BytesIO
            df = pd.read_parquet(BytesIO(dataset_bytes))
            
            logger.info(f"HISTORICAL_DATA: dataset_loaded rows={len(df)} columns={len(df.columns)}")
            
            # Validate dataset structure
            required_columns = set(BASE_FEATURE_ORDER)
            available_columns = set(df.columns)
            missing_columns = required_columns - available_columns
            
            if missing_columns:
                logger.warning(f"HISTORICAL_DATA: missing_columns {missing_columns}")
            
            # Convert to training experiences (sample a subset to avoid overwhelming the buffer)
            max_historical = min(1000, len(df))  # Limit to 1000 historical experiences
            sampled_df = df.sample(n=max_historical, random_state=42) if len(df) > max_historical else df
            
            action_map = {0: "Scale Down", 1: "Keep Same", 2: "Scale Up"}
            loaded_count = 0
            validation_failures = 0
            
            for _, row in sampled_df.iterrows():
                try:
                    # Validate row data
                    if row.isnull().sum() > len(row) * 0.5:  # More than 50% null values
                        validation_failures += 1
                        continue
                    
                    # Create state from features (adapt to current feature order)
                    state_dict = {}
                    for feat in self.feature_order:
                        if feat in row and pd.notna(row[feat]):
                            value = float(row[feat])
                            # Validate feature values
                            if not np.isfinite(value):
                                validation_failures += 1
                                continue
                            state_dict[feat] = value
                        else:
                            state_dict[feat] = 0.0  # Default value for missing features
                    
                    # Validate scaling action
                    scaling_action = row.get('scaling_action', 1)
                    if not isinstance(scaling_action, (int, float)) or scaling_action not in [0, 1, 2]:
                        scaling_action = 1  # Default to "Keep Same"
                    
                    # Create a simple next state (assume minimal change)
                    next_state_dict = state_dict.copy()
                    
                    experience_dict = {
                        'state': state_dict,
                        'action': action_map.get(int(scaling_action), "Keep Same"),
                        'reward': 0.0,  # Historical reward unknown, use neutral
                        'next_state': next_state_dict
                    }
                    
                    # Convert and add to replay buffer
                    state_vec = self._dict_to_feature_vector(state_dict)
                    next_state_vec = self._dict_to_feature_vector(next_state_dict)
                    
                    # Validate feature vectors
                    if not (np.isfinite(state_vec).all() and np.isfinite(next_state_vec).all()):
                        validation_failures += 1
                        continue
                    
                    training_exp = TrainingExperience(
                        state=state_vec,
                        action=int(scaling_action),
                        reward=0.0,
                        next_state=next_state_vec,
                        done=False
                    )
                    
                    self.memory.push(training_exp)
                    loaded_count += 1
                    
                except Exception as exp_error:
                    validation_failures += 1
                    logger.debug(f"HISTORICAL_DATA: experience_validation_failed error={exp_error}")
                    continue  # Skip malformed experiences
            
            logger.info(f"HISTORICAL_DATA: loading_complete "
                       f"loaded_experiences={loaded_count} "
                       f"validation_failures={validation_failures} "
                       f"success_rate={loaded_count/(loaded_count+validation_failures)*100:.1f}%")
            
        except Exception as e:
            logger.warning(f"HISTORICAL_DATA: load_failed error={e}")
            # Don't raise - historical data is optional

    async def _generate_evaluation_outputs(self):
        """Generate and save all evaluation outputs."""
        try:
            if not evaluator:
                return
            
            logger.info("EVALUATION: generating_outputs")
            
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
            
            logger.info(f"EVALUATION: completed files_generated={len(saved_files)}")
            
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
lstm_predictor = None  # LSTM workload predictor for temporal features

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

# --- Feature Engineering Helper ---
def _ensure_raw_features(metrics: Dict[str, float]) -> Dict[str, float]:
    """Ensure Kubernetes state features have fallback values - no computed features."""
    
    # Kubernetes state feature defaults (current state indicators)
    feature_defaults = {
        'kube_deployment_status_replicas_unavailable': 0.0,                 # Default: no unavailable replicas (healthy)
        'kube_pod_container_status_ready': 1.0,                             # Default: containers ready (healthy)
        'kube_deployment_spec_replicas': 1.0,                               # Default: 1 desired replica
        'kube_pod_container_resource_limits_cpu': 0.5,                      # Default: 0.5 CPU cores
        'kube_pod_container_resource_limits_memory': 536870912,             # Default: 512MB in bytes
        'kube_pod_container_status_running': 1.0,                           # Default: containers running (healthy)
        'kube_deployment_status_observed_generation': 1.0,                  # Default: deployment generation 1
        'node_network_up': 1.0,                                             # Default: network up (healthy)
        'kube_pod_container_status_last_terminated_exitcode': 0.0,          # Default: 0 exit code (successful termination)
    }
    
    # Apply defaults for missing Kubernetes state features only
    for feature, default_value in feature_defaults.items():
        if feature not in metrics or metrics[feature] == 0:
            metrics[feature] = default_value
    
    return metrics

# --- LangGraph Nodes ---
async def get_live_metrics(state: ScalingState, is_next_state: bool = False) -> Dict[str, Any]:
    node_name = "observe_next_state" if is_next_state else "get_live_metrics"
    logger.info(f"NODE_START: {node_name}")
    logger.info("=" * 60)
    
    from datetime import datetime
    current_time = datetime.now()
    
    # FIXED: Multi-dimensional Kubernetes state queries with proper aggregation
    queries = {
        # 1. Deployment unavailable replicas (single value per deployment, deduplicate across scraping sources)
        "kube_deployment_status_replicas_unavailable": f'max(kube_deployment_status_replicas_unavailable{{deployment="{TARGET_DEPLOYMENT}",namespace="{TARGET_NAMESPACE}"}}) or vector(0)',
        
        # 2. Pod container readiness (sum across all consumer containers, handle multiple pods)
        "kube_pod_container_status_ready": f'sum(kube_pod_container_status_ready{{namespace="{TARGET_NAMESPACE}",pod=~"{TARGET_DEPLOYMENT}-.*"}}) or sum(kube_pod_container_status_ready{{namespace="{TARGET_NAMESPACE}"}})',
        
        # 3. Desired replica count (single value per deployment, deduplicate)
        "kube_deployment_spec_replicas": f'max(kube_deployment_spec_replicas{{deployment="{TARGET_DEPLOYMENT}",namespace="{TARGET_NAMESPACE}"}}) or vector(1)',
        
        # 4. Running containers (sum across all consumer containers)
        "kube_pod_container_status_running": f'sum(kube_pod_container_status_running{{namespace="{TARGET_NAMESPACE}",pod=~"{TARGET_DEPLOYMENT}-.*"}}) or sum(kube_pod_container_status_running{{namespace="{TARGET_NAMESPACE}"}})',
        
        # 5. Deployment generation (single value per deployment, deduplicate)
        "kube_deployment_status_observed_generation": f'max(kube_deployment_status_observed_generation{{deployment="{TARGET_DEPLOYMENT}",namespace="{TARGET_NAMESPACE}"}}) or vector(1)',
        
        # 6. CPU resource limits (sum across all consumer containers with CPU resource)
        "kube_pod_container_resource_limits_cpu": f'sum(kube_pod_container_resource_limits{{resource="cpu",namespace="{TARGET_NAMESPACE}",pod=~"{TARGET_DEPLOYMENT}-.*"}}) or sum(kube_pod_container_resource_limits{{resource="cpu",namespace="{TARGET_NAMESPACE}"}}) or vector(0.5)',
        
        # 7. Memory resource limits (sum across all consumer containers with memory resource)
        "kube_pod_container_resource_limits_memory": f'sum(kube_pod_container_resource_limits{{resource="memory",namespace="{TARGET_NAMESPACE}",pod=~"{TARGET_DEPLOYMENT}-.*"}}) or sum(kube_pod_container_resource_limits{{resource="memory",namespace="{TARGET_NAMESPACE}"}}) or vector(536870912)',
        
        # 8. Network status (check if any nodes have network up - cluster-wide indicator)
        "node_network_up": 'sum(up{job=~".*node.*"}) or sum(up{job="node-exporter"}) or sum(node_network_up) or vector(1)',
        
        # 9. Container last terminated exit code (termination health indicator)
        "kube_pod_container_status_last_terminated_exitcode": f'max(kube_pod_container_status_last_terminated_exitcode{{namespace=\"{TARGET_NAMESPACE}\",pod=~\"{TARGET_DEPLOYMENT}-.*\"}}) or max(kube_pod_container_status_last_terminated_exitcode{{namespace=\"{TARGET_NAMESPACE}\"}}) or vector(0)',
    }
    # CLEAN ARCHITECTURE: No computed temporal features (moving averages, deviations, etc.)
    # All temporal intelligence now comes from LSTM predictions
    
    tasks = {name: prometheus_client.query(query) for name, query in queries.items()}
    
    # Also get current replicas separately
    current_replicas_query = f'kube_deployment_status_replicas{{deployment="{TARGET_DEPLOYMENT}",namespace="{TARGET_NAMESPACE}"}}'
    tasks['current_replicas'] = prometheus_client.query(current_replicas_query)
    
    results = await asyncio.gather(*tasks.values())
    
    metrics = dict(zip(tasks.keys(), results))
    current_replicas = int(metrics.pop('current_replicas', 1))
    
    # Ensure raw features have defaults (no computations)
    metrics = _ensure_raw_features(metrics)
    
    # Add observation to LSTM predictor and get temporal features
    lstm_features = {}
    if lstm_predictor:
        try:
            # Add current observation to LSTM buffer
            await lstm_predictor.add_observation_async(metrics)
            
            # Get LSTM-enhanced features
            lstm_features = await lstm_predictor.get_lstm_features_async()
            
            # Add LSTM features to metrics
            metrics.update(lstm_features)
            
            # Log LSTM readiness
            buffer_info = lstm_predictor.get_buffer_info()
            lstm_ready = buffer_info['model_ready']
            buffer_utilization = buffer_info['buffer_stats']['utilization']
            
            logger.info(f"LSTM_STATUS: ready={lstm_ready} buffer_util={buffer_utilization:.1%} "
                       f"trend={lstm_features.get('workload_trend_direction', 0):.2f} "
                       f"confidence={lstm_features.get('pattern_confidence', 0):.2f}")
            
        except Exception as e:
            logger.warning(f"LSTM_FEATURES: extraction_failed error={e}")
            # Add default LSTM features as fallback
            default_lstm_features = {
                'current_load_pressure': 0.5,
                'load_stability': 0.8,
                'next_30sec_pressure': 0.5,
                'next_60sec_pressure': 0.5,
                'workload_trend_direction': 0.0,
                'pattern_confidence': 0.5,
                'anomaly_score': 0.0,
                'optimal_replicas_forecast': float(current_replicas)
            }
            metrics.update(default_lstm_features)
    else:
        logger.debug("LSTM_FEATURES: using_defaults predictor_not_initialized")
        # Add default LSTM features when LSTM is not available
        default_lstm_features = {
            'current_load_pressure': 0.5,
            'load_stability': 0.8,
            'next_30sec_pressure': 0.5,
            'next_60sec_pressure': 0.5,
            'workload_trend_direction': 0.0,
            'pattern_confidence': 0.5,
            'anomaly_score': 0.0,
            'optimal_replicas_forecast': float(current_replicas)
        }
        metrics.update(default_lstm_features)
    
    CURRENT_REPLICAS_GAUGE.set(current_replicas)
    
    # Enhanced logging with Kubernetes state architecture
    replicas_unavailable = metrics.get('kube_deployment_status_replicas_unavailable', 0)
    pods_ready = metrics.get('kube_pod_container_status_ready', 1)
    desired_replicas = metrics.get('kube_deployment_spec_replicas', 1)
    containers_running = metrics.get('kube_pod_container_status_running', 1)
    deployment_generation = metrics.get('kube_deployment_status_observed_generation', 1)
    cpu_limits = metrics.get('kube_pod_container_resource_limits_cpu', 0.5)
    memory_limits = metrics.get('kube_pod_container_resource_limits_memory', 536870912)
    network_up = metrics.get('node_network_up', 1)
    terminated_exitcode = metrics.get('kube_pod_container_status_last_terminated_exitcode', 0)
    
    logger.info(f"KUBERNETES_STATE: unavailable_replicas={replicas_unavailable} "
                f"pods_ready={pods_ready} desired_replicas={desired_replicas} "
                f"containers_running={containers_running} deployment_gen={deployment_generation} "
                f"current_replicas={current_replicas}")
    
    logger.info(f"RESOURCE_LIMITS: cpu_cores={cpu_limits:.2f} "
                f"memory_mb={memory_limits/1000000:.1f} network_up={network_up} "
                f"last_exit_code={terminated_exitcode}")
    
    logger.info(f"MULTI_DIMENSIONAL: aggregated_across_pods=consumer-* "
                f"cpu_sum={cpu_limits:.2f} memory_sum={memory_limits/1000000:.1f}MB "
                f"ready_containers={pods_ready} running_containers={containers_running} "
                f"termination_health={terminated_exitcode}")
    
    # Log LSTM proactive analysis if available
    if lstm_features:
        logger.info(f"LSTM_ANALYSIS: current_pressure={lstm_features.get('current_load_pressure', 0):.2f} "
                   f"30sec_pressure={lstm_features.get('next_30sec_pressure', 0):.2f} "
                   f"60sec_pressure={lstm_features.get('next_60sec_pressure', 0):.2f} "
                   f"stability={lstm_features.get('load_stability', 0):.2f} "
                   f"optimal_replicas={lstm_features.get('optimal_replicas_forecast', 0):.1f}")
    
    # Log feature availability for debugging (clean architecture: no redundancy)
    # Check if features exist (not None) rather than > 0, since 0 can be a valid value
    base_features_available = sum(1 for feat in BASE_FEATURE_ORDER if feat in metrics)
    lstm_features_available = sum(1 for feat in LSTM_FEATURE_ORDER if feat in metrics)
    total_features_available = base_features_available + lstm_features_available
    
    logger.info(f"FEATURE_AVAILABILITY: total={total_features_available}/{len(FEATURE_ORDER)} "
               f"base={base_features_available}/{len(BASE_FEATURE_ORDER)} lstm={lstm_features_available}/{len(LSTM_FEATURE_ORDER)}")
    
    # Debug: Log missing features if any
    if total_features_available < len(FEATURE_ORDER):
        missing_base = [feat for feat in BASE_FEATURE_ORDER if feat not in metrics]
        missing_lstm = [feat for feat in LSTM_FEATURE_ORDER if feat not in metrics]
        if missing_base:
            logger.warning(f"MISSING_BASE_FEATURES: {missing_base}")
        if missing_lstm:
            logger.warning(f"MISSING_LSTM_FEATURES: {missing_lstm}")
    
    logger.info("=" * 60)
    logger.info(f"NODE_END: {node_name}")
    return {"current_metrics": metrics, "current_replicas": current_replicas}

async def get_dqn_recommendation(state: ScalingState) -> Dict[str, Any]:
    logger.info("NODE_START: get_dqn_recommendation")
    logger.info("=" * 60)
    if not scaler:
        logger.error("DQN_PREDICTION: scaler_not_loaded")
        return {"error": "SCALER_NOT_LOADED"}

    metrics = state['current_metrics'].copy()
    current_replicas = state['current_replicas']
    
    # FIXED: Only scale the 9 raw base features from Prometheus with proper feature names
    raw_features = [metrics.get(feat, 0.0) for feat in BASE_FEATURE_ORDER]
    
    # Transform numpy array to maintain consistency with how scaler was fitted (no feature names)
    raw_features_array = np.array([raw_features])
    scaled_raw_features = scaler.transform(raw_features_array)
    
    # Get LSTM features separately (these are already in appropriate ranges)
    lstm_features = [metrics.get(feat, 0.0) for feat in LSTM_FEATURE_ORDER]
    
    # Combine scaled raw features + unscaled LSTM features = 11 total for DQN
    final_feature_vector = scaled_raw_features[0].tolist() + lstm_features
    
    logger.info(f"FEATURE_COMPOSITION: scaled_raw={len(scaled_raw_features[0])} "
                f"lstm={len(lstm_features)} total={len(final_feature_vector)}")
    
    # Count available feature types (check existence, not value > 0)
    base_available = sum(1 for feat in BASE_FEATURE_ORDER if feat in metrics)
    lstm_available = sum(1 for feat in LSTM_FEATURE_ORDER if feat in metrics)
    
    logger.info(f"SYSTEM_STATE: replicas={current_replicas} "
               f"unavailable={metrics.get('kube_deployment_status_replicas_unavailable', 0)} "
               f"readiness={metrics.get('kube_pod_container_status_ready', 1.0):.3f}")
    
    # Log LSTM insights if available
    proactive_override = None
    if lstm_available > 0:
        next_30sec = metrics.get('next_30sec_pressure', 0)
        next_60sec = metrics.get('next_60sec_pressure', 0)
        trend_velocity = metrics.get('trend_velocity', 0)
        optimal_replicas = metrics.get('optimal_replicas_forecast', current_replicas)
        
        # PROACTIVE SCALING LOGIC - New LSTM architecture
        pressure_increase_30s = next_30sec - 0.5  # 0.5 is baseline/neutral pressure
        pressure_increase_60s = next_60sec - 0.5
        max_pressure_increase = max(pressure_increase_30s, pressure_increase_60s)
        
        # Strong proactive signals that should override DQN conservatism
        if max_pressure_increase > 0.3 and trend_velocity > 0.1:  # Strong load increase predicted
            proactive_override = "Scale Up"
            override_reason = f"LSTM predicts strong load increase: {max_pressure_increase:+.2f} pressure, {trend_velocity:+.2f} velocity"
        elif max_pressure_increase < -0.3 and trend_velocity < -0.1 and current_replicas > 1:  # Strong load decrease
            proactive_override = "Scale Down" 
            override_reason = f"LSTM predicts strong load decrease: {max_pressure_increase:+.2f} pressure, {trend_velocity:+.2f} velocity"
        elif abs(optimal_replicas - current_replicas) >= 2:  # LSTM strongly disagrees with current replica count
            if optimal_replicas > current_replicas:
                proactive_override = "Scale Up"
                override_reason = f"LSTM optimal replicas ({optimal_replicas:.1f}) significantly higher than current ({current_replicas})"
            else:
                proactive_override = "Scale Down"
                override_reason = f"LSTM optimal replicas ({optimal_replicas:.1f}) significantly lower than current ({current_replicas})"
        
        proactive_signal = "SCALE_UP_SOON" if max_pressure_increase > 0.2 else "STABLE" if max_pressure_increase > -0.2 else "SCALE_DOWN_SOON"
        
        logger.info(f"LSTM_INSIGHTS: 30sec_pressure={next_30sec:.2f} "
                   f"60sec_pressure={next_60sec:.2f} "
                   f"pressure_change={max_pressure_increase:+.2f} "
                   f"trend_velocity={trend_velocity:+.2f} "
                   f"optimal_replicas={optimal_replicas:.1f} "
                   f"proactive_signal={proactive_signal}")
        
        if proactive_override:
            logger.warning(f"PROACTIVE_OVERRIDE: {proactive_override} - {override_reason}")
    
    try:
        if dqn_model is not None:
            # Use local PyTorch model with epsilon-greedy exploration
            global current_epsilon, decision_count
            
            device = next(dqn_model.parameters()).device
            input_tensor = torch.FloatTensor([final_feature_vector]).to(device)
            
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
            
            # Epsilon-greedy exploration with detailed reasoning
            if random.random() < current_epsilon:
                action_index = random.randint(0, 2)  # Random action
                exploration_type = "exploration"
                logger.info(f"DQN_EXPLORATION: random_action epsilon={current_epsilon:.3f}")
            else:
                action_index = np.argmax(q_values)  # Greedy action
                exploration_type = "exploitation"  
                logger.info(f"DQN_EXPLOITATION: best_action epsilon={current_epsilon:.3f}")
            
            # Decay epsilon
            current_epsilon = max(EPSILON_END, current_epsilon * EPSILON_DECAY)
            decision_count += 1
            
            action_map = {0: "Scale Down", 1: "Keep Same", 2: "Scale Up"}
            dqn_action_name = action_map.get(action_index, "Unknown")
            
            # Let DQN make the decision - no overrides
            action_name = dqn_action_name
            override_applied = False
            
            # Log LSTM insights for training feedback
            if proactive_override and proactive_override != action_name:
                logger.info(f"LSTM_SIGNAL: suggests={proactive_override} dqn_chose={action_name} (training_opportunity)")
            
            # Generate comprehensive explanation using explainable AI system
            explanation = decision_reasoning.explain_dqn_decision(
                metrics=metrics,
                q_values=q_values.tolist(),
                action_name=action_name,
                exploration_type=exploration_type,
                epsilon=current_epsilon,
                current_replicas=current_replicas
            )
            
            # Add override information to explanation
            if override_applied:
                explanation['proactive_override'] = {
                    'applied': True,
                    'dqn_recommendation': dqn_action_name,
                    'lstm_override': action_name,
                    'reason': override_reason
                }
            
            # Log detailed reasoning
            decision_reasoning.log_decision_reasoning(explanation, metrics, q_values.tolist())
            
            # Enhanced logging with confidence and reasoning
            confidence = explanation['confidence_metrics']['decision_confidence']
            risk_level = explanation['risk_assessment']
            confidence_gap = explanation['confidence_metrics']['confidence_gap']
            
            if override_applied:
                logger.info(f"FINAL_DECISION: action={action_name} (DQN={dqn_action_name}, LSTM_OVERRIDE) confidence={confidence} risk={risk_level}")
            else:
                logger.info(f"DQN_DECISION: action={action_name} confidence={confidence} risk={risk_level}")
            logger.info(f"DQN_Q_VALUES: values=[{','.join(f'{q:.3f}' for q in q_values)}] gap={confidence_gap:.3f}")
            logger.info(f"DQN_REASONING: factors_count={len(explanation['reasoning_factors'])}")
            
            # Update Prometheus metrics
            DQN_DECISIONS_COUNTER.inc()
            DQN_EPSILON_GAUGE.set(current_epsilon)
            
            # Convert confidence string to numeric value for Prometheus
            confidence_numeric = 0.5  # Default medium confidence
            if isinstance(confidence, str):
                if confidence.lower() == 'high':
                    confidence_numeric = 1.0
                elif confidence.lower() == 'medium':
                    confidence_numeric = 0.5
                elif confidence.lower() == 'low':
                    confidence_numeric = 0.0
            else:
                confidence_numeric = float(confidence)
            
            DQN_DECISION_CONFIDENCE_GAUGE.set(confidence_numeric)
            DQN_Q_VALUE_SCALE_UP_GAUGE.set(q_values[2])  # Scale up is action 2
            DQN_Q_VALUE_SCALE_DOWN_GAUGE.set(q_values[0])  # Scale down is action 0
            DQN_Q_VALUE_KEEP_SAME_GAUGE.set(q_values[1])  # Keep same is action 1
            
            # Update action counters
            if action_name == "Scale Up":
                DQN_ACTION_SCALE_UP_COUNTER.inc()
            elif action_name == "Scale Down":
                DQN_ACTION_SCALE_DOWN_COUNTER.inc()
            else:  # Keep Same
                DQN_ACTION_KEEP_SAME_COUNTER.inc()
            
            # Update exploration/exploitation counters
            if exploration_type == "exploration":
                DQN_EXPLORATION_COUNTER.inc()
            else:
                DQN_EXPLOITATION_COUNTER.inc()
            
            # Update LSTM feature metrics if available
            if 'next_30sec_pressure' in metrics:
                DQN_LSTM_NEXT_30SEC_GAUGE.set(metrics['next_30sec_pressure'])
            if 'next_60sec_pressure' in metrics:
                DQN_LSTM_NEXT_60SEC_GAUGE.set(metrics['next_60sec_pressure'])
            if 'trend_velocity' in metrics:
                DQN_LSTM_TREND_VELOCITY_GAUGE.set(metrics['trend_velocity'])
            if 'pattern_confidence' in metrics:
                DQN_LSTM_PATTERN_CONFIDENCE_GAUGE.set(metrics['pattern_confidence'])
            
            # Log specific reasoning for this decision
            if ENABLE_DETAILED_REASONING:
                logger.info("DQN_FACTORS: summary_top3")
                for i, factor in enumerate(explanation['reasoning_factors'][:3]):  # Top 3 factors
                    logger.info(f"DQN_FACTOR_{i+1}: {factor}")
            
            experience_update = {"state": metrics, "action": action_name}
            
            logger.info("=" * 60)
            logger.info("NODE_END: get_dqn_recommendation")
            return {
                "dqn_prediction": {
                    "action_name": action_name, 
                    "q_values": q_values.tolist(), 
                    "epsilon": current_epsilon, 
                    "exploration_type": exploration_type,
                    "explanation": explanation,
                    "confidence": confidence,
                    "risk_assessment": risk_level
                }, 
                "experience": experience_update
            }
        else:
            # Enhanced fallback: Rule-based logic with balanced selected features
            logger.info("DQN_FALLBACK: using_rule_based_logic model_not_available")
            
            # Get current Kubernetes state features for fallback decision
            replicas_unavailable = metrics.get('kube_deployment_status_replicas_unavailable', 0)
            pods_ready = metrics.get('kube_pod_container_status_ready', 1)
            desired_replicas = metrics.get('kube_deployment_spec_replicas', 1)
            containers_running = metrics.get('kube_pod_container_status_running', 1)
            deployment_generation = metrics.get('kube_deployment_status_observed_generation', 1)
            cpu_limits = metrics.get('kube_pod_container_resource_limits_cpu', 0.5)
            memory_limits = metrics.get('kube_pod_container_resource_limits_memory', 536870912)
            network_up = metrics.get('node_network_up', 1)
            terminated_exitcode = metrics.get('kube_pod_container_status_last_terminated_exitcode', 0)
            
            # Generate analysis for fallback decision
            analysis = decision_reasoning.analyze_metrics(metrics, current_replicas)
            
            # Enhanced rule-based decision with Kubernetes state features
            reasoning_factors = []
            
            # Decision logic based on multi-dimensional Kubernetes state
            memory_mb = memory_limits / 1000000  # Convert to MB for readability
            
            if replicas_unavailable > 0 or pods_ready < 0.8 or containers_running < 0.8:  # Health issues
                action_name = "Scale Up"
                reasoning_factors.append(f"Health issues detected: {replicas_unavailable} unavailable, {pods_ready:.1f} ready ratio, {containers_running:.1f} running ratio")
                reasoning_factors.append("Scaling up to improve system health and availability")
                risk_level = "high"
            elif cpu_limits > 2.0 or memory_mb > 800 or terminated_exitcode > 0:  # Resource pressure
                action_name = "Scale Up"
                reasoning_factors.append(f"Resource pressure: {cpu_limits:.1f} CPU cores, {memory_mb:.1f}MB memory, exit code {terminated_exitcode}")
                reasoning_factors.append("Scaling up to handle resource pressure")
                risk_level = "medium"
            elif cpu_limits < 1.0 and memory_mb < 300 and current_replicas > 1 and network_up > 0.8:  # Low utilization
                action_name = "Scale Down"
                reasoning_factors.append(f"Low utilization: {cpu_limits:.1f} CPU cores, {memory_mb:.1f}MB memory, healthy network")
                reasoning_factors.append(f"System has excess capacity with {current_replicas} replicas - can optimize costs")
                risk_level = "low"
            else:
                action_name = "Keep Same"
                reasoning_factors.append(f"Balanced state: {cpu_limits:.1f} CPU cores, {memory_mb:.1f}MB memory, {desired_replicas} replicas")
                reasoning_factors.append("System operating within acceptable parameters")
                risk_level = "low"
            
            # Create fallback explanation
            fallback_explanation = {
                'timestamp': datetime.now().isoformat(),
                'decision_type': 'FALLBACK_RULE_BASED',
                'recommended_action': action_name,
                'exploration_strategy': 'rule_based',
                'confidence_metrics': {
                    'decision_confidence': 'medium',
                    'rule_based': True
                },
                'reasoning_factors': reasoning_factors,
                'risk_assessment': risk_level,
                'system_analysis': analysis
            }
            
            logger.info(f"FALLBACK_DECISION: action={action_name} risk={risk_level}")
            logger.info(f"FALLBACK_REASONING: {reasoning_factors[0]}")
            
            if ENABLE_DETAILED_REASONING:
                logger.info("FALLBACK_ANALYSIS: detailed_factors")
                for i, factor in enumerate(reasoning_factors):
                    logger.info(f"FALLBACK_FACTOR_{i+1}: {factor}")
            
            experience_update = {"state": metrics, "action": action_name}
            
            logger.info("=" * 60)
            logger.info("NODE_END: get_dqn_recommendation")
            return {
                "dqn_prediction": {
                    "action_name": action_name, 
                    "q_values": [0.0, 1.0, 0.0],
                    "explanation": fallback_explanation,
                    "confidence": "medium",
                    "risk_assessment": risk_level,
                    "fallback_mode": True
                }, 
                "experience": experience_update
            }
            
    except Exception as e:
        logger.error(f"DQN inference failed: {e}", exc_info=True)
        logger.info("=" * 60)
        logger.info("NODE_END: get_dqn_recommendation")
        return {"error": f"DQN_INFERENCE_FAILED: {e}"}

async def validate_with_llm(state: ScalingState) -> Dict[str, Any]:
    logger.info("NODE_START: validate_with_llm")
    logger.info("=" * 60)
    
    # Check if LLM validation is enabled
    if not ENABLE_LLM_VALIDATION:
        logger.info("SAFETY_MONITOR: disabled_by_config skipping")
        return {"llm_validation_response": {"approved": True, "reason": "Safety monitor disabled by configuration.", "confidence": "medium"}}
    
    if not validator_agent:
        if ENABLE_LLM_VALIDATION:
            logger.warning("SAFETY_MONITOR: agent_not_initialized skipping")
            return {"llm_validation_response": {"approved": True, "reason": "Safety monitor enabled but agent not available.", "confidence": "low"}}
        else:
            # This case should not happen since we check ENABLE_LLM_VALIDATION earlier, but adding for safety
            logger.debug("SAFETY_MONITOR: agent_not_initialized validation_disabled")
            return {"llm_validation_response": {"approved": True, "reason": "Safety monitor disabled by configuration.", "confidence": "medium"}}

    # Handle missing dqn_prediction gracefully
    dqn_prediction = state.get('dqn_prediction', {'action_name': 'Keep Same'})
    action_name = dqn_prediction.get('action_name', 'Keep Same')
    dqn_confidence = dqn_prediction.get('confidence', 'unknown')
    dqn_risk = dqn_prediction.get('risk_assessment', 'unknown')
    dqn_explanation = dqn_prediction.get('explanation', {})

    # Extract LSTM predictions for LLM context
    metrics = state['current_metrics']
    next_30sec = metrics.get('next_30sec_pressure', 0)
    next_60sec = metrics.get('next_60sec_pressure', 0)
    trend_velocity = metrics.get('trend_velocity', 0)
    optimal_replicas = metrics.get('optimal_replicas_forecast', state['current_replicas'])
    
    # Calculate pressure changes for context
    baseline_pressure = 0.5  # Neutral pressure level
    pressure_change_30s = next_30sec - baseline_pressure
    pressure_change_60s = next_60sec - baseline_pressure
    
    # Create comprehensive validation prompt with LSTM intelligence
    prompt = f"""You are a SAFETY MONITOR for Kubernetes autoscaling. Your ONLY role is to detect and prevent EXTREME or DANGEROUS scaling decisions that could harm the cluster.

🚨 CRITICAL: Only intervene for EXTREME decisions. Allow normal DQN learning to proceed uninterrupted.

DQN SCALING DECISION TO EVALUATE:
- Recommended Action: {action_name}
- Current Replicas: {state['current_replicas']}
- DQN Confidence: {dqn_confidence}
- DQN Risk Assessment: {dqn_risk}

DQN REASONING FACTORS:
{chr(10).join(f"- {factor}" for factor in dqn_explanation.get('reasoning_factors', ['No DQN reasoning available']))}

CURRENT SYSTEM METRICS:
- Pod Readiness: {metrics.get('kube_pod_container_status_ready', 1.0):.1%}
- Unavailable Replicas: {metrics.get('kube_deployment_status_replicas_unavailable', 0)}
- CPU Usage: {metrics.get('process_cpu_seconds_total', 0):.1f}%
- HTTP Latency: {metrics.get('http_request_duration_highr_seconds_sum', 0):.3f}s
- Container Restarts: {metrics.get('kube_pod_container_status_restarts_total', 0)}

LSTM PREDICTIVE INTELLIGENCE:
- Next 30sec Load Pressure: {next_30sec:.2f} (change: {pressure_change_30s:+.2f})
- Next 60sec Load Pressure: {next_60sec:.2f} (change: {pressure_change_60s:+.2f})
- Trend Velocity: {trend_velocity:+.2f}
- LSTM Optimal Replicas: {optimal_replicas:.1f}

🔍 EXTREME DECISION CRITERIA (Block these ONLY):

1. **RUNAWAY SCALING**: Scaling to >15 replicas without clear justification
2. **RESOURCE EXHAUSTION**: Scaling up when cluster resources are constrained  
3. **MASSIVE OVER-SCALING**: Requesting 3x+ more replicas than LSTM suggests
4. **DANGEROUS DOWN-SCALING**: Scaling to 0 or very low when load is increasing
5. **RAPID OSCILLATION**: Frequent large scaling changes without stabilization
6. **CONTRADICTS LSTM**: Scaling opposite to strong LSTM trend predictions
7. **IGNORES HIGH RISK**: DQN marked decision as "high" risk but proceeding anyway

⚠️  **DEFAULT: APPROVE** - Only block if decision meets extreme criteria above.

🛡️  **MCP ACCESS**: You have read-only access to Kubernetes cluster via MCP tools if needed for safety assessment.

SAFETY ASSESSMENT QUESTIONS:
- Does the DQN reasoning justify this decision?
- Is this decision likely to cause cluster instability?
- Does this risk resource exhaustion or service outage?
- Is the DQN risk assessment reasonable given the circumstances?
- Does this contradict strong LSTM predictions without good reason?

IMPORTANT: You may use READ-ONLY Kubernetes tools to inspect cluster state if needed.
DO NOT attempt to scale, modify, or perform any write operations.

CRITICAL: Respond with ONLY a valid JSON object. No markdown, no explanations.

Example for NORMAL decision (approve):
{{
    "approved": true,
    "confidence": "high", 
    "reasoning": "DQN reasoning is sound and decision is within safe parameters",
    "safety_risk": "none",
    "extreme_factors": [],
    "cluster_check_performed": false
}}

Example for EXTREME decision (block):
{{
    "approved": false,
    "confidence": "high",
    "reasoning": "EXTREME: Scaling to 25 replicas risks cluster resource exhaustion despite DQN confidence",
    "safety_risk": "high",
    "extreme_factors": ["runaway_scaling", "resource_exhaustion_risk"],
    "alternative_suggestion": "Scale more gradually to 8-10 replicas first",
    "cluster_check_performed": true
}}

Your JSON response:"""

    try:
        logger.info(f"SAFETY_MONITOR: evaluating action={action_name} dqn_confidence={dqn_confidence}")
        
        # Invoke the validator agent
        response = await validator_agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
        last_message = response['messages'][-1].content
        
        logger.info(f"SAFETY_MONITOR: response_received chars={len(last_message)}")
        
        # Debug: Log first 200 chars of response to see what we're getting
        if len(last_message) > 0:
            logger.debug(f"SAFETY_RAW_RESPONSE: {last_message[:200]}...")
        
        # Enhanced JSON parsing with fallback
        validation_result = parse_llm_json_response(last_message, action_name)
        
        # Enhanced logging with safety assessment outcome
        safety_status = "BLOCKED" if not validation_result['approved'] else "APPROVED"
        safety_risk = validation_result.get('safety_risk', 'none')
        logger.info(f"SAFETY_MONITOR: {safety_status} risk_level={safety_risk}")
        
        # Log safety details only if decision was blocked or high risk detected
        if not validation_result['approved'] or safety_risk in ['high', 'medium']:
            logger.warning("SAFETY_ALERT: detailed_analysis")
            logger.warning(f"SAFETY_DECISION: approved={validation_result['approved']} "
                         f"confidence={validation_result['confidence']} "
                         f"risk={safety_risk}")
            
            logger.warning(f"SAFETY_REASONING: {validation_result['reasoning']}")
            
            if validation_result.get('extreme_factors'):
                logger.warning(f"EXTREME_FACTORS: count={len(validation_result['extreme_factors'])}")
                for i, factor in enumerate(validation_result['extreme_factors']):
                    logger.warning(f"EXTREME_FACTOR_{i+1}: {factor}")
            
            if validation_result.get('alternative_suggestion'):
                logger.info(f"SAFETY_ALTERNATIVE: {validation_result['alternative_suggestion']}")
        else:
            # Normal approval - minimal logging to avoid noise
            logger.info(f"SAFETY_APPROVED: confidence={validation_result['confidence']}")
        
        if not validation_result['approved']:
            logger.error(f"SCALING_BLOCKED: reason={validation_result['reasoning']}")
            if validation_result.get('alternative_suggestion'):
                logger.info(f"SUGGESTED_ALTERNATIVE: {validation_result['alternative_suggestion']}")
        
        logger.info("=" * 60)
        logger.info("NODE_END: validate_with_llm")
        return {"llm_validation_response": validation_result}
        
    except Exception as e:
        logger.error(f"SAFETY_MONITOR: failed error={e}")
        
        # Check if this is a permission-related error or JSON parsing issue
        error_str = str(e).lower()
        is_permission_error = any(keyword in error_str for keyword in
                                 ['permission', 'forbidden', 'unauthorized', 'rbac', 'access denied'])
        is_parsing_error = 'json' in error_str or 'parse' in error_str
        
        if is_permission_error:
            logger.warning("SAFETY_MONITOR: permission_error_detected suggesting_disable")
            logger.warning("SAFETY_MONITOR: consider_setting ENABLE_LLM_VALIDATION=false")
        elif is_parsing_error:
            logger.warning("SAFETY_MONITOR: json_parsing_issues_detected")
            logger.warning("SAFETY_MONITOR: llm_not_following_json_format")
        
        # Enhanced fallback - APPROVE by default for safety (preserve DQN learning)
        fallback_result = {
            "approved": True,  # Safety-first: only block explicitly dangerous decisions
            "confidence": "low",
            "reasoning": f"Safety monitor error: {str(e)}. Defaulting to APPROVAL to preserve DQN learning.",
            "safety_risk": "unknown",
            "extreme_factors": ["safety_monitor_unavailable"],
            "alternative_suggestion": "Monitor system closely - safety checks temporarily disabled",
            "cluster_check_performed": False,
            "validation_score": 0.3,
            "fallback_mode": True,
            "permission_error": is_permission_error
        }
        
        logger.warning("SAFETY_MONITOR: fallback_approved preserving_dqn_learning")
        logger.info("=" * 60)
        logger.info("NODE_END: validate_with_llm")
        return {"llm_validation_response": fallback_result}

def parse_llm_json_response(response_text: str, action_name: str) -> Dict[str, Any]:
    """Parse LLM JSON response with robust error handling and safety-first fallbacks."""
    try:
        import re
        
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)
            
            # Validate required fields for safety monitoring
            required_fields = ['approved', 'confidence', 'reasoning']
            if all(field in parsed for field in required_fields):
                
                # Ensure all expected safety fields exist with defaults
                safety_defaults = {
                    'safety_risk': parsed.get('safety_risk', 'none'),
                    'extreme_factors': parsed.get('extreme_factors', []),
                    'alternative_suggestion': parsed.get('alternative_suggestion', ''),
                    'cluster_check_performed': parsed.get('cluster_check_performed', False),
                    'validation_score': 0.8 if parsed.get('approved', True) else 0.3
                }
                
                # Merge parsed response with safety defaults
                result = {**parsed, **safety_defaults}
                
                # Ensure confidence is valid
                if result['confidence'] not in ['low', 'medium', 'high']:
                    result['confidence'] = 'medium'
                
                return result
        
        # If JSON parsing failed, log and use safety fallback
        logger.warning(f"LLM_PARSING: json_extraction_failed response_preview={response_text[:100]}")
        
    except json.JSONDecodeError as e:
        logger.warning(f"LLM_PARSING: json_decode_error position={e.pos}")
        
    except Exception as e:
        logger.warning(f"LLM_PARSING: general_parsing_error error={e}")
    
    # SAFETY FALLBACK: Default to APPROVE for any parsing failures
    # This preserves DQN learning and only blocks when LLM explicitly identifies extreme risk
    return {
        'approved': True,  # Safety-first: approve unless explicitly dangerous
        'confidence': 'low',
        'reasoning': f'Parsing failure, defaulting to APPROVE for safety. Response: {response_text[:200]}...',
        'safety_risk': 'unknown',
        'extreme_factors': ['parsing_failure'],
        'alternative_suggestion': 'Monitor decision closely due to validation parsing failure',
        'cluster_check_performed': False,
        'validation_score': 0.5,
        'fallback_mode': True
    }

def plan_final_action(state: ScalingState) -> Dict[str, Any]:
    logger.info("NODE_START: plan_final_action")
    logger.info("=" * 60)
    current_replicas = state['current_replicas']
    
    # Handle missing dqn_prediction gracefully
    dqn_prediction = state.get('dqn_prediction', {'action_name': 'Keep Same'})
    action_name = dqn_prediction.get('action_name', 'Keep Same')
    dqn_confidence = dqn_prediction.get('confidence', 'unknown')
    dqn_risk = dqn_prediction.get('risk_assessment', 'unknown')
    
    # Get LLM validation results
    llm_validation = state.get('llm_validation_response', {})
    llm_approved = llm_validation.get('approved', True)
    llm_confidence = llm_validation.get('confidence', 'unknown')
    llm_reasoning = llm_validation.get('reasoning', 'No validation reasoning available')
    
    # Calculate new replica count
    new_replicas = current_replicas
    if action_name == 'Scale Up': 
        new_replicas += 1
    elif action_name == 'Scale Down': 
        new_replicas -= 1
    
    # SAFETY OVERRIDE: If safety monitor blocks the decision, keep current replicas
    if not llm_approved:
        logger.warning(f"SAFETY_OVERRIDE: blocking_{action_name} maintaining_current_replicas={current_replicas}")
        final_decision = current_replicas  # KEEP CURRENT - NO SCALING
        decision_explanation['decision_factors'].append("SAFETY OVERRIDE: Extreme scaling decision blocked - maintaining current replica count")
        decision_explanation['risk_mitigation'].append("CRITICAL: Dangerous scaling prevented by safety monitor")
    else:
        final_decision = max(1, min(20, new_replicas))  # Normal decision with safety constraints
    
    # Ensure we never go below 1 replica regardless of safety decisions
    final_decision = max(1, final_decision)
    
    # Create comprehensive decision explanation
    decision_explanation = {
        'timestamp': datetime.now().isoformat(),
        'decision_pipeline': {
            'dqn_recommendation': {
                'action': action_name,
                'confidence': dqn_confidence,
                'risk_assessment': dqn_risk
            },
            'llm_validation': {
                'approved': llm_approved,
                'confidence': llm_confidence,
                'reasoning_summary': llm_reasoning
            },
            'final_decision': {
                'from_replicas': current_replicas,
                'to_replicas': final_decision,
                'action_executed': 'Scale Up' if final_decision > current_replicas else 'Scale Down' if final_decision < current_replicas else 'Keep Same'
            }
        },
        'decision_factors': [],
        'risk_mitigation': [],
        'expected_outcomes': []
    }
    
    # Add decision factors
    if llm_approved:
        decision_explanation['decision_factors'].append("Safety monitor found no extreme risks - DQN decision approved")
    else:
        decision_explanation['decision_factors'].append("SAFETY ALERT: Decision blocked due to extreme risk factors")
        decision_explanation['risk_mitigation'].append("CRITICAL: Extreme scaling decision prevented by safety monitor")
    
    # Add confidence assessment
    overall_confidence = 'high'
    if dqn_confidence == 'low' or llm_confidence == 'low':
        overall_confidence = 'low'
    elif dqn_confidence == 'medium' or llm_confidence == 'medium':
        overall_confidence = 'medium'
    
    decision_explanation['overall_confidence'] = overall_confidence
    
    # Add expected outcomes based on action
    if final_decision > current_replicas:
        decision_explanation['expected_outcomes'].extend([
            "Improved response times and reduced latency",
            "Better handling of increased load",
            "Higher resource costs"
        ])
        decision_explanation['risk_mitigation'].append("Monitor for over-provisioning")
    elif final_decision < current_replicas:
        decision_explanation['expected_outcomes'].extend([
            "Reduced resource costs",
            "Optimized resource utilization",
            "Potential slight increase in response times"
        ])
        decision_explanation['risk_mitigation'].append("Monitor for performance degradation")
    else:
        decision_explanation['expected_outcomes'].append("Maintained current performance and cost balance")
    
    # Update the Prometheus gauge
    DESIRED_REPLICAS_GAUGE.set(final_decision)
    
    # Create comprehensive audit trail
    if 'dqn_prediction' in state and 'explanation' in dqn_prediction:
        # Simple audit trail creation (inline)
        audit_trail = {
            'decision_id': f"dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'current_replicas': current_replicas,
            'final_decision': final_decision,
            'dqn_action': action_name,
            'dqn_confidence': dqn_confidence,
            'safety_approved': llm_approved,
            'safety_reasoning': llm_reasoning
        }
        decision_explanation['audit_trail_id'] = audit_trail['decision_id']
    
    # Enhanced logging with complete decision reasoning
    logger.info(f"FINAL_DECISION: scale_from={current_replicas} scale_to={final_decision}")
    logger.info(f"FINAL_CONFIDENCE: overall={overall_confidence} dqn={dqn_confidence} safety={llm_confidence}")
    logger.info(f"SAFETY_STATUS: approved={llm_approved}")
    logger.info(f"PROMETHEUS: gauge_updated value={final_decision}")
    
    if ENABLE_DETAILED_REASONING:
        logger.info("FINAL_DECISION_ANALYSIS: detailed_breakdown")
        logger.info(f"FINAL_DECISION_ANALYSIS: action_path={action_name}_to_{decision_explanation['decision_pipeline']['final_decision']['action_executed']}")
        logger.info(f"FINAL_DECISION_ANALYSIS: expected_outcomes_count={len(decision_explanation['expected_outcomes'])}")
        for i, outcome in enumerate(decision_explanation['expected_outcomes']):
            logger.info(f"FINAL_DECISION_ANALYSIS: outcome_{i+1}={outcome}")
        
        if decision_explanation['risk_mitigation']:
            logger.info(f"FINAL_DECISION_ANALYSIS: risk_mitigation_count={len(decision_explanation['risk_mitigation'])}")
            for i, mitigation in enumerate(decision_explanation['risk_mitigation']):
                logger.info(f"FINAL_DECISION_ANALYSIS: mitigation_{i+1}={mitigation}")
    
    # Warning for low confidence decisions
    if overall_confidence == 'low':
        logger.warning("LOW_CONFIDENCE: enhanced_monitoring_recommended")
        logger.warning("LOW_CONFIDENCE: consider_manual_review")
    
    # CRITICAL warning for safety monitor blocking (extreme decisions only)
    if not llm_approved:
        logger.error("EXTREME_DECISION_BLOCKED: safety_monitor_intervention")
        logger.error(f"SAFETY_BLOCK_REASON: {llm_reasoning}")
        logger.error("EXTREME_DECISION_BLOCKED: this_indicates_potentially_dangerous_scaling")
    
    logger.info("=" * 60)
    logger.info("NODE_END: plan_final_action")
    return {
        "final_decision": final_decision,
        "decision_explanation": decision_explanation,
        "overall_confidence": overall_confidence,
        "llm_approved": llm_approved
    }

async def wait_for_system_to_stabilize(state: ScalingState) -> None:
    logger.info("NODE_START: wait_for_system_to_stabilize")
    logger.info("=" * 60)
    logger.info(f"STABILIZATION: waiting_seconds={STABILIZATION_PERIOD_SECONDS}")
    await asyncio.sleep(STABILIZATION_PERIOD_SECONDS)
    logger.info("=" * 60)
    logger.info("NODE_END: wait_for_system_to_stabilize")
    return {}

async def observe_next_state_and_calculate_reward(state: ScalingState) -> Dict[str, Any]:
    logger.info("NODE_START: observe_next_state_and_calculate_reward")
    logger.info("=" * 60)
    
    # Get next state data
    next_state_data = await get_live_metrics(state, is_next_state=True)
    next_state_metrics = next_state_data.get("current_metrics", {})
    current_replicas = next_state_data.get("current_replicas", 1)
    
    # Get current state for comparison
    current_state_metrics = state.get("current_metrics", {})
    
    # Extract action for context - handle missing experience gracefully
    experience = state.get('experience')
    if not experience:
        logger.error("REWARD_CALCULATION: experience_missing from_state, cannot_calculate_reward")
        return {"error": "Experience data not available for reward calculation"}
    
    action = experience.get('action', 'Keep Same')
    
    # Get LSTM predictions from the experience (if available)
    lstm_predictions = None
    if 'state' in experience and experience['state']:
        # Extract LSTM features that were available at decision time
        lstm_predictions = {
            'next_30sec_pressure': experience['state'].get('next_30sec_pressure'),
            'next_60sec_pressure': experience['state'].get('next_60sec_pressure'), 
            'trend_velocity': experience['state'].get('trend_velocity'),
            'pattern_type_spike': experience['state'].get('pattern_type_spike'),
            'pattern_type_gradual': experience['state'].get('pattern_type_gradual'),
            'pattern_type_cyclical': experience['state'].get('pattern_type_cyclical')
        }
    
    # Calculate improved reward using the new multi-objective function
    reward = calculate_stable_reward(current_state_metrics, next_state_metrics, action, 
                                   current_replicas, lstm_predictions)
    
    experience['reward'] = reward
    experience['next_state'] = {**next_state_metrics, 'current_replicas': current_replicas}
    
    # Update reward metrics (assuming reward components are available)
    DQN_REWARD_TOTAL_GAUGE.set(reward)
    if isinstance(reward, dict):
        # If reward is broken down into components
        DQN_REWARD_PERFORMANCE_GAUGE.set(reward.get('performance', 0))
        DQN_REWARD_RESOURCE_GAUGE.set(reward.get('resource', 0))
        DQN_REWARD_HEALTH_GAUGE.set(reward.get('health', 0))
        DQN_REWARD_COST_GAUGE.set(reward.get('cost', 0))
    else:
        # If reward is a single value, estimate components based on weights
        performance_component = reward * REWARD_PERFORMANCE_WEIGHT
        resource_component = reward * REWARD_RESOURCE_WEIGHT
        health_component = reward * REWARD_HEALTH_WEIGHT
        cost_component = reward * REWARD_COST_WEIGHT
        
        DQN_REWARD_PERFORMANCE_GAUGE.set(performance_component)
        DQN_REWARD_RESOURCE_GAUGE.set(resource_component)
        DQN_REWARD_HEALTH_GAUGE.set(health_component)
        DQN_REWARD_COST_GAUGE.set(cost_component)
    
    logger.info("=" * 60)
    logger.info("NODE_END: observe_next_state_and_calculate_reward")
    return {"experience": experience}

def calculate_stable_reward(current_state, next_state, action, current_replicas, lstm_predictions=None):
    """
    IMPROVED: Multi-objective reward function that avoids circular dependencies.
    Uses outcome-based metrics separate from DQN input features.
    """
    
    # === PERFORMANCE COMPONENT (40%) ===
    # Use derived metrics that measure actual system outcomes, not raw input features
    
    # 1. Response time improvement (derived from HTTP metrics)
    current_http_bucket = current_state.get('http_request_duration_highr_seconds_bucket', 0)
    next_http_bucket = next_state.get('http_request_duration_highr_seconds_bucket', 0)
    current_response_size = current_state.get('http_response_size_bytes_sum', 0)
    next_response_size = next_state.get('http_response_size_bytes_sum', 0)
    
    # Calculate response efficiency (lower bucket count + reasonable response size = better performance)
    current_efficiency = calculate_response_efficiency(current_http_bucket, current_response_size)
    next_efficiency = calculate_response_efficiency(next_http_bucket, next_response_size)
    performance_improvement = (next_efficiency - current_efficiency) * 4.0
    
    # === RESOURCE EFFICIENCY COMPONENT (30%) ===
    # Measure resource utilization effectiveness
    
    current_memory = current_state.get('process_resident_memory_bytes', 100000000) / 1000000  # MB
    next_memory = next_state.get('process_resident_memory_bytes', 100000000) / 1000000
    current_cpu = current_state.get('process_cpu_seconds_total', 0)
    next_cpu = next_state.get('process_cpu_seconds_total', 0)
    
    # Resource efficiency: stable memory + appropriate CPU usage
    memory_stability = max(0, 3.0 - abs(next_memory - current_memory) * 0.002)  # Reward stability
    cpu_efficiency = calculate_cpu_efficiency(current_cpu, next_cpu, current_replicas)
    resource_efficiency = (memory_stability + cpu_efficiency) / 2.0
    
    # === SYSTEM HEALTH COMPONENT (20%) ===
    # Use health indicators that reflect system reliability
    
    current_scrape = current_state.get('scrape_samples_scraped', 300)
    next_scrape = next_state.get('scrape_samples_scraped', 300)
    
    # Health score based on monitoring effectiveness and system stability
    health_score = calculate_health_score(current_scrape, next_scrape, next_memory, next_http_bucket)
    
    # === COST OPTIMIZATION COMPONENT (10%) ===
    # Economic efficiency with non-linear scaling costs
    
    base_cost = 1.0
    replica_cost = base_cost * (current_replicas ** 1.2)  # Non-linear cost growth
    cost_efficiency = max(0, 5.0 - replica_cost * 0.05)  # Diminishing returns for more replicas
    
    # === PROACTIVE INTELLIGENCE BONUS ===
    # FIXED: Use LSTM predictions from CURRENT state, not next state
    proactive_bonus = 0.0
    if lstm_predictions:
        proactive_bonus = calculate_proactive_bonus(action, lstm_predictions, 
                                                   current_state, next_state)
    
    # === STABILITY PENALTY ===
    # Penalize excessive scaling actions to encourage stability
    action_penalty = calculate_action_penalty(action, current_replicas)
    
    # === WEIGHTED COMBINATION ===
    # Use configurable reward component weights
    total_reward = (
        performance_improvement * REWARD_PERFORMANCE_WEIGHT +    # Configurable: User experience
        resource_efficiency * REWARD_RESOURCE_WEIGHT +          # Configurable: Resource optimization  
        health_score * REWARD_HEALTH_WEIGHT +                   # Configurable: System reliability
        cost_efficiency * REWARD_COST_WEIGHT +                  # Configurable: Economic efficiency
        proactive_bonus +                                       # Bonus: Intelligent prediction alignment
        action_penalty                                          # Penalty: Excessive scaling
    )
    
    # Normalize to [-10, 10] range for stable training
    total_reward = np.clip(total_reward, -10.0, 10.0)
    
    logger.info(f"REWARD_BREAKDOWN: performance={performance_improvement:.2f}({REWARD_PERFORMANCE_WEIGHT:.0%}) "
               f"resource={resource_efficiency:.2f}({REWARD_RESOURCE_WEIGHT:.0%}) health={health_score:.2f}({REWARD_HEALTH_WEIGHT:.0%}) "
               f"cost={cost_efficiency:.2f}({REWARD_COST_WEIGHT:.0%}) proactive={proactive_bonus:.2f} "
               f"action_penalty={action_penalty:.2f} total={total_reward:.2f}")
    
    return total_reward


def calculate_response_efficiency(http_bucket_count, response_size_sum):
    """Calculate response efficiency score (higher is better)."""
    # Normalize metrics and combine for efficiency score
    bucket_score = max(0, 5.0 - http_bucket_count * 0.002)  # Lower bucket count = better
    size_score = max(0, 3.0 - response_size_sum * 0.00001)  # Reasonable response size
    return (bucket_score + size_score) / 2.0


def calculate_cpu_efficiency(current_cpu, next_cpu, replicas):
    """Calculate CPU utilization efficiency."""
    cpu_usage_rate = max(0.1, next_cpu - current_cpu)  # CPU usage in time period
    cpu_per_replica = cpu_usage_rate / max(1, replicas)
    
    # Optimal CPU usage per replica (not too low, not too high)
    if 0.5 <= cpu_per_replica <= 2.0:  # Sweet spot
        return 2.0
    elif cpu_per_replica < 0.5:  # Under-utilized
        return 1.0 - (0.5 - cpu_per_replica)
    else:  # Over-utilized
        return max(0, 2.0 - (cpu_per_replica - 2.0) * 0.5)


def calculate_health_score(current_scrape, next_scrape, memory_mb, http_buckets):
    """Calculate overall system health score."""
    # Scrape sample health (monitoring effectiveness)
    scrape_health = min(3.0, next_scrape / 100.0)  # 300+ samples = full score
    
    # System stress indicators
    memory_stress = max(0, 2.0 - max(0, memory_mb - 500) * 0.002)  # Penalty for >500MB
    traffic_stress = max(0, 2.0 - max(0, http_buckets - 100) * 0.001)  # Penalty for high traffic
    
    return (scrape_health + memory_stress + traffic_stress) / 3.0


def calculate_proactive_bonus(action, lstm_predictions, current_state, next_state):
    """
    FIXED: Reward DQN decisions that align with LSTM predictions.
    Uses LSTM predictions from decision time, not future state.
    """
    if not lstm_predictions:
        return 0.0
    
    # Get LSTM predictions that were available at decision time
    predicted_30sec = lstm_predictions.get('next_30sec_pressure', 0.5)
    predicted_60sec = lstm_predictions.get('next_60sec_pressure', 0.5)
    trend_velocity = lstm_predictions.get('trend_velocity', 0.0)
    
    # Calculate actual load change to validate LSTM predictions
    current_load = current_state.get('http_request_duration_highr_seconds_bucket', 0)
    next_load = next_state.get('http_request_duration_highr_seconds_bucket', 0)
    actual_load_change = (next_load - current_load) / max(1, current_load)  # Percentage change
    
    bonus = 0.0
    
    # Reward proactive scaling up when LSTM predicted load increase AND it actually happened
    if action == "Scale Up":
        if predicted_30sec > 0.6 and actual_load_change > 0.1:  # Predicted high load, actually increased
            bonus = 2.0 * (predicted_30sec - 0.5)  # Stronger bonus for accurate predictions
        elif predicted_30sec > 0.6 and actual_load_change <= 0:  # Predicted load but prevented it
            bonus = 1.5  # Bonus for successful prevention
    
    # Reward proactive scaling down when LSTM predicted load decrease AND it was safe
    elif action == "Scale Down":
        if predicted_30sec < 0.4 and actual_load_change <= 0.1:  # Predicted low load, stayed low
            bonus = 1.5 * (0.5 - predicted_30sec)
    
    # Reward keeping same when LSTM predicted stability
    elif action == "Keep Same":
        if 0.4 <= predicted_30sec <= 0.6 and abs(actual_load_change) < 0.1:  # Predicted stable, stayed stable
            bonus = 1.0
    
    return bonus


def calculate_action_penalty(action, current_replicas):
    """Penalize excessive or inappropriate scaling actions."""
    penalty = 0.0
    
    # Penalize scaling down when already at minimum
    if action == "Scale Down" and current_replicas <= 1:
        penalty = -2.0
    
    # ENHANCED: Progressive penalty for excessive scaling up
    elif action == "Scale Up":
        if current_replicas >= 15:
            penalty = -5.0  # Very strong penalty for scaling beyond 15
        elif current_replicas >= 10:
            penalty = -3.0  # Strong penalty for scaling beyond 10  
        elif current_replicas >= 8:
            penalty = -1.5  # Moderate penalty for scaling beyond 8
        elif current_replicas >= 5:
            penalty = -0.5  # Light penalty for scaling beyond 5
    
    # Small penalty for any scaling action to encourage stability
    elif action != "Keep Same":
        penalty = -0.2
    
    return penalty
    
async def log_experience(state: ScalingState) -> Dict[str, Any]:
    logger.info("NODE_START: log_experience")
    logger.info("=" * 60)
    exp = state['experience']
    
    # Combined approach: immediate training + Redis backup + research logging
    try:
        # 1. Trigger immediate training (primary)
        if dqn_trainer:
            await dqn_trainer.add_experience_for_training(exp)
            logger.info("EXPERIENCE: queued_for_training")
        
        # 2. Also log to Redis as backup (for monitoring/debugging)
        if redis_client:
            experience_json = json.dumps(exp)
            redis_client.lpush(REPLAY_BUFFER_KEY, experience_json)
            redis_client.ltrim(REPLAY_BUFFER_KEY, 0, 99)  # Keep last 100 for monitoring
            logger.info("EXPERIENCE: logged_to_redis monitoring_enabled")
        
        # 3. Add to evaluator for analysis
        if evaluator and ENABLE_EVALUATION_OUTPUTS:
            evaluator.add_experience(exp)
        
        # 4. Update Prometheus experience counter
        DQN_EXPERIENCES_COUNTER.inc()
        
    except Exception as e:
        logger.error(f"EXPERIENCE: logging_failed error={e}")
    
    logger.info("=" * 60)
    logger.info("NODE_END: log_experience")
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
    # Configure Kopf to use a different port to avoid conflict with our metrics server
    settings.networking.health_listening_port = 8081  # Use port 8081 for Kopf health checks
    
    # Use annotations storage to avoid K8s 1.16+ schema pruning issues
    # This prevents "Patching failed with inconsistencies" errors
    settings.persistence.progress_storage = kopf.AnnotationsProgressStorage()
    settings.persistence.diffbase_storage = kopf.AnnotationsDiffBaseStorage()
    
    # Production logging format - structured and regex-extractable
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger.info("STARTUP: NimbusGuard DQN Operator initializing components")
    logger.info("KOPF: using_annotations_storage to_avoid_schema_pruning")
    logger.info(f"IMPROVEMENTS: stabilization_period={STABILIZATION_PERIOD_SECONDS}s improved_rewards={ENABLE_IMPROVED_REWARDS}")
    
    # Log all configurable hyperparameters for research transparency
    logger.info("=== HYPERPARAMETER CONFIGURATION ===")
    logger.info(f"DQN_EXPLORATION: epsilon_start={EPSILON_START} epsilon_end={EPSILON_END} epsilon_decay={EPSILON_DECAY}")
    logger.info(f"DQN_TRAINING: gamma={GAMMA} lr={LR} memory_capacity={MEMORY_CAPACITY} batch_size={BATCH_SIZE}")
    logger.info(f"DQN_ARCHITECTURE: hidden_dims={DQN_HIDDEN_DIMS}")
    logger.info(f"LSTM_CONFIG: hidden_dim={LSTM_HIDDEN_DIM} num_layers={LSTM_NUM_LAYERS} dropout={LSTM_DROPOUT} sequence_length={LSTM_SEQUENCE_LENGTH}")
    logger.info(f"REWARD_WEIGHTS: performance={REWARD_PERFORMANCE_WEIGHT:.0%} resource={REWARD_RESOURCE_WEIGHT:.0%} health={REWARD_HEALTH_WEIGHT:.0%} cost={REWARD_COST_WEIGHT:.0%}")
    logger.info("=== END HYPERPARAMETER CONFIG ===")
    
    # Initialize all global clients
    global prometheus_client, llm, scaler, dqn_model, validator_agent, redis_client, minio_client, metrics_server, dqn_trainer, evaluator, lstm_predictor
    
    prometheus_client = PrometheusClient()
    
    # Only initialize LLM if validation is enabled
    if ENABLE_LLM_VALIDATION:
        try:
            llm = ChatOpenAI(model=AI_MODEL, temperature=AI_TEMPERATURE)
            logger.info(f"LLM: initialized_successfully model={AI_MODEL}")
        except Exception as e:
            logger.error(f"LLM: initialization_failed error={e}")
            logger.warning("LLM: consider_disabling_validation ENABLE_LLM_VALIDATION=false")
            raise kopf.PermanentError(f"LLM initialization failed but validation is enabled: {e}")
    else:
        llm = None
        logger.info("LLM: skipped_initialization validation_disabled")

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
            logger.info(f"MINIO: bucket_created name={BUCKET_NAME}")
        logger.info("MINIO: connection_established")
    except Exception as e:
        logger.error(f"Failed to connect to MinIO: {e}")
        raise kopf.PermanentError(f"MinIO connection failed: {e}")

    # Load scaler from MinIO (upload from local file if not exists)
    try:
        logger.info(f"SCALER: loading from_minio={SCALER_NAME}")
        response = minio_client.get_object(BUCKET_NAME, SCALER_NAME)
        buffer = BytesIO(response.read())
        scaler = joblib.load(buffer)
        logger.info(f"SCALER: loaded_successfully type=RobustScaler features={len(FEATURE_ORDER)}")
    except Exception as e:
        logger.warning(f"SCALER: not_found_in_minio error={e}")
        logger.info("SCALER: attempting_local_upload")
        
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
                logger.info(f"SCALER: uploaded_to_minio name={SCALER_NAME} size={len(scaler_data)}")
                
                # Now load it
                scaler = joblib.load(BytesIO(scaler_data))
                logger.info(f"SCALER: local_upload_loaded type=RobustScaler features={len(FEATURE_ORDER)}")
            else:
                logger.error(f"SCALER: local_file_missing path={local_scaler_path}")
                raise kopf.PermanentError(f"Feature scaler not available locally or in MinIO")
                
        except Exception as upload_error:
            logger.error(f"SCALER: upload_failed error={upload_error}")
            raise kopf.PermanentError(f"Feature scaler loading failed: {upload_error}")

    # Upload training dataset to MinIO if not exists (for research and retraining)
    try:
        # Check if dataset exists in MinIO
        minio_client.stat_object(BUCKET_NAME, "dqn_features.parquet")
        logger.info("DATASET: already_exists_in_minio name=dqn_features.parquet")
    except Exception:
        logger.info("DATASET: uploading_to_minio")
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
                logger.info(f"DATASET: uploaded_successfully name=dqn_features.parquet size={len(dataset_data)}")
            else:
                logger.warning(f"DATASET: local_file_missing path={local_dataset_path}")
        except Exception as dataset_upload_error:
            logger.warning(f"DATASET: upload_failed error={dataset_upload_error}")
            # Not a fatal error - the system can still operate without historical data

    # Load DQN model from MinIO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        logger.info("DQN_MODEL: loading from_minio=dqn_model.pt")
        response = minio_client.get_object(BUCKET_NAME, "dqn_model.pt")
        buffer = BytesIO(response.read())
        checkpoint = torch.load(buffer, map_location=device)
        
        # Initialize model with clean architecture (5 base + 6 LSTM = 11 features)
        state_dim = len(FEATURE_ORDER)  # Now 11 features total
        action_dim = 3
        dqn_model = EnhancedQNetwork(state_dim, action_dim, DQN_HIDDEN_DIMS).to(device)
        logger.info(f"DQN_MODEL: architecture_initialized input_features={state_dim} output_actions={action_dim}")
        
        if isinstance(checkpoint, dict) and 'policy_net_state_dict' in checkpoint:
            dqn_model.load_state_dict(checkpoint['policy_net_state_dict'])
        else:
            dqn_model.load_state_dict(checkpoint)
            
        dqn_model.eval()  # Set to evaluation mode
        logger.info(f"DQN_MODEL: loaded_successfully device={device}")
        
        # Initialize combined trainer for real-time learning
        dqn_trainer = CombinedDQNTrainer(dqn_model, device, FEATURE_ORDER)
        
        # Ensure model is saved to MinIO (in case it was loaded but trainer is new)
        try:
            await dqn_trainer._save_model()
            logger.info("DQN_MODEL: saved_to_minio")
        except Exception as save_error:
            logger.error(f"DQN_MODEL: save_failed error={save_error}")
        
        # Start background training loop
        asyncio.create_task(dqn_trainer.continuous_training_loop())
        logger.info("DQN_TRAINER: background_loop_started")
        
    except Exception as e:
        logger.warning(f"DQN_MODEL: load_failed error={e}")
        logger.info("DQN_MODEL: using_fallback_logic")
        
        # Even without a pre-trained model, we can start with a fresh model
        try:
            state_dim = len(FEATURE_ORDER)  # Clean architecture: 11 features (5 base + 6 LSTM)
            action_dim = 3
            dqn_model = EnhancedQNetwork(state_dim, action_dim, DQN_HIDDEN_DIMS).to(device)
            logger.info(f"DQN_MODEL: fresh_model_created input_features={state_dim} output_actions={action_dim}")
            
            # Initialize trainer with fresh model
            logger.info("DQN_TRAINER: initializing")
            dqn_trainer = CombinedDQNTrainer(dqn_model, device, FEATURE_ORDER)
            logger.info("DQN_TRAINER: initialized_successfully")
            
            # Save fresh model to MinIO immediately (synchronous to catch errors)
            logger.info("DQN_MODEL: saving_fresh_model")
            try:
                await dqn_trainer._save_model()
                logger.info("DQN_MODEL: fresh_model_saved")
            except Exception as save_error:
                logger.error(f"DQN_MODEL: save_failed error={save_error}")
                # Continue anyway - the system can still function without saving
            
            asyncio.create_task(dqn_trainer.continuous_training_loop())
            logger.info("DQN_TRAINER: fresh_model_loop_started")
            
        except Exception as fresh_model_error:
            logger.error(f"DQN_MODEL: fresh_creation_failed error={fresh_model_error}")
            dqn_trainer = None

    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        logger.info("REDIS: connection_established")
    except Exception as e:
        logger.error(f"REDIS: connection_failed error={e}")
        # Not raising a permanent error, as the operator might still function for decision-making
        
    # Initialize MCP validation with LLM supervisor (only if LLM validation is enabled)
    if ENABLE_LLM_VALIDATION and llm is not None:
        if MCP_SERVER_URL:
            try:
                logger.info(f"MCP: connecting url={MCP_SERVER_URL}")
                mcp_client = MultiServerMCPClient({
                    "kubernetes": {
                        "url": f"{MCP_SERVER_URL}/sse/",
                        "transport": "sse"  # Use SSE transport instead of streamable_http
                    }
                })
                tools = await mcp_client.get_tools()
                validator_agent = create_react_agent(llm, tools=tools)
                logger.info(f"MCP: validator_initialized tools_count={len(tools)}")
                logger.info(f"MCP: available_tools=[{','.join(tool.name for tool in tools)}]")
            except Exception as e:
                logger.error(f"MCP: initialization_failed error={e}")
                logger.info("MCP: fallback_to_llm_only")
                validator_agent = create_react_agent(llm, tools=[])
        else:
            logger.info("MCP: url_not_set using_llm_only")
            validator_agent = create_react_agent(llm, tools=[])
    else:
        validator_agent = None
        logger.info("VALIDATOR: skipped_initialization llm_validation_disabled")
    
    # Initialize evaluator
    if ENABLE_EVALUATION_OUTPUTS:
        try:
            evaluator = DQNEvaluator(minio_client, bucket_name="evaluation-outputs")
            logger.info("EVALUATOR: initialized")
        except Exception as e:
            logger.error(f"EVALUATOR: initialization_failed error={e}")
            evaluator = None
    else:
        logger.info("EVALUATOR: disabled")
        evaluator = None

    # Initialize LSTM workload predictor for temporal features
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lstm_predictor = create_lstm_predictor(
            feature_names=BASE_FEATURE_ORDER,  # Use base features for LSTM input
            device=str(device)
        )
        logger.info(f"LSTM: initialized device={device}")
        logger.info(f"LSTM: input_features={len(BASE_FEATURE_ORDER)} output_features={len(LSTM_FEATURE_ORDER)}")
        logger.info("LSTM: proactive_mode sequence_length=2min predictions=30sec+60sec_ahead")
    except Exception as e:
        logger.error(f"LSTM: initialization_failed error={e}")
        logger.info("LSTM: operating_without_temporal_features")
        lstm_predictor = None

    # Initialize gauges to a default value
    CURRENT_REPLICAS_GAUGE.set(1)
    DESIRED_REPLICAS_GAUGE.set(1)
    
    # Initialize DQN metrics
    DQN_TRAINING_LOSS_GAUGE.set(0.0)
    DQN_EPSILON_GAUGE.set(EPSILON_START)
    DQN_BUFFER_SIZE_GAUGE.set(0)
    DQN_TRAINING_STEPS_GAUGE.set(0)
    DQN_DECISION_CONFIDENCE_GAUGE.set(0.0)
    DQN_Q_VALUE_SCALE_UP_GAUGE.set(0.0)
    DQN_Q_VALUE_SCALE_DOWN_GAUGE.set(0.0)
    DQN_Q_VALUE_KEEP_SAME_GAUGE.set(0.0)
    DQN_LSTM_NEXT_30SEC_GAUGE.set(0.0)
    DQN_LSTM_NEXT_60SEC_GAUGE.set(0.0)
    DQN_LSTM_TREND_VELOCITY_GAUGE.set(0.0)
    DQN_LSTM_PATTERN_CONFIDENCE_GAUGE.set(0.0)
    DQN_REWARD_TOTAL_GAUGE.set(0.0)
    DQN_REWARD_PERFORMANCE_GAUGE.set(0.0)
    DQN_REWARD_RESOURCE_GAUGE.set(0.0)
    DQN_REWARD_HEALTH_GAUGE.set(0.0)
    DQN_REWARD_COST_GAUGE.set(0.0)
    
    # Run initial DQN decision to get a baseline
    try:
        logger.info("DECISION: running_initial_baseline")
        await run_intelligent_scaling_decision()
        logger.info("DECISION: initial_completed")
    except Exception as e:
        logger.warning(f"DECISION: initial_failed error={e} will_retry_with_timer")
    
    # Add HTTP/2 error log suppression
    aiohttp_server_logger = logging.getLogger("aiohttp.server")
    aiohttp_server_logger.addFilter(HTTP2ErrorFilter())
    
    # Start HTTP server for essential endpoints only
    app = web.Application()
    app.router.add_get('/metrics', metrics_handler)  # Essential: Prometheus scraping
    
    # Add a root handler
    async def root_handler(request):
        return web.json_response({
            "message": "NimbusGuard DQN Adapter", 
            "status": "running",
            "services": {
                "http": "health+metrics",
                "grpc": "keda_external_scaler"
            }
        })
    
    # Add health check endpoint
    async def health_handler(request):
        health_status = {
            "status": "healthy",
            "service": "dqn-adapter",
            "components": {
                "dqn_trainer": dqn_trainer is not None,
                "evaluator": evaluator is not None,
                "redis": redis_client is not None,
                "prometheus": prometheus_client is not None,
                "lstm_predictor": lstm_predictor is not None
            },
            "metrics": {
                "decisions_total": int(DQN_DECISIONS_COUNTER._value._value),
                "training_steps": int(DQN_TRAINING_STEPS_GAUGE._value._value),
                "buffer_size": int(DQN_BUFFER_SIZE_GAUGE._value._value),
                "current_epsilon": float(DQN_EPSILON_GAUGE._value._value)
            }
        }
        return web.json_response(health_status)
    
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
    app.router.add_get('/healthz', health_handler)
    app.router.add_post('/evaluate', evaluation_trigger_handler)
    
    # Start HTTP server
    metrics_server = web.AppRunner(app)
    await metrics_server.setup()
    site = web.TCPSite(metrics_server, '0.0.0.0', 8080)
    await site.start()
    logger.info("HTTP_SERVER: started port=8080 endpoints=[metrics,health,evaluate]")
    
    # Start gRPC server for KEDA External Scaler (TRULY DYNAMIC!)
    from grpc_scaler import start_grpc_server_async
    grpc_thread = start_grpc_server_async(
        dqn_desired_gauge=DESIRED_REPLICAS_GAUGE,
        current_replicas_gauge=CURRENT_REPLICAS_GAUGE,
        logger=logger,
        port=9091
    )
    logger.info("GRPC_SERVER: started port=9091 for_keda_external_scaler")
        
    logger.info("STARTUP: complete watching_scaledobject_events")

@kopf.on.cleanup()
async def cleanup(**kwargs):
    logger.info("SHUTDOWN: operator_stopping")
    if redis_client:
        try:
            redis_client.close()
        except Exception as e:
            logger.error(f"SHUTDOWN: redis_close_failed error={e}")
    
    if metrics_server:
        try:
            await metrics_server.cleanup()
            logger.info("SHUTDOWN: metrics_server_stopped")
        except Exception as e:
            logger.error(f"SHUTDOWN: metrics_server_error error={e}")

async def run_intelligent_scaling_decision():
    """Execute the intelligent scaling decision using DQN"""
    logger.info("DECISION: starting_intelligent_scaling")
    graph = create_graph()
    try:
        final_state = await graph.ainvoke({}, {"recursion_limit": 15})
        final_replicas = final_state.get('final_decision', 'N/A')
        logger.info(f"DECISION: completed final_replicas={final_replicas}")
    except Exception as e:
        logger.error(f"DECISION: critical_error error={e}", exc_info=True)

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

# --- Explainable AI Helper Functions ---
class DecisionReasoning:
    """Comprehensive decision reasoning and explanation system."""
    
    def __init__(self):
        self.decision_history = []
        self.reasoning_logger = logging.getLogger("AI_Reasoning")
        
        # Set reasoning log level
        if REASONING_LOG_LEVEL == "DEBUG":
            self.reasoning_logger.setLevel(logging.DEBUG)
        else:
            self.reasoning_logger.setLevel(logging.INFO)
    
    def analyze_metrics(self, metrics: Dict[str, float], current_replicas: int) -> Dict[str, Any]:
        """Analyze current metrics and provide detailed insights."""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'current_replicas': current_replicas,
            'raw_metrics': metrics.copy(),
            'insights': {},
            'risk_factors': [],
            'performance_indicators': {}
        }
        
        # Analyze key performance indicators using balanced selected features
        http_bucket = metrics.get('http_request_duration_highr_seconds_bucket', 0)
        memory_bytes = metrics.get('process_resident_memory_bytes', 100000000)
        cpu_usage = metrics.get('process_cpu_seconds_total', 0)
        scrape_samples = metrics.get('scrape_samples_scraped', 300)
        response_size = metrics.get('http_response_size_bytes_sum', 0)
        
        memory_mb = memory_bytes / 1000000  # Convert to MB
        
        # Latency analysis (HTTP request duration buckets = response time performance)
        if http_bucket > 1000:
            analysis['insights']['latency'] = "HIGH LATENCY"
            analysis['risk_factors'].append(f"HTTP latency bucket count {http_bucket:.0f} exceeds 1000 - slow responses")
            analysis['performance_indicators']['latency_severity'] = 'critical'
        elif http_bucket > 500:
            analysis['insights']['latency'] = "ELEVATED LATENCY"
            analysis['risk_factors'].append(f"HTTP latency bucket count {http_bucket:.0f} approaching 1000 - degraded performance")
            analysis['performance_indicators']['latency_severity'] = 'warning'
        else:
            analysis['insights']['latency'] = "LATENCY NORMAL"
            analysis['performance_indicators']['latency_severity'] = 'normal'
        
        # Memory analysis
        if memory_mb > 1000:
            analysis['insights']['memory'] = "HIGH MEMORY USAGE"
            analysis['risk_factors'].append(f"Memory usage {memory_mb:.1f}MB exceeds 1GB")
            analysis['performance_indicators']['memory_severity'] = 'critical'
        elif memory_mb > 500:
            analysis['insights']['memory'] = "ELEVATED MEMORY USAGE"
            analysis['risk_factors'].append(f"Memory usage {memory_mb:.1f}MB approaching 1GB")
            analysis['performance_indicators']['memory_severity'] = 'warning'
        else:
            analysis['insights']['memory'] = "MEMORY USAGE NORMAL"
            analysis['performance_indicators']['memory_severity'] = 'normal'
        
        # Health monitoring analysis
        if scrape_samples < 100:
            analysis['insights']['health'] = "POOR HEALTH MONITORING"
            analysis['risk_factors'].append(f"Scrape samples {scrape_samples:.0f} below 100")
            analysis['performance_indicators']['health_severity'] = 'critical'
        elif scrape_samples < 200:
            analysis['insights']['health'] = "DEGRADED HEALTH MONITORING"
            analysis['risk_factors'].append(f"Scrape samples {scrape_samples:.0f} below 200")
            analysis['performance_indicators']['health_severity'] = 'warning'
        else:
            analysis['insights']['health'] = "HEALTH MONITORING NORMAL"
            analysis['performance_indicators']['health_severity'] = 'normal'
        
        # CPU analysis (unchanged)
        if cpu_usage > 90:
            analysis['insights']['cpu'] = "HIGH CPU USAGE"
            analysis['risk_factors'].append(f"CPU usage {cpu_usage:.1f}% exceeds 90%")
            analysis['performance_indicators']['cpu_severity'] = 'critical'
        elif cpu_usage > 70:
            analysis['insights']['cpu'] = "ELEVATED CPU USAGE"
            analysis['risk_factors'].append(f"CPU usage {cpu_usage:.1f}% approaching 90%")
            analysis['performance_indicators']['cpu_severity'] = 'warning'
        else:
            analysis['insights']['cpu'] = "CPU USAGE NORMAL"
            analysis['performance_indicators']['cpu_severity'] = 'normal'
        
        return analysis
    
    def explain_dqn_decision(self, metrics: Dict[str, float], q_values: List[float], 
                           action_name: str, exploration_type: str, epsilon: float,
                           current_replicas: int) -> Dict[str, Any]:
        """Provide detailed explanation of DQN decision."""
        explanation = {
            'timestamp': datetime.now().isoformat(),
            'decision_type': 'DQN_RECOMMENDATION',
            'recommended_action': action_name,
            'exploration_strategy': exploration_type,
            'confidence_metrics': {},
            'reasoning_factors': [],
            'model_analysis': {},
            'risk_assessment': 'low'
        }
        
        # Q-value analysis
        action_map = {0: "Scale Down", 1: "Keep Same", 2: "Scale Up"}
        q_value_analysis = {}
        for i, (action, q_val) in enumerate(zip(action_map.values(), q_values)):
            q_value_analysis[action] = {
                'q_value': q_val,
                'confidence': 'high' if q_val > 0.7 else 'medium' if q_val > 0.3 else 'low'
            }
        
        explanation['model_analysis']['q_values'] = q_value_analysis
        
        # Confidence calculation
        max_q = max(q_values)
        second_max_q = sorted(q_values, reverse=True)[1] if len(q_values) > 1 else 0
        confidence_gap = max_q - second_max_q
        
        explanation['confidence_metrics'] = {
            'max_q_value': max_q,
            'confidence_gap': confidence_gap,
            'decision_confidence': 'high' if confidence_gap > 0.3 else 'medium' if confidence_gap > 0.1 else 'low',
            'epsilon': epsilon,
            'exploration_probability': epsilon
        }
        
        # Reasoning factors based on metrics
        analysis = self.analyze_metrics(metrics, current_replicas)
        explanation['reasoning_factors'] = [
            f"Current system state: {len(analysis['risk_factors'])} risk factors detected",
            f"Latency status: {analysis['insights'].get('latency', 'unknown')}",
            f"Memory status: {analysis['insights'].get('memory', 'unknown')}",
            f"Decision confidence: {explanation['confidence_metrics']['decision_confidence']}",
            f"Exploration strategy: {exploration_type} (ε={epsilon:.3f})"
        ]
        
        # Risk assessment
        critical_risks = len([r for r in analysis['risk_factors'] if 'critical' in r.lower() or 'exceeds' in r.lower()])
        if critical_risks > 0:
            explanation['risk_assessment'] = 'high'
        elif len(analysis['risk_factors']) > 0:
            explanation['risk_assessment'] = 'medium'
        
        # Add reasoning based on action
        if action_name == 'Scale Up':
            explanation['reasoning_factors'].append("SCALE UP reasoning: System may be under-provisioned, scaling up to meet demand")
        elif action_name == 'Scale Down':
            explanation['reasoning_factors'].append("SCALE DOWN reasoning: System has excess capacity and can operate efficiently with fewer resources")
        else:
            explanation['reasoning_factors'].append("KEEP SAME reasoning: System is operating within acceptable parameters")
        
        explanation['reasoning_factors'] = sorted(explanation['reasoning_factors'])
        return explanation
    
    def log_decision_reasoning(self, explanation: Dict[str, Any], metrics: Dict[str, float], q_values: List[float] = None) -> None:
        """Log detailed decision reasoning in a structured, regex-friendly format."""
        
        self.reasoning_logger.info("AI_REASONING: analysis_start")
        self.reasoning_logger.info(f"AI_REASONING: timestamp={datetime.now().isoformat()}")
        self.reasoning_logger.info(f"AI_REASONING: recommended_action={explanation['recommended_action']}")
        self.reasoning_logger.info(f"AI_REASONING: exploration_strategy={explanation['exploration_strategy']}")
        self.reasoning_logger.info(f"AI_REASONING: risk_assessment={explanation['risk_assessment'].upper()}")
        self.reasoning_logger.info(f"AI_REASONING: decision_confidence={explanation['confidence_metrics']['decision_confidence'].upper()} confidence_gap={explanation['confidence_metrics']['confidence_gap']:.3f}")
        self.reasoning_logger.info(f"AI_REASONING: exploration_rate={explanation['confidence_metrics']['epsilon']:.3f}")

        # Q-Value analysis (if available)
        if q_values and len(q_values) >= 3:
            q_confidences = [self.get_q_value_confidence(q) for q in q_values]
            self.reasoning_logger.info(f"AI_REASONING: q_value_scale_down={q_values[0]:.3f} confidence={q_confidences[0]}")
            self.reasoning_logger.info(f"AI_REASONING: q_value_keep_same={q_values[1]:.3f} confidence={q_confidences[1]}")
            self.reasoning_logger.info(f"AI_REASONING: q_value_scale_up={q_values[2]:.3f} confidence={q_confidences[2]}")
        else:
            self.reasoning_logger.info("AI_REASONING: q_values=unavailable fallback_mode=true")

        # Key reasoning factors
        for i, factor in enumerate(explanation['reasoning_factors']):
            clean_factor = factor.lstrip('•').strip()
            self.reasoning_logger.info(f"AI_REASONING: factor_{i+1}={clean_factor}")

        # Key metrics at time of decision (balanced selected features)
        self.reasoning_logger.info(f"AI_REASONING: raw_feature name=http_bucket value={metrics.get('http_request_duration_highr_seconds_bucket', 'N/A')}")
        self.reasoning_logger.info(f"AI_REASONING: raw_feature name=memory_mb value={metrics.get('process_resident_memory_bytes', 0)/1000000:.1f}")
        self.reasoning_logger.info(f"AI_REASONING: raw_feature name=cpu_usage_percent value={metrics.get('process_cpu_seconds_total', 0):.1f}")
        self.reasoning_logger.info(f"AI_REASONING: raw_feature name=scrape_samples value={metrics.get('scrape_samples_scraped', 'N/A')}")
        self.reasoning_logger.info(f"AI_REASONING: raw_feature name=response_size value={metrics.get('http_response_size_bytes_sum', 'N/A')}")
        
        self.reasoning_logger.info("AI_REASONING: analysis_end")

    def get_q_value_confidence(self, q_value: float) -> str:
        """Get confidence level based on Q-value magnitude."""
        if abs(q_value) > 5: return "high"
        if abs(q_value) > 2: return "medium"
        return "low"

    def create_audit_trail(self, explanation: Dict[str, Any], final_decision: int, llm_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Create a structured audit trail for the decision."""
        import uuid
        
        audit_trail = {
            'decision_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'dqn_explanation': explanation,
            'llm_validation': llm_validation,
            'final_decision': final_decision,
            'explainable': True,
            'auditable': True,
            'reversible': True
        }
        
        # Store in decision history (keep last 100)
        self.decision_history.append(audit_trail)
        if len(self.decision_history) > 100:
            self.decision_history.pop(0)
        
        return audit_trail

# Initialize decision reasoning system
decision_reasoning = DecisionReasoning()

# --- Event-Driven DQN Decision Making (Learning from Timer Approach) ---

# Convert the old timer to proper event-driven handlers targeting the SAME resource!
# OLD: @kopf.timer('keda.sh', 'v1alpha1', 'scaledobjects', labels={'component': 'keda-dqn'}, interval=5)
# NEW: Event-driven reactions to the SAME ScaledObject

@kopf.on.field('keda.sh', 'v1alpha1', 'scaledobjects', 
               field='status.lastActiveTime', 
               labels={'component': 'keda-dqn'})
async def on_keda_evaluation(old, new, meta, spec, status, **kwargs):
    """
    React to KEDA external scaler evaluations (event-driven version of the old timer!)
    
    This targets the SAME resource as the old timer but reacts to actual changes
    instead of polling every 5 seconds. Much more efficient and kubernetes-native.
    """
    logger.info(f"EVENT: keda_evaluation_triggered scaledobject={meta['name']} "
               f"namespace={meta.get('namespace')} last_active_time={new}")
    
    # Only process our specific DQN ScaledObject (same filter as old timer)
    if meta['name'] != 'consumer-scaler-dqn':
        return
    
    # Get current scaling status from KEDA (same logic as old timer)
    current_replicas = status.get('originalReplicaCount', 1)
    is_active = any(condition.get('type') == 'Active' and condition.get('status') == 'True' 
                   for condition in status.get('conditions', []))
    
    logger.info(f"KEDA_EVALUATION: replicas={current_replicas} active={is_active} "
               f"triggered_by_field_change (was_timer_every_5s)")
    
    try:
        # Run DQN decision making process (SAME function as old timer!)
        await run_intelligent_scaling_decision()
        logger.info("EVENT: decision_completed (was_timer_success)")
    except Exception as e:
        logger.error(f"EVENT: decision_failed error={e} (was_timer_error)", exc_info=True)

@kopf.on.update('keda.sh', 'v1alpha1', 'scaledobjects', 
                labels={'component': 'keda-dqn'})
async def on_scaledobject_update(old, new, meta, **kwargs):
    """
    React to any changes in the ScaledObject configuration or status.
    
    This catches broader changes that might not trigger the field watcher.
    """
    # Only process our specific DQN ScaledObject
    if meta['name'] != 'consumer-scaler-dqn':
        return
    
    logger.info(f"EVENT: scaledobject_updated name={meta['name']} "
               f"namespace={meta.get('namespace')} generation={meta.get('generation', 'unknown')}")
    
    # Check if this is a significant status change
    old_replica_count = old.get('status', {}).get('originalReplicaCount', 0)
    new_replica_count = new.get('status', {}).get('originalReplicaCount', 0)
    
    if old_replica_count != new_replica_count:
        logger.info(f"EVENT: scaledobject_replica_change from={old_replica_count} to={new_replica_count}")
        # Update our gauge immediately
        CURRENT_REPLICAS_GAUGE.set(new_replica_count)

@kopf.on.field('apps', 'v1', 'deployments', 
               field='status.replicas')
async def on_replica_count_change(old, new, meta, **kwargs):
    """
    React to actual replica count changes in our target deployment.
    
    This provides immediate feedback when scaling actions complete,
    enabling faster reward calculation and experience generation.
    """
    # Only monitor our target deployment in the nimbusguard namespace
    if meta.get('namespace') != 'nimbusguard' or meta.get('name') != TARGET_DEPLOYMENT:
        return
        
    logger.info(f"EVENT: replica_change_detected deployment={meta['name']} "
               f"namespace={meta.get('namespace')} replicas_changed={old}→{new}")
    
    # Update our current replicas gauge immediately  
    CURRENT_REPLICAS_GAUGE.set(new or 0)
    
    # This could trigger additional learning if the change was significant
    if old and abs((new or 0) - old) >= 2:
        logger.info(f"EVENT: significant_scaling_detected trigger_additional_learning")
        try:
            await run_intelligent_scaling_decision()
        except Exception as e:
            logger.error(f"EVENT: additional_learning_failed error={e}")

@kopf.on.create('keda.sh', 'v1alpha1', 'scaledobjects',
                labels={'component': 'keda-dqn'})
async def on_scaledobject_created(meta, spec, **kwargs):
    """
    React to DQN ScaledObject creation.
    
    Initializes DQN decision-making when our ScaledObject is first created.
    """
    logger.info(f"EVENT: scaledobject_created name={meta['name']} "
               f"namespace={meta.get('namespace')} target={spec.get('scaleTargetRef', {}).get('name', 'unknown')}")
    
    if meta['name'] != 'consumer-scaler-dqn':
        return
    
    # Run initial DQN decision for the new ScaledObject
    try:
        logger.info("EVENT: running_initial_dqn_decision")
        await run_intelligent_scaling_decision()
        logger.info("EVENT: initial_decision_completed")
    except Exception as e:
        logger.warning(f"EVENT: initial_decision_failed error={e}")

# Event-driven architecture: Same resource targeting as timer but reactive instead of polling!