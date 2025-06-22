#!/usr/bin/env python3
"""
Continuous Training Script for NimbusGuard DQN
Collects real metrics from Prometheus and trains the model
"""

import os
import sys
import time
import json
import logging
import requests
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def collect_prometheus_metrics():
    """Collect training data from Prometheus"""
    prometheus_url = os.getenv('PROMETHEUS_ENDPOINT', 'http://prometheus.monitoring.svc.cluster.local:9090')
    collection_hours = int(os.getenv('COLLECTION_HOURS', '4'))
    min_samples = int(os.getenv('MIN_SAMPLES', '100'))
    
    logger.info(f"Collecting metrics from {prometheus_url} for last {collection_hours} hours")
    
    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=collection_hours)
    
    # Convert to Unix timestamps
    start_ts = int(start_time.timestamp())
    end_ts = int(end_time.timestamp())
    
    # Queries for DQN state features
    queries = {
        'cpu_usage': 'avg(rate(container_cpu_usage_seconds_total[5m])) by (namespace)',
        'memory_usage': 'avg(container_memory_working_set_bytes / container_spec_memory_limit_bytes) by (namespace)',
        'pod_count': 'count(kube_pod_info{namespace="nimbusguard"}) by (namespace)',
        'request_rate': 'avg(rate(nimbusguard_http_requests_total[5m]))',
        'error_rate': 'avg(rate(nimbusguard_http_requests_total{status=~"5.."}[5m]))',
        'response_time': 'avg(nimbusguard_http_request_duration_seconds)',
        'kserve_latency': 'avg(nimbusguard_kserve_latency_seconds)',
        'scaling_decisions': 'increase(nimbusguard_dqn_decisions_total[5m])',
    }
    
    training_data = []
    
    try:
        # Collect data points every 5 minutes over the time range
        step = 300  # 5 minutes
        for ts in range(start_ts, end_ts, step):
            state_vector = []
            
            # Collect each metric
            for metric_name, query in queries.items():
                try:
                    response = requests.get(
                        f"{prometheus_url}/api/v1/query",
                        params={'query': query, 'time': ts},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data['data']['result']:
                            # Take first result or average if multiple
                            values = [float(r['value'][1]) for r in data['data']['result']]
                            value = np.mean(values) if values else 0.0
                        else:
                            value = 0.0
                    else:
                        logger.warning(f"Failed to query {metric_name}: {response.status_code}")
                        value = 0.0
                        
                    state_vector.append(value)
                    
                except Exception as e:
                    logger.warning(f"Error querying {metric_name}: {e}")
                    state_vector.append(0.0)
            
            # Pad or truncate to 11 dimensions
            while len(state_vector) < 11:
                state_vector.append(0.0)
            state_vector = state_vector[:11]
            
            # Generate synthetic action and reward for now
            # In a real implementation, you'd get this from actual scaling events
            action = np.random.randint(0, 5)  # 0-4 for the 5 scaling actions
            reward = np.random.normal(0.1, 0.2)  # Placeholder reward
            
            training_data.append({
                'timestamp': ts,
                'state': state_vector,
                'action': action,
                'reward': reward
            })
    
    except Exception as e:
        logger.error(f"Error collecting metrics: {e}")
        return None
    
    logger.info(f"Collected {len(training_data)} data points")
    
    if len(training_data) < min_samples:
        logger.warning(f"Not enough samples ({len(training_data)} < {min_samples}). Using synthetic data.")
        return generate_synthetic_data(min_samples)
    
    return training_data

def generate_synthetic_data(n_samples=100):
    """Generate synthetic training data for testing"""
    logger.info(f"Generating {n_samples} synthetic training samples")
    
    training_data = []
    for i in range(n_samples):
        # Generate realistic state vector
        state = [
            np.random.uniform(0.1, 0.9),  # CPU usage
            np.random.uniform(0.2, 0.8),  # Memory usage  
            np.random.uniform(1, 10),     # Pod count
            np.random.uniform(0, 100),    # Request rate
            np.random.uniform(0, 0.1),    # Error rate
            np.random.uniform(0.01, 0.5), # Response time
            np.random.uniform(0.01, 0.1), # KServe latency
            np.random.uniform(0, 5),      # Scaling decisions
            np.random.uniform(0, 1),      # Additional metrics
            np.random.uniform(0, 1),
            np.random.uniform(0, 1)
        ]
        
        # Generate action based on state (more realistic)
        cpu_usage = state[0]
        if cpu_usage > 0.8:
            action = 3 or 4  # Scale up
        elif cpu_usage < 0.3:
            action = 0 or 1  # Scale down
        else:
            action = 2  # No action
            
        # Generate reward based on action appropriateness
        if action == 2:  # NO_ACTION
            reward = np.random.normal(0.3, 0.1)
        elif action in [1, 3]:  # Small scaling
            reward = np.random.normal(0.1, 0.2)
        else:  # Large scaling
            reward = np.random.normal(-0.1, 0.3)
        
        training_data.append({
            'timestamp': int(time.time()) - (n_samples - i) * 300,
            'state': state,
            'action': action,
            'reward': reward
        })
    
    return training_data

def train_dqn_model(training_data):
    """Train DQN model with collected data"""
    logger.info("Starting DQN training...")
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        logger.error("PyTorch not available. Cannot train model.")
        return False
    
    # Prepare training data
    states = np.array([d['state'] for d in training_data])
    actions = np.array([d['action'] for d in training_data])
    rewards = np.array([d['reward'] for d in training_data])
    
    logger.info(f"Training data shape: states={states.shape}, actions={actions.shape}, rewards={rewards.shape}")
    
    # DQN model matching the transformer architecture
    class DQN(nn.Module):
        def __init__(self, state_dim=11, action_dim=5, hidden_dim=128):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, action_dim)
            )
        
        def forward(self, x):
            return self.network(x)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DQN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training parameters
    epochs = 100
    batch_size = min(32, len(training_data))
    
    logger.info(f"Training on {device} for {epochs} epochs with batch size {batch_size}")
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        # Sample batch
        indices = np.random.choice(len(states), batch_size, replace=False)
        batch_states = torch.FloatTensor(states[indices]).to(device)
        batch_actions = torch.LongTensor(actions[indices]).to(device)
        batch_rewards = torch.FloatTensor(rewards[indices]).to(device)
        
        # Forward pass
        q_values = model(batch_states)
        action_q_values = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze()
        
        # Simple Q-learning update
        target_q_values = batch_rewards
        loss = criterion(action_q_values, target_q_values)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
    
    # Save model in KServe expected format
    model_path = os.getenv('MODEL_PATH', '/models')
    model_name = os.getenv('KSERVE_MODEL_NAME', 'nimbusguard-dqn')
    
    # Create directory structure expected by KServe
    model_dir = f"{model_path}/{model_name}"
    latest_dir = f"{model_dir}/latest"
    os.makedirs(latest_dir, exist_ok=True)
    
    # Also save timestamped version for history
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamped_dir = f"{model_dir}/{timestamp}"
    os.makedirs(timestamped_dir, exist_ok=True)
    
    # Prepare model checkpoint with metadata
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'state_dim': 11,
        'action_dim': 5,
        'metadata': {
            'model_version': int(timestamp),
            'training_timestamp': timestamp,
            'total_samples': len(training_data),
            'final_loss': loss.item(),
            'model_type': 'DQN'
        }
    }
    
    # Save timestamped version
    timestamped_file = f"{timestamped_dir}/model.pth"
    torch.save(checkpoint, timestamped_file)
    
    # Save latest version (what KServe will load)
    latest_file = f"{latest_dir}/model.pth"
    torch.save(checkpoint, latest_file)
    
    # Also keep the old format for backward compatibility
    old_format_file = f"{model_path}/dqn_model_{timestamp}.pt"
    torch.save(checkpoint, old_format_file)
    
    # Create/update symlink to latest model (old format)
    latest_symlink = f"{model_path}/dqn_model_latest.pt"
    if os.path.exists(latest_symlink):
        os.remove(latest_symlink)
    os.symlink(os.path.basename(old_format_file), latest_symlink)
    
    logger.info(f"Model saved to {latest_file}")
    logger.info(f"Latest model symlink: {latest_symlink}")
    
    return True

def main():
    """Main training function"""
    logger.info("🤖 Starting NimbusGuard Continuous Training")
    logger.info("=" * 50)
    
    # Collect training data
    training_data = collect_prometheus_metrics()
    if not training_data:
        logger.error("Failed to collect training data")
        sys.exit(1)
    
    # Train model
    if train_dqn_model(training_data):
        logger.info("✅ Training completed successfully!")
    else:
        logger.error("❌ Training failed!")
        sys.exit(1)
    
    logger.info("🎉 Continuous training finished!")

if __name__ == "__main__":
    main() 