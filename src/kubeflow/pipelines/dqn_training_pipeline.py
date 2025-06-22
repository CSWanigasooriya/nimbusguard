"""
NimbusGuard DQN Training Pipeline for Kubeflow
This pipeline automates the training and deployment of DQN models for autoscaling decisions.
"""

import kfp
from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics
from typing import Dict, Any

import pandas as pd


@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "torch==2.0.0", 
        "numpy==1.24.0", 
        "requests==2.31.0", 
        "prometheus-client==0.19.0",
        "pandas==2.0.0",
        "scikit-learn==1.3.0"
    ]
)
def collect_training_data(
    prometheus_endpoint: str,
    collection_hours: int,
    output_dataset: Output[Dataset]
) -> str:
    """Collect training data from Prometheus metrics for DQN training"""
    import json
    import requests
    import numpy as np
    from datetime import datetime, timedelta
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Collecting {collection_hours} hours of data from {prometheus_endpoint}")
    
    # Define metrics to collect for DQN state representation
    metrics_queries = {
        'cpu_utilization': 'avg(rate(container_cpu_usage_seconds_total[5m])) by (pod)',
        'memory_utilization': 'avg(container_memory_working_set_bytes) by (pod)',
        'network_io': 'rate(container_network_transmit_bytes_total[5m])',
        'request_rate': 'rate(http_requests_total[5m])',
        'pod_count': 'count(kube_pod_info{namespace="nimbusguard"})',
        'scaling_events': 'increase(nimbusguard_scaling_events_total[1h])',
        'response_time': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))'
    }
    
    # Time range for data collection
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=collection_hours)
    
    training_samples = []
    
    try:
        for metric_name, query in metrics_queries.items():
            logger.info(f"Collecting metric: {metric_name}")
            
            params = {
                'query': query,
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'step': '30s'
            }
            
            response = requests.get(f"{prometheus_endpoint}/api/v1/query_range", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] == 'success':
                results = data['data']['result']
                for result in results:
                    values = result['values']
                    for timestamp, value in values:
                        try:
                            sample = {
                                'timestamp': timestamp,
                                'metric': metric_name,
                                'value': float(value),
                                'labels': result.get('metric', {})
                            }
                            training_samples.append(sample)
                        except (ValueError, TypeError):
                            continue
    
    except Exception as e:
        logger.error(f"Error collecting metrics: {e}")
        # Create minimal dataset if collection fails
        training_samples = [{'timestamp': datetime.now().timestamp(), 'metric': 'dummy', 'value': 0.0}]
    
    # Process and structure data for DQN training
    processed_data = process_metrics_for_dqn(training_samples)
    
    # Save to output
    with open(output_dataset.path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    logger.info(f"Collected {len(processed_data['states'])} training samples")
    return f"Collected {len(processed_data['states'])} samples"


def process_metrics_for_dqn(raw_samples):
    """Process raw metrics into DQN training format"""
    import pandas as pd
    import numpy as np
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(raw_samples)
    
    # Group by timestamp to create state vectors
    states = []
    actions = []
    rewards = []
    
    # This is a simplified version - you'd implement your actual state representation logic
    timestamps = df['timestamp'].unique()
    
    for i, ts in enumerate(timestamps[:-1]):  # Exclude last timestamp for next_state
        # Get metrics for this timestamp
        ts_data = df[df['timestamp'] == ts]
        
        # Create 11-dimensional state vector (matching your current DQN)
        state_vector = [
            ts_data[ts_data['metric'] == 'cpu_utilization']['value'].mean() or 0.0,
            ts_data[ts_data['metric'] == 'memory_utilization']['value'].mean() or 0.0,
            ts_data[ts_data['metric'] == 'network_io']['value'].mean() or 0.0,
            ts_data[ts_data['metric'] == 'request_rate']['value'].mean() or 0.0,
            ts_data[ts_data['metric'] == 'pod_count']['value'].mean() or 1.0,
            ts_data[ts_data['metric'] == 'response_time']['value'].mean() or 0.0,
            # Add more features as needed to reach 11 dimensions
            0.0, 0.0, 0.0, 0.0, 0.0  # Placeholder features
        ]
        
        # Simulate action (in real implementation, get from historical decisions)
        action = np.random.randint(0, 5)  # 5 possible actions
        
        # Simulate reward (in real implementation, calculate based on performance)
        reward = np.random.uniform(-1, 1)
        
        states.append(state_vector)
        actions.append(action)
        rewards.append(reward)
    
    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'metadata': {
            'collection_timestamp': pd.Timestamp.now().isoformat(),
            'sample_count': len(states)
        }
    }


@component(
    base_image="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",
    packages_to_install=[
        "numpy==1.24.0",
        "scikit-learn==1.3.0",
        "matplotlib==3.7.0"
    ]
)
def train_dqn_model(
    input_dataset: Input[Dataset],
    learning_rate: float,
    gamma: float,
    epsilon_decay: float,
    batch_size: int,
    epochs: int,
    trained_model: Output[Model],
    training_metrics: Output[Metrics]
) -> Dict[str, float]:
    """Train DQN model with hyperparameters from Katib"""
    import json
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from pathlib import Path
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load training data
    with open(input_dataset.path, 'r') as f:
        data = json.load(f)
    
    states = np.array(data['states'])
    actions = np.array(data['actions'])
    rewards = np.array(data['rewards'])
    
    logger.info(f"Training with {len(states)} samples")
    logger.info(f"Hyperparameters: lr={learning_rate}, gamma={gamma}, decay={epsilon_decay}, batch={batch_size}")
    
    # Define DQN model (simplified version of your DoubleDQNModel)
    class SimpleDQN(nn.Module):
        def __init__(self, state_dim=11, action_dim=5, hidden_dim=128):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        
        def forward(self, x):
            return self.network(x)
    
    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleDQN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        # Simple batch training
        indices = np.random.choice(len(states), min(batch_size, len(states)), replace=False)
        batch_states = torch.FloatTensor(states[indices]).to(device)
        batch_actions = torch.LongTensor(actions[indices]).to(device)
        batch_rewards = torch.FloatTensor(rewards[indices]).to(device)
        
        # Forward pass
        q_values = model(batch_states)
        action_q_values = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze()
        
        # Simple loss (in real implementation, use your DQN logic)
        target_q_values = batch_rewards  # Simplified target
        loss = criterion(action_q_values, target_q_values)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
    
    # Save model
    model_path = Path(trained_model.path)
    model_path.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': {
            'learning_rate': learning_rate,
            'gamma': gamma,
            'epsilon_decay': epsilon_decay,
            'batch_size': batch_size
        },
        'training_metadata': {
            'epochs': epochs,
            'final_loss': losses[-1],
            'samples_trained': len(states)
        }
    }, model_path / 'dqn_model.pth')
    
    # Calculate metrics
    avg_loss = np.mean(losses)
    final_loss = losses[-1]
    
    # Save metrics
    metrics = {
        'average_loss': avg_loss,
        'final_loss': final_loss,
        'epochs_trained': epochs,
        'samples_used': len(states)
    }
    
    with open(training_metrics.path, 'w') as f:
        json.dump(metrics, f)
    
    logger.info(f"Training completed. Average loss: {avg_loss:.4f}")
    
    return metrics


@component(
    base_image="python:3.11-slim",
    packages_to_install=["torch==2.0.0", "numpy==1.24.0", "pandas==2.0.0"]
)
def validate_model(
    trained_model: Input[Model],
    validation_threshold: float,
    model_validation: Output[Metrics]
) -> Dict[str, Any]:
    """Validate the trained model and decide if it should be deployed"""
    import json
    import torch
    import numpy as np
    from pathlib import Path
    import logging
    import pandas as pd
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load model
    model_path = Path(trained_model.path) / 'dqn_model.pth'
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Simple validation metrics
    final_loss = checkpoint['training_metadata']['final_loss']
    epochs_trained = checkpoint['training_metadata']['epochs_trained']
    
    # Validation logic
    is_valid = final_loss < validation_threshold
    
    validation_results = {
        'is_valid': is_valid,
        'final_loss': final_loss,
        'validation_threshold': validation_threshold,
        'epochs_trained': epochs_trained,
        'deployment_ready': is_valid,
        'validation_timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(model_validation.path, 'w') as f:
        json.dump(validation_results, f)
    
    logger.info(f"Model validation: {'PASSED' if is_valid else 'FAILED'} (loss: {final_loss:.4f})")
    
    return validation_results


@pipeline(
    name="nimbusguard-dqn-training-pipeline",
    description="Complete DQN training pipeline for NimbusGuard autoscaling"
)
def dqn_training_pipeline(
    prometheus_endpoint: str = "http://prometheus.nimbusguard.svc.cluster.local:9090",
    collection_hours: int = 24,
    learning_rate: float = 1e-4,
    gamma: float = 0.95,
    epsilon_decay: float = 0.995,
    batch_size: int = 32,
    epochs: int = 100,
    validation_threshold: float = 0.5
):
    """
    Complete pipeline for training DQN models
    
    Args:
        prometheus_endpoint: Prometheus server URL for data collection
        collection_hours: Hours of historical data to collect
        learning_rate: Learning rate for DQN training
        gamma: Discount factor for Q-learning
        epsilon_decay: Epsilon decay rate
        batch_size: Training batch size
        epochs: Number of training epochs
        validation_threshold: Maximum loss threshold for model validation
    """
    
    # Step 1: Collect training data
    collect_task = collect_training_data(
        prometheus_endpoint=prometheus_endpoint,
        collection_hours=collection_hours
    )
    
    # Step 2: Train DQN model
    train_task = train_dqn_model(
        input_dataset=collect_task.outputs['output_dataset'],
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        epochs=epochs
    )
    
    # Step 3: Validate model
    validate_task = validate_model(
        trained_model=train_task.outputs['trained_model'],
        validation_threshold=validation_threshold
    )
    
    # Set resource requirements
    collect_task.set_cpu_request("100m").set_memory_request("512Mi")
    train_task.set_cpu_request("1").set_memory_request("2Gi")
    validate_task.set_cpu_request("100m").set_memory_request("256Mi")
    
    return {
        'training_metrics': train_task.outputs['training_metrics'],
        'validation_results': validate_task.outputs['model_validation']
    }


if __name__ == "__main__":
    # Compile pipeline
    kfp.compiler.Compiler().compile(
        pipeline_func=dqn_training_pipeline,
        package_path="nimbusguard_dqn_pipeline.yaml"
    )
    print("Pipeline compiled successfully!")
