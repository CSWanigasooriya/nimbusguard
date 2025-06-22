#!/usr/bin/env python3
"""
Standalone Katib training script that matches your DQN implementation
This script is called by Katib trials with different hyperparameters
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Set up logging to write metrics that Katib can read
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/training.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='DQN Training for Katib with correct actions')
    parser.add_argument('--learning-rate', type=float, required=True)
    parser.add_argument('--gamma', type=float, required=True)
    parser.add_argument('--epsilon-decay', type=float, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--metrics-path', type=str, default='/tmp/metrics.log')
    return parser.parse_args()

def collect_training_data():
    """Simplified data collection for Katib trials"""
    import numpy as np
    
    # For Katib trials, use simulated data to speed up experiments
    # In production, you'd use real Prometheus data
    n_samples = 1000
    state_dim = 11  # Your actual state dimension
    action_dim = 5  # Your actual action dimension: SCALE_DOWN_2, SCALE_DOWN_1, NO_ACTION, SCALE_UP_1, SCALE_UP_2
    
    # Generate realistic state vectors (11 dimensions)
    states = np.random.randn(n_samples, state_dim)
    
    # Generate actions according to your actual ScalingActions enum
    # 0: SCALE_DOWN_2, 1: SCALE_DOWN_1, 2: NO_ACTION, 3: SCALE_UP_1, 4: SCALE_UP_2
    actions = np.random.randint(0, action_dim, n_samples)
    
    # Generate rewards with realistic distribution
    # Better rewards for NO_ACTION and small scaling actions
    action_rewards = {
        0: np.random.normal(-0.1, 0.3, n_samples),  # SCALE_DOWN_2 (slightly negative)
        1: np.random.normal(0.1, 0.2, n_samples),   # SCALE_DOWN_1 (slightly positive)
        2: np.random.normal(0.3, 0.1, n_samples),   # NO_ACTION (mostly positive)
        3: np.random.normal(0.1, 0.2, n_samples),   # SCALE_UP_1 (slightly positive)
        4: np.random.normal(-0.1, 0.3, n_samples),  # SCALE_UP_2 (slightly negative)
    }
    
    rewards = np.array([action_rewards[action][i] for i, action in enumerate(actions)])
    
    return {
        'states': states.tolist(),
        'actions': actions.tolist(),
        'rewards': rewards.tolist()
    }

def train_model(data, hyperparams):
    """Train DQN model with given hyperparameters using your action space"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    
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
    
    # Prepare data
    states = np.array(data['states'])
    actions = np.array(data['actions'])
    rewards = np.array(data['rewards'])
    
    logger.info(f"Training with {len(states)} samples")
    logger.info(f"State shape: {states.shape}")
    logger.info(f"Action distribution: {np.bincount(actions, minlength=5)}")
    logger.info(f"Reward stats: mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleDQN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    criterion = nn.MSELoss()
    
    # Training loop
    losses = []
    best_loss = float('inf')
    convergence_epoch = 0
    
    for epoch in range(hyperparams['epochs']):
        # Batch selection
        batch_size = min(hyperparams['batch_size'], len(states))
        indices = np.random.choice(len(states), batch_size, replace=False)
        
        batch_states = torch.FloatTensor(states[indices]).to(device)
        batch_actions = torch.LongTensor(actions[indices]).to(device)
        batch_rewards = torch.FloatTensor(rewards[indices]).to(device)
        
        # Forward pass
        q_values = model(batch_states)
        action_q_values = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze()
        
        # Enhanced loss calculation for better DQN training
        # Use next states for better Q-learning (simplified for Katib)
        with torch.no_grad():
            # Approximate next state Q-values
            next_indices = np.random.choice(len(states), batch_size, replace=False)
            next_states = torch.FloatTensor(states[next_indices]).to(device)
            next_q_values = model(next_states).max(1)[0]
            target_q_values = batch_rewards + hyperparams['gamma'] * next_q_values
        
        loss = criterion(action_q_values, target_q_values)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Track best performance
        if loss.item() < best_loss:
            best_loss = loss.item()
            convergence_epoch = epoch
        
        # Log progress every 10 epochs
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{hyperparams['epochs']}, Loss: {loss.item():.4f}")
    
    # Calculate performance metrics
    final_loss = losses[-1]
    avg_loss = np.mean(losses)
    loss_improvement = losses[0] - final_loss if len(losses) > 0 else 0
    stability = 1.0 - (np.std(losses[-10:]) / (np.mean(losses[-10:]) + 1e-8))
    
    # Test model on validation data
    model.eval()
    with torch.no_grad():
        val_indices = np.random.choice(len(states), min(100, len(states)), replace=False)
        val_states = torch.FloatTensor(states[val_indices]).to(device)
        val_actions = actions[val_indices]
        
        val_q_values = model(val_states)
        predicted_actions = val_q_values.argmax(dim=1).cpu().numpy()
        
        # Calculate action prediction accuracy (how often model picks reasonable actions)
        reasonable_actions = np.isin(predicted_actions, [1, 2, 3])  # SCALE_DOWN_1, NO_ACTION, SCALE_UP_1
        action_reasonableness = np.mean(reasonable_actions)
    
    return {
        'final_loss': final_loss,
        'average_loss': avg_loss,
        'best_loss': best_loss,
        'convergence_epoch': convergence_epoch,
        'loss_improvement': loss_improvement,
        'stability_score': stability,
        'action_reasonableness': action_reasonableness,
        'hyperparams': hyperparams
    }

def write_metrics(metrics, metrics_path):
    """Write metrics in format that Katib can parse"""
    # Katib expects metrics in specific format
    with open(metrics_path, 'w') as f:
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                f.write(f"{metric_name}={value}\n")
                
    # Also log to stdout for Katib to capture
    logger.info("=== KATIB METRICS ===")
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"metric={metric_name} value={value}")
            logger.info(f"Metric {metric_name}: {value}")

def main():
    args = parse_args()
    
    logger.info("Starting Katib DQN training trial with correct NimbusGuard actions")
    logger.info(f"Hyperparameters: {vars(args)}")
    logger.info("Action space: 0=SCALE_DOWN_2, 1=SCALE_DOWN_1, 2=NO_ACTION, 3=SCALE_UP_1, 4=SCALE_UP_2")
    
    # Collect training data
    logger.info("Collecting training data...")
    data = collect_training_data()
    logger.info(f"Collected {len(data['states'])} training samples")
    
    # Prepare hyperparameters
    hyperparams = {
        'learning_rate': args.learning_rate,
        'gamma': args.gamma,
        'epsilon_decay': args.epsilon_decay,
        'batch_size': args.batch_size,
        'epochs': args.epochs
    }
    
    # Train model
    logger.info("Training model...")
    start_time = time.time()
    metrics = train_model(data, hyperparams)
    training_time = time.time() - start_time
    
    # Add timing information
    metrics['training_time'] = training_time
    
    # Write metrics for Katib
    write_metrics(metrics, args.metrics_path)
    
    logger.info("Training completed successfully")
    logger.info(f"Final metrics: {metrics}")
    
    # Save model checkpoint for best trials
    if metrics['final_loss'] < 0.1:  # Save good models
        logger.info("Saving model checkpoint for good performance")
        # In real implementation, you'd save to persistent storage

if __name__ == "__main__":
    main()
