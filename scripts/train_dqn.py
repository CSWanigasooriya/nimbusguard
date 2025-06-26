#!/usr/bin/env python3
"""
Train a Deep Q-Network (DQN) to predict Kubernetes scaling actions.

Dataset: scripts/processed_data/engineered_features.parquet
Action column: 0=Scale Down, 1=Keep Same, 2=Scale Up
Reward: negative weighted sum of normalized cost & performance metrics.
The trained model is saved to models/dqn_model.pt
"""
import os
from pathlib import Path
from collections import deque
import random
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -----------------------------
# Hyper-parameters (can be overridden via CLI)
# -----------------------------
DEFAULT_HPARAMS = dict(
    episodes=250,
    batch_size=64,
    gamma=0.99,
    lr=1e-4,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=0.995,
    target_update=10,
    memory_capacity=5000,
    alpha=0.5,  # weight for cost
    beta=0.5,   # weight for performance penalty
    test_size=0.2,  # 80/20 train/test split
    random_state=42,  # for reproducible splits
)


class ReplayMemory:
    """Circular replay buffer"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(tuple(transition))

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


def compute_reward(row: pd.Series, scalers: dict, alpha: float, beta: float) -> float:
    """Compute reward: negative weighted cost & performance penalty (to minimize)."""
    # Normalize metrics using provided MinMax scalers (0-1)
    cost = scalers["memory_utilization_mean"].transform([[row.memory_utilization_mean]])[0, 0]
    perf_penalty_1 = scalers["avg_response_time"].transform([[row.avg_response_time]])[0, 0]
    perf_penalty_2 = scalers["total_anomaly_score"].transform([[row.total_anomaly_score]])[0, 0]
    reward = -(alpha * cost + beta * (0.5 * perf_penalty_1 + 0.5 * perf_penalty_2))
    return reward


def prepare_data(dataset_path: str, test_size: float = 0.2, random_state: int = 42):
    """Load dataframe, split into features (X) and target (y), fit scaler."""
    df = pd.read_parquet(dataset_path)

    # Identify columns
    action_col = "scaling_action"
    reward_cols = ["memory_utilization_mean", "avg_response_time", "total_anomaly_score"]
    
    # Exclude non-numeric columns (datetime, string, etc.) and target columns
    exclude_cols = [action_col] + reward_cols + ["optimal_pod_count"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = numeric_cols.difference(exclude_cols)

    # Fill any NaNs and convert to float32
    X = df[feature_cols].fillna(0).astype(np.float32).values
    y = df[action_col].astype(np.int64).values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Split dataframe correspondingly for reward computation
    train_indices, test_indices = train_test_split(
        range(len(df)), test_size=test_size, random_state=random_state, stratify=y
    )
    df_train = df.iloc[train_indices].reset_index(drop=True)
    df_test = df.iloc[test_indices].reset_index(drop=True)

    # Standardize features for stability (fit on train, transform both)
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    # Fit MinMax scalers for reward metrics (fit on train only)
    reward_scalers = {}
    for col in reward_cols:
        mm = MinMaxScaler()
        reward_scalers[col] = mm.fit(df_train[col].astype(np.float32).values.reshape(-1, 1))

    return (X_train_scaled, X_test_scaled, y_train, y_test, 
            df_train, df_test, feature_scaler, reward_scalers)


def save_scalers(feature_scaler, save_dir: Path):
    """Persist scaler parameters for inference."""
    import joblib
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_scaler, save_dir / "feature_scaler.gz")


def train_dqn(dataset_path: str, output_path: str, **hparams):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data with train/test split
    (X_train, X_test, y_train, y_test, 
     df_train, df_test, feature_scaler, reward_scalers) = prepare_data(
        dataset_path, hparams["test_size"], hparams["random_state"]
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Feature dimensions: {X_train.shape[1]}")
    
    state_dim = X_train.shape[1]
    action_dim = int(y_train.max() + 1)

    policy_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=hparams["lr"])
    memory = ReplayMemory(hparams["memory_capacity"])

    epsilon = hparams["epsilon_start"]

    # Convert training data to tensors for efficiency
    states = torch.tensor(X_train, dtype=torch.float32, device=device)
    action_tensor = torch.tensor(y_train, dtype=torch.long, device=device)

    # Pre-compute rewards for each training row (offline RL)
    rewards_np = np.array([
        compute_reward(df_train.iloc[i], reward_scalers, hparams["alpha"], hparams["beta"])
        for i in range(len(df_train))
    ], dtype=np.float32)
    rewards = torch.tensor(rewards_np, dtype=torch.float32, device=device)

    print(f"Starting training for {hparams['episodes']} episodes...")
    
    # Training loop (offline, replaying training dataset)
    for episode in range(hparams["episodes"]):
        # Replay training dataset sequentially
        for idx in range(len(states)):
            state = states[idx]
            action = action_tensor[idx]
            reward = rewards[idx]
            next_state = states[idx + 1] if idx + 1 < len(states) else torch.zeros_like(state)
            done = idx + 1 == len(states)
            memory.push(state, action, reward, next_state, done)

            # Learn every step if enough samples
            if len(memory) >= hparams["batch_size"]:
                batch = memory.sample(hparams["batch_size"])
                b_state, b_action, b_reward, b_next_state, b_done = zip(*batch)
                b_state = torch.stack(b_state)
                b_action = torch.stack(b_action)
                b_reward = torch.stack(b_reward)
                b_next_state = torch.stack(b_next_state)
                b_done = torch.tensor(b_done, dtype=torch.bool, device=device)

                # Q(s,a)
                state_action_values = policy_net(b_state).gather(1, b_action.unsqueeze(1)).squeeze()

                # max_a' Q_target(s', a')
                with torch.no_grad():
                    next_state_values = target_net(b_next_state).max(1)[0]
                    next_state_values[b_done] = 0.0
                expected_values = b_reward + hparams["gamma"] * next_state_values

                loss = F.smooth_l1_loss(state_action_values, expected_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Epsilon decay (not used in offline learning but kept for completeness)
        epsilon = max(hparams["epsilon_end"], epsilon * hparams["epsilon_decay"])

        # Update target network
        if (episode + 1) % hparams["target_update"] == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{hparams['episodes']} | Loss: {loss.item():.4f}")

    # Evaluate on test set
    print("\n" + "="*50)
    print("TRAINING COMPLETED - EVALUATING ON TEST SET")
    print("="*50)
    
    policy_net.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        test_q_values = policy_net(X_test_tensor)
        test_predictions = test_q_values.argmax(dim=1).cpu().numpy()
    
    # Calculate test accuracy
    test_accuracy = (test_predictions == y_test).mean()
    print(f"Test Accuracy: {test_accuracy:.3f}")
    
    # Per-class test performance
    from sklearn.metrics import classification_report
    print("\nTest Set Performance:")
    print(classification_report(y_test, test_predictions, 
                              target_names=['Scale Down', 'Keep Same', 'Scale Up']))

    # Save model & scalers
    save_dir = Path(output_path).expanduser().resolve().parent
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy_net.state_dict(), output_path)
    save_scalers(feature_scaler, save_dir)
    
    # Save test set for evaluation script
    test_data_path = save_dir / "test_data.parquet"
    df_test.to_parquet(test_data_path)
    
    print(f"\nModel saved to: {output_path}")
    print(f"Feature scaler saved to: {save_dir / 'feature_scaler.gz'}")
    print(f"Test data saved to: {test_data_path}")
    
    return {
        'test_accuracy': test_accuracy,
        'test_predictions': test_predictions,
        'test_targets': y_test
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN for Kubernetes scaling actions")
    # Default path is relative to this script's directory so it works regardless of current working dir
    default_data_path = Path(__file__).parent / "processed_data" / "engineered_features.parquet"
    parser.add_argument("--data", default=str(default_data_path), help="Path to dataset parquet file")
    parser.add_argument("--output", default="models/dqn_model.pt", help="Path to save the trained model (.pt)")
    # Hyper-parameters overrides
    for k, v in DEFAULT_HPARAMS.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)

    args = parser.parse_args()
    hp = {k: getattr(args, k) for k in DEFAULT_HPARAMS}
    # Resolve data path in case user calls the script from a different cwd
    data_path = Path(args.data)
    if not data_path.exists():
        # Try interpreting as relative to the script directory
        alt_path = Path(__file__).parent / args.data
        if alt_path.exists():
            data_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {args.data} or {alt_path}")

    train_dqn(str(data_path), args.output, **hp) 