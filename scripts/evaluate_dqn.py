#!/usr/bin/env python3
"""
Evaluate a trained DQN model for Kubernetes scaling actions.

This script computes multiple evaluation metrics:
1. Classification metrics (accuracy, precision, recall, F1)
2. Business/operational metrics (cost efficiency, performance impact)
3. Policy-specific metrics (action distribution, reward analysis)
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Import the QNetwork from train_dqn
import sys
sys.path.append(str(Path(__file__).parent))
from train_dqn import QNetwork, compute_reward, prepare_data


def load_model_and_scalers(model_path: str, scaler_path: str):
    """Load trained model and feature scaler."""
    # Load feature scaler
    feature_scaler = joblib.load(scaler_path)
    
    # Load model (need to know dimensions first)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We'll determine state_dim from the scaler
    state_dim = feature_scaler.n_features_in_
    action_dim = 3  # Scale Down, Keep Same, Scale Up
    
    model = QNetwork(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    return model, feature_scaler, device


def evaluate_predictions(model, X, y_true, device):
    """Evaluate model predictions vs ground truth."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        q_values = model(X_tensor)
        y_pred = q_values.argmax(dim=1).cpu().numpy()
    
    # Classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class metrics
    class_report = classification_report(y_true, y_pred, 
                                       target_names=['Scale Down', 'Keep Same', 'Scale Up'],
                                       output_dict=True)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'predictions': y_pred,
        'q_values': q_values.cpu().numpy()
    }


def evaluate_business_metrics(df, y_true, y_pred, reward_scalers, alpha=0.5, beta=0.5):
    """Evaluate business/operational impact of predictions."""
    
    # Simulate rewards for true vs predicted actions
    true_rewards = []
    pred_rewards = []
    
    for i, row in df.iterrows():
        # Compute reward for actual action taken
        true_reward = compute_reward(row, reward_scalers, alpha, beta)
        true_rewards.append(true_reward)
        
        # Simulate reward for predicted action (approximation)
        # This is a simplified approach - in reality, we'd need to simulate the environment
        pred_reward = true_reward  # Placeholder - same as true for now
        pred_rewards.append(pred_reward)
    
    true_rewards = np.array(true_rewards)
    pred_rewards = np.array(pred_rewards)
    
    # Action distribution analysis
    action_names = ['Scale Down', 'Keep Same', 'Scale Up']
    true_dist = pd.Series(y_true).value_counts().sort_index()
    pred_dist = pd.Series(y_pred).value_counts().sort_index()
    
    # Cost efficiency metrics
    memory_util = df['memory_utilization_mean'].values
    response_time = df['avg_response_time'].values
    anomaly_score = df['total_anomaly_score'].values
    
    results = {
        'reward_analysis': {
            'true_rewards_mean': true_rewards.mean(),
            'true_rewards_std': true_rewards.std(),
            'pred_rewards_mean': pred_rewards.mean(),
            'pred_rewards_std': pred_rewards.std(),
        },
        'action_distribution': {
            'true': dict(zip(action_names, true_dist.values)),
            'predicted': dict(zip(action_names, pred_dist.values))
        },
        'operational_metrics': {
            'avg_memory_utilization': memory_util.mean(),
            'avg_response_time': response_time.mean(),
            'avg_anomaly_score': anomaly_score.mean(),
            'memory_util_std': memory_util.std(),
            'response_time_std': response_time.std(),
            'anomaly_score_std': anomaly_score.std(),
        }
    }
    
    return results


def evaluate_policy_quality(q_values, y_true, y_pred):
    """Evaluate the quality of the learned policy."""
    
    # Q-value confidence (higher max Q-value = more confident)
    max_q_values = np.max(q_values, axis=1)
    confidence_scores = max_q_values - np.mean(q_values, axis=1)
    
    # Value function analysis
    q_value_stats = {
        'mean_max_q': max_q_values.mean(),
        'std_max_q': max_q_values.std(),
        'mean_confidence': confidence_scores.mean(),
        'std_confidence': confidence_scores.std(),
    }
    
    # Action value analysis per class
    action_q_analysis = {}
    for action in range(3):
        mask = y_true == action
        if mask.sum() > 0:
            action_q_analysis[action] = {
                'mean_q_value': q_values[mask, action].mean(),
                'std_q_value': q_values[mask, action].std(),
                'correct_predictions': (y_pred[mask] == action).sum(),
                'total_samples': mask.sum()
            }
    
    return {
        'q_value_stats': q_value_stats,
        'action_q_analysis': action_q_analysis,
        'confidence_scores': confidence_scores
    }


def plot_evaluation_results(eval_results, business_results, policy_results, save_dir: Path):
    """Generate evaluation plots."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(eval_results['confusion_matrix'], 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Scale Down', 'Keep Same', 'Scale Up'],
                yticklabels=['Scale Down', 'Keep Same', 'Scale Up'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Action Distribution Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    true_dist = business_results['action_distribution']['true']
    pred_dist = business_results['action_distribution']['predicted']
    
    actions = list(true_dist.keys())
    true_counts = list(true_dist.values())
    pred_counts = list(pred_dist.values())
    
    x = np.arange(len(actions))
    width = 0.35
    
    ax1.bar(x - width/2, true_counts, width, label='True', alpha=0.8)
    ax1.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
    ax1.set_xlabel('Actions')
    ax1.set_ylabel('Count')
    ax1.set_title('Action Distribution Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(actions)
    ax1.legend()
    
    # 3. Q-value Distribution
    q_stats = policy_results['q_value_stats']
    ax2.hist(policy_results['confidence_scores'], bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Q-value Confidence Distribution')
    ax2.axvline(q_stats['mean_confidence'], color='red', linestyle='--', 
                label=f'Mean: {q_stats["mean_confidence"]:.3f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'action_and_confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def print_evaluation_summary(eval_results, business_results, policy_results):
    """Print comprehensive evaluation summary."""
    print("=" * 60)
    print("DQN MODEL EVALUATION SUMMARY")
    print("=" * 60)
    
    # Classification Performance
    print("\n📊 CLASSIFICATION PERFORMANCE:")
    print(f"  Accuracy:  {eval_results['accuracy']:.3f}")
    print(f"  Precision: {eval_results['precision']:.3f}")
    print(f"  Recall:    {eval_results['recall']:.3f}")
    print(f"  F1-Score:  {eval_results['f1']:.3f}")
    
    # Per-class performance
    print("\n📈 PER-CLASS PERFORMANCE:")
    class_report = eval_results['classification_report']
    for action_name in ['Scale Down', 'Keep Same', 'Scale Up']:
        if action_name in class_report:
            metrics = class_report[action_name]
            print(f"  {action_name:12s}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    # Business Metrics
    print("\n💰 BUSINESS METRICS:")
    reward_analysis = business_results['reward_analysis']
    print(f"  Avg Reward:        {reward_analysis['true_rewards_mean']:.3f} ± {reward_analysis['true_rewards_std']:.3f}")
    
    operational = business_results['operational_metrics']
    print(f"  Memory Utilization: {operational['avg_memory_utilization']:.3f} ± {operational['memory_util_std']:.3f}")
    print(f"  Response Time:     {operational['avg_response_time']:.3f} ± {operational['response_time_std']:.3f}")
    print(f"  Anomaly Score:     {operational['avg_anomaly_score']:.3f} ± {operational['anomaly_score_std']:.3f}")
    
    # Policy Quality
    print("\n🎯 POLICY QUALITY:")
    q_stats = policy_results['q_value_stats']
    print(f"  Mean Max Q-value:  {q_stats['mean_max_q']:.3f} ± {q_stats['std_max_q']:.3f}")
    print(f"  Mean Confidence:   {q_stats['mean_confidence']:.3f} ± {q_stats['std_confidence']:.3f}")
    
    # Action Distribution
    print("\n📋 ACTION DISTRIBUTION:")
    true_dist = business_results['action_distribution']['true']
    pred_dist = business_results['action_distribution']['predicted']
    for action in ['Scale Down', 'Keep Same', 'Scale Up']:
        true_count = true_dist.get(action, 0)
        pred_count = pred_dist.get(action, 0)
        print(f"  {action:12s}: True={true_count:3d}, Pred={pred_count:3d}, Diff={pred_count-true_count:+3d}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DQN model")
    parser.add_argument("--model", default="models/dqn_model.pt", help="Path to trained model")
    parser.add_argument("--scaler", default="models/feature_scaler.gz", help="Path to feature scaler")
    parser.add_argument("--data", default="processed_data/engineered_features.parquet", help="Path to dataset")
    parser.add_argument("--output", default="evaluation_results", help="Output directory for results")
    parser.add_argument("--plots", action="store_true", help="Generate evaluation plots")
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    model_path = script_dir / args.model
    scaler_path = script_dir / args.scaler
    data_path = script_dir / args.data
    output_dir = script_dir / args.output
    
    print(f"Loading model from: {model_path}")
    print(f"Loading scaler from: {scaler_path}")
    print(f"Loading data from: {data_path}")
    
    # Load model and data
    model, feature_scaler, device = load_model_and_scalers(str(model_path), str(scaler_path))
    
    # Use the test data that was saved during training
    test_data_path = model_path.parent / "test_data.parquet"
    if test_data_path.exists():
        print(f"Using saved test data: {test_data_path}")
        df = pd.read_parquet(test_data_path)
        
        # Prepare features from test data
        action_col = "scaling_action"
        reward_cols = ["memory_utilization_mean", "avg_response_time", "total_anomaly_score"]
        exclude_cols = [action_col] + reward_cols + ["optimal_pod_count"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = numeric_cols.difference(exclude_cols)
        
        X = df[feature_cols].fillna(0).astype(np.float32).values
        y_true = df[action_col].astype(np.int64).values
        
        # Transform features using the saved scaler
        X = feature_scaler.transform(X)
        
        # Create reward scalers (fit on test data for evaluation purposes)
        reward_scalers = {}
        for col in reward_cols:
            mm = MinMaxScaler()
            reward_scalers[col] = mm.fit(df[col].astype(np.float32).values.reshape(-1, 1))
    else:
        print(f"Test data not found, using full dataset for evaluation")
        # Fall back to using the full dataset
        (X_train, X_test, y_train, y_test, 
         df_train, df_test, _, reward_scalers) = prepare_data(str(data_path))
        X, y_true, df = X_test, y_test, df_test
    
    print(f"Dataset shape: {X.shape}")
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Run evaluations
    print("\nRunning evaluation...")
    eval_results = evaluate_predictions(model, X, y_true, device)
    business_results = evaluate_business_metrics(df, y_true, eval_results['predictions'], reward_scalers)
    policy_results = evaluate_policy_quality(eval_results['q_values'], y_true, eval_results['predictions'])
    
    # Print summary
    print_evaluation_summary(eval_results, business_results, policy_results)
    
    # Generate plots if requested
    if args.plots:
        print(f"\nGenerating plots in: {output_dir}")
        plot_evaluation_results(eval_results, business_results, policy_results, output_dir)
        print("Plots saved successfully!")
    
    # Save detailed results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_summary = {
        'classification': eval_results,
        'business': business_results,
        'policy': policy_results
    }
    
    # Save as JSON for programmatic access
    import json
    with open(output_dir / 'evaluation_summary.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        json.dump(convert_numpy(results_summary), f, indent=2)
    
    print(f"\nDetailed results saved to: {output_dir / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()