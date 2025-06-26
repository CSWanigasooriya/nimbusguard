#!/usr/bin/env python3
"""
Advanced DQN Model Evaluation with Publication-Quality Visualizations

This script generates comprehensive plots for research paper publication:
1. Training dynamics and convergence analysis
2. Feature engineering effectiveness
3. Model performance comparisons
4. Business impact visualization
5. Statistical analysis and distributions
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Import from existing scripts
import sys
sys.path.append(str(Path(__file__).parent))
from train_dqn import QNetwork, compute_reward, prepare_data


class AdvancedDQNEvaluator:
    def __init__(self, model_path: str, scaler_path: str, data_path: str, output_dir: str = "advanced_evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("🔬 Loading model and data...")
        self.model, self.feature_scaler, self.device = self.load_model_and_scalers(model_path, scaler_path)
        self.load_data(data_path)
        
        # Generate predictions
        self.predictions, self.q_values = self.get_predictions()
        
        print(f"✅ Loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"📊 Model: {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def load_model_and_scalers(self, model_path: str, scaler_path: str):
        """Load trained model and feature scaler."""
        feature_scaler = joblib.load(scaler_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        state_dim = feature_scaler.n_features_in_
        action_dim = 3  # Scale Down, Keep Same, Scale Up
        
        model = QNetwork(state_dim, action_dim)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
        
        return model, feature_scaler, device
    
    def load_data(self, data_path: str):
        """Load and prepare test data."""
        # Try to use saved test data first
        test_data_path = Path(data_path).parent / "models" / "test_data.parquet"
        if test_data_path.exists():
            print(f"📂 Using saved test data: {test_data_path}")
            self.df = pd.read_parquet(test_data_path)
        else:
            print(f"📂 Using full dataset for evaluation")
            (_, X_test, _, y_test, _, df_test, _, _) = prepare_data(data_path)
            self.df = df_test
        
        # Prepare features
        action_col = "scaling_action"
        reward_cols = ["memory_utilization_mean", "avg_response_time", "total_anomaly_score"]
        exclude_cols = [action_col] + reward_cols + ["optimal_pod_count"]
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        feature_cols = numeric_cols.difference(exclude_cols)
        
        self.X = self.df[feature_cols].fillna(0).astype(np.float32).values
        self.y_true = self.df[action_col].astype(np.int64).values
        self.X = self.feature_scaler.transform(self.X)
        
        # Prepare reward scalers
        self.reward_scalers = {}
        for col in reward_cols:
            mm = MinMaxScaler()
            self.reward_scalers[col] = mm.fit(self.df[col].astype(np.float32).values.reshape(-1, 1))
        
        self.action_names = ['Scale Down', 'Keep Same', 'Scale Up']
        self.feature_names = feature_cols.tolist()
    
    def get_predictions(self):
        """Generate model predictions and Q-values."""
        with torch.no_grad():
            X_tensor = torch.tensor(self.X, dtype=torch.float32, device=self.device)
            q_values = self.model(X_tensor).cpu().numpy()
            predictions = q_values.argmax(axis=1)
        return predictions, q_values
    
    def plot_confusion_matrix_advanced(self):
        """Enhanced confusion matrix with normalization options."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        cm = confusion_matrix(self.y_true, self.predictions)
        cm_norm_true = confusion_matrix(self.y_true, self.predictions, normalize='true')
        cm_norm_pred = confusion_matrix(self.y_true, self.predictions, normalize='pred')
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.action_names, yticklabels=self.action_names, ax=axes[0])
        axes[0].set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        
        # Normalized by true labels (recall)
        sns.heatmap(cm_norm_true, annot=True, fmt='.2f', cmap='Greens',
                   xticklabels=self.action_names, yticklabels=self.action_names, ax=axes[1])
        axes[1].set_title('Normalized by True Labels (Recall)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        
        # Normalized by predictions (precision)
        sns.heatmap(cm_norm_pred, annot=True, fmt='.2f', cmap='Oranges',
                   xticklabels=self.action_names, yticklabels=self.action_names, ax=axes[2])
        axes[2].set_title('Normalized by Predictions (Precision)', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('True Label', fontsize=12)
        axes[2].set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix_advanced.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curves(self):
        """Multi-class ROC curves (One-vs-Rest)."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Binarize labels for multi-class ROC
        y_bin = label_binarize(self.y_true, classes=[0, 1, 2])
        n_classes = y_bin.shape[1]
        
        # ROC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], self.q_values[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            axes[0].plot(fpr[i], tpr[i], label=f'{self.action_names[i]} (AUC = {roc_auc[i]:.2f})', linewidth=2)
        
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate', fontsize=12)
        axes[0].set_ylabel('True Positive Rate', fontsize=12)
        axes[0].set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
        axes[0].legend(loc="lower right")
        axes[0].grid(True, alpha=0.3)
        
        # Precision-Recall curves
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_bin[:, i], self.q_values[:, i])
            pr_auc = auc(recall, precision)
            axes[1].plot(recall, precision, label=f'{self.action_names[i]} (AUC = {pr_auc:.2f})', linewidth=2)
        
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('Recall', fontsize=12)
        axes[1].set_ylabel('Precision', fontsize=12)
        axes[1].set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        axes[1].legend(loc="lower left")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self):
        """Feature importance analysis using Random Forest as baseline."""
        print("🌳 Computing feature importance with Random Forest...")
        
        # Train a Random Forest for feature importance comparison
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(self.X, self.y_true)
        
        # Get top features
        feature_importance = rf.feature_importances_
        top_indices = np.argsort(feature_importance)[-20:][::-1]  # Top 20
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Feature importance bar plot
        axes[0, 0].barh(range(len(top_indices)), feature_importance[top_indices])
        axes[0, 0].set_yticks(range(len(top_indices)))
        axes[0, 0].set_yticklabels([self.feature_names[i][:30] for i in top_indices], fontsize=8)
        axes[0, 0].set_xlabel('Feature Importance', fontsize=12)
        axes[0, 0].set_title('Top 20 Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
        axes[0, 0].invert_yaxis()
        
        # Feature importance distribution
        axes[0, 1].hist(feature_importance, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(np.mean(feature_importance), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(feature_importance):.4f}')
        axes[0, 1].set_xlabel('Feature Importance', fontsize=12)
        axes[0, 1].set_ylabel('Count', fontsize=12)
        axes[0, 1].set_title('Feature Importance Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
        
        # Cumulative feature importance
        sorted_importance = np.sort(feature_importance)[::-1]
        cumsum_importance = np.cumsum(sorted_importance)
        axes[1, 0].plot(range(1, len(cumsum_importance) + 1), cumsum_importance / cumsum_importance[-1])
        axes[1, 0].axhline(0.8, color='red', linestyle='--', label='80% threshold')
        axes[1, 0].axhline(0.9, color='orange', linestyle='--', label='90% threshold')
        axes[1, 0].set_xlabel('Number of Features', fontsize=12)
        axes[1, 0].set_ylabel('Cumulative Importance', fontsize=12)
        axes[1, 0].set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Feature correlation heatmap (top features)
        top_features_data = self.X[:, top_indices[:10]]
        corr_matrix = np.corrcoef(top_features_data.T)
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   xticklabels=[self.feature_names[i][:15] for i in top_indices[:10]],
                   yticklabels=[self.feature_names[i][:15] for i in top_indices[:10]],
                   ax=axes[1, 1])
        axes[1, 1].set_title('Top 10 Features Correlation', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return top_indices, feature_importance
    
    def plot_dimensionality_reduction(self):
        """t-SNE and PCA visualization of feature space."""
        print("🎯 Computing dimensionality reduction...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sample data for t-SNE (computationally expensive)
        n_samples = min(500, len(self.X))
        indices = np.random.choice(len(self.X), n_samples, replace=False)
        X_sample = self.X[indices]
        y_sample = self.y_true[indices]
        pred_sample = self.predictions[indices]
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_sample)
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_sample)
        
        # t-SNE plot colored by true labels
        scatter1 = axes[0, 0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, 
                                     cmap='viridis', alpha=0.7, s=50)
        axes[0, 0].set_title('t-SNE: True Labels', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('t-SNE 1', fontsize=12)
        axes[0, 0].set_ylabel('t-SNE 2', fontsize=12)
        plt.colorbar(scatter1, ax=axes[0, 0], ticks=[0, 1, 2], 
                    label='Action', shrink=0.8)
        
        # t-SNE plot colored by predictions
        scatter2 = axes[0, 1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=pred_sample, 
                                     cmap='viridis', alpha=0.7, s=50)
        axes[0, 1].set_title('t-SNE: Predicted Labels', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('t-SNE 1', fontsize=12)
        axes[0, 1].set_ylabel('t-SNE 2', fontsize=12)
        plt.colorbar(scatter2, ax=axes[0, 1], ticks=[0, 1, 2], 
                    label='Predicted Action', shrink=0.8)
        
        # PCA plot colored by true labels
        scatter3 = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_sample, 
                                     cmap='viridis', alpha=0.7, s=50)
        axes[1, 0].set_title(f'PCA: True Labels\n(Explained Variance: {pca.explained_variance_ratio_.sum():.2%})', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
        axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
        plt.colorbar(scatter3, ax=axes[1, 0], ticks=[0, 1, 2], 
                    label='Action', shrink=0.8)
        
        # PCA explained variance
        pca_full = PCA()
        pca_full.fit(X_sample)
        cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
        
        axes[1, 1].plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-', markersize=4)
        axes[1, 1].axhline(0.8, color='red', linestyle='--', label='80% variance')
        axes[1, 1].axhline(0.9, color='orange', linestyle='--', label='90% variance')
        axes[1, 1].set_xlabel('Number of Components', fontsize=12)
        axes[1, 1].set_ylabel('Cumulative Explained Variance', fontsize=12)
        axes[1, 1].set_title('PCA Explained Variance', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dimensionality_reduction.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_q_value_analysis(self):
        """Deep analysis of Q-values and confidence."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Q-value distributions per action
        for i, action_name in enumerate(self.action_names):
            axes[0, i].hist(self.q_values[:, i], bins=30, alpha=0.7, edgecolor='black', 
                           color=sns.color_palette()[i])
            axes[0, i].set_title(f'Q-values: {action_name}', fontsize=12, fontweight='bold')
            axes[0, i].set_xlabel('Q-value', fontsize=10)
            axes[0, i].set_ylabel('Frequency', fontsize=10)
            axes[0, i].axvline(np.mean(self.q_values[:, i]), color='red', linestyle='--',
                              label=f'Mean: {np.mean(self.q_values[:, i]):.3f}')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
        
        # Q-value confidence vs accuracy
        max_q = np.max(self.q_values, axis=1)
        confidence = max_q - np.mean(self.q_values, axis=1)
        correct_predictions = (self.predictions == self.y_true)
        
        # Confidence distribution
        axes[1, 0].hist(confidence[correct_predictions], bins=20, alpha=0.7, 
                       label='Correct', color='green', edgecolor='black')
        axes[1, 0].hist(confidence[~correct_predictions], bins=20, alpha=0.7, 
                       label='Incorrect', color='red', edgecolor='black')
        axes[1, 0].set_xlabel('Confidence Score', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Confidence: Correct vs Incorrect', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-value scatter matrix
        q_df = pd.DataFrame(self.q_values, columns=self.action_names)
        q_df['True_Action'] = self.y_true
        
        # Pairwise Q-value relationships
        axes[1, 1].scatter(self.q_values[:, 0], self.q_values[:, 1], 
                          c=self.y_true, cmap='viridis', alpha=0.6, s=30)
        axes[1, 1].set_xlabel(f'Q({self.action_names[0]})', fontsize=12)
        axes[1, 1].set_ylabel(f'Q({self.action_names[1]})', fontsize=12)
        axes[1, 1].set_title('Q-value Relationships', fontsize=14, fontweight='bold')
        
        # Confidence vs Max Q-value
        axes[1, 2].scatter(max_q, confidence, c=correct_predictions, 
                          cmap='RdYlGn', alpha=0.6, s=30)
        axes[1, 2].set_xlabel('Max Q-value', fontsize=12)
        axes[1, 2].set_ylabel('Confidence Score', fontsize=12)
        axes[1, 2].set_title('Confidence vs Max Q-value', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'q_value_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_business_metrics(self):
        """Business impact and operational metrics visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Compute rewards
        rewards = []
        for _, row in self.df.iterrows():
            reward = compute_reward(row, self.reward_scalers, 0.5, 0.5)
            rewards.append(reward)
        rewards = np.array(rewards)
        
        # Reward distribution by action
        for i, action_name in enumerate(self.action_names):
            mask = self.y_true == i
            if mask.sum() > 0:
                axes[0, i].hist(rewards[mask], bins=15, alpha=0.7, 
                               color=sns.color_palette()[i], edgecolor='black')
                axes[0, i].set_title(f'Reward Distribution: {action_name}', 
                                    fontsize=12, fontweight='bold')
                axes[0, i].set_xlabel('Reward', fontsize=10)
                axes[0, i].set_ylabel('Frequency', fontsize=10)
                axes[0, i].axvline(np.mean(rewards[mask]), color='red', linestyle='--',
                                  label=f'Mean: {np.mean(rewards[mask]):.3f}')
                axes[0, i].legend()
                axes[0, i].grid(True, alpha=0.3)
        
        # Operational metrics
        memory_util = self.df['memory_utilization_mean'].values
        response_time = self.df['avg_response_time'].values
        anomaly_score = self.df['total_anomaly_score'].values
        
        # Memory utilization vs predictions
        axes[1, 0].boxplot([memory_util[self.y_true == i] for i in range(3)], 
                          labels=self.action_names)
        axes[1, 0].set_title('Memory Utilization by True Action', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Memory Utilization', fontsize=10)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Response time vs predictions
        axes[1, 1].boxplot([response_time[self.y_true == i] for i in range(3)], 
                          labels=self.action_names)
        axes[1, 1].set_title('Response Time by True Action', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Response Time', fontsize=10)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Action prediction accuracy vs confidence
        confidence = np.max(self.q_values, axis=1) - np.mean(self.q_values, axis=1)
        correct = (self.predictions == self.y_true).astype(int)
        
        # Bin confidence and compute accuracy
        confidence_bins = np.percentile(confidence, [0, 20, 40, 60, 80, 100])
        bin_accuracy = []
        bin_centers = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (confidence >= confidence_bins[i]) & (confidence < confidence_bins[i + 1])
            if mask.sum() > 0:
                bin_accuracy.append(correct[mask].mean())
                bin_centers.append((confidence_bins[i] + confidence_bins[i + 1]) / 2)
        
        axes[1, 2].plot(bin_centers, bin_accuracy, 'bo-', linewidth=2, markersize=8)
        axes[1, 2].set_xlabel('Confidence Score', fontsize=12)
        axes[1, 2].set_ylabel('Accuracy', fontsize=12)
        axes[1, 2].set_title('Accuracy vs Confidence', fontsize=14, fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'business_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_architecture_summary(self):
        """Model architecture and parameter analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Model architecture diagram (simplified)
        layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Output']
        layer_sizes = [self.X.shape[1], 256, 128, 3]
        
        # Network architecture visualization
        max_size = max(layer_sizes)
        for i, (name, size) in enumerate(zip(layer_names, layer_sizes)):
            height = size / max_size
            axes[0, 0].barh(i, height, alpha=0.7, 
                           color=sns.color_palette()[i % len(sns.color_palette())])
            axes[0, 0].text(height/2, i, f'{name}\n({size})', 
                           ha='center', va='center', fontweight='bold')
        
        axes[0, 0].set_yticks(range(len(layer_names)))
        axes[0, 0].set_yticklabels(layer_names)
        axes[0, 0].set_xlabel('Relative Size', fontsize=12)
        axes[0, 0].set_title('DQN Architecture', fontsize=14, fontweight='bold')
        
        # Parameter count breakdown
        param_counts = []
        param_names = []
        for name, param in self.model.named_parameters():
            param_counts.append(param.numel())
            param_names.append(name.replace('net.', '').replace('.weight', ' W').replace('.bias', ' b'))
        
        axes[0, 1].pie(param_counts, labels=param_names, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Parameter Distribution', fontsize=14, fontweight='bold')
        
        # Training vs Test performance comparison (mock data since we don't have training history)
        epochs = range(1, 21)
        train_acc = np.random.normal(0.7, 0.1, 20).cumsum() * 0.01  # Mock training accuracy
        test_acc = np.random.normal(0.65, 0.08, 20).cumsum() * 0.01  # Mock test accuracy
        train_acc = np.clip(train_acc, 0, 1)
        test_acc = np.clip(test_acc, 0, 1)
        
        axes[1, 0].plot(epochs, train_acc, 'b-', label='Training', linewidth=2)
        axes[1, 0].plot(epochs, test_acc, 'r-', label='Validation', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Accuracy', fontsize=12)
        axes[1, 0].set_title('Learning Curves (Simulated)', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Class distribution
        unique, counts = np.unique(self.y_true, return_counts=True)
        axes[1, 1].bar(self.action_names, counts, alpha=0.7, 
                      color=sns.color_palette()[:len(self.action_names)])
        axes[1, 1].set_ylabel('Count', fontsize=12)
        axes[1, 1].set_title('Class Distribution in Test Set', fontsize=14, fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for i, count in enumerate(counts):
            axes[1, 1].text(i, count + 1, str(count), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_architecture.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self):
        """Generate all visualizations and a summary report."""
        print("\n🎨 Generating comprehensive evaluation report...")
        
        # Generate all plots
        plots = [
            ("Confusion Matrix Analysis", self.plot_confusion_matrix_advanced),
            ("ROC and Precision-Recall Curves", self.plot_roc_curves),
            ("Feature Importance Analysis", self.plot_feature_importance),
            ("Dimensionality Reduction", self.plot_dimensionality_reduction),
            ("Q-value Analysis", self.plot_q_value_analysis),
            ("Business Metrics", self.plot_business_metrics),
            ("Model Architecture", self.plot_model_architecture_summary),
        ]
        
        for plot_name, plot_func in plots:
            print(f"  📊 Generating: {plot_name}")
            try:
                plot_func()
                print(f"  ✅ Completed: {plot_name}")
            except Exception as e:
                print(f"  ❌ Error in {plot_name}: {str(e)}")
        
        # Generate summary statistics
        self.generate_summary_stats()
        
        print(f"\n🎉 All visualizations saved to: {self.output_dir}")
        print("📁 Generated files:")
        for file in sorted(self.output_dir.glob("*.png")):
            print(f"  - {file.name}")
    
    def generate_summary_stats(self):
        """Generate comprehensive statistics summary."""
        stats = {
            'model_info': {
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'architecture': 'DQN with 256-128 hidden layers',
                'input_features': self.X.shape[1],
                'output_classes': 3
            },
            'dataset_info': {
                'test_samples': len(self.X),
                'feature_dimensions': self.X.shape[1],
                'class_distribution': {
                    name: int(count) for name, count in 
                    zip(self.action_names, np.bincount(self.y_true))
                }
            },
            'performance_metrics': {
                'accuracy': float(accuracy_score(self.y_true, self.predictions)),
                'per_class_metrics': {}
            }
        }
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(self.y_true, self.predictions)
        for i, name in enumerate(self.action_names):
            stats['performance_metrics']['per_class_metrics'][name] = {
                'precision': float(precision[i]) if i < len(precision) else 0.0,
                'recall': float(recall[i]) if i < len(recall) else 0.0,
                'f1_score': float(f1[i]) if i < len(f1) else 0.0,
                'support': int(support[i]) if i < len(support) else 0
            }
        
        # Save as JSON
        import json
        with open(self.output_dir / 'comprehensive_summary.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save as LaTeX table for papers
        self.generate_latex_table(stats)
    
    def generate_latex_table(self, stats):
        """Generate LaTeX table for research papers."""
        latex_content = """
\\begin{table}[htbp]
\\centering
\\caption{DQN Model Performance on Kubernetes Scaling Task}
\\label{tab:dqn_performance}
\\begin{tabular}{lcccc}
\\hline
\\textbf{Action Class} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} & \\textbf{Support} \\\\
\\hline
"""
        
        for action_name in self.action_names:
            metrics = stats['performance_metrics']['per_class_metrics'].get(action_name, {})
            latex_content += f"{action_name} & "
            latex_content += f"{metrics.get('precision', 0):.3f} & "
            latex_content += f"{metrics.get('recall', 0):.3f} & "
            latex_content += f"{metrics.get('f1_score', 0):.3f} & "
            latex_content += f"{metrics.get('support', 0)} \\\\\n"
        
        latex_content += """\\hline
\\textbf{Overall Accuracy} & \\multicolumn{4}{c}{""" + f"{stats['performance_metrics']['accuracy']:.3f}" + """} \\\\
\\hline
\\end{tabular}
\\end{table}
"""
        
        with open(self.output_dir / 'performance_table.tex', 'w') as f:
            f.write(latex_content)


def main():
    parser = argparse.ArgumentParser(description="Advanced DQN Model Evaluation")
    parser.add_argument("--model", default="models/dqn_model.pt", help="Path to trained model")
    parser.add_argument("--scaler", default="models/feature_scaler.gz", help="Path to feature scaler")
    parser.add_argument("--data", default="processed_data/engineered_features.parquet", help="Path to dataset")
    parser.add_argument("--output", default="advanced_evaluation", help="Output directory")
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    model_path = script_dir / args.model
    scaler_path = script_dir / args.scaler
    data_path = script_dir / args.data
    
    print("🚀 Starting Advanced DQN Evaluation")
    print("=" * 60)
    
    evaluator = AdvancedDQNEvaluator(
        str(model_path), str(scaler_path), str(data_path), args.output
    )
    
    evaluator.generate_comprehensive_report()
    
    print("\n🎓 Evaluation complete! Ready for research publication.")


if __name__ == "__main__":
    main() 