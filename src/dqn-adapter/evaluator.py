#!/usr/bin/env python3
"""
Evaluation Module for DQN Adapter

This module generates publication-quality diagrams and evaluation metrics
for the combined DQN architecture, then saves them to MinIO for persistence.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from io import BytesIO
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

class DQNEvaluator:
    """
    Comprehensive evaluation system for DQN adapter.
    Generates publication-quality diagrams and saves to MinIO.
    """
    
    def __init__(self, minio_client, bucket_name: str = "research-outputs"):
        self.minio_client = minio_client
        self.bucket_name = bucket_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure bucket exists
        if not self.minio_client.bucket_exists(self.bucket_name):
            self.minio_client.make_bucket(self.bucket_name)
            logger.info(f"Created MinIO bucket: {self.bucket_name}")
        
        # Initialize data storage
        self.experiences = []
        self.training_metrics = []
        self.model_checkpoints = []
        
        logger.info(f"🔬 DQN Evaluator initialized (timestamp: {self.timestamp})")
    
    def add_experience(self, experience: Dict[str, Any]):
        """Add experience data for analysis."""
        experience['timestamp'] = time.time()
        self.experiences.append(experience)
    
    def add_training_metrics(self, loss: float, epsilon: float, batch_size: int, buffer_size: int):
        """Add training metrics for analysis."""
        metrics = {
            'timestamp': time.time(),
            'loss': loss,
            'epsilon': epsilon,
            'batch_size': batch_size,
            'buffer_size': buffer_size
        }
        self.training_metrics.append(metrics)
    
    def generate_learning_curve_analysis(self) -> str:
        """Generate comprehensive learning curve analysis."""
        logger.info("📈 Generating learning curve analysis...")
        
        if not self.training_metrics:
            logger.warning("No training metrics available for learning curve")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract metrics
        timestamps = [m['timestamp'] for m in self.training_metrics]
        losses = [m['loss'] for m in self.training_metrics]
        epsilons = [m['epsilon'] for m in self.training_metrics]
        buffer_sizes = [m['buffer_size'] for m in self.training_metrics]
        
        # Convert to relative time (minutes from start)
        start_time = min(timestamps)
        relative_times = [(t - start_time) / 60 for t in timestamps]
        
        # 1. Training Loss Over Time
        axes[0, 0].plot(relative_times, losses, 'b-', linewidth=2, alpha=0.8)
        axes[0, 0].set_xlabel('Time (minutes)', fontsize=12)
        axes[0, 0].set_ylabel('Training Loss', fontsize=12)
        axes[0, 0].set_title('Training Loss Evolution', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add trend line
        if len(losses) > 10:
            z = np.polyfit(relative_times, losses, 1)
            p = np.poly1d(z)
            axes[0, 0].plot(relative_times, p(relative_times), "r--", alpha=0.8, 
                           label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
            axes[0, 0].legend()
        
        # 2. Epsilon Decay
        axes[0, 1].plot(relative_times, epsilons, 'g-', linewidth=2, alpha=0.8)
        axes[0, 1].set_xlabel('Time (minutes)', fontsize=12)
        axes[0, 1].set_ylabel('Epsilon (Exploration Rate)', fontsize=12)
        axes[0, 1].set_title('Exploration vs Exploitation Balance', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Buffer Growth
        axes[1, 0].plot(relative_times, buffer_sizes, 'purple', linewidth=2, alpha=0.8)
        axes[1, 0].set_xlabel('Time (minutes)', fontsize=12)
        axes[1, 0].set_ylabel('Replay Buffer Size', fontsize=12)
        axes[1, 0].set_title('Experience Accumulation', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Learning Efficiency (Loss vs Buffer Size)
        if len(losses) > 5:
            scatter = axes[1, 1].scatter(buffer_sizes, losses, c=relative_times, 
                                       cmap='viridis', alpha=0.7, s=50)
            axes[1, 1].set_xlabel('Replay Buffer Size', fontsize=12)
            axes[1, 1].set_ylabel('Training Loss', fontsize=12)
            axes[1, 1].set_title('Learning Efficiency Over Time', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=axes[1, 1])
            cbar.set_label('Time (minutes)', fontsize=10)
        
        plt.tight_layout()
        
        # Save to MinIO
        filename = f"learning_curves_{self.timestamp}.png"
        return self._save_plot_to_minio(fig, filename)
    
    def generate_action_distribution_analysis(self) -> str:
        """Analyze action distribution and decision patterns."""
        logger.info("🎯 Generating action distribution analysis...")
        
        if not self.experiences:
            logger.warning("No experiences available for action analysis")
            return None
        
        # Extract actions and rewards
        actions = [exp['action'] for exp in self.experiences]
        rewards = [exp['reward'] for exp in self.experiences]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Action Distribution
        action_counts = pd.Series(actions).value_counts()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        bars = axes[0, 0].bar(action_counts.index, action_counts.values, 
                             color=colors, alpha=0.8, edgecolor='black')
        axes[0, 0].set_xlabel('Scaling Action', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Action Distribution', fontsize=14, fontweight='bold')
        
        # Add percentage labels
        total = sum(action_counts.values)
        for bar, count in zip(bars, action_counts.values):
            pct = (count / total) * 100
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{count}\n({pct:.1f}%)', ha='center', fontweight='bold')
        
        # 2. Reward Distribution by Action
        action_rewards = {action: [] for action in set(actions)}
        for action, reward in zip(actions, rewards):
            action_rewards[action].append(reward)
        
        box_data = [action_rewards[action] for action in action_counts.index]
        bp = axes[0, 1].boxplot(box_data, labels=action_counts.index, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        axes[0, 1].set_xlabel('Scaling Action', fontsize=12)
        axes[0, 1].set_ylabel('Reward', fontsize=12)
        axes[0, 1].set_title('Reward Distribution by Action', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Action Sequence Analysis
        if len(actions) > 1:
            # Create transition matrix
            action_list = ['Scale Down', 'Keep Same', 'Scale Up']
            transition_matrix = np.zeros((3, 3))
            
            action_to_idx = {action: i for i, action in enumerate(action_list)}
            
            for i in range(len(actions) - 1):
                curr_idx = action_to_idx.get(actions[i], 1)
                next_idx = action_to_idx.get(actions[i + 1], 1)
                transition_matrix[curr_idx, next_idx] += 1
            
            # Normalize to probabilities
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            transition_matrix = np.divide(transition_matrix, row_sums, 
                                        out=np.zeros_like(transition_matrix), 
                                        where=row_sums!=0)
            
            im = axes[1, 0].imshow(transition_matrix, cmap='Blues', aspect='auto')
            axes[1, 0].set_xticks(range(3))
            axes[1, 0].set_yticks(range(3))
            axes[1, 0].set_xticklabels(action_list, rotation=45)
            axes[1, 0].set_yticklabels(action_list)
            axes[1, 0].set_xlabel('Next Action', fontsize=12)
            axes[1, 0].set_ylabel('Current Action', fontsize=12)
            axes[1, 0].set_title('Action Transition Probabilities', fontsize=14, fontweight='bold')
            
            # Add probability labels
            for i in range(3):
                for j in range(3):
                    axes[1, 0].text(j, i, f'{transition_matrix[i, j]:.2f}', 
                                   ha='center', va='center', fontweight='bold')
            
            plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Reward Evolution Over Time
        if len(rewards) > 1:
            window_size = min(10, len(rewards) // 5)
            if window_size > 1:
                smoothed_rewards = pd.Series(rewards).rolling(window=window_size).mean()
                axes[1, 1].plot(range(len(rewards)), rewards, 'lightblue', alpha=0.5, label='Raw Rewards')
                axes[1, 1].plot(range(len(rewards)), smoothed_rewards, 'darkblue', 
                               linewidth=2, label=f'Moving Average (window={window_size})')
            else:
                axes[1, 1].plot(range(len(rewards)), rewards, 'darkblue', linewidth=2)
            
            axes[1, 1].set_xlabel('Decision Number', fontsize=12)
            axes[1, 1].set_ylabel('Reward', fontsize=12)
            axes[1, 1].set_title('Reward Evolution', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            if window_size > 1:
                axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save to MinIO
        filename = f"action_analysis_{self.timestamp}.png"
        return self._save_plot_to_minio(fig, filename)
    
    def generate_performance_dashboard(self, model_state: Dict[str, Any]) -> str:
        """Generate comprehensive performance dashboard."""
        logger.info("📊 Generating performance dashboard...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Model Architecture Visualization
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_model_architecture(ax1, model_state)
        
        # 2. Training Progress
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_training_progress(ax2)
        
        # 3. Experience Statistics
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_experience_statistics(ax3)
        
        # 4. Reward Analysis
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_reward_analysis(ax4)
        
        # 5. System Metrics
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_system_metrics(ax5)
        
        # 6. Decision Confidence
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_decision_confidence(ax6)
        
        # 7. Performance Summary Table
        ax7 = fig.add_subplot(gs[3, :])
        self._create_performance_summary_table(ax7, model_state)
        
        # Save to MinIO
        filename = f"performance_dashboard_{self.timestamp}.png"
        return self._save_plot_to_minio(fig, filename)
    
    def _plot_model_architecture(self, ax, model_state):
        """Plot model architecture diagram."""
        layers = ['Input\n(4249)', 'Hidden 1\n(512)', 'Hidden 2\n(256)', 'Hidden 3\n(128)', 'Output\n(3)']
        sizes = [4249, 512, 256, 128, 3]
        colors = ['red', 'orange', 'yellow', 'green', 'blue']
        
        for i, (layer, size, color) in enumerate(zip(layers, sizes, colors)):
            # Draw nodes
            radius = np.sqrt(size) / 100
            circle = plt.Circle((i, 0), radius, color=color, alpha=0.7, ec='black')
            ax.add_patch(circle)
            
            # Connect layers
            if i < len(layers) - 1:
                ax.arrow(i + radius, 0, 1 - 2*radius, 0, 
                        head_width=0.05, head_length=0.05, fc='black', ec='black')
            
            # Labels
            ax.text(i, -0.4, layer, ha='center', va='center', fontsize=9, fontweight='bold')
        
        ax.set_xlim(-0.5, len(layers) - 0.5)
        ax.set_ylim(-0.6, 0.6)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('DQN Architecture', fontsize=14, fontweight='bold')
    
    def _plot_training_progress(self, ax):
        """Plot training progress."""
        if not self.training_metrics:
            ax.text(0.5, 0.5, 'No training data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            return
        
        losses = [m['loss'] for m in self.training_metrics]
        ax.plot(losses, 'b-', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Progress', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_experience_statistics(self, ax):
        """Plot experience statistics."""
        if not self.experiences:
            ax.text(0.5, 0.5, 'No experience data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return
        
        # Experience accumulation over time
        timestamps = [exp['timestamp'] for exp in self.experiences]
        start_time = min(timestamps)
        relative_times = [(t - start_time) / 60 for t in timestamps]
        
        ax.plot(relative_times, range(1, len(self.experiences) + 1), 'g-', linewidth=2)
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Total Experiences')
        ax.set_title('Experience Accumulation', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_reward_analysis(self, ax):
        """Plot reward analysis."""
        if not self.experiences:
            ax.text(0.5, 0.5, 'No reward data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return
        
        rewards = [exp['reward'] for exp in self.experiences]
        ax.hist(rewards, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(rewards), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(rewards):.2f}')
        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Reward Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_system_metrics(self, ax):
        """Plot system metrics from experiences."""
        if not self.experiences:
            ax.text(0.5, 0.5, 'No system data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return
        
        # Extract latency metrics
        latencies = []
        for exp in self.experiences:
            if 'state' in exp and 'request_latency_p95' in exp['state']:
                latencies.append(exp['state']['request_latency_p95'])
        
        if latencies:
            ax.plot(latencies, 'purple', linewidth=2)
            ax.set_xlabel('Decision Number')
            ax.set_ylabel('Latency (P95)')
            ax.set_title('System Latency Trend', fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    def _plot_decision_confidence(self, ax):
        """Plot decision confidence over time."""
        # This would require Q-values from experiences
        ax.text(0.5, 0.5, 'Decision Confidence\n(Q-values analysis)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Decision Confidence', fontweight='bold')
    
    def _create_performance_summary_table(self, ax, model_state):
        """Create performance summary table."""
        # Calculate summary statistics
        total_experiences = len(self.experiences)
        total_training_steps = len(self.training_metrics)
        avg_reward = np.mean([exp['reward'] for exp in self.experiences]) if self.experiences else 0
        
        # Create table data
        data = [
            ['Total Experiences', f'{total_experiences:,}'],
            ['Training Steps', f'{total_training_steps:,}'],
            ['Average Reward', f'{avg_reward:.3f}'],
            ['Model Parameters', f'{model_state.get("total_params", "N/A"):,}' if isinstance(model_state.get("total_params"), int) else 'N/A'],
            ['Current Epsilon', f'{self.training_metrics[-1]["epsilon"]:.3f}' if self.training_metrics else 'N/A'],
            ['Buffer Size', f'{self.training_metrics[-1]["buffer_size"]:,}' if self.training_metrics else 'N/A']
        ]
        
        # Create table
        table = ax.table(cellText=data, colLabels=['Metric', 'Value'],
                        cellLoc='center', loc='center', colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4ECDC4')
                    cell.set_text_props(weight='bold')
                else:
                    cell.set_facecolor('#F0F0F0' if i % 2 == 0 else 'white')
        
        ax.axis('off')
        ax.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report."""
        logger.info("📋 Generating summary report...")
        
        # Compile all metrics
        summary = {
            'timestamp': self.timestamp,
            'evaluation_metrics': {
                'total_experiences': len(self.experiences),
                'total_training_steps': len(self.training_metrics),
                'average_reward': np.mean([exp['reward'] for exp in self.experiences]) if self.experiences else 0,
                'reward_std': np.std([exp['reward'] for exp in self.experiences]) if self.experiences else 0,
            },
            'action_distribution': {},
            'training_performance': {},
            'system_metrics': {}
        }
        
        # Action distribution
        if self.experiences:
            actions = [exp['action'] for exp in self.experiences]
            action_counts = pd.Series(actions).value_counts()
            summary['action_distribution'] = action_counts.to_dict()
        
        # Training performance
        if self.training_metrics:
            losses = [m['loss'] for m in self.training_metrics]
            summary['training_performance'] = {
                'final_loss': losses[-1] if losses else 0,
                'loss_improvement': losses[0] - losses[-1] if len(losses) > 1 else 0,
                'convergence_rate': np.mean(np.diff(losses)) if len(losses) > 1 else 0
            }
        
        # Save JSON report
        report_filename = f"evaluation_summary_{self.timestamp}.json"
        json_buffer = BytesIO()
        json_buffer.write(json.dumps(summary, indent=2, default=str).encode())
        json_buffer.seek(0)
        
        self.minio_client.put_object(
            self.bucket_name,
            report_filename,
            data=json_buffer,
            length=json_buffer.getbuffer().nbytes,
            content_type='application/json'
        )
        
        logger.info(f"📊 Evaluation summary saved: {report_filename}")
        return report_filename
    
    def _save_plot_to_minio(self, fig, filename: str) -> str:
        """Save matplotlib figure to MinIO."""
        try:
            # Save to buffer
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            
            # Upload to MinIO
            self.minio_client.put_object(
                self.bucket_name,
                filename,
                data=buffer,
                length=buffer.getbuffer().nbytes,
                content_type='image/png'
            )
            
            plt.close(fig)  # Free memory
            logger.info(f"📈 Evaluation diagram saved: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save plot to MinIO: {e}")
            plt.close(fig)
            return None
    
    def generate_all_outputs(self, model_state: Dict[str, Any]) -> List[str]:
        """Generate all evaluation outputs and return list of saved files."""
        logger.info("🔬 Generating comprehensive evaluation...")
        
        saved_files = []
        
        # Generate all visualizations
        files = [
            self.generate_learning_curve_analysis(),
            self.generate_action_distribution_analysis(), 
            self.generate_performance_dashboard(model_state),
            self.generate_summary_report()
        ]
        
        saved_files.extend([f for f in files if f is not None])
        
        logger.info(f"✅ Evaluation complete. Generated {len(saved_files)} files:")
        for file in saved_files:
            logger.info(f"  📄 {file}")
        
        return saved_files 