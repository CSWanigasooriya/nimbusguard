#!/usr/bin/env python3
"""
Evaluation Module for DQN Adapter

This module generates publication-quality diagrams and evaluation metrics
for the clean 5-feature DQN architecture, then saves them to MinIO for persistence.

Feature Architecture:
- 9 Base Features: Scientifically-selected Kubernetes state metrics from Prometheus
- Total: 9 scientifically-selected features for optimal DQN performance
"""

import os
import json
import time
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from io import BytesIO

# Configure matplotlib for headless environments (containers)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for containers
import matplotlib.pyplot as plt

# Configure seaborn with error handling
try:
    import seaborn as sns
    # Use updated seaborn style (seaborn-v0_8 is deprecated)
    plt.style.use('default')
    sns.set_palette("husl")
    SEABORN_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("EVALUATOR: seaborn_configured_successfully")
except ImportError as e:
    SEABORN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"EVALUATOR: seaborn_not_available error={e}")
except Exception as e:
    SEABORN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"EVALUATOR: seaborn_configuration_failed error={e}")
    plt.style.use('default')  # Fallback to default style

# Configure plotly with error handling
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
    logger.info("EVALUATOR: plotly_configured_successfully")
except ImportError as e:
    PLOTLY_AVAILABLE = False
    logger.warning(f"EVALUATOR: plotly_not_available error={e}")
except Exception as e:
    PLOTLY_AVAILABLE = False
    logger.warning(f"EVALUATOR: plotly_configuration_failed error={e}")

# Configure scientific libraries with error handling
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    logger.info("EVALUATOR: torch_configured_successfully")
except ImportError as e:
    TORCH_AVAILABLE = False
    logger.warning(f"EVALUATOR: torch_not_available error={e}")

try:
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, confusion_matrix,
        classification_report, roc_curve, auc, precision_recall_curve
    )
    from sklearn.preprocessing import label_binarize
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
    logger.info("EVALUATOR: sklearn_configured_successfully")
except ImportError as e:
    SKLEARN_AVAILABLE = False
    logger.warning(f"EVALUATOR: sklearn_not_available error={e}")

logger = logging.getLogger(__name__)

class DQNEvaluator:
    """
    Comprehensive evaluation system for DQN adapter.
    Generates publication-quality diagrams and saves to MinIO.
    """
    
    def __init__(self, minio_client, bucket_name: str = "research-outputs"):
        try:
            self.minio_client = minio_client
            self.bucket_name = bucket_name
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Check library availability
            self.matplotlib_available = True
            self.seaborn_available = SEABORN_AVAILABLE
            self.plotly_available = PLOTLY_AVAILABLE
            self.sklearn_available = SKLEARN_AVAILABLE
            self.torch_available = TORCH_AVAILABLE
            
            logger.info(f"EVALUATOR: library_status matplotlib={self.matplotlib_available} "
                       f"seaborn={self.seaborn_available} plotly={self.plotly_available} "
                       f"sklearn={self.sklearn_available} torch={self.torch_available}")
            
            # Ensure bucket exists with error handling
            try:
                if not self.minio_client.bucket_exists(self.bucket_name):
                    self.minio_client.make_bucket(self.bucket_name)
                    logger.info(f"EVALUATOR: bucket_created name={self.bucket_name}")
                else:
                    logger.info(f"EVALUATOR: bucket_exists name={self.bucket_name}")
            except Exception as bucket_error:
                logger.error(f"EVALUATOR: bucket_setup_failed error={bucket_error}")
                raise
            
            # Initialize data storage with resource limits
            self.experiences = []
            self.training_metrics = []
            self.model_checkpoints = []
            
            # Resource limits to prevent memory leaks
            self.max_experiences = 10000
            self.max_training_metrics = 5000
            self.max_q_value_history = 5000
            self.max_episode_metrics = 1000
            
            # RESEARCH ENHANCEMENT: Add research-focused data tracking
            self.episode_metrics = []           # Episode-level statistics
            self.q_value_history = []           # Q-value evolution tracking
            self.hyperparameter_runs = {}       # Multiple configuration results
            self.policy_snapshots = []          # Policy state at intervals
            self.current_episode_id = 0         # Episode boundary tracking
            self.current_episode_rewards = []   # Rewards within current episode
            
            logger.info(f"EVALUATOR: initialized_successfully timestamp={self.timestamp}")
            logger.info("EVALUATOR: research_enhancement_enabled episode_tracking=True q_value_tracking=True")
            
        except Exception as init_error:
            logger.error(f"EVALUATOR: initialization_failed error={init_error}")
            raise
    
    def add_experience(self, experience: Dict[str, Any]):
        """Add experience data for analysis with research enhancements."""
        experience['timestamp'] = time.time()
        experience['episode_id'] = self.current_episode_id
        
        # Track episode rewards for episode-level analysis
        if 'reward' in experience:
            self.current_episode_rewards.append(experience['reward'])
        
        self.experiences.append(experience)
        # Enforce memory limits
        if len(self.experiences) > self.max_experiences:
            self.experiences = self.experiences[-self.max_experiences:]
    
    def add_q_values(self, q_values: List[float], action_taken: int, state_features: Dict[str, float]):
        """Add Q-value data for research analysis."""
        try:
            q_value_entry = {
                'timestamp': time.time(),
                'episode_id': self.current_episode_id,
                'q_values': q_values.copy() if isinstance(q_values, list) else list(q_values),
                'action_taken': action_taken,
                'max_q_value': float(max(q_values)) if q_values else 0.0,
                'min_q_value': float(min(q_values)) if q_values else 0.0,
                'avg_q_value': float(np.mean(q_values)) if q_values else 0.0,
                'q_value_variance': float(np.var(q_values)) if len(q_values) > 1 else 0.0,
                'state_features': state_features.copy()
            }
            self.q_value_history.append(q_value_entry)
            # Enforce memory limits
            if len(self.q_value_history) > self.max_q_value_history:
                self.q_value_history = self.q_value_history[-self.max_q_value_history:]
            
            if len(self.q_value_history) % 10 == 0:
                logger.info(f"EVALUATOR: q_values_tracked count={len(self.q_value_history)} "
                           f"avg_q={q_value_entry['avg_q_value']:.3f}")
        except Exception as e:
            logger.error(f"EVALUATOR: add_q_values_failed error={e}")
    
    def end_episode(self, episode_success: bool = True, episode_info: Dict[str, Any] = None):
        """Mark end of episode and calculate episode-level metrics."""
        try:
            if not self.current_episode_rewards:
                logger.warning(f"EVALUATOR: episode_end_no_rewards episode_id={self.current_episode_id}")
                self.current_episode_id += 1
                return
            
            # Calculate episode statistics
            episode_stats = {
                'episode_id': self.current_episode_id,
                'timestamp': time.time(),
                'episode_length': len(self.current_episode_rewards),
                'total_reward': sum(self.current_episode_rewards),
                'avg_reward': np.mean(self.current_episode_rewards),
                'max_reward': max(self.current_episode_rewards),
                'min_reward': min(self.current_episode_rewards),
                'reward_variance': np.var(self.current_episode_rewards),
                'episode_success': episode_success,
                'episode_info': episode_info or {}
            }
            
            self.episode_metrics.append(episode_stats)
            # Enforce memory limits
            if len(self.episode_metrics) > self.max_episode_metrics:
                self.episode_metrics = self.episode_metrics[-self.max_episode_metrics:]
            
            logger.info(f"EVALUATOR: episode_completed episode_id={self.current_episode_id} "
                       f"length={episode_stats['episode_length']} "
                       f"total_reward={episode_stats['total_reward']:.3f}")
            
            # Reset for next episode
            self.current_episode_id += 1
            self.current_episode_rewards = []
            
        except Exception as e:
            logger.error(f"EVALUATOR: end_episode_failed error={e}")
    
    def add_hyperparameter_run(self, run_id: str, hyperparams: Dict[str, Any], 
                              performance_metrics: Dict[str, float]):
        """Add hyperparameter experiment results for sensitivity analysis."""
        try:
            self.hyperparameter_runs[run_id] = {
                'hyperparams': hyperparams.copy(),
                'performance': performance_metrics.copy(),
                'timestamp': time.time()
            }
            logger.info(f"EVALUATOR: hyperparameter_run_added run_id={run_id}")
        except Exception as e:
            logger.error(f"EVALUATOR: add_hyperparameter_run_failed error={e}")
    
    def add_training_metrics(self, loss: float, epsilon: float, batch_size: int, buffer_size: int):
        """Add training metrics for analysis with comprehensive validation."""
        try:
            # Validate inputs before adding
            if not isinstance(loss, (int, float)) or not np.isfinite(loss):
                logger.warning(f"EVALUATOR: invalid_loss_value loss={loss} type={type(loss)} - skipping")
                return
                
            if not (0 <= loss <= 1000):  # Reasonable loss range for DQN
                logger.warning(f"EVALUATOR: loss_out_of_range loss={loss} expected_range=[0,1000] - skipping")
                return
                
            if not isinstance(epsilon, (int, float)) or not (0 <= epsilon <= 1):
                logger.warning(f"EVALUATOR: invalid_epsilon epsilon={epsilon} expected_range=[0,1] - skipping")
                return
                
            if not isinstance(batch_size, int) or not (1 <= batch_size <= 1000):
                logger.warning(f"EVALUATOR: invalid_batch_size batch_size={batch_size} expected_range=[1,1000] - skipping")
                return
                
            if not isinstance(buffer_size, int) or not (0 <= buffer_size <= 100000):
                logger.warning(f"EVALUATOR: invalid_buffer_size buffer_size={buffer_size} expected_range=[0,100000] - skipping")
                return
            
            # Create validated metrics entry
            metrics = {
                'timestamp': time.time(),
                'loss': float(loss),
                'epsilon': float(epsilon),
                'batch_size': int(batch_size),
                'buffer_size': int(buffer_size)
            }
            
            # Check for reasonable progression (detect potential data corruption)
            if len(self.training_metrics) > 0:
                last_metrics = self.training_metrics[-1]
                time_diff = metrics['timestamp'] - last_metrics['timestamp']
                
                # Check for reasonable time progression (not too fast/slow)
                if time_diff < 0.1:  # Less than 100ms between training steps
                    logger.debug(f"EVALUATOR: very_fast_training_step time_diff={time_diff:.3f}s")
                elif time_diff > 3600:  # More than 1 hour between steps
                    logger.warning(f"EVALUATOR: large_time_gap time_diff={time_diff:.1f}s")
                
                # Check for reasonable buffer growth
                buffer_growth = metrics['buffer_size'] - last_metrics['buffer_size']
                if buffer_growth < 0:
                    logger.warning(f"EVALUATOR: buffer_size_decreased from={last_metrics['buffer_size']} to={metrics['buffer_size']}")
                elif buffer_growth > 1000:  # More than 1000 experiences added at once
                    logger.warning(f"EVALUATOR: large_buffer_growth growth={buffer_growth}")
            
            # Add to metrics list
            self.training_metrics.append(metrics)
            
            # Log every 10th metric for monitoring
            if len(self.training_metrics) % 10 == 0:
                logger.info(f"EVALUATOR: training_metrics_added count={len(self.training_metrics)} "
                           f"loss={loss:.4f} buffer_size={buffer_size} epsilon={epsilon:.4f}")
            
            # Enforce memory limits
            if len(self.training_metrics) > self.max_training_metrics:
                self.training_metrics = self.training_metrics[-self.max_training_metrics:]
                logger.info(f"EVALUATOR: training_metrics_trimmed kept_last={self.max_training_metrics}")
                
        except Exception as e:
            logger.error(f"EVALUATOR: add_training_metrics_failed error={e}")
            # Don't raise - just log and continue
    
    def generate_learning_curve_analysis(self) -> str:
        """Generate comprehensive learning curve analysis."""
        logger.info("EVALUATOR: generating_learning_curve_analysis")
        
        try:
            if not self.training_metrics:
                logger.warning("EVALUATOR: no_training_metrics_available")
                return None
            
            if not self.matplotlib_available:
                logger.warning("EVALUATOR: matplotlib_not_available skipping_learning_curves")
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
            
            # 4. Learning Efficiency (Loss vs Buffer Size) - IMPROVED WITH OUTLIER DETECTION
            if len(losses) > 5:
                # Remove extreme outliers using IQR method
                losses_array = np.array(losses)
                buffer_sizes_array = np.array(buffer_sizes)
                relative_times_array = np.array(relative_times)
                
                # Outlier detection for losses
                q75_loss, q25_loss = np.percentile(losses_array, [75, 25])
                iqr_loss = q75_loss - q25_loss
                lower_bound_loss = q25_loss - 1.5 * iqr_loss
                upper_bound_loss = q75_loss + 1.5 * iqr_loss
                
                # Outlier detection for buffer sizes
                q75_buffer, q25_buffer = np.percentile(buffer_sizes_array, [75, 25])
                iqr_buffer = q75_buffer - q25_buffer
                lower_bound_buffer = q25_buffer - 1.5 * iqr_buffer
                upper_bound_buffer = q75_buffer + 1.5 * iqr_buffer
                
                # Create mask for valid data points (no outliers)
                valid_loss_mask = (losses_array >= lower_bound_loss) & (losses_array <= upper_bound_loss)
                valid_buffer_mask = (buffer_sizes_array >= lower_bound_buffer) & (buffer_sizes_array <= upper_bound_buffer)
                valid_mask = valid_loss_mask & valid_buffer_mask
                
                # Filter data
                filtered_losses = losses_array[valid_mask]
                filtered_buffer_sizes = buffer_sizes_array[valid_mask]
                filtered_times = relative_times_array[valid_mask]
                
                # Log outlier information
                outliers_removed = len(losses) - len(filtered_losses)
                if outliers_removed > 0:
                    logger.info(f"EVALUATOR: outliers_removed count={outliers_removed} "
                               f"loss_range=[{lower_bound_loss:.2f}, {upper_bound_loss:.2f}] "
                               f"buffer_range=[{lower_bound_buffer:.0f}, {upper_bound_buffer:.0f}]")
                
                if len(filtered_losses) > 3:  # Need at least 3 points for meaningful plot
                    scatter = axes[1, 1].scatter(filtered_buffer_sizes, filtered_losses, c=filtered_times, 
                                               cmap='viridis', alpha=0.7, s=50)
                    axes[1, 1].set_xlabel('Replay Buffer Size', fontsize=12)
                    axes[1, 1].set_ylabel('Training Loss', fontsize=12)
                    axes[1, 1].set_title(f'Learning Efficiency Over Time\n(Outliers removed: {outliers_removed})', 
                                        fontsize=14, fontweight='bold')
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=axes[1, 1])
                    cbar.set_label('Time (minutes)', fontsize=10)
                    
                    # Add statistics text
                    if len(filtered_losses) > 1:
                        buffer_growth = filtered_buffer_sizes[-1] - filtered_buffer_sizes[0]
                        loss_change = filtered_losses[-1] - filtered_losses[0]
                        stats_text = f'Buffer Growth: {buffer_growth:.0f}\nLoss Change: {loss_change:.3f}'
                        axes[1, 1].text(0.02, 0.98, stats_text, transform=axes[1, 1].transAxes,
                                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    # Fallback when too many outliers
                    axes[1, 1].text(0.5, 0.5, f'Insufficient valid data\n({len(filtered_losses)} points after outlier removal)', 
                                   ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
                    axes[1, 1].set_title('Learning Efficiency Over Time\n(Data quality issues)', fontsize=14, fontweight='bold')
            else:
                axes[1, 1].text(0.5, 0.5, f'Insufficient training data\n({len(losses)} data points)', 
                               ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
                axes[1, 1].set_title('Learning Efficiency Over Time', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Save to MinIO
            filename = f"learning_curves_{self.timestamp}.png"
            return self._save_plot_to_minio(fig, filename)
            
        except Exception as e:
            logger.error(f"EVALUATOR: learning_curve_generation_failed error={e}")
            return None
    
    def generate_q_value_analysis(self) -> str:
        """Generate comprehensive Q-value convergence analysis for research."""
        logger.info("EVALUATOR: generating_q_value_analysis")
        
        try:
            if not self.q_value_history:
                logger.warning("EVALUATOR: no_q_value_data_available")
                return None
            
            if not self.matplotlib_available:
                logger.warning("EVALUATOR: matplotlib_not_available skipping_q_value_analysis")
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Extract Q-value data
            timestamps = [q['timestamp'] for q in self.q_value_history]
            start_time = min(timestamps)
            relative_times = [(t - start_time) / 60 for t in timestamps]
            
            avg_q_values = [q['avg_q_value'] for q in self.q_value_history]
            max_q_values = [q['max_q_value'] for q in self.q_value_history]
            min_q_values = [q['min_q_value'] for q in self.q_value_history]
            q_variances = [q['q_value_variance'] for q in self.q_value_history]
            
            # 1. Q-Value Evolution Over Time
            axes[0, 0].plot(relative_times, avg_q_values, 'b-', linewidth=2, label='Average Q-Value', alpha=0.8)
            axes[0, 0].fill_between(relative_times, min_q_values, max_q_values, alpha=0.3, color='blue')
            axes[0, 0].set_xlabel('Time (minutes)', fontsize=12)
            axes[0, 0].set_ylabel('Q-Value', fontsize=12)
            axes[0, 0].set_title('Q-Value Evolution & Convergence', fontsize=14, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # Add convergence trend line
            if len(avg_q_values) > 10:
                z = np.polyfit(relative_times, avg_q_values, 1)
                p = np.poly1d(z)
                axes[0, 0].plot(relative_times, p(relative_times), "r--", alpha=0.8, 
                               label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
                axes[0, 0].legend()
            
            # 2. Q-Value Variance (Stability Indicator)
            axes[0, 1].plot(relative_times, q_variances, 'purple', linewidth=2, alpha=0.8)
            axes[0, 1].set_xlabel('Time (minutes)', fontsize=12)
            axes[0, 1].set_ylabel('Q-Value Variance', fontsize=12)
            axes[0, 1].set_title('Q-Value Stability (Lower = More Stable)', fontsize=14, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add moving average
            if len(q_variances) > 5:
                window = min(10, len(q_variances) // 3)
                smoothed_variance = pd.Series(q_variances).rolling(window=window).mean()
                axes[0, 1].plot(relative_times, smoothed_variance, 'red', linewidth=2, 
                               label=f'Moving Avg (window={window})')
                axes[0, 1].legend()
            
            # 3. Per-Action Q-Value Distribution
            if len(self.q_value_history) > 10:
                action_q_values = {0: [], 1: [], 2: []}  # Scale Down, Keep Same, Scale Up
                
                for q_entry in self.q_value_history:
                    if len(q_entry['q_values']) >= 3:
                        for action_idx, q_val in enumerate(q_entry['q_values']):
                            if action_idx < 3:
                                action_q_values[action_idx].append(q_val)
                
                action_labels = ['Scale Down', 'Keep Same', 'Scale Up']
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                
                box_data = [action_q_values[i] for i in range(3) if action_q_values[i]]
                if box_data:
                    bp = axes[1, 0].boxplot(box_data, labels=action_labels[:len(box_data)], 
                                           patch_artist=True)
                    for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    axes[1, 0].set_xlabel('Action', fontsize=12)
                    axes[1, 0].set_ylabel('Q-Value', fontsize=12)
                    axes[1, 0].set_title('Q-Value Distribution by Action', fontsize=14, fontweight='bold')
                    axes[1, 0].grid(True, alpha=0.3)
                else:
                    axes[1, 0].text(0.5, 0.5, 'Insufficient Q-value data\nfor action analysis', 
                                   ha='center', va='center', transform=axes[1, 0].transAxes)
            
            # 4. Q-Value Convergence Metrics
            if len(avg_q_values) > 20:
                # Calculate convergence metrics
                recent_window = avg_q_values[-10:]  # Last 10 values
                earlier_window = avg_q_values[-20:-10]  # Previous 10 values
                
                recent_mean = np.mean(recent_window)
                earlier_mean = np.mean(earlier_window)
                recent_std = np.std(recent_window)
                convergence_rate = abs(recent_mean - earlier_mean)
                
                # Plot convergence analysis
                window_size = min(10, len(avg_q_values) // 5)
                moving_avg = pd.Series(avg_q_values).rolling(window=window_size).mean()
                moving_std = pd.Series(avg_q_values).rolling(window=window_size).std()
                
                axes[1, 1].plot(relative_times, moving_avg, 'darkblue', linewidth=2, label='Moving Average')
                axes[1, 1].fill_between(relative_times, 
                                       moving_avg - moving_std, 
                                       moving_avg + moving_std, 
                                       alpha=0.3, color='lightblue', label='±1 Std Dev')
                
                axes[1, 1].set_xlabel('Time (minutes)', fontsize=12)
                axes[1, 1].set_ylabel('Q-Value', fontsize=12)
                axes[1, 1].set_title(f'Q-Value Convergence Analysis\nRecent Stability: ±{recent_std:.3f}', 
                                    fontsize=14, fontweight='bold')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].legend()
                
                # Add convergence status text
                if recent_std < 0.1:
                    convergence_status = "CONVERGED"
                    status_color = "green"
                elif recent_std < 0.5:
                    convergence_status = "STABILIZING"
                    status_color = "orange"
                else:
                    convergence_status = "UNSTABLE"
                    status_color = "red"
                
                axes[1, 1].text(0.02, 0.98, f'Status: {convergence_status}', 
                               transform=axes[1, 1].transAxes, fontweight='bold',
                               color=status_color, fontsize=12,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor convergence analysis\n(need >20 samples)', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
            
            plt.tight_layout()
            
            # Save to MinIO
            filename = f"q_value_analysis_{self.timestamp}.png"
            return self._save_plot_to_minio(fig, filename)
            
        except Exception as e:
            logger.error(f"EVALUATOR: q_value_analysis_failed error={e}")
            return None
    
    def generate_action_distribution_analysis(self) -> str:
        """Analyze action distribution and decision patterns."""
        logger.info("EVALUATOR: generating_action_distribution_analysis")
        
        try:
            if not self.experiences:
                logger.warning("EVALUATOR: no_experiences_available")
                return None
            
            if not self.matplotlib_available:
                logger.warning("EVALUATOR: matplotlib_not_available skipping_action_analysis")
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
            
        except Exception as e:
            logger.error(f"EVALUATOR: action_distribution_analysis_failed error={e}")
            return None
    
    def generate_episode_analysis(self) -> str:
        """Generate episode-based reward and performance analysis for research."""
        logger.info("EVALUATOR: generating_episode_analysis")
        
        try:
            if not self.episode_metrics:
                logger.warning("EVALUATOR: no_episode_data_available")
                return None
            
            if not self.matplotlib_available:
                logger.warning("EVALUATOR: matplotlib_not_available skipping_episode_analysis")
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Extract episode data
            episode_ids = [ep['episode_id'] for ep in self.episode_metrics]
            total_rewards = [ep['total_reward'] for ep in self.episode_metrics]
            avg_rewards = [ep['avg_reward'] for ep in self.episode_metrics]
            episode_lengths = [ep['episode_length'] for ep in self.episode_metrics]
            timestamps = [ep['timestamp'] for ep in self.episode_metrics]
            
            # Convert to relative time
            start_time = min(timestamps)
            relative_times = [(t - start_time) / 3600 for t in timestamps]  # Hours
            
            # 1. Episode Reward Evolution
            axes[0, 0].plot(episode_ids, total_rewards, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.8)
            axes[0, 0].set_xlabel('Episode Number', fontsize=12)
            axes[0, 0].set_ylabel('Total Episode Reward', fontsize=12)
            axes[0, 0].set_title('Episode Reward Evolution', fontsize=14, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add moving average trend
            if len(total_rewards) > 5:
                window = min(10, len(total_rewards) // 3)
                smoothed_rewards = pd.Series(total_rewards).rolling(window=window).mean()
                axes[0, 0].plot(episode_ids, smoothed_rewards, 'red', linewidth=2, 
                               label=f'Moving Avg (window={window})')
                axes[0, 0].legend()
            
            # 2. Episode Length Analysis
            axes[0, 1].plot(episode_ids, episode_lengths, 'green', linewidth=2, marker='s', markersize=4, alpha=0.8)
            axes[0, 1].set_xlabel('Episode Number', fontsize=12)
            axes[0, 1].set_ylabel('Episode Length (Steps)', fontsize=12)
            axes[0, 1].set_title('Episode Length Evolution', fontsize=14, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add statistics
            avg_length = np.mean(episode_lengths)
            axes[0, 1].axhline(y=avg_length, color='red', linestyle='--', 
                              label=f'Average: {avg_length:.1f} steps')
            axes[0, 1].legend()
            
            # 3. Performance Distribution
            reward_bins = 20
            axes[1, 0].hist(total_rewards, bins=reward_bins, alpha=0.7, edgecolor='black', color='skyblue')
            axes[1, 0].axvline(np.mean(total_rewards), color='red', linestyle='--', linewidth=2,
                              label=f'Mean: {np.mean(total_rewards):.2f}')
            axes[1, 0].axvline(np.median(total_rewards), color='orange', linestyle='--', linewidth=2,
                              label=f'Median: {np.median(total_rewards):.2f}')
            axes[1, 0].set_xlabel('Total Episode Reward', fontsize=12)
            axes[1, 0].set_ylabel('Frequency', fontsize=12)
            axes[1, 0].set_title('Episode Reward Distribution', fontsize=14, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Learning Progress Analysis
            if len(total_rewards) > 10:
                # Calculate learning metrics
                first_quarter = total_rewards[:len(total_rewards)//4]
                last_quarter = total_rewards[-len(total_rewards)//4:]
                
                improvement = np.mean(last_quarter) - np.mean(first_quarter)
                improvement_pct = (improvement / abs(np.mean(first_quarter))) * 100 if np.mean(first_quarter) != 0 else 0
                
                # Create learning progress visualization
                quarter_size = len(total_rewards) // 4
                quarters = ['Q1', 'Q2', 'Q3', 'Q4']
                quarter_means = []
                quarter_stds = []
                
                for i in range(4):
                    start_idx = i * quarter_size
                    end_idx = (i + 1) * quarter_size if i < 3 else len(total_rewards)
                    quarter_data = total_rewards[start_idx:end_idx]
                    quarter_means.append(np.mean(quarter_data))
                    quarter_stds.append(np.std(quarter_data))
                
                x_pos = np.arange(len(quarters))
                axes[1, 1].bar(x_pos, quarter_means, yerr=quarter_stds, alpha=0.7, 
                              color=['lightcoral', 'lightblue', 'lightgreen', 'gold'],
                              edgecolor='black', capsize=5)
                axes[1, 1].set_xlabel('Training Quarter', fontsize=12)
                axes[1, 1].set_ylabel('Average Episode Reward', fontsize=12)
                axes[1, 1].set_title(f'Learning Progress by Quarter\nImprovement: {improvement:+.2f} ({improvement_pct:+.1f}%)', 
                                    fontsize=14, fontweight='bold')
                axes[1, 1].set_xticks(x_pos)
                axes[1, 1].set_xticklabels(quarters)
                axes[1, 1].grid(True, alpha=0.3)
                
                # Add trend arrow
                if improvement > 0:
                    axes[1, 1].annotate('📈 Improving', xy=(0.7, 0.9), xycoords='axes fraction',
                                       fontsize=12, fontweight='bold', color='green')
                elif improvement < 0:
                    axes[1, 1].annotate('📉 Declining', xy=(0.7, 0.9), xycoords='axes fraction',
                                       fontsize=12, fontweight='bold', color='red')
                else:
                    axes[1, 1].annotate('📊 Stable', xy=(0.7, 0.9), xycoords='axes fraction',
                                       fontsize=12, fontweight='bold', color='blue')
                
                # Add statistics table
                stats_text = f"""Episode Statistics:
Total Episodes: {len(self.episode_metrics)}
Avg Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}
Best Episode: {max(total_rewards):.2f}
Worst Episode: {min(total_rewards):.2f}
Success Rate: {sum(1 for ep in self.episode_metrics if ep.get('episode_success', True))/len(self.episode_metrics)*100:.1f}%"""
                
                axes[1, 1].text(0.02, 0.02, stats_text, transform=axes[1, 1].transAxes,
                               verticalalignment='bottom', fontsize=9,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                axes[1, 1].text(0.5, 0.5, 'Insufficient episode data\nfor learning analysis\n(need >10 episodes)', 
                               ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
            
            plt.tight_layout()
            
            # Save to MinIO
            filename = f"episode_analysis_{self.timestamp}.png"
            return self._save_plot_to_minio(fig, filename)
            
        except Exception as e:
            logger.error(f"EVALUATOR: episode_analysis_failed error={e}")
            return None
    
    def generate_hyperparameter_sensitivity_analysis(self) -> str:
        """Generate hyperparameter sensitivity analysis for research."""
        logger.info("EVALUATOR: generating_hyperparameter_sensitivity_analysis")
        
        try:
            if not self.hyperparameter_runs:
                logger.warning("EVALUATOR: no_hyperparameter_data_available")
                return None
            
            if not self.matplotlib_available:
                logger.warning("EVALUATOR: matplotlib_not_available skipping_hyperparameter_analysis")
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Extract hyperparameter data
            runs = list(self.hyperparameter_runs.values())
            run_ids = list(self.hyperparameter_runs.keys())
            
            # 1. Learning Rate Impact Analysis
            lr_data = {}
            gamma_data = {}
            batch_size_data = {}
            
            for run_id, run_data in self.hyperparameter_runs.items():
                hyperparams = run_data['hyperparams']
                performance = run_data['performance']
                
                # Group by learning rate
                lr = hyperparams.get('learning_rate', 0.001)
                if lr not in lr_data:
                    lr_data[lr] = []
                lr_data[lr].append(performance.get('avg_reward', 0))
                
                # Group by gamma (discount factor)
                gamma = hyperparams.get('gamma', 0.95)
                if gamma not in gamma_data:
                    gamma_data[gamma] = []
                gamma_data[gamma].append(performance.get('avg_reward', 0))
                
                # Group by batch size
                batch_size = hyperparams.get('batch_size', 32)
                if batch_size not in batch_size_data:
                    batch_size_data[batch_size] = []
                batch_size_data[batch_size].append(performance.get('avg_reward', 0))
            
            # Plot Learning Rate Impact
            if len(lr_data) > 1:
                lr_values = sorted(lr_data.keys())
                lr_means = [np.mean(lr_data[lr]) for lr in lr_values]
                lr_stds = [np.std(lr_data[lr]) if len(lr_data[lr]) > 1 else 0 for lr in lr_values]
                
                axes[0, 0].bar(range(len(lr_values)), lr_means, yerr=lr_stds, 
                              alpha=0.7, color='lightblue', edgecolor='black', capsize=5)
                axes[0, 0].set_xlabel('Learning Rate', fontsize=12)
                axes[0, 0].set_ylabel('Average Reward', fontsize=12)
                axes[0, 0].set_title('Learning Rate Impact', fontsize=14, fontweight='bold')
                axes[0, 0].set_xticks(range(len(lr_values)))
                axes[0, 0].set_xticklabels([f'{lr:.4f}' for lr in lr_values])
                axes[0, 0].grid(True, alpha=0.3)
                
                # Highlight best learning rate
                best_lr_idx = np.argmax(lr_means)
                axes[0, 0].bar(best_lr_idx, lr_means[best_lr_idx], 
                              color='gold', edgecolor='black', alpha=0.9)
                axes[0, 0].text(best_lr_idx, lr_means[best_lr_idx] + lr_stds[best_lr_idx], 
                               '★ Best', ha='center', va='bottom', fontweight='bold', color='red')
            else:
                axes[0, 0].text(0.5, 0.5, 'Insufficient learning rate\nvariations for analysis', 
                               ha='center', va='center', transform=axes[0, 0].transAxes)
            
            # Plot Gamma (Discount Factor) Impact
            if len(gamma_data) > 1:
                gamma_values = sorted(gamma_data.keys())
                gamma_means = [np.mean(gamma_data[gamma]) for gamma in gamma_values]
                gamma_stds = [np.std(gamma_data[gamma]) if len(gamma_data[gamma]) > 1 else 0 for gamma in gamma_values]
                
                axes[0, 1].bar(range(len(gamma_values)), gamma_means, yerr=gamma_stds, 
                              alpha=0.7, color='lightgreen', edgecolor='black', capsize=5)
                axes[0, 1].set_xlabel('Discount Factor (γ)', fontsize=12)
                axes[0, 1].set_ylabel('Average Reward', fontsize=12)
                axes[0, 1].set_title('Discount Factor Impact', fontsize=14, fontweight='bold')
                axes[0, 1].set_xticks(range(len(gamma_values)))
                axes[0, 1].set_xticklabels([f'{gamma:.2f}' for gamma in gamma_values])
                axes[0, 1].grid(True, alpha=0.3)
                
                # Highlight best gamma
                best_gamma_idx = np.argmax(gamma_means)
                axes[0, 1].bar(best_gamma_idx, gamma_means[best_gamma_idx], 
                              color='gold', edgecolor='black', alpha=0.9)
                axes[0, 1].text(best_gamma_idx, gamma_means[best_gamma_idx] + gamma_stds[best_gamma_idx], 
                               '★ Best', ha='center', va='bottom', fontweight='bold', color='red')
            else:
                axes[0, 1].text(0.5, 0.5, 'Insufficient gamma\nvariations for analysis', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
            
            # Plot Batch Size Impact
            if len(batch_size_data) > 1:
                batch_values = sorted(batch_size_data.keys())
                batch_means = [np.mean(batch_size_data[batch]) for batch in batch_values]
                batch_stds = [np.std(batch_size_data[batch]) if len(batch_size_data[batch]) > 1 else 0 for batch in batch_values]
                
                axes[1, 0].bar(range(len(batch_values)), batch_means, yerr=batch_stds, 
                              alpha=0.7, color='lightcoral', edgecolor='black', capsize=5)
                axes[1, 0].set_xlabel('Batch Size', fontsize=12)
                axes[1, 0].set_ylabel('Average Reward', fontsize=12)
                axes[1, 0].set_title('Batch Size Impact', fontsize=14, fontweight='bold')
                axes[1, 0].set_xticks(range(len(batch_values)))
                axes[1, 0].set_xticklabels([str(batch) for batch in batch_values])
                axes[1, 0].grid(True, alpha=0.3)
                
                # Highlight best batch size
                best_batch_idx = np.argmax(batch_means)
                axes[1, 0].bar(best_batch_idx, batch_means[best_batch_idx], 
                              color='gold', edgecolor='black', alpha=0.9)
                axes[1, 0].text(best_batch_idx, batch_means[best_batch_idx] + batch_stds[best_batch_idx], 
                               '★ Best', ha='center', va='bottom', fontweight='bold', color='red')
            else:
                axes[1, 0].text(0.5, 0.5, 'Insufficient batch size\nvariations for analysis', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
            
            # Overall Hyperparameter Performance Comparison
            if len(runs) > 1:
                run_rewards = [run['performance'].get('avg_reward', 0) for run in runs]
                run_labels = [f"Run {i+1}" for i in range(len(run_rewards))]
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(run_rewards)))
                bars = axes[1, 1].bar(range(len(run_rewards)), run_rewards, 
                                     color=colors, alpha=0.7, edgecolor='black')
                
                axes[1, 1].set_xlabel('Hyperparameter Configuration', fontsize=12)
                axes[1, 1].set_ylabel('Average Reward', fontsize=12)
                axes[1, 1].set_title('Overall Configuration Performance', fontsize=14, fontweight='bold')
                axes[1, 1].set_xticks(range(len(run_labels)))
                axes[1, 1].set_xticklabels(run_labels, rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
                
                # Highlight best configuration
                best_run_idx = np.argmax(run_rewards)
                bars[best_run_idx].set_color('gold')
                bars[best_run_idx].set_edgecolor('red')
                bars[best_run_idx].set_linewidth(2)
                
                # Add configuration details as text
                best_run = runs[best_run_idx]
                best_config = best_run['hyperparams']
                config_text = f"""Best Configuration:
LR: {best_config.get('learning_rate', 'N/A')}
γ: {best_config.get('gamma', 'N/A')}
Batch: {best_config.get('batch_size', 'N/A')}
Reward: {best_run['performance'].get('avg_reward', 0):.3f}"""
                
                axes[1, 1].text(0.02, 0.98, config_text, transform=axes[1, 1].transAxes,
                               verticalalignment='top', fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            else:
                axes[1, 1].text(0.5, 0.5, 'Insufficient configurations\nfor comparison analysis', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
            
            plt.tight_layout()
            
            # Save to MinIO
            filename = f"hyperparameter_sensitivity_{self.timestamp}.png"
            return self._save_plot_to_minio(fig, filename)
            
        except Exception as e:
            logger.error(f"EVALUATOR: hyperparameter_sensitivity_failed error={e}")
            return None
    
    def generate_feature_architecture_analysis(self) -> str:
        """Generate dedicated feature architecture analysis."""
        logger.info("EVALUATOR: generating_feature_architecture_analysis")
        
        try:
            if not self.matplotlib_available:
                logger.warning("EVALUATOR: matplotlib_not_available skipping_feature_architecture")
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Feature Architecture Breakdown
            self._plot_feature_architecture(axes[0, 0])
            
            # 2. Feature Type Comparison
            consumer_count = 9
            total_count = 9
            
            categories = ['Consumer\nPerformance', 'Total\nFeatures']
            values = [consumer_count, total_count]
            colors = ['#FF6B6B', '#4ECDC4']
            
            bars = axes[0, 1].bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
            axes[0, 1].set_ylabel('Feature Count')
            axes[0, 1].set_title('Feature Count: Consumer Performance Focus', fontweight='bold')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                               f'{value}', ha='center', fontweight='bold', fontsize=12)
            
            axes[0, 1].set_ylim(0, max(values) + 2)
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Data Flow Diagram
            axes[1, 0].text(0.5, 0.9, 'CONSUMER PERFORMANCE DATA FLOW', ha='center', fontweight='bold', 
                           fontsize=14, transform=axes[1, 0].transAxes)
            
            flow_text = """
            ┌─────────────────────┐
            │  CONSUMER PODS      │
            │   (Port 8000)       │
            │                     │
            │ • CPU Rate          │
            │ • GC Activity       │
            │ • HTTP Metrics      │
            │ • File Descriptors  │
            │ • Response Sizes    │
            └─────────────────────┘
                    │
                    ▼
            ┌─────────────────────┐
            │   PROMETHEUS        │
            │ (Scrape & Store)    │
            └─────────────────────┘
                    │
                    ▼
            ┌─────────────────────┐
            │  ROBUST SCALER      │
            │ (9 Features)        │
            └─────────────────────┘
                    │
                    ▼
            ┌─────────────────────┐
            │   DQN MODEL         │
            │ (Consumer Focused)  │
            │ Scientifically      │
            │   Selected          │
            └─────────────────────┘
            """
            
            axes[1, 0].text(0.5, 0.5, flow_text, ha='center', va='center',
                           transform=axes[1, 0].transAxes, fontsize=9, family='monospace',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
            axes[1, 0].axis('off')
            
            # 4. Benefits of Consumer Performance Focus
            benefits = [
                'Direct Performance Impact',
                'Real-time Consumer Metrics', 
                'Optimal Scaling Decisions',
                'Scientific Feature Selection',
                'Efficient Resource Usage'
            ]
            
            y_positions = np.arange(len(benefits))
            colors_benefits = plt.cm.Set3(np.linspace(0, 1, len(benefits)))
            
            bars = axes[1, 1].barh(y_positions, [1]*len(benefits), color=colors_benefits, alpha=0.8)
            axes[1, 1].set_yticks(y_positions)
            axes[1, 1].set_yticklabels(benefits)
            axes[1, 1].set_xlabel('Consumer Performance Benefits')
            axes[1, 1].set_title('Benefits of Consumer-Focused Architecture', fontweight='bold')
            axes[1, 1].set_xlim(0, 1.2)
            
            # Remove x-axis ticks for cleaner look
            axes[1, 1].set_xticks([])
            
            # Add checkmarks
            for i, benefit in enumerate(benefits):
                axes[1, 1].text(1.05, i, '✓', fontsize=16, fontweight='bold', 
                               color='green', ha='center', va='center')
            
            plt.tight_layout()
            
            # Save to MinIO
            filename = f"consumer_feature_architecture_{self.timestamp}.png"
            return self._save_plot_to_minio(fig, filename)
            
        except Exception as e:
            logger.error(f"EVALUATOR: feature_architecture_analysis_failed error={e}")
            return None
    
    def generate_performance_dashboard(self, model_state: Dict[str, Any]) -> str:
        """Generate comprehensive performance dashboard with clean architecture visualization."""
        logger.info("EVALUATOR: generating_performance_dashboard")
        
        try:
            if not self.matplotlib_available:
                logger.warning("EVALUATOR: matplotlib_not_available skipping_performance_dashboard")
                return None
            
            fig = plt.figure(figsize=(24, 20))
            gs = fig.add_gridspec(5, 4, hspace=0.4, wspace=0.3)
            
            # 1. Feature Architecture Breakdown (NEW)
            ax1 = fig.add_subplot(gs[0, :2])
            self._plot_feature_architecture(ax1)
            
            # 2. Model Architecture Visualization
            ax2 = fig.add_subplot(gs[0, 2:])
            self._plot_model_architecture(ax2, model_state)
            
            # 3. Training Progress
            ax3 = fig.add_subplot(gs[1, :2])
            self._plot_training_progress(ax3)
            
            # 4. Experience Statistics
            ax4 = fig.add_subplot(gs[1, 2:])
            self._plot_experience_statistics(ax4)
            
            # 5. Reward Analysis
            ax5 = fig.add_subplot(gs[2, :2])
            self._plot_reward_analysis(ax5)
            
            # 6. System Metrics (Updated for raw features)
            ax6 = fig.add_subplot(gs[2, 2:])
            self._plot_system_metrics(ax6)
            
            # 7. Decision Confidence
            ax7 = fig.add_subplot(gs[3, :2])
            self._plot_decision_confidence(ax7)
            
            # 8. Performance Summary Table  
            ax8 = fig.add_subplot(gs[4, :])
            self._create_performance_summary_table(ax8, model_state)
            
            # Save to MinIO
            filename = f"performance_dashboard_{self.timestamp}.png"
            return self._save_plot_to_minio(fig, filename)
            
        except Exception as e:
            logger.error(f"EVALUATOR: performance_dashboard_failed error={e}")
            return None
    
    def _plot_model_architecture(self, ax, model_state):
        """Plot model architecture diagram for consumer performance features."""
        layers = ['Input\n(9)\nConsumer Performance\nFeatures', 'Hidden 1\n(64)', 'Hidden 2\n(32)', 'Output\n(3)\nScaling Actions']
        sizes = [9, 64, 32, 3]
        colors = ['red', 'orange', 'yellow', 'green']
        
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
        ax.set_title('DQN Architecture: Consumer Performance Focus', fontsize=14, fontweight='bold')
    
    def _plot_feature_architecture(self, ax):
        """Plot feature architecture breakdown showing 9 scientifically-selected consumer performance features."""
        # Define feature groups (scientifically-selected consumer performance features)
        consumer_features = [
            'process_cpu_seconds_total_rate',
            'python_gc_collections_total_rate',
            'python_gc_objects_collected_total_rate',
            'http_request_duration_seconds_sum_rate',
            'http_requests_total_rate',
            'http_request_duration_seconds_count_rate',
            'process_open_fds',
            'http_response_size_bytes_sum_rate',
            'http_request_size_bytes_count_rate'
        ]
        
        # Create bar chart showing feature list
        y_positions = range(len(consumer_features))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98FB98', '#F0E68C', '#FFB6C1']
        
        bars = ax.barh(y_positions, [1]*len(consumer_features), color=colors, alpha=0.8)
        
        # Create readable labels
        feature_labels = [
            'CPU Usage Rate',
            'GC Collections Rate', 
            'GC Objects Rate',
            'HTTP Duration Rate',
            'HTTP Requests Rate',
            'HTTP Count Rate',
            'Open File Descriptors',
            'Response Size Rate',
            'Request Count Rate'
        ]
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(feature_labels, fontsize=10)
        ax.set_xlabel('Feature Importance')
        ax.set_title('Consumer Performance Features: 9 Scientifically-Selected Metrics', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(0, 1.2)
        ax.set_xticks([])
        
        # Add checkmarks for each feature
        for i in range(len(consumer_features)):
            ax.text(1.05, i, '✓', fontsize=16, fontweight='bold', 
                   color='green', ha='center', va='center')
        
        # Add feature description text below
        feature_text = (
            "Consumer Performance Features (Scientifically Selected):\n" + 
            "• CPU utilization rate (current usage)\n" +
            "• Memory pressure rates (GC activity)\n" +
            "• HTTP performance rates (request processing)\n" +
            "• I/O load indicators (file descriptors)\n" +
            "• Network payload rates (response/request sizes)"
        )
        
        ax.text(0.5, -0.3, feature_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
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
        """Plot system metrics from experiences using new raw features."""
        if not self.experiences:
            ax.text(0.5, 0.5, 'No system data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return
        
        # Extract balanced selected feature metrics
        http_bucket_metrics = []
        memory_metrics = []
        scrape_health_metrics = []
        
        for exp in self.experiences:
            if 'state' in exp:
                # Use our balanced selected features
                if 'http_request_duration_highr_seconds_bucket' in exp['state']:
                    http_bucket_metrics.append(exp['state']['http_request_duration_highr_seconds_bucket'])
                if 'process_resident_memory_bytes' in exp['state']:
                    memory_metrics.append(exp['state']['process_resident_memory_bytes'] / 1000000)  # Convert to MB
                if 'scrape_samples_scraped' in exp['state']:
                    scrape_health_metrics.append(exp['state']['scrape_samples_scraped'])
        
        if http_bucket_metrics:
            ax.plot(http_bucket_metrics, 'purple', linewidth=2, label='HTTP Bucket Count', alpha=0.8)
            ax.set_xlabel('Decision Number')
            ax.set_ylabel('HTTP Bucket Count', color='purple')
            ax.set_title('System Performance Trends (Balanced Features)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add secondary y-axis for memory usage if available
            if memory_metrics and len(memory_metrics) == len(http_bucket_metrics):
                ax2 = ax.twinx()
                ax2.plot(memory_metrics, 'green', linewidth=2, label='Memory Usage (MB)', alpha=0.8)
                ax2.set_ylabel('Memory Usage (MB)', color='green')
                
                # Add legend
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, ['HTTP Bucket Count', 'Memory Usage (MB)'], loc='upper left')
        elif memory_metrics:
            ax.plot(memory_metrics, 'green', linewidth=2)
            ax.set_xlabel('Decision Number')
            ax.set_ylabel('Memory Usage (MB)')
            ax.set_title('Memory Usage Trend', fontweight='bold')
            ax.grid(True, alpha=0.3)
        elif scrape_health_metrics:
            ax.plot(scrape_health_metrics, 'orange', linewidth=2)
            ax.set_xlabel('Decision Number')
            ax.set_ylabel('Scrape Samples')
            ax.set_title('Health Monitoring Trend', fontweight='bold')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No balanced feature data available\nfor system metrics', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
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
        
        # Create table data with consumer performance architecture info
        data = [
            ['Feature Architecture', '9 Consumer Performance Metrics (Scientifically Selected)'],
            ['Consumer Features', '9 (CPU, Memory, HTTP, I/O Performance)'],
            ['Feature Source', 'Consumer Pod Metrics (Port 8000)'],
            ['Architecture Focus', 'Direct Performance Impact'],
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
        ax.set_title('Performance Summary: Consumer-Focused DQN', fontsize=14, fontweight='bold', pad=20)
    
    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report."""
        logger.info("EVALUATOR: generating_summary_report")
        
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
        
        logger.info(f"EVALUATOR: summary_saved file={report_filename}")
        return report_filename
    
    def _save_plot_to_minio(self, fig, filename: str) -> str:
        """Save matplotlib figure to MinIO with robust error handling."""
        buffer = None
        try:
            logger.info(f"EVALUATOR: saving_plot filename={filename}")
            
            # Validate inputs
            if fig is None:
                logger.error("EVALUATOR: figure_is_none cannot_save")
                return None
                
            if not filename:
                logger.error("EVALUATOR: filename_empty cannot_save")
                return None
            
            # Create buffer and save figure
            buffer = BytesIO()
            
            # Use lower DPI for containers to reduce memory usage
            dpi = 150 if os.getenv('KUBERNETES_SERVICE_HOST') else 300
            
            # Save with error handling
            fig.savefig(
                buffer, 
                format='png', 
                dpi=dpi, 
                bbox_inches='tight', 
                facecolor='white',
                edgecolor='none',
                pad_inches=0.1
            )
            buffer.seek(0)
            
            # Validate buffer
            if buffer.getbuffer().nbytes == 0:
                logger.error("EVALUATOR: empty_buffer generated")
                return None
            
            logger.info(f"EVALUATOR: plot_generated size_bytes={buffer.getbuffer().nbytes}")
            
            # Upload to MinIO with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.minio_client.put_object(
                        self.bucket_name,
                        filename,
                        data=buffer,
                        length=buffer.getbuffer().nbytes,
                        content_type='image/png'
                    )
                    logger.info(f"EVALUATOR: plot_uploaded_successfully filename={filename} attempt={attempt + 1}")
                    break
                except Exception as upload_error:
                    logger.warning(f"EVALUATOR: upload_attempt_failed attempt={attempt + 1} error={upload_error}")
                    if attempt == max_retries - 1:
                        raise upload_error
                    time.sleep(1)  # Brief delay before retry
            
            return filename
            
        except Exception as e:
            logger.error(f"EVALUATOR: plot_save_failed filename={filename} error={e}")
            return None
            
        finally:
            # Clean up resources
            try:
                if buffer:
                    buffer.close()
                if fig:
                    plt.close(fig)  # Free memory
                    logger.debug("EVALUATOR: plot_memory_freed")
            except Exception as cleanup_error:
                logger.warning(f"EVALUATOR: cleanup_failed error={cleanup_error}")
    
    def generate_comprehensive_research_dashboard(self) -> List[str]:
        """Generate all research-focused analyses for publication-quality results."""
        logger.info("EVALUATOR: generating_comprehensive_research_dashboard")
        
        research_outputs = []
        
        try:
            if not self.matplotlib_available:
                logger.warning("EVALUATOR: matplotlib_not_available skipping_research_dashboard")
                return research_outputs
            
            # Research-focused analysis generators
            research_generators = [
                ("q_value_convergence", self.generate_q_value_analysis),
                ("episode_performance", self.generate_episode_analysis),
                ("hyperparameter_optimization", self.generate_hyperparameter_sensitivity_analysis),
                ("learning_dynamics", self.generate_learning_curve_analysis),
                ("system_architecture", self.generate_feature_architecture_analysis)
            ]
            
            logger.info(f"EVALUATOR: research_data_status episodes={len(self.episode_metrics)} "
                       f"q_values={len(self.q_value_history)} "
                       f"hyperparameter_runs={len(self.hyperparameter_runs)}")
            
            # Generate each research analysis
            for analysis_name, generator_func in research_generators:
                try:
                    logger.info(f"EVALUATOR: generating_research_analysis type={analysis_name}")
                    result = generator_func()
                    
                    if result:
                        research_outputs.append(result)
                        logger.info(f"EVALUATOR: research_analysis_success type={analysis_name} file={result}")
                    else:
                        logger.warning(f"EVALUATOR: research_analysis_skipped type={analysis_name} reason=no_data_or_error")
                        
                except Exception as e:
                    logger.error(f"EVALUATOR: research_analysis_failed type={analysis_name} error={e}")
            
            # Generate summary of research findings
            if research_outputs:
                try:
                    summary_file = self.generate_summary_report()
                    if summary_file:
                        research_outputs.append(summary_file)
                        logger.info("EVALUATOR: research_summary_generated")
                except Exception as e:
                    logger.error(f"EVALUATOR: research_summary_failed error={e}")
            
            logger.info(f"EVALUATOR: research_dashboard_complete generated={len(research_outputs)} files")
            
            return research_outputs
            
        except Exception as e:
            logger.error(f"EVALUATOR: research_dashboard_failed error={e}")
            return research_outputs
    
    def generate_all_outputs(self, model_state: Dict[str, Any]) -> List[str]:
        """Generate all evaluation outputs and return list of saved files."""
        logger.info("EVALUATOR: generating_comprehensive_evaluation")
        
        saved_files = []
        failed_outputs = []
        
        try:
            # Check system readiness
            logger.info(f"EVALUATOR: system_status experiences={len(self.experiences)} "
                       f"training_metrics={len(self.training_metrics)} "
                       f"matplotlib_available={self.matplotlib_available}")
            
            if not self.matplotlib_available:
                logger.warning("EVALUATOR: matplotlib_not_available skipping_all_visualizations")
                # Still try to generate JSON report
                try:
                    json_file = self.generate_summary_report()
                    if json_file:
                        saved_files.append(json_file)
                        logger.info("EVALUATOR: json_report_generated_successfully")
                except Exception as json_error:
                    logger.error(f"EVALUATOR: json_report_failed error={json_error}")
                    failed_outputs.append("summary_report")
                return saved_files
            
            # Define all output generators with their names - ENHANCED FOR RESEARCH
            output_generators = [
                # Core Analysis
                ("learning_curves", self.generate_learning_curve_analysis),
                ("action_analysis", self.generate_action_distribution_analysis), 
                ("feature_architecture", self.generate_feature_architecture_analysis),
                ("performance_dashboard", lambda: self.generate_performance_dashboard(model_state)),
                
                # RESEARCH ENHANCEMENTS: Advanced DQN Analysis
                ("q_value_analysis", self.generate_q_value_analysis),
                ("episode_analysis", self.generate_episode_analysis),
                ("hyperparameter_sensitivity", self.generate_hyperparameter_sensitivity_analysis),
                
                # Final Summary
                ("summary_report", self.generate_summary_report)
            ]
            
            # Generate each output with individual error handling
            for output_name, generator_func in output_generators:
                try:
                    logger.info(f"EVALUATOR: generating output_type={output_name}")
                    result = generator_func()
                    
                    if result:
                        saved_files.append(result)
                        logger.info(f"EVALUATOR: output_success type={output_name} file={result}")
                    else:
                        failed_outputs.append(output_name)
                        logger.warning(f"EVALUATOR: output_returned_none type={output_name}")
                        
                except Exception as output_error:
                    failed_outputs.append(output_name)
                    logger.error(f"EVALUATOR: output_failed type={output_name} error={output_error}")
            
            # Log final results
            logger.info(f"EVALUATOR: evaluation_complete "
                       f"successful_outputs={len(saved_files)} "
                       f"failed_outputs={len(failed_outputs)}")
            
            if saved_files:
                logger.info("EVALUATOR: successful_files:")
                for file in saved_files:
                    logger.info(f"  - {file}")
            
            if failed_outputs:
                logger.warning("EVALUATOR: failed_outputs:")
                for output in failed_outputs:
                    logger.warning(f"  - {output}")
            
            return saved_files
            
        except Exception as e:
            logger.error(f"EVALUATOR: comprehensive_evaluation_failed error={e}")
            return saved_files  # Return whatever we managed to generate
    
    def cleanup_corrupted_data(self):
        """Clean up corrupted training metrics data."""
        try:
            if not self.training_metrics:
                logger.info("EVALUATOR: no_training_metrics_to_cleanup")
                return
            
            original_count = len(self.training_metrics)
            cleaned_metrics = []
            
            for metrics in self.training_metrics:
                try:
                    # Validate each metric entry
                    loss = metrics.get('loss', 0)
                    epsilon = metrics.get('epsilon', 0)
                    batch_size = metrics.get('batch_size', 32)
                    buffer_size = metrics.get('buffer_size', 0)
                    timestamp = metrics.get('timestamp', time.time())
                    
                    # Check if all values are valid
                    if (isinstance(loss, (int, float)) and np.isfinite(loss) and 0 <= loss <= 1000 and
                        isinstance(epsilon, (int, float)) and 0 <= epsilon <= 1 and
                        isinstance(batch_size, int) and 1 <= batch_size <= 1000 and
                        isinstance(buffer_size, int) and 0 <= buffer_size <= 100000 and
                        isinstance(timestamp, (int, float)) and timestamp > 0):
                        
                        cleaned_metrics.append(metrics)
                
                except Exception:
                    continue  # Skip corrupted entries
            
            # Replace with cleaned data
            self.training_metrics = cleaned_metrics
            removed_count = original_count - len(cleaned_metrics)
            
            logger.info(f"EVALUATOR: data_cleanup_complete "
                       f"original_count={original_count} "
                       f"cleaned_count={len(cleaned_metrics)} "
                       f"removed_count={removed_count}")
            
        except Exception as e:
            logger.error(f"EVALUATOR: cleanup_failed error={e}")
    
    def get_data_quality_stats(self) -> Dict[str, Any]:
        """Get data quality statistics for debugging."""
        try:
            if not self.training_metrics:
                return {"status": "no_data"}
            
            losses = [m['loss'] for m in self.training_metrics]
            buffer_sizes = [m['buffer_size'] for m in self.training_metrics]
            epsilons = [m['epsilon'] for m in self.training_metrics]
            timestamps = [m['timestamp'] for m in self.training_metrics]
            
            # Calculate statistics
            stats = {
                "total_metrics": len(self.training_metrics),
                "loss_stats": {
                    "min": float(np.min(losses)),
                    "max": float(np.max(losses)),
                    "mean": float(np.mean(losses)),
                    "std": float(np.std(losses)),
                    "outliers": int(np.sum((np.array(losses) > np.mean(losses) + 3*np.std(losses)) | 
                                         (np.array(losses) < np.mean(losses) - 3*np.std(losses))))
                },
                "buffer_stats": {
                    "min": int(np.min(buffer_sizes)),
                    "max": int(np.max(buffer_sizes)),
                    "growth": int(np.max(buffer_sizes) - np.min(buffer_sizes)),
                    "unique_values": len(set(buffer_sizes))
                },
                "epsilon_stats": {
                    "min": float(np.min(epsilons)),
                    "max": float(np.max(epsilons)),
                    "decay_trend": float(epsilons[-1] - epsilons[0]) if len(epsilons) > 1 else 0.0
                },
                "time_stats": {
                    "duration_minutes": float((np.max(timestamps) - np.min(timestamps)) / 60),
                    "avg_interval_seconds": float(np.mean(np.diff(timestamps))) if len(timestamps) > 1 else 0.0
                }
            }
            
            logger.info(f"EVALUATOR: data_quality_stats {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"EVALUATOR: data_quality_stats_failed error={e}")
            return {"status": "error", "error": str(e)} 