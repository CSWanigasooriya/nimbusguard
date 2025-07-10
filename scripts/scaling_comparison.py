#!/usr/bin/env python3
"""
SCALING APPROACHES COMPARISON ANALYSIS
======================================

This script provides comprehensive comparison between three autoscaling approaches:
1. DQN-based intelligent scaling (NimbusGuard)
2. Traditional HPA (Horizontal Pod Autoscaler)
3. KEDA event-driven scaling

The analysis includes:
- Performance metrics comparison
- Resource utilization efficiency
- Scaling decision patterns
- Response time analysis
- Cost-effectiveness evaluation
- System stability metrics

Generated outputs suitable for research papers and presentations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.shadow': False,
    'legend.framealpha': 1.0,
    'legend.edgecolor': 'black',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

class ScalingComparisonAnalyzer:
    """Comprehensive analysis and comparison of different scaling approaches."""
    
    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define scaling approach directories
        self.dqn_dir = self.data_dir / "prometheus_data_dqn"
        self.hpa_dir = self.data_dir / "prometheus_data_hpa"
        self.keda_dir = self.data_dir / "prometheus_data_keda"
        
        # Professional color scheme for research papers
        self.colors = {
            'dqn': '#1f77b4',      # Blue - NimbusGuard DQN
            'hpa': '#ff7f0e',      # Orange - Traditional HPA
            'keda': '#2ca02c',     # Green - KEDA
            'accent': '#d62728',   # Red - Highlights
            'secondary': '#9467bd', # Purple - Secondary metrics
            'gray': '#7f7f7f'      # Gray - Neutral
        }
        
        print(f"📊 Scaling Comparison Analyzer initialized")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        
    def load_prometheus_data(self, data_dir: Path, approach_name: str) -> dict:
        """Load and process Prometheus data from CSV files."""
        if not data_dir.exists():
            print(f"⚠️ Directory not found: {data_dir}")
            return {}
            
        print(f"📈 Loading {approach_name} data from {data_dir}")
        
        # Key metrics to extract
        key_metrics = {
            'replicas': 'kube_deployment_status_replicas.csv',
            'desired_replicas': 'kube_deployment_spec_replicas.csv',
            'unavailable_replicas': 'kube_deployment_status_replicas_unavailable.csv',
            'pod_ready': 'kube_pod_container_status_ready.csv',
            'cpu_usage': 'kube_pod_container_resource_limits_cpu.csv',
            'memory_usage': 'kube_pod_container_resource_limits_memory.csv',
            'container_running': 'kube_pod_container_status_running.csv',
            'network_up': 'node_network_up.csv'
        }
        
        # DQN-specific metrics
        if approach_name.lower() == 'dqn':
            dqn_metrics = {
                'dqn_decisions': 'dqn_decisions_total.csv',
                'dqn_actions_scale_up': 'dqn_action_scale_up_total.csv',
                'dqn_actions_scale_down': 'dqn_action_scale_down_total.csv',
                'dqn_actions_keep_same': 'dqn_action_keep_same_total.csv',
                'dqn_reward': 'dqn_reward_total.csv',
                'dqn_confidence': 'dqn_decision_confidence_avg.csv',
                'lstm_pressure_30s': 'dqn_lstm_next_30sec_pressure.csv',
                'lstm_pressure_60s': 'dqn_lstm_next_60sec_pressure.csv'
            }
            key_metrics.update(dqn_metrics)
        
        data = {}
        for metric_name, filename in key_metrics.items():
            file_path = data_dir / filename
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    if not df.empty and 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.sort_values('timestamp')
                        # Take mean value if multiple entries per timestamp
                        if 'value' in df.columns:
                            df = df.groupby('timestamp')['value'].mean().reset_index()
                        data[metric_name] = df
                        print(f"  ✓ Loaded {metric_name}: {len(df)} samples")
                    else:
                        print(f"  ⚠️ Empty or invalid data: {filename}")
                except Exception as e:
                    print(f"  ❌ Error loading {filename}: {e}")
            else:
                print(f"  ⚠️ File not found: {filename}")
        
        return data
    
    def calculate_performance_metrics(self, data: dict, approach_name: str) -> dict:
        """Calculate key performance metrics for each scaling approach."""
        metrics = {
            'approach': approach_name,
            'total_samples': 0,
            'avg_replicas': 0,
            'avg_desired_replicas': 0,
            'avg_unavailable_replicas': 0,
            'avg_pod_readiness': 0,
            'avg_cpu_per_replica': 0,
            'avg_memory_per_replica': 0,
            'scaling_stability': 0,
            'resource_efficiency': 0,
            'availability_score': 0
        }
        
        if not data:
            return metrics
        
        # Basic replica metrics
        if 'replicas' in data:
            replicas_df = data['replicas']
            metrics['total_samples'] = len(replicas_df)
            metrics['avg_replicas'] = replicas_df['value'].mean()
            
            # Calculate scaling stability (lower variance = more stable)
            if len(replicas_df) > 1:
                replica_changes = abs(replicas_df['value'].diff()).sum()
                metrics['scaling_stability'] = 1.0 / (1.0 + replica_changes / len(replicas_df))
        
        if 'desired_replicas' in data:
            metrics['avg_desired_replicas'] = data['desired_replicas']['value'].mean()
        
        if 'unavailable_replicas' in data:
            metrics['avg_unavailable_replicas'] = data['unavailable_replicas']['value'].mean()
        
        if 'pod_ready' in data:
            metrics['avg_pod_readiness'] = data['pod_ready']['value'].mean()
            # Availability score based on pod readiness
            metrics['availability_score'] = min(1.0, metrics['avg_pod_readiness'])
        
        # Resource efficiency calculations
        if 'cpu_usage' in data and 'replicas' in data:
            cpu_df = data['cpu_usage']
            replica_df = data['replicas']
            
            # Merge on timestamp for accurate per-replica calculations
            merged = pd.merge(cpu_df, replica_df, on='timestamp', suffixes=('_cpu', '_replicas'))
            if not merged.empty:
                merged['cpu_per_replica'] = merged['value_cpu'] / np.maximum(1, merged['value_replicas'])
                metrics['avg_cpu_per_replica'] = merged['cpu_per_replica'].mean()
        
        if 'memory_usage' in data and 'replicas' in data:
            memory_df = data['memory_usage']
            replica_df = data['replicas']
            
            merged = pd.merge(memory_df, replica_df, on='timestamp', suffixes=('_memory', '_replicas'))
            if not merged.empty:
                merged['memory_per_replica'] = merged['value_memory'] / np.maximum(1, merged['value_replicas'])
                metrics['avg_memory_per_replica'] = merged['memory_per_replica'].mean() / 1e6  # Convert to MB
        
        # Resource efficiency score (normalized)
        optimal_cpu_range = (0.5, 2.0)  # Optimal CPU cores per replica
        optimal_memory_range = (100, 500)  # Optimal MB per replica
        
        cpu_efficiency = 1.0
        if metrics['avg_cpu_per_replica'] > 0:
            if optimal_cpu_range[0] <= metrics['avg_cpu_per_replica'] <= optimal_cpu_range[1]:
                cpu_efficiency = 1.0
            else:
                cpu_efficiency = max(0.1, 1.0 - abs(metrics['avg_cpu_per_replica'] - 1.25) * 0.5)
        
        memory_efficiency = 1.0
        if metrics['avg_memory_per_replica'] > 0:
            if optimal_memory_range[0] <= metrics['avg_memory_per_replica'] <= optimal_memory_range[1]:
                memory_efficiency = 1.0
            else:
                memory_efficiency = max(0.1, 1.0 - abs(metrics['avg_memory_per_replica'] - 300) * 0.002)
        
        metrics['resource_efficiency'] = (cpu_efficiency + memory_efficiency) / 2.0
        
        # DQN-specific metrics
        if approach_name.lower() == 'dqn':
            if 'dqn_decisions' in data:
                metrics['total_dqn_decisions'] = data['dqn_decisions']['value'].iloc[-1] if not data['dqn_decisions'].empty else 0
            
            if 'dqn_confidence' in data:
                metrics['avg_decision_confidence'] = data['dqn_confidence']['value'].mean()
            
            if 'dqn_reward' in data:
                metrics['avg_reward'] = data['dqn_reward']['value'].mean()
        
        return metrics
    
    def create_time_series_comparison(self, dqn_data, hpa_data, keda_data):
        """Create time series comparison charts with proper handling of different evaluation periods."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Scaling Approaches: Individual Time Series Analysis\n(Note: Different evaluation periods)', fontsize=16, fontweight='bold')
        
        # 1. Individual Replica Count Time Series
        ax1 = axes[0, 0]
        
        # Create normalized time series (minutes from start)
        def normalize_timestamps(df):
            if df.empty:
                return df
            df_copy = df.copy()
            start_time = df_copy['timestamp'].min()
            df_copy['minutes_from_start'] = (df_copy['timestamp'] - start_time).dt.total_seconds() / 60
            return df_copy
        
        # Plot each approach separately with normalized time
        if 'replicas' in dqn_data and not dqn_data['replicas'].empty:
            df_norm = normalize_timestamps(dqn_data['replicas'])
            ax1.plot(df_norm['minutes_from_start'], df_norm['value'], 
                    color=self.colors['dqn'], label='DQN (30min)', linewidth=2, alpha=0.8)
        
        if 'replicas' in hpa_data and not hpa_data['replicas'].empty:
            df_norm = normalize_timestamps(hpa_data['replicas'])
            ax1.plot(df_norm['minutes_from_start'], df_norm['value'], 
                    color=self.colors['hpa'], label='HPA (longer)', linewidth=2, alpha=0.8)
        
        if 'replicas' in keda_data and not keda_data['replicas'].empty:
            df_norm = normalize_timestamps(keda_data['replicas'])
            ax1.plot(df_norm['minutes_from_start'], df_norm['value'], 
                    color=self.colors['keda'], label='KEDA (longest)', linewidth=2, alpha=0.8)
        
        ax1.set_title('Replica Count Over Time (Normalized)', fontweight='bold')
        ax1.set_xlabel('Minutes from Start')
        ax1.set_ylabel('Number of Replicas')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Scaling Behavior Patterns (Distribution)
        ax2 = axes[0, 1]
        
        replica_data = []
        labels = []
        colors = []
        
        if 'replicas' in dqn_data and not dqn_data['replicas'].empty:
            replica_data.append(dqn_data['replicas']['value'].values)
            labels.append('DQN')
            colors.append(self.colors['dqn'])
        
        if 'replicas' in hpa_data and not hpa_data['replicas'].empty:
            replica_data.append(hpa_data['replicas']['value'].values)
            labels.append('HPA')
            colors.append(self.colors['hpa'])
        
        if 'replicas' in keda_data and not keda_data['replicas'].empty:
            replica_data.append(keda_data['replicas']['value'].values)
            labels.append('KEDA')
            colors.append(self.colors['keda'])
        
        if replica_data:
            bp = ax2.boxplot(replica_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax2.set_title('Replica Count Distribution', fontweight='bold')
        ax2.set_ylabel('Number of Replicas')
        ax2.grid(True, alpha=0.3)
        
        # 3. Availability Comparison (Normalized)
        ax3 = axes[1, 0]
        
        if 'pod_ready' in dqn_data and not dqn_data['pod_ready'].empty:
            df_norm = normalize_timestamps(dqn_data['pod_ready'])
            ax3.plot(df_norm['minutes_from_start'], df_norm['value'], 
                    color=self.colors['dqn'], label='DQN', linewidth=2, alpha=0.8)
        
        if 'pod_ready' in hpa_data and not hpa_data['pod_ready'].empty:
            df_norm = normalize_timestamps(hpa_data['pod_ready'])
            ax3.plot(df_norm['minutes_from_start'], df_norm['value'], 
                    color=self.colors['hpa'], label='HPA', linewidth=2, alpha=0.8)
        
        if 'pod_ready' in keda_data and not keda_data['pod_ready'].empty:
            df_norm = normalize_timestamps(keda_data['pod_ready'])
            ax3.plot(df_norm['minutes_from_start'], df_norm['value'], 
                    color=self.colors['keda'], label='KEDA', linewidth=2, alpha=0.8)
        
        ax3.set_title('Pod Readiness Over Time (Normalized)', fontweight='bold')
        ax3.set_xlabel('Minutes from Start')
        ax3.set_ylabel('Pod Readiness Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Evaluation Period Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create evaluation period summary
        summary_text = "EVALUATION PERIODS:\n\n"
        
        if 'replicas' in dqn_data and not dqn_data['replicas'].empty:
            dqn_duration = (dqn_data['replicas']['timestamp'].max() - 
                          dqn_data['replicas']['timestamp'].min()).total_seconds() / 60
            dqn_samples = len(dqn_data['replicas'])
            summary_text += f"🔵 DQN (NimbusGuard):\n"
            summary_text += f"   Duration: {dqn_duration:.1f} minutes\n"
            summary_text += f"   Samples: {dqn_samples}\n"
            summary_text += f"   Avg Replicas: {dqn_data['replicas']['value'].mean():.2f}\n\n"
        
        if 'replicas' in hpa_data and not hpa_data['replicas'].empty:
            hpa_duration = (hpa_data['replicas']['timestamp'].max() - 
                          hpa_data['replicas']['timestamp'].min()).total_seconds() / 60
            hpa_samples = len(hpa_data['replicas'])
            summary_text += f"🟠 Traditional HPA:\n"
            summary_text += f"   Duration: {hpa_duration:.1f} minutes\n"
            summary_text += f"   Samples: {hpa_samples}\n"
            summary_text += f"   Avg Replicas: {hpa_data['replicas']['value'].mean():.2f}\n\n"
        
        if 'replicas' in keda_data and not keda_data['replicas'].empty:
            keda_duration = (keda_data['replicas']['timestamp'].max() - 
                           keda_data['replicas']['timestamp'].min()).total_seconds() / 60
            keda_samples = len(keda_data['replicas'])
            summary_text += f"🟢 KEDA:\n"
            summary_text += f"   Duration: {keda_duration:.1f} minutes\n"
            summary_text += f"   Samples: {keda_samples}\n"
            summary_text += f"   Avg Replicas: {keda_data['replicas']['value'].mean():.2f}\n\n"
        
        summary_text += "⚠️ Note: Different evaluation periods\n"
        summary_text += "Comparisons based on aggregate metrics"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        ax4.set_title('Evaluation Period Summary', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "time_series_comparison.png", dpi=300)
        plt.close()
        print("✅ Created normalized time series comparison")
    
    def create_individual_time_series(self, dqn_data, hpa_data, keda_data):
        """Create individual time series for each approach to show actual temporal behavior."""
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('Individual Scaling Approach Time Series\n(Actual Evaluation Periods)', fontsize=16, fontweight='bold')
        
        # 1. DQN Time Series
        ax1 = axes[0]
        if 'replicas' in dqn_data and not dqn_data['replicas'].empty:
            df = dqn_data['replicas']
            ax1.plot(df['timestamp'], df['value'], color=self.colors['dqn'], linewidth=2)
            ax1.fill_between(df['timestamp'], df['value'], alpha=0.3, color=self.colors['dqn'])
            
            # Add statistics
            mean_replicas = df['value'].mean()
            ax1.axhline(mean_replicas, color='red', linestyle='--', alpha=0.7, 
                       label=f'Mean: {mean_replicas:.2f}')
            
            duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60
            ax1.set_title(f'DQN (NimbusGuard) - {duration:.1f} minutes', fontweight='bold')
            ax1.set_ylabel('Replicas')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'DQN data not available', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=12)
        
        # 2. HPA Time Series
        ax2 = axes[1]
        if 'replicas' in hpa_data and not hpa_data['replicas'].empty:
            df = hpa_data['replicas']
            ax2.plot(df['timestamp'], df['value'], color=self.colors['hpa'], linewidth=2)
            ax2.fill_between(df['timestamp'], df['value'], alpha=0.3, color=self.colors['hpa'])
            
            # Add statistics
            mean_replicas = df['value'].mean()
            ax2.axhline(mean_replicas, color='red', linestyle='--', alpha=0.7, 
                       label=f'Mean: {mean_replicas:.2f}')
            
            duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60
            ax2.set_title(f'Traditional HPA - {duration:.1f} minutes', fontweight='bold')
            ax2.set_ylabel('Replicas')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'HPA data not available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
        
        # 3. KEDA Time Series
        ax3 = axes[2]
        if 'replicas' in keda_data and not keda_data['replicas'].empty:
            df = keda_data['replicas']
            ax3.plot(df['timestamp'], df['value'], color=self.colors['keda'], linewidth=2)
            ax3.fill_between(df['timestamp'], df['value'], alpha=0.3, color=self.colors['keda'])
            
            # Add statistics
            mean_replicas = df['value'].mean()
            ax3.axhline(mean_replicas, color='red', linestyle='--', alpha=0.7, 
                       label=f'Mean: {mean_replicas:.2f}')
            
            duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60
            ax3.set_title(f'KEDA - {duration:.1f} minutes', fontweight='bold')
            ax3.set_ylabel('Replicas')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'KEDA data not available', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
        
        # Format x-axis for better readability
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "individual_time_series.png", dpi=300)
        plt.close()
        print("✅ Created individual time series charts")
    
    def create_performance_metrics_comparison(self, dqn_metrics, hpa_metrics, keda_metrics):
        """Create performance metrics comparison charts."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Scaling Approaches: Performance Metrics Comparison', fontsize=16, fontweight='bold')
        
        approaches = ['DQN', 'HPA', 'KEDA']
        colors = [self.colors['dqn'], self.colors['hpa'], self.colors['keda']]
        
        # 1. Average Metrics Bar Chart
        ax1 = axes[0, 0]
        
        metrics_to_compare = ['avg_replicas', 'avg_unavailable_replicas', 'avg_pod_readiness']
        metric_labels = ['Avg Replicas', 'Avg Unavailable', 'Avg Pod Readiness']
        
        x = np.arange(len(metrics_to_compare))
        width = 0.25
        
        dqn_values = [dqn_metrics.get(m, 0) for m in metrics_to_compare]
        hpa_values = [hpa_metrics.get(m, 0) for m in metrics_to_compare]
        keda_values = [keda_metrics.get(m, 0) for m in metrics_to_compare]
        
        ax1.bar(x - width, dqn_values, width, label='DQN', color=colors[0], alpha=0.8)
        ax1.bar(x, hpa_values, width, label='HPA', color=colors[1], alpha=0.8)
        ax1.bar(x + width, keda_values, width, label='KEDA', color=colors[2], alpha=0.8)
        
        ax1.set_title('Key Metrics Comparison', fontweight='bold')
        ax1.set_ylabel('Metric Value')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metric_labels, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Resource Efficiency Comparison
        ax2 = axes[0, 1]
        
        efficiency_metrics = [dqn_metrics.get('resource_efficiency', 0),
                             hpa_metrics.get('resource_efficiency', 0),
                             keda_metrics.get('resource_efficiency', 0)]
        
        bars = ax2.bar(approaches, efficiency_metrics, color=colors, alpha=0.8)
        ax2.set_title('Resource Efficiency Score', fontweight='bold')
        ax2.set_ylabel('Efficiency Score (0-1)')
        ax2.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, value in zip(bars, efficiency_metrics):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.grid(True, alpha=0.3)
        
        # 3. Scaling Stability Comparison
        ax3 = axes[1, 0]
        
        stability_metrics = [dqn_metrics.get('scaling_stability', 0),
                            hpa_metrics.get('scaling_stability', 0),
                            keda_metrics.get('scaling_stability', 0)]
        
        bars = ax3.bar(approaches, stability_metrics, color=colors, alpha=0.8)
        ax3.set_title('Scaling Stability Score', fontweight='bold')
        ax3.set_ylabel('Stability Score (0-1)')
        ax3.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, value in zip(bars, stability_metrics):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax3.grid(True, alpha=0.3)
        
        # 4. Overall Performance Radar Chart
        ax4 = axes[1, 1]
        
        # Calculate overall scores for radar chart
        performance_categories = ['Resource\nEfficiency', 'Scaling\nStability', 'Availability', 'Responsiveness']
        
        dqn_scores = [
            dqn_metrics.get('resource_efficiency', 0),
            dqn_metrics.get('scaling_stability', 0),
            dqn_metrics.get('availability_score', 0),
            min(1.0, 1.0 / max(0.1, dqn_metrics.get('avg_unavailable_replicas', 1)))  # Responsiveness
        ]
        
        hpa_scores = [
            hpa_metrics.get('resource_efficiency', 0),
            hpa_metrics.get('scaling_stability', 0),
            hpa_metrics.get('availability_score', 0),
            min(1.0, 1.0 / max(0.1, hpa_metrics.get('avg_unavailable_replicas', 1)))
        ]
        
        keda_scores = [
            keda_metrics.get('resource_efficiency', 0),
            keda_metrics.get('scaling_stability', 0),
            keda_metrics.get('availability_score', 0),
            min(1.0, 1.0 / max(0.1, keda_metrics.get('avg_unavailable_replicas', 1)))
        ]
        
        # Simple bar chart instead of radar for clarity
        x = np.arange(len(performance_categories))
        width = 0.25
        
        ax4.bar(x - width, dqn_scores, width, label='DQN', color=colors[0], alpha=0.8)
        ax4.bar(x, hpa_scores, width, label='HPA', color=colors[1], alpha=0.8)
        ax4.bar(x + width, keda_scores, width, label='KEDA', color=colors[2], alpha=0.8)
        
        ax4.set_title('Overall Performance Comparison', fontweight='bold')
        ax4.set_ylabel('Performance Score (0-1)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(performance_categories, fontsize=9)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_metrics_comparison.png", dpi=300)
        plt.close()
        print("✅ Created performance metrics comparison")
    
    def create_dqn_specific_analysis(self, dqn_data):
        """Create DQN-specific analysis charts."""
        if not dqn_data:
            print("⚠️ No DQN data available for specific analysis")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('DQN-Specific Intelligence Analysis', fontsize=16, fontweight='bold')
        
        # 1. DQN Decision Actions Over Time
        ax1 = axes[0, 0]
        
        if all(metric in dqn_data for metric in ['dqn_actions_scale_up', 'dqn_actions_scale_down', 'dqn_actions_keep_same']):
            scale_up_df = dqn_data['dqn_actions_scale_up']
            scale_down_df = dqn_data['dqn_actions_scale_down']
            keep_same_df = dqn_data['dqn_actions_keep_same']
            
            ax1.plot(scale_up_df['timestamp'], scale_up_df['value'], 
                    color=self.colors['accent'], label='Scale Up', linewidth=2)
            ax1.plot(scale_down_df['timestamp'], scale_down_df['value'], 
                    color=self.colors['dqn'], label='Scale Down', linewidth=2)
            ax1.plot(keep_same_df['timestamp'], keep_same_df['value'], 
                    color=self.colors['gray'], label='Keep Same', linewidth=2)
            
            ax1.set_title('DQN Action Distribution Over Time', fontweight='bold')
            ax1.set_ylabel('Cumulative Actions')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'DQN action data\nnot available', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        
        # 2. DQN Decision Confidence Over Time
        ax2 = axes[0, 1]
        
        if 'dqn_confidence' in dqn_data:
            confidence_df = dqn_data['dqn_confidence']
            ax2.plot(confidence_df['timestamp'], confidence_df['value'], 
                    color=self.colors['secondary'], linewidth=2)
            ax2.set_title('DQN Decision Confidence', fontweight='bold')
            ax2.set_ylabel('Confidence Score')
            ax2.grid(True, alpha=0.3)
            
            # Add confidence level annotations
            mean_confidence = confidence_df['value'].mean()
            ax2.axhline(mean_confidence, color='red', linestyle='--', 
                       label=f'Mean: {mean_confidence:.3f}')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'DQN confidence data\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        
        # 3. LSTM Pressure Predictions
        ax3 = axes[1, 0]
        
        if 'lstm_pressure_30s' in dqn_data and 'lstm_pressure_60s' in dqn_data:
            pressure_30s = dqn_data['lstm_pressure_30s']
            pressure_60s = dqn_data['lstm_pressure_60s']
            
            ax3.plot(pressure_30s['timestamp'], pressure_30s['value'], 
                    color=self.colors['dqn'], label='30s Forecast', linewidth=2)
            ax3.plot(pressure_60s['timestamp'], pressure_60s['value'], 
                    color=self.colors['secondary'], label='60s Forecast', linewidth=2)
            
            ax3.set_title('LSTM Load Pressure Predictions', fontweight='bold')
            ax3.set_ylabel('Predicted Load Pressure')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'LSTM prediction data\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        
        # 4. DQN Reward Signal
        ax4 = axes[1, 1]
        
        if 'dqn_reward' in dqn_data:
            reward_df = dqn_data['dqn_reward']
            ax4.plot(reward_df['timestamp'], reward_df['value'], 
                    color=self.colors['accent'], linewidth=2)
            ax4.set_title('DQN Reward Signal', fontweight='bold')
            ax4.set_ylabel('Reward Value')
            ax4.grid(True, alpha=0.3)
            
            # Add reward statistics
            mean_reward = reward_df['value'].mean()
            ax4.axhline(mean_reward, color='blue', linestyle='--', 
                       label=f'Mean: {mean_reward:.3f}')
            ax4.axhline(0, color='gray', linestyle='-', alpha=0.5, label='Zero Line')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'DQN reward data\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        
        # Format x-axis for better readability
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "dqn_specific_analysis.png", dpi=300)
        plt.close()
        print("✅ Created DQN-specific analysis")
    
    def create_summary_report(self, dqn_metrics, hpa_metrics, keda_metrics):
        """Create comprehensive summary report."""
        
        # Create summary comparison table
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Scaling Approaches: Comprehensive Summary', fontsize=16, fontweight='bold')
        
        # 1. Metrics Summary Table
        metrics_data = {
            'Metric': [
                'Avg Replicas',
                'Avg Unavailable',
                'Pod Readiness',
                'Resource Efficiency',
                'Scaling Stability',
                'Availability Score'
            ],
            'DQN': [
                f"{dqn_metrics.get('avg_replicas', 0):.2f}",
                f"{dqn_metrics.get('avg_unavailable_replicas', 0):.2f}",
                f"{dqn_metrics.get('avg_pod_readiness', 0):.3f}",
                f"{dqn_metrics.get('resource_efficiency', 0):.3f}",
                f"{dqn_metrics.get('scaling_stability', 0):.3f}",
                f"{dqn_metrics.get('availability_score', 0):.3f}"
            ],
            'HPA': [
                f"{hpa_metrics.get('avg_replicas', 0):.2f}",
                f"{hpa_metrics.get('avg_unavailable_replicas', 0):.2f}",
                f"{hpa_metrics.get('avg_pod_readiness', 0):.3f}",
                f"{hpa_metrics.get('resource_efficiency', 0):.3f}",
                f"{hpa_metrics.get('scaling_stability', 0):.3f}",
                f"{hpa_metrics.get('availability_score', 0):.3f}"
            ],
            'KEDA': [
                f"{keda_metrics.get('avg_replicas', 0):.2f}",
                f"{keda_metrics.get('avg_unavailable_replicas', 0):.2f}",
                f"{keda_metrics.get('avg_pod_readiness', 0):.3f}",
                f"{keda_metrics.get('resource_efficiency', 0):.3f}",
                f"{keda_metrics.get('scaling_stability', 0):.3f}",
                f"{keda_metrics.get('availability_score', 0):.3f}"
            ]
        }
        
        # Create table
        table_data = []
        for i, metric in enumerate(metrics_data['Metric']):
            table_data.append([metric, metrics_data['DQN'][i], 
                             metrics_data['HPA'][i], metrics_data['KEDA'][i]])
        
        ax1.axis('tight')
        ax1.axis('off')
        table = ax1.table(cellText=table_data,
                         colLabels=['Metric', 'DQN', 'HPA', 'KEDA'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(metrics_data['Metric']) + 1):
            for j in range(4):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    if j == 1:  # DQN column
                        cell.set_facecolor('#e6f2ff')
                    elif j == 2:  # HPA column
                        cell.set_facecolor('#fff2e6')
                    elif j == 3:  # KEDA column
                        cell.set_facecolor('#e6ffe6')
        
        ax1.set_title('Performance Metrics Summary', fontweight='bold', pad=20)
        
        # 2. Overall Performance Score
        ax2.axis('off')
        
        # Calculate overall performance scores
        def calculate_overall_score(metrics):
            weights = {
                'resource_efficiency': 0.3,
                'scaling_stability': 0.25,
                'availability_score': 0.25,
                'responsiveness': 0.2
            }
            
            responsiveness = min(1.0, 1.0 / max(0.1, metrics.get('avg_unavailable_replicas', 1)))
            
            score = (
                metrics.get('resource_efficiency', 0) * weights['resource_efficiency'] +
                metrics.get('scaling_stability', 0) * weights['scaling_stability'] +
                metrics.get('availability_score', 0) * weights['availability_score'] +
                responsiveness * weights['responsiveness']
            )
            return score
        
        dqn_overall = calculate_overall_score(dqn_metrics)
        hpa_overall = calculate_overall_score(hpa_metrics)
        keda_overall = calculate_overall_score(keda_metrics)
        
        # Create performance score visualization
        approaches = ['DQN\n(NimbusGuard)', 'Traditional\nHPA', 'KEDA\nEvent-Driven']
        scores = [dqn_overall, hpa_overall, keda_overall]
        colors = [self.colors['dqn'], self.colors['hpa'], self.colors['keda']]
        
        y_pos = np.arange(len(approaches))
        bars = ax2.barh(y_pos, scores, color=colors, alpha=0.8)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(approaches)
        ax2.set_xlabel('Overall Performance Score')
        ax2.set_title('Overall Performance Ranking', fontweight='bold')
        ax2.set_xlim(0, 1.1)
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            width = bar.get_width()
            ax2.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', ha='left', va='center', fontweight='bold')
        
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "summary_report.png", dpi=300)
        plt.close()
        print("✅ Created summary report")
        
        # Save detailed metrics to JSON
        summary_data = {
            'comparison_timestamp': datetime.now().isoformat(),
            'approaches': {
                'dqn': dqn_metrics,
                'hpa': hpa_metrics,
                'keda': keda_metrics
            },
            'overall_scores': {
                'dqn': dqn_overall,
                'hpa': hpa_overall,
                'keda': keda_overall
            },
            'winner': 'DQN' if dqn_overall == max(scores) else ('HPA' if hpa_overall == max(scores) else 'KEDA')
        }
        
        with open(self.output_dir / "comparison_summary.json", 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print("✅ Saved detailed comparison summary")
    
    def run_comprehensive_analysis(self):
        """Run the complete comparison analysis."""
        print("\n🔍 STARTING COMPREHENSIVE SCALING COMPARISON ANALYSIS")
        print("=" * 70)
        
        # Load data from all approaches
        print("\n📊 Loading data from all scaling approaches...")
        dqn_data = self.load_prometheus_data(self.dqn_dir, 'DQN')
        hpa_data = self.load_prometheus_data(self.hpa_dir, 'HPA')
        keda_data = self.load_prometheus_data(self.keda_dir, 'KEDA')
        
        # Calculate performance metrics
        print("\n📈 Calculating performance metrics...")
        dqn_metrics = self.calculate_performance_metrics(dqn_data, 'DQN')
        hpa_metrics = self.calculate_performance_metrics(hpa_data, 'HPA')
        keda_metrics = self.calculate_performance_metrics(keda_data, 'KEDA')
        
        # Create visualizations
        print("\n🎨 Creating comparison visualizations...")
        self.create_time_series_comparison(dqn_data, hpa_data, keda_data)
        self.create_individual_time_series(dqn_data, hpa_data, keda_data)
        self.create_performance_metrics_comparison(dqn_metrics, hpa_metrics, keda_metrics)
        self.create_dqn_specific_analysis(dqn_data)
        self.create_summary_report(dqn_metrics, hpa_metrics, keda_metrics)
        
        print("\n" + "=" * 70)
        print("✅ COMPREHENSIVE SCALING COMPARISON ANALYSIS COMPLETED!")
        print(f"📁 All outputs saved to: {self.output_dir}")
        print("📊 Generated files:")
        for file in sorted(self.output_dir.glob("*.png")):
            print(f"   - {file.name}")
        for file in sorted(self.output_dir.glob("*.json")):
            print(f"   - {file.name}")
        print("=" * 70)

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare DQN, HPA, and KEDA scaling approaches")
    parser.add_argument("--data-dir", type=str, default=".",
                        help="Directory containing prometheus_data_* subdirectories")
    parser.add_argument("--output-dir", type=str, default="scaling_comparison",
                        help="Output directory for comparison visualizations")
    
    args = parser.parse_args()
    
    # Create comparison analyzer
    analyzer = ScalingComparisonAnalyzer(Path(args.data_dir), Path(args.output_dir))
    
    # Run comprehensive analysis
    analyzer.run_comprehensive_analysis()

if __name__ == "__main__":
    main() 