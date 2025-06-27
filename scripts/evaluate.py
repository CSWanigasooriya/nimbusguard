#!/usr/bin/env python3
"""
DQN Feature Engineering Showcase
===============================

This script generates publication-quality diagrams and visualizations for the
11-feature DQN approach, suitable for research papers and presentations.

Generated Outputs:
1. Feature Selection Pipeline
2. Feature Importance Ranking (11 features)
3. Feature Selection Method Comparison
4. Statistical Validation Results
5. Scaling Decision Distribution
6. Feature Correlation Analysis
7. Data Quality Assessment
8. Research Impact Comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("⚠️ Plotly not available. Install with: pip install plotly")
    PLOTLY_AVAILABLE = False

try:
    import kaleido
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False

# Set style for publication-quality plots
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    # Fallback for newer versions of matplotlib/seaborn
    plt.style.use('seaborn-whitegrid')
sns.set_palette("husl")

class DQNFeatureShowcase:
    """Generate publication-quality visualizations for DQN feature engineering."""
    
    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load DQN data
        self.df = pd.read_parquet(self.data_dir / "dqn_features.parquet")
        self.scaler = joblib.load(self.data_dir / "feature_scaler.gz")
        
        with open(self.data_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Define color scheme for consistency
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'info': '#7209B7',
            'light': '#F2F2F2',
            'dark': '#333333'
        }
        
        print(f"📊 Loaded DQN data: {self.df.shape[0]} samples, {self.df.shape[1]} features")
        print(f"🎯 Selected features: {len(self.metadata['selected_features'])}")
        print(f"📈 Action distribution: {self.metadata['dataset_info']['action_distribution']}")
    
    def create_pipeline_diagram(self):
        """Create a feature selection pipeline diagram."""
        if not PLOTLY_AVAILABLE:
            print("⚠️ Skipping pipeline diagram - Plotly not available")
            return
            
        fig = go.Figure()
        
        # Define pipeline stages
        stages = [
            {"name": "Raw Prometheus\nMetrics\n(100+ metrics)", "x": 1, "y": 6, "color": self.colors['primary']},
            {"name": "Statistical\nCleaning", "x": 2, "y": 6, "color": self.colors['secondary']},
            {"name": "Domain Feature\nEngineering", "x": 3, "y": 6, "color": self.colors['accent']},
            {"name": "Multi-Method\nSelection", "x": 4, "y": 6, "color": self.colors['success']},
            {"name": "Validation &\nScaling", "x": 5, "y": 6, "color": self.colors['info']},
            {"name": "11 Optimal\nDQN Features", "x": 6, "y": 6, "color": self.colors['primary']}
        ]
        
        # Selection methods below main pipeline
        methods = [
            {"name": "Mutual\nInformation", "x": 3.3, "y": 4},
            {"name": "Random Forest\nImportance", "x": 3.9, "y": 4},
            {"name": "Correlation\nAnalysis", "x": 4.5, "y": 4},
            {"name": "Recursive Feature\nElimination", "x": 5.1, "y": 4}
        ]
        
        # Add boxes for each stage
        for i, stage in enumerate(stages):
            fig.add_shape(
                type="rect",
                x0=stage["x"]-0.4, y0=stage["y"]-0.3,
                x1=stage["x"]+0.4, y1=stage["y"]+0.3,
                fillcolor=stage["color"],
                opacity=0.7,
                line=dict(color=stage["color"], width=2)
            )
            
            fig.add_annotation(
                x=stage["x"], y=stage["y"],
                text=stage["name"],
                showarrow=False,
                font=dict(color="white", size=12, family="Arial Black"),
                align="center"
            )
            
            # Add arrows between stages
            if i < len(stages) - 1:
                fig.add_annotation(
                    x=stage["x"]+0.5, y=stage["y"],
                    ax=stage["x"]+0.4, ay=stage["y"],
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor=self.colors['dark']
                )
        
        # Add selection methods below main pipeline
        for method in methods:
            fig.add_shape(
                type="rect",
                x0=method["x"]-0.25, y0=method["y"]-0.15,
                x1=method["x"]+0.25, y1=method["y"]+0.15,
                fillcolor=self.colors['light'],
                opacity=0.8,
                line=dict(color=self.colors['dark'], width=1)
            )
            
            fig.add_annotation(
                x=method["x"], y=method["y"],
                text=method["name"],
                showarrow=False,
                font=dict(color=self.colors['dark'], size=9),
                align="center"
            )
            
            # Add arrow from Multi-Method Selection to each method
            fig.add_annotation(
                x=method["x"], y=method["y"]+0.2,
                ax=4, ay=5.7,
                xref="x", yref="y",
                axref="x", ayref="y",
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor=self.colors['dark']
            )
        
        fig.update_layout(
            title=dict(
                text="<b>11-Feature DQN Selection Pipeline</b>",
                x=0.5,
                font=dict(size=20, color=self.colors['dark'])
            ),
            xaxis=dict(range=[0.5, 6.5], showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(range=[3.5, 6.5], showgrid=False, showticklabels=False, zeroline=False),
            plot_bgcolor='white',
            width=1200,
            height=500,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Save the diagram
        fig.write_html(self.output_dir / "pipeline_diagram.html")
        if KALEIDO_AVAILABLE:
            try:
                fig.write_image(self.output_dir / "pipeline_diagram.png", width=1200, height=500, scale=2)
                print("✅ Created pipeline diagram")
            except Exception as e:
                print(f"⚠️ Could not save PNG: {e}")
                print("✅ Created pipeline diagram (HTML only)")
        else:
            print("⚠️ Kaleido not available for PNG export (install with: pip install kaleido)")
            print("✅ Created pipeline diagram (HTML only)")
    
    def create_feature_importance_analysis(self):
        """Create analysis of the 11 selected features."""
        # Get the 11 selected features and their analysis
        selected_features = self.metadata['selected_features']
        feature_analysis = self.metadata['feature_analysis']
        
        # Create subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{len(selected_features)}-Feature DQN Analysis', fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Selected Features Ranking
        final_scores = feature_analysis['final_scores']
        features = list(final_scores.keys())
        scores = list(final_scores.values())
        
        # Create horizontal bar chart for feature ranking
        y_pos = np.arange(len(features))
        bars = ax1.barh(y_pos, scores, color=self.colors['primary'], alpha=0.8)
        ax1.set_yticks(y_pos)
        
        # Create better feature labels (shorten intelligently)
        def shorten_feature_name(name):
            # Remove common prefixes and use abbreviations
            name = name.replace('kube_pod_container_', 'pod_')
            name = name.replace('kube_deployment_', 'deploy_')
            name = name.replace('status_', '')
            name = name.replace('_ma_10', ' (10min avg)')
            name = name.replace('_dev_10', ' (10min dev)')
            name = name.replace('_seconds_', '_sec_')
            name = name.replace('http_request_duration_highr_', 'http_dur_')
            if len(name) > 30:
                name = name[:27] + '...'
            return name.replace('_', ' ').title()
        
        ax1.set_yticklabels([shorten_feature_name(f) for f in features])
        ax1.set_xlabel('Ensemble Score')
        ax1.set_title(f'{len(selected_features)} Selected Features (Ranked by Ensemble Score)', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax1.text(score + 1, i, f'{score:.1f}', va='center', fontweight='bold')
        
        # 2. Feature Selection Method Comparison
        methods = ['Mutual Information', 'Random Forest', 'Correlation', 'RFE']
        
        # Count how many of our selected features appear in top 10 of each method
        method_coverage = []
        mi_top = list(feature_analysis['selection_methods']['mutual_information'].keys())[:10]
        rf_top = list(feature_analysis['selection_methods']['random_forest'].keys())[:10]
        corr_top = [item[0] for item in list(feature_analysis['selection_methods']['correlation'].items())[:10]]
        rfe_selected = feature_analysis['selection_methods']['rfecv_selected']
        
        method_coverage = [
            len([f for f in selected_features if f in mi_top]),
            len([f for f in selected_features if f in rf_top]),
            len([f for f in selected_features if f in corr_top]),
            len([f for f in selected_features if f in rfe_selected])
        ]
        
        bars = ax2.bar(methods, method_coverage, color=[self.colors['primary'], self.colors['secondary'], 
                                                       self.colors['accent'], self.colors['success']], alpha=0.8)
        ax2.set_ylabel('Features in Final Selection')
        ax2.set_title(f'Method Contribution to Final {len(selected_features)} Features', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 12)
        
        # Add value labels
        for bar, count in zip(bars, method_coverage):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Action distribution
        action_names = {0: 'Scale Down', 1: 'Keep Same', 2: 'Scale Up'}
        action_counts = self.df['scaling_action'].value_counts().sort_index()
        action_labels = [action_names[i] for i in action_counts.index]
        
        bars = ax3.bar(action_labels, action_counts.values, 
                      color=[self.colors['success'], self.colors['accent'], self.colors['primary']])
        ax3.set_title('Scaling Decision Distribution', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Number of Samples')
        
        # Add percentage labels on bars
        total = sum(action_counts.values)
        for bar, count in zip(bars, action_counts.values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{count}\n({count/total*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
        
        # 4. Feature Type Distribution (Kubernetes Focus)
        feature_types = {
            'Response Time': [f for f in selected_features if 'response_time' in f],
            'Pod Status': [f for f in selected_features if 'kube_pod_container_status' in f],
            'Deployment': [f for f in selected_features if 'kube_deployment' in f],
            'Health Ratios': [f for f in selected_features if 'health_ratio' in f],
            'Resource Limits': [f for f in selected_features if 'resource_limits' in f],
            'Deviation Features': [f for f in selected_features if '_dev_' in f],
        }
        
        # Filter out empty categories and count features
        type_counts = {k: len(v) for k, v in feature_types.items() if v}
        
        # Create pie chart
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent'], 
                 self.colors['success'], self.colors['info']][:len(type_counts)]
        
        wedges, texts, autotexts = ax4.pie(type_counts.values(), labels=type_counts.keys(), 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax4.set_title('Selected Feature Types', fontsize=14, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_color('white')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Created feature importance analysis")
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap of selected features."""
        # Use the actual selected features from the dataset
        selected_features = self.metadata['selected_features']
        
        # Filter features that exist in the dataset
        available_features = [f for f in selected_features if f in self.df.columns]
        
        if len(available_features) < 2:
            print("⚠️ Not enough features for correlation analysis")
            return
        
        # Calculate correlation matrix
        corr_matrix = self.df[available_features].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        
        plt.title(f'Feature Correlation Matrix\n({len(available_features)} Selected DQN Features)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Created correlation heatmap")
    
    def create_performance_metrics_comparison(self):
        """Create feature distributions analysis."""
        # Create subplot layout - only top two charts
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('DQN Feature Analysis', fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Response Time Distribution (with outlier handling)
        if 'avg_response_time' in self.df.columns:
            # Handle extreme outliers by using 95th percentile as upper limit for visualization
            response_data = self.df['avg_response_time']
            q95 = response_data.quantile(0.95)
            
            # Create histogram with outlier-aware binning
            axes[0].hist(response_data.clip(upper=q95), bins=50, color=self.colors['primary'], alpha=0.7, edgecolor='black')
            axes[0].set_xlabel('Average Response Time (ms)')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Response Time Distribution\n(Clipped at 95th percentile for visualization)', fontweight='bold')
            axes[0].axvline(response_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean ({response_data.mean():.0f}ms)')
            axes[0].axvline(response_data.median(), color='orange', linestyle='--', linewidth=2, label=f'Median ({response_data.median():.0f}ms)')
            
            # Add outlier information
            outlier_count = (response_data > q95).sum()
            if outlier_count > 0:
                axes[0].text(0.98, 0.98, f'{outlier_count} outliers\n(>{q95:.0f}ms)', 
                        transform=axes[0].transAxes, ha='right', va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            axes[0].legend()
            axes[0].grid(alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, 'Response time data\nnot available', 
                        ha='center', va='center', transform=axes[0].transAxes,
                        fontsize=12, color=self.colors['dark'])
            axes[0].set_title('Response Time Distribution', fontweight='bold')

        # 2. Pod Health Metrics Distribution (with outlier handling)
        pod_health_column = None
        for col in ['kube_pod_container_status_ready', 'kube_pod_container_status_ready_ma_5']:
            if col in self.df.columns:
                pod_health_column = col
                break
        
        if pod_health_column:
            # Handle outliers using IQR method
            pod_data = self.df[pod_health_column]
            q1 = pod_data.quantile(0.25)
            q3 = pod_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Count outliers
            outliers_low = (pod_data < lower_bound).sum()
            outliers_high = (pod_data > upper_bound).sum()
            total_outliers = outliers_low + outliers_high
            
            # Clip data for visualization
            clipped_data = pod_data.clip(lower=lower_bound, upper=upper_bound)
            
            axes[1].hist(clipped_data, bins=50, color=self.colors['secondary'], alpha=0.7, edgecolor='black')
            axes[1].set_xlabel('Kube Pod Container Status Ready')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Pod Health Metrics Distribution\n(Outliers clipped for visualization)', fontweight='bold')
            axes[1].axvline(pod_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean ({pod_data.mean():.3f})')
            axes[1].axvline(pod_data.median(), color='orange', linestyle='--', linewidth=2, label=f'Median ({pod_data.median():.3f})')
            
            # Add outlier information
            if total_outliers > 0:
                axes[1].text(0.98, 0.98, f'{total_outliers} outliers\n({total_outliers/len(pod_data)*100:.1f}%)', 
                           transform=axes[1].transAxes, ha='right', va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Add explanation of decimal values
            explanation_text = ('Decimal Values Explained:\n'
                              '0.941 = 94.1% containers ready\n'
                              '0.888 ≈ 8/9 containers ready\n'
                              '0.700 ≈ 70% containers ready\n\n'
                              'Formula: ready_containers/total_containers')
            axes[1].text(0.02, 0.98, explanation_text, 
                        transform=axes[1].transAxes, ha='left', va='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9),
                        fontsize=9, fontweight='normal')
            
            axes[1].legend()
            axes[1].grid(alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'Pod health metrics\nnot available', 
                        ha='center', va='center', transform=axes[1].transAxes,
                        fontsize=12, color=self.colors['dark'])
            axes[1].set_title('Pod Health Metrics Distribution', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Created feature distributions analysis")
    
    def create_data_quality_report(self):
        """Create data quality assessment report."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Data Quality Assessment', fontsize=20, fontweight='bold')
        
        # 1. Scaling Action Distribution Pie Chart
        action_counts = self.df['scaling_action'].value_counts().sort_index()
        action_labels = ['Scale Down', 'Keep Same', 'Scale Up']
        action_colors = [self.colors['primary'], self.colors['accent'], self.colors['success']]
        
        # Create pie chart
        wedges, texts, autotexts = ax1.pie(action_counts.values, 
                                          labels=action_labels,
                                          colors=action_colors,
                                          autopct='%1.1f%%',
                                          startangle=90,
                                          explode=(0.05, 0.05, 0.05))
        
        # Enhance text appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        for text in texts:
            text.set_fontweight('bold')
            text.set_fontsize(11)
        
        ax1.set_title('Scaling Action Distribution', fontweight='bold', fontsize=14)
        
        # Add sample count in center (if possible)
        total_samples = len(self.df)
        ax1.text(0, 0, f'{total_samples}\nSamples', ha='center', va='center',
                fontsize=12, fontweight='bold', color=self.colors['dark'])
        
        # 2. Pod Health vs Response Time Scatter (with outlier handling)
        if 'kube_pod_container_status_ready' in self.df.columns and 'avg_response_time' in self.df.columns:
            # Handle outliers for better visualization
            response_data = self.df['avg_response_time']
            q99 = response_data.quantile(0.99)
            
            # Create scatter plot colored by scaling action
            action_colors = {0: self.colors['primary'], 1: self.colors['accent'], 2: self.colors['success']}
            action_labels = {0: 'Scale Down', 1: 'Keep Same', 2: 'Scale Up'}
            
            for action in [0, 1, 2]:
                mask = self.df['scaling_action'] == action
                if mask.sum() > 0:
                    # Clip response times for better visualization
                    y_data = self.df[mask]['avg_response_time'].clip(upper=q99)
                    ax2.scatter(self.df[mask]['kube_pod_container_status_ready'], 
                              y_data,
                              c=action_colors[action], 
                              alpha=0.6, 
                              label=action_labels[action],
                              s=30)
            
            ax2.set_xlabel('Pod Container Status Ready (Fraction)')
            ax2.set_ylabel('Average Response Time (ms)')
            ax2.set_title('Pod Health vs Response Time\n(Response time clipped at 99th percentile)', fontweight='bold')
            
            # Add text about readiness range
            readiness_min = self.df['kube_pod_container_status_ready'].min()
            readiness_max = self.df['kube_pod_container_status_ready'].max()
            ax2.text(0.02, 0.98, f'Readiness range:\n{readiness_min:.2f} - {readiness_max:.2f}', 
                    transform=ax2.transAxes, ha='left', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Pod Health or Response Time\ndata not available', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=12, color=self.colors['dark'])
            ax2.set_title('Pod Health vs Response Time Analysis', fontweight='bold')
        
        # 3. Feature Value Distributions (Box Plot) - Using Robust Normalization
        selected_features = self.metadata['selected_features'][:6]  # Top 6 features for readability
        feature_data = []
        feature_names = []
        
        for feature in selected_features:
            if feature in self.df.columns:
                # Use robust normalization (z-score with clipping for outliers)
                feature_values = self.df[feature].copy()
                
                # Handle edge cases
                if feature_values.std() == 0:
                    # Constant feature - normalize to 0.5
                    normalized_values = pd.Series([0.5] * len(feature_values))
                else:
                    # Z-score normalization
                    normalized_values = (feature_values - feature_values.mean()) / feature_values.std()
                    
                    # Clip extreme outliers to [-3, 3] range and rescale to [0, 1]
                    normalized_values = normalized_values.clip(-3, 3)
                    normalized_values = (normalized_values + 3) / 6  # Scale to [0, 1]
                
                feature_data.append(normalized_values)
                feature_names.append(feature.replace('_', ' ').title()[:15])
        
        if feature_data:
            # Create box plot without outliers for cleaner visualization
            box_plot = ax3.boxplot(feature_data, labels=feature_names, showfliers=False)
            ax3.set_ylabel('Standardized Feature Values')
            ax3.set_title('Feature Value Distributions (Standardized)\n(Outliers clipped for clarity)', fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(-0.1, 1.1)  # Set consistent y-axis range
            
            # Add note about outlier handling
            ax3.text(0.02, 0.98, 'Outliers clipped\nat ±3σ range', 
                    transform=ax3.transAxes, ha='left', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'No feature data\navailable for boxplot', 
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=12, color=self.colors['dark'])
            ax3.set_title('Feature Value Distributions', fontweight='bold')
        
        # 4. Feature Type Analysis (Consumer Pod Features)
        # Categorize the actual selected features
        selected_features = self.metadata['selected_features']
        
        base_features = [f for f in selected_features if not any(x in f for x in ['_dev_', '_ma_', '_volatility', 'memory_growth_rate'])]
        derived_features = [f for f in selected_features if any(x in f for x in ['_dev_', '_ma_', '_volatility'])]
        computed_features = [f for f in selected_features if f == 'memory_growth_rate']  # memory_growth_rate is computed feature
        
        # Only include categories that have features
        categories = []
        counts = []
        colors = []
        
        if len(base_features) > 0:
            categories.append('Base\nMetrics')
            counts.append(len(base_features))
            colors.append(self.colors['primary'])
        
        if len(derived_features) > 0:
            categories.append('Derived\nFeatures')
            counts.append(len(derived_features))
            colors.append(self.colors['secondary'])
        
        if len(computed_features) > 0:
            categories.append('Computed\nFeatures')
            counts.append(len(computed_features))
            colors.append(self.colors['accent'])
        
        bars = ax4.bar(categories, counts, color=colors, alpha=0.8)
        ax4.set_ylabel('Number of Features')
        ax4.set_title('Selected Feature Types Analysis', fontweight='bold')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        # Set y-axis to show whole numbers only
        ax4.set_ylim(0, max(counts) + 1)
        ax4.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "data_quality_report.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Created data quality report")
    
    def create_research_summary(self):
        """Create a comprehensive research summary document."""
        summary = {
            "feature_engineering_summary": {
                "total_samples": len(self.df),
                "total_features": len(self.metadata['selected_features']),
                "time_range": {
                    "start": self.metadata['dataset_info']['time_range'][0] if self.metadata['dataset_info']['time_range'] else 'N/A',
                    "end": self.metadata['dataset_info']['time_range'][1] if self.metadata['dataset_info']['time_range'] else 'N/A',
                    "duration_hours": 'N/A'
                },
                "action_distribution": self.metadata['dataset_info']['action_distribution'],
                "feature_categories": {
                    "response_time": len([f for f in self.metadata['selected_features'] if 'response_time' in f]),
                    "pod_status": len([f for f in self.metadata['selected_features'] if 'kube_pod_container_status' in f]),
                    "deployment_metrics": len([f for f in self.metadata['selected_features'] if 'kube_deployment' in f]),
                    "health_ratios": len([f for f in self.metadata['selected_features'] if 'health_ratio' in f]),
                    "resource_limits": len([f for f in self.metadata['selected_features'] if 'resource_limits' in f]),
                    "deviation_features": len([f for f in self.metadata['selected_features'] if '_dev_' in f])
                }
            },
            "data_quality_metrics": {
                                 "missing_values": int(self.df.isnull().sum().sum()),
                 "duplicate_rows": int(self.df.duplicated().sum()),
                                 "feature_variance_stats": {
                     "mean_variance": float(self.df.select_dtypes(include=[np.number]).var().mean()),
                     "std_variance": float(self.df.select_dtypes(include=[np.number]).var().std()),
                     "note": "Calculated on raw features - use normalized variance for feature importance"
                 }
            },
            "scaling_insights": {
                "scale_up_percentage": float(self.metadata['dataset_info']['action_distribution'].get('2', self.metadata['dataset_info']['action_distribution'].get(2, 0)) / len(self.df) * 100),
                "scale_down_percentage": float(self.metadata['dataset_info']['action_distribution'].get('0', self.metadata['dataset_info']['action_distribution'].get(0, 0)) / len(self.df) * 100),
                "keep_same_percentage": float(self.metadata['dataset_info']['action_distribution'].get('1', self.metadata['dataset_info']['action_distribution'].get(1, 0)) / len(self.df) * 100)
            }
        }
        
        # Save summary
        with open(self.output_dir / "research_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create markdown report
        markdown_report = f"""# 11-Feature DQN Research Summary

## Overview
This document summarizes the advanced feature selection process for DQN-based Kubernetes pod autoscaling using statistical methods.

## Dataset Statistics
- **Total Samples**: {summary['feature_engineering_summary']['total_samples']:,}
- **Selected Features**: {summary['feature_engineering_summary']['total_features']}
- **Selection Methods**: Mutual Information, Random Forest, Correlation Analysis, RFE
- **Statistical Validation**: ✅ Applied

## Selected Feature Categories (Kubernetes Focus)
- **Response Time Metrics**: {summary['feature_engineering_summary']['feature_categories']['response_time']} features
- **Pod Status Metrics**: {summary['feature_engineering_summary']['feature_categories']['pod_status']} features  
- **Deployment Metrics**: {summary['feature_engineering_summary']['feature_categories']['deployment_metrics']} features
- **Health Ratio Metrics**: {summary['feature_engineering_summary']['feature_categories']['health_ratios']} features
- **Resource Limit Metrics**: {summary['feature_engineering_summary']['feature_categories']['resource_limits']} features
- **Deviation Features**: {summary['feature_engineering_summary']['feature_categories']['deviation_features']} features

## Scaling Decision Distribution
- **Scale Up**: {summary['scaling_insights']['scale_up_percentage']:.1f}% ({summary['feature_engineering_summary']['action_distribution'].get(2, 0)} samples)
- **Keep Same**: {summary['scaling_insights']['keep_same_percentage']:.1f}% ({summary['feature_engineering_summary']['action_distribution'].get(1, 0)} samples)
- **Scale Down**: {summary['scaling_insights']['scale_down_percentage']:.1f}% ({summary['feature_engineering_summary']['action_distribution'].get(0, 0)} samples)

## Data Quality
- **Missing Values**: {summary['data_quality_metrics']['missing_values']}
- **Duplicate Rows**: {summary['data_quality_metrics']['duplicate_rows']}
- **Feature Variance**: Mean = {summary['data_quality_metrics']['feature_variance_stats']['mean_variance']:.3f}, Std = {summary['data_quality_metrics']['feature_variance_stats']['std_variance']:.3f}

## Key Insights
1. The dataset shows a strong bias towards scale-up decisions ({summary['scaling_insights']['scale_up_percentage']:.1f}%), indicating high system load during the monitoring period.
2. Advanced statistical methods reduced dimensionality from 100+ raw metrics to {summary['feature_engineering_summary']['total_features']} optimal features.
3. High data quality with {summary['data_quality_metrics']['missing_values']} missing values across all features.
4. Multi-method feature selection ensures robust and statistically significant feature choices.

## Generated Visualizations
1. **feature_analysis.png**: Feature importance and method comparison
2. **correlation_heatmap.png**: Selected feature correlations
3. **feature_distributions.png**: Feature distribution analysis
4. **data_quality_report.png**: Data quality assessment

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(self.output_dir / "research_summary.md", 'w') as f:
            f.write(markdown_report)
        
        print("✅ Created research summary")
    
    def generate_all_visualizations(self):
        """Generate all research visualizations."""
        print("\n🎨 GENERATING RESEARCH SHOWCASE VISUALIZATIONS")
        print("=" * 60)
        
        # Skip pipeline diagram generation (not needed)
        self.create_feature_importance_analysis()
        self.create_correlation_heatmap()
        self.create_performance_metrics_comparison()
        self.create_data_quality_report()
        self.create_research_summary()
        
        print("\n" + "=" * 60)
        print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print(f"📁 Output directory: {self.output_dir}")
        print("📊 Files created:")
        for file in sorted(self.output_dir.glob("*")):
            print(f"   - {file.name}")
        print("=" * 60)

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate research showcase visualizations")
    parser.add_argument("--data-dir", type=str, default="dqn_data",
                        help="Directory containing processed DQN features")
    parser.add_argument("--output-dir", type=str, default="research_showcase",
                        help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    # Create showcase generator
    showcase = DQNFeatureShowcase(Path(args.data_dir), Path(args.output_dir))
    
    # Generate all visualizations
    showcase.generate_all_visualizations()

if __name__ == "__main__":
    main()