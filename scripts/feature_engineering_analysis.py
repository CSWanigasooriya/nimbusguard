#!/usr/bin/env python3
"""
Feature Engineering Analysis and Visualization

This script analyzes the feature engineering pipeline and demonstrates
the effectiveness of different feature engineering techniques applied
to the Kubernetes metrics dataset.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class FeatureEngineeringAnalyzer:
    def __init__(self, raw_data_path: str, engineered_data_path: str, output_dir: str = "feature_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("📊 Loading raw and engineered datasets...")
        self.load_datasets(raw_data_path, engineered_data_path)
        
        print(f"✅ Raw data: {self.raw_df.shape[0]} samples, {self.raw_df.shape[1]} features")
        print(f"✅ Engineered data: {self.eng_df.shape[0]} samples, {self.eng_df.shape[1]} features")
        print(f"🔧 Feature expansion: {self.eng_df.shape[1] - self.raw_df.shape[1]} new features")
    
    def load_datasets(self, raw_data_path: str, engineered_data_path: str):
        """Load both raw and engineered datasets."""
        self.raw_df = pd.read_parquet(raw_data_path)
        self.eng_df = pd.read_parquet(engineered_data_path)
        
        # Identify action column
        self.action_col = "scaling_action"
        self.action_names = ['Scale Down', 'Keep Same', 'Scale Up']
        
        # Separate features and target
        self.y = self.eng_df[self.action_col].values
        
        # Get numeric features only
        exclude_cols = [self.action_col, "optimal_pod_count"]
        
        # Raw features (intersection with engineered to ensure compatibility)
        raw_numeric = self.raw_df.select_dtypes(include=[np.number]).columns
        raw_features = raw_numeric.difference(exclude_cols)
        
        eng_numeric = self.eng_df.select_dtypes(include=[np.number]).columns  
        eng_features = eng_numeric.difference(exclude_cols)
        
        # Find common features for comparison
        self.common_features = list(set(raw_features) & set(eng_features))
        self.new_features = list(set(eng_features) - set(raw_features))
        
        print(f"🔗 Common features: {len(self.common_features)}")
        print(f"🆕 New features: {len(self.new_features)}")
        
        # Prepare feature matrices
        self.X_raw = self.raw_df[self.common_features].fillna(0).values
        self.X_eng = self.eng_df[eng_features].fillna(0).values
        self.X_common = self.eng_df[self.common_features].fillna(0).values
        
        self.raw_feature_names = self.common_features
        self.eng_feature_names = list(eng_features)
    
    def plot_feature_distribution_comparison(self):
        """Compare distributions of original vs engineered features."""
        # Select a subset of features for visualization
        sample_features = self.common_features[:12]  # Top 12 for visualization
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, feature in enumerate(sample_features):
            if i >= len(axes):
                break
                
            # Raw feature distribution
            raw_data = self.raw_df[feature].dropna()
            eng_data = self.eng_df[feature].dropna()
            
            axes[i].hist(raw_data, bins=30, alpha=0.6, label='Raw Data', 
                        color='skyblue', edgecolor='black', density=True)
            axes[i].hist(eng_data, bins=30, alpha=0.6, label='Engineered Data', 
                        color='orange', edgecolor='black', density=True)
            
            axes[i].set_title(f'{feature[:30]}...', fontsize=10, fontweight='bold')
            axes[i].set_xlabel('Value', fontsize=9)
            axes[i].set_ylabel('Density', fontsize=9)
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
            
            # Add statistics
            raw_mean, raw_std = raw_data.mean(), raw_data.std()
            eng_mean, eng_std = eng_data.mean(), eng_data.std()
            
            axes[i].text(0.02, 0.98, f'Raw: μ={raw_mean:.2e}, σ={raw_std:.2e}\\nEng: μ={eng_mean:.2e}, σ={eng_std:.2e}',
                        transform=axes[i].transAxes, verticalalignment='top',
                        fontsize=7, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance_comparison(self):
        """Compare feature importance between raw and engineered features."""
        print("🌳 Computing feature importance comparison...")
        
        # Train Random Forest on both feature sets
        rf_raw = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_eng = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Standardize features
        scaler_raw = StandardScaler()
        scaler_eng = StandardScaler()
        
        X_raw_scaled = scaler_raw.fit_transform(self.X_raw)
        X_eng_scaled = scaler_eng.fit_transform(self.X_eng)
        
        # Fit models
        rf_raw.fit(X_raw_scaled, self.y)
        rf_eng.fit(X_eng_scaled, self.y)
        
        # Get performance scores
        raw_score = rf_raw.score(X_raw_scaled, self.y)
        eng_score = rf_eng.score(X_eng_scaled, self.y)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Performance comparison
        performance_data = {'Raw Features': raw_score, 'Engineered Features': eng_score}
        bars = axes[0, 0].bar(performance_data.keys(), performance_data.values(), 
                             color=['skyblue', 'orange'], alpha=0.8, edgecolor='black')
        axes[0, 0].set_ylabel('Accuracy Score', fontsize=12)
        axes[0, 0].set_title('Model Performance: Raw vs Engineered Features', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, performance_data.values()):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', fontweight='bold')
        
        # Feature count comparison
        feature_counts = {'Raw Features': len(self.raw_feature_names), 
                         'Engineered Features': len(self.eng_feature_names)}
        bars = axes[0, 1].bar(feature_counts.keys(), feature_counts.values(), 
                             color=['skyblue', 'orange'], alpha=0.8, edgecolor='black')
        axes[0, 1].set_ylabel('Number of Features', fontsize=12)
        axes[0, 1].set_title('Feature Count Comparison', fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, value in zip(bars, feature_counts.values()):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                           f'{value:,}', ha='center', fontweight='bold')
        
        # Top feature importance for raw features
        top_raw_indices = np.argsort(rf_raw.feature_importances_)[-8:][::-1]  # Reduce to 8 for better readability
        axes[1, 0].barh(range(len(top_raw_indices)), rf_raw.feature_importances_[top_raw_indices],
                       color='skyblue', alpha=0.8, edgecolor='black')
        axes[1, 0].set_yticks(range(len(top_raw_indices)))
        # Truncate feature names and add ellipsis
        truncated_raw_names = [name[:25] + '...' if len(name) > 25 else name 
                              for i, name in enumerate([self.raw_feature_names[i] for i in top_raw_indices])]
        axes[1, 0].set_yticklabels(truncated_raw_names, fontsize=9)
        axes[1, 0].set_xlabel('Feature Importance', fontsize=12)
        axes[1, 0].set_title('Top Raw Features (RF)', fontsize=12, fontweight='bold')
        axes[1, 0].invert_yaxis()
        
        # Top feature importance for engineered features  
        top_eng_indices = np.argsort(rf_eng.feature_importances_)[-8:][::-1]  # Reduce to 8 for better readability
        axes[1, 1].barh(range(len(top_eng_indices)), rf_eng.feature_importances_[top_eng_indices],
                       color='orange', alpha=0.8, edgecolor='black')
        axes[1, 1].set_yticks(range(len(top_eng_indices)))
        # Truncate feature names and add ellipsis
        truncated_eng_names = [name[:25] + '...' if len(name) > 25 else name 
                              for i, name in enumerate([self.eng_feature_names[i] for i in top_eng_indices])]
        axes[1, 1].set_yticklabels(truncated_eng_names, fontsize=9)
        axes[1, 1].set_xlabel('Feature Importance', fontsize=12)
        axes[1, 1].set_title('Top Engineered Features (RF)', fontsize=12, fontweight='bold')
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return raw_score, eng_score
    
    def plot_dimensionality_analysis(self):
        """Analyze dimensionality and separability improvement."""
        print("🎯 Analyzing dimensionality and class separability...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Standardize data
        scaler_raw = StandardScaler()
        scaler_eng = StandardScaler()
        
        X_raw_scaled = scaler_raw.fit_transform(self.X_raw)
        X_eng_scaled = scaler_eng.fit_transform(self.X_eng)
        
        # PCA analysis
        pca_raw = PCA()
        pca_eng = PCA()
        
        pca_raw.fit(X_raw_scaled)
        pca_eng.fit(X_eng_scaled)
        
        # Explained variance comparison
        cumvar_raw = np.cumsum(pca_raw.explained_variance_ratio_)
        cumvar_eng = np.cumsum(pca_eng.explained_variance_ratio_)
        
        axes[0, 0].plot(range(1, min(51, len(cumvar_raw) + 1)), cumvar_raw[:50], 
                       'b-', label='Raw Features', linewidth=2)
        axes[0, 0].plot(range(1, min(51, len(cumvar_eng) + 1)), cumvar_eng[:50], 
                       'r-', label='Engineered Features', linewidth=2)
        axes[0, 0].axhline(0.8, color='green', linestyle='--', alpha=0.7, label='80% variance')
        axes[0, 0].axhline(0.9, color='orange', linestyle='--', alpha=0.7, label='90% variance')
        axes[0, 0].set_xlabel('Number of Components', fontsize=12)
        axes[0, 0].set_ylabel('Cumulative Explained Variance', fontsize=12)
        axes[0, 0].set_title('PCA: Explained Variance Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Components needed for 80% and 90% variance
        components_80_raw = np.argmax(cumvar_raw >= 0.8) + 1
        components_90_raw = np.argmax(cumvar_raw >= 0.9) + 1
        components_80_eng = np.argmax(cumvar_eng >= 0.8) + 1
        components_90_eng = np.argmax(cumvar_eng >= 0.9) + 1
        
        # Component efficiency comparison
        component_data = {
            '80% Variance': [components_80_raw, components_80_eng],
            '90% Variance': [components_90_raw, components_90_eng]
        }
        
        x = np.arange(len(component_data))
        width = 0.35
        
        for i, (threshold, values) in enumerate(component_data.items()):
            axes[0, 1].bar(x[i] - width/2, values[0], width, label='Raw' if i == 0 else "", 
                          color='skyblue', alpha=0.8, edgecolor='black')
            axes[0, 1].bar(x[i] + width/2, values[1], width, label='Engineered' if i == 0 else "", 
                          color='orange', alpha=0.8, edgecolor='black')
            
            # Add value labels
            axes[0, 1].text(x[i] - width/2, values[0] + 5, str(values[0]), 
                           ha='center', fontweight='bold')
            axes[0, 1].text(x[i] + width/2, values[1] + 5, str(values[1]), 
                           ha='center', fontweight='bold')
        
        axes[0, 1].set_ylabel('Components Needed', fontsize=12)
        axes[0, 1].set_title('Dimensionality Efficiency', fontsize=14, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(component_data.keys())
        axes[0, 1].legend()
        
        # 2D visualizations using first 2 PCA components
        # Sample data for visualization
        n_samples = min(500, len(X_raw_scaled))
        indices = np.random.choice(len(X_raw_scaled), n_samples, replace=False)
        
        pca_2d_raw = PCA(n_components=2)
        pca_2d_eng = PCA(n_components=2)
        
        X_raw_2d = pca_2d_raw.fit_transform(X_raw_scaled[indices])
        X_eng_2d = pca_2d_eng.fit_transform(X_eng_scaled[indices])
        y_sample = self.y[indices]
        
        # Raw features PCA visualization
        scatter1 = axes[0, 2].scatter(X_raw_2d[:, 0], X_raw_2d[:, 1], c=y_sample, 
                                     cmap='viridis', alpha=0.6, s=30)
        axes[0, 2].set_xlabel(f'PC1 ({pca_2d_raw.explained_variance_ratio_[0]:.1%})', fontsize=12)
        axes[0, 2].set_ylabel(f'PC2 ({pca_2d_raw.explained_variance_ratio_[1]:.1%})', fontsize=12)
        axes[0, 2].set_title('PCA: Raw Features', fontsize=14, fontweight='bold')
        plt.colorbar(scatter1, ax=axes[0, 2], ticks=[0, 1, 2], shrink=0.8)
        
        # Engineered features PCA visualization
        scatter2 = axes[1, 0].scatter(X_eng_2d[:, 0], X_eng_2d[:, 1], c=y_sample, 
                                     cmap='viridis', alpha=0.6, s=30)
        axes[1, 0].set_xlabel(f'PC1 ({pca_2d_eng.explained_variance_ratio_[0]:.1%})', fontsize=12)
        axes[1, 0].set_ylabel(f'PC2 ({pca_2d_eng.explained_variance_ratio_[1]:.1%})', fontsize=12)
        axes[1, 0].set_title('PCA: Engineered Features', fontsize=14, fontweight='bold')
        plt.colorbar(scatter2, ax=axes[1, 0], ticks=[0, 1, 2], shrink=0.8)
        
        # Class separability analysis using between/within class variance ratio
        def compute_separability(X, y):
            """Compute Fisher's discriminant ratio for class separability."""
            classes = np.unique(y)
            overall_mean = np.mean(X, axis=0)
            
            # Between-class scatter
            S_B = np.zeros((X.shape[1], X.shape[1]))
            # Within-class scatter  
            S_W = np.zeros((X.shape[1], X.shape[1]))
            
            for cls in classes:
                X_cls = X[y == cls]
                n_cls = len(X_cls)
                mean_cls = np.mean(X_cls, axis=0)
                
                # Between-class
                diff = (mean_cls - overall_mean).reshape(-1, 1)
                S_B += n_cls * np.dot(diff, diff.T)
                
                # Within-class
                S_W += np.dot((X_cls - mean_cls).T, (X_cls - mean_cls))
            
            # Compute ratio of traces (simplified Fisher criterion)
            ratio = np.trace(S_B) / (np.trace(S_W) + 1e-8)
            return ratio
        
        sep_raw = compute_separability(X_raw_scaled, self.y)
        sep_eng = compute_separability(X_eng_scaled, self.y)
        
        # Separability comparison
        sep_data = {'Raw Features': sep_raw, 'Engineered Features': sep_eng}
        bars = axes[1, 1].bar(sep_data.keys(), sep_data.values(), 
                             color=['skyblue', 'orange'], alpha=0.8, edgecolor='black')
        axes[1, 1].set_ylabel('Fisher Discriminant Ratio', fontsize=12)
        axes[1, 1].set_title('Class Separability Analysis', fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, value in zip(bars, sep_data.values()):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{value:.3f}', ha='center', fontweight='bold')
        
        # Feature type analysis - show as horizontal bar chart instead of pie
        feature_categories = self.categorize_engineered_features()
        category_counts = {cat: len(features) for cat, features in feature_categories.items() if len(features) > 0}
        
        # Sort by count for better visualization
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        categories, counts = zip(*sorted_categories) if sorted_categories else ([], [])
        
        # Horizontal bar chart with better spacing
        if categories:
            y_pos = np.arange(len(categories))
            bars = axes[1, 2].barh(y_pos, counts, color=sns.color_palette('Set2', len(categories)), 
                                  alpha=0.8, edgecolor='black')
            axes[1, 2].set_yticks(y_pos)
            axes[1, 2].set_yticklabels([cat.replace(' ', '\n') for cat in categories], fontsize=9)
            axes[1, 2].set_xlabel('Number of Features', fontsize=12)
            axes[1, 2].set_title('Engineered Feature Types', fontsize=14, fontweight='bold')
            axes[1, 2].invert_yaxis()
            
            # Add count labels on bars
            for i, (bar, count) in enumerate(zip(bars, counts)):
                axes[1, 2].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                               str(count), ha='left', va='center', fontweight='bold', fontsize=9)
        else:
            axes[1, 2].text(0.5, 0.5, 'No categorized features', ha='center', va='center',
                           transform=axes[1, 2].transAxes, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dimensionality_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return sep_raw, sep_eng
    
    def categorize_engineered_features(self):
        """Categorize engineered features by type with better classification."""
        categories = {
            'Moving Averages': [],
            'Anomaly Detection': [],
            'Trend Analysis': [],
            'Rate/Derivatives': [],
            'Statistical Features': [],
            'Performance Metrics': [],
            'Z-Score Normalization': [],
            'Other Engineering': []
        }
        
        for feature in self.new_features:
            feature_lower = feature.lower()
            
            # More specific categorization based on actual engineered features
            if any(keyword in feature_lower for keyword in ['_ma_', 'moving_avg', 'rolling']):
                categories['Moving Averages'].append(feature)
            elif any(keyword in feature_lower for keyword in ['is_anomaly', '_anomaly', 'outlier']):
                categories['Anomaly Detection'].append(feature)
            elif any(keyword in feature_lower for keyword in ['_trend', 'trend_', 'slope']):
                categories['Trend Analysis'].append(feature)
            elif any(keyword in feature_lower for keyword in ['_rate', 'rate_', 'derivative', '_diff']):
                categories['Rate/Derivatives'].append(feature)
            elif any(keyword in feature_lower for keyword in ['_zscore', 'zscore_', 'normalized']):
                categories['Z-Score Normalization'].append(feature)
            elif any(keyword in feature_lower for keyword in ['response_time', 'latency', 'duration', 'throughput']):
                categories['Performance Metrics'].append(feature)
            elif any(keyword in feature_lower for keyword in ['_mean', '_max', '_min', '_std', '_var', '_sum']):
                categories['Statistical Features'].append(feature)
            else:
                categories['Other Engineering'].append(feature)
        
        return categories
    
    def plot_mutual_information_analysis(self):
        """Analyze mutual information between features and target."""
        print("🔍 Computing mutual information analysis...")
        
        # Compute mutual information for both feature sets
        mi_raw = mutual_info_classif(self.X_raw, self.y, random_state=42)
        mi_eng = mutual_info_classif(self.X_eng, self.y, random_state=42)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Mutual information distribution comparison
        axes[0, 0].hist(mi_raw, bins=30, alpha=0.7, label='Raw Features', 
                       color='skyblue', edgecolor='black', density=True)
        axes[0, 0].hist(mi_eng, bins=30, alpha=0.7, label='Engineered Features', 
                       color='orange', edgecolor='black', density=True)
        axes[0, 0].set_xlabel('Mutual Information', fontsize=12)
        axes[0, 0].set_ylabel('Density', fontsize=12)
        axes[0, 0].set_title('Mutual Information Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Average mutual information comparison
        avg_mi_data = {'Raw Features': np.mean(mi_raw), 'Engineered Features': np.mean(mi_eng)}
        bars = axes[0, 1].bar(avg_mi_data.keys(), avg_mi_data.values(), 
                             color=['skyblue', 'orange'], alpha=0.8, edgecolor='black')
        axes[0, 1].set_ylabel('Average Mutual Information', fontsize=12)
        axes[0, 1].set_title('Average Mutual Information Comparison', fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, value in zip(bars, avg_mi_data.values()):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{value:.4f}', ha='center', fontweight='bold')
        
        # Top mutual information features (raw)
        top_mi_raw_indices = np.argsort(mi_raw)[-8:][::-1]
        axes[1, 0].barh(range(len(top_mi_raw_indices)), mi_raw[top_mi_raw_indices],
                       color='skyblue', alpha=0.8, edgecolor='black')
        axes[1, 0].set_yticks(range(len(top_mi_raw_indices)))
        # Better truncation for mutual information plot
        truncated_mi_raw_names = [name[:22] + '...' if len(name) > 22 else name 
                                 for name in [self.raw_feature_names[i] for i in top_mi_raw_indices]]
        axes[1, 0].set_yticklabels(truncated_mi_raw_names, fontsize=9)
        axes[1, 0].set_xlabel('Mutual Information', fontsize=12)
        axes[1, 0].set_title('Top Raw Features (MI)', fontsize=12, fontweight='bold')
        axes[1, 0].invert_yaxis()
        
        # Top mutual information features (engineered)
        top_mi_eng_indices = np.argsort(mi_eng)[-8:][::-1]
        axes[1, 1].barh(range(len(top_mi_eng_indices)), mi_eng[top_mi_eng_indices],
                       color='orange', alpha=0.8, edgecolor='black')
        axes[1, 1].set_yticks(range(len(top_mi_eng_indices)))
        # Better truncation for mutual information plot
        truncated_mi_eng_names = [name[:22] + '...' if len(name) > 22 else name 
                                 for name in [self.eng_feature_names[i] for i in top_mi_eng_indices]]
        axes[1, 1].set_yticklabels(truncated_mi_eng_names, fontsize=9)
        axes[1, 1].set_xlabel('Mutual Information', fontsize=12)
        axes[1, 1].set_title('Top Engineered Features (MI)', fontsize=12, fontweight='bold')
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mutual_information_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return np.mean(mi_raw), np.mean(mi_eng)
    
    def generate_comprehensive_report(self):
        """Generate all feature engineering analysis visualizations."""
        print("\n🎨 Generating comprehensive feature engineering analysis...")
        
        analyses = [
            ("Feature Distribution Comparison", self.plot_feature_distribution_comparison),
            ("Feature Importance Comparison", self.plot_feature_importance_comparison),
            ("Dimensionality Analysis", self.plot_dimensionality_analysis),
            ("Mutual Information Analysis", self.plot_mutual_information_analysis),
        ]
        
        results = {}
        
        for analysis_name, analysis_func in analyses:
            print(f"  📊 Generating: {analysis_name}")
            try:
                result = analysis_func()
                if result:
                    results[analysis_name] = result
                print(f"  ✅ Completed: {analysis_name}")
            except Exception as e:
                print(f"  ❌ Error in {analysis_name}: {str(e)}")
        
        # Generate summary report
        self.generate_summary_report(results)
        
        print(f"\n🎉 All feature engineering analysis saved to: {self.output_dir}")
        print("📁 Generated files:")
        for file in sorted(self.output_dir.glob("*.png")):
            print(f"  - {file.name}")
    
    def generate_summary_report(self, results):
        """Generate a comprehensive summary of feature engineering effectiveness."""
        feature_categories = self.categorize_engineered_features()
        
        summary = {
            'feature_engineering_summary': {
                'original_features': len(self.raw_feature_names),
                'engineered_features': len(self.eng_feature_names),
                'new_features_added': len(self.new_features),
                'feature_expansion_ratio': len(self.eng_feature_names) / len(self.raw_feature_names),
                'feature_categories': {cat: len(features) for cat, features in feature_categories.items()}
            },
            'performance_improvements': results
        }
        
        # Save as JSON
        import json
        with open(self.output_dir / 'feature_engineering_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else x)
        
        # Generate LaTeX summary table
        self.generate_feature_latex_table(summary)
    
    def generate_feature_latex_table(self, summary):
        """Generate LaTeX table for feature engineering results."""
        latex_content = """
\\begin{table}[htbp]
\\centering
\\caption{Feature Engineering Impact Analysis}
\\label{tab:feature_engineering}
\\begin{tabular}{lc}
\\hline
\\textbf{Metric} & \\textbf{Value} \\\\
\\hline
Original Features & """ + f"{summary['feature_engineering_summary']['original_features']}" + """ \\\\
Engineered Features & """ + f"{summary['feature_engineering_summary']['engineered_features']}" + """ \\\\
New Features Added & """ + f"{summary['feature_engineering_summary']['new_features_added']}" + """ \\\\
Feature Expansion Ratio & """ + f"{summary['feature_engineering_summary']['feature_expansion_ratio']:.2f}" + """x \\\\
\\hline
\\end{tabular}
\\end{table}
"""
        
        with open(self.output_dir / 'feature_engineering_table.tex', 'w') as f:
            f.write(latex_content)


def main():
    parser = argparse.ArgumentParser(description="Feature Engineering Analysis")
    parser.add_argument("--raw", default="processed_data/dataset.parquet", 
                       help="Path to raw dataset")
    parser.add_argument("--engineered", default="processed_data/engineered_features.parquet", 
                       help="Path to engineered dataset")
    parser.add_argument("--output", default="feature_analysis", 
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    raw_path = script_dir / args.raw
    eng_path = script_dir / args.engineered
    
    print("🔬 Starting Feature Engineering Analysis")
    print("=" * 60)
    
    analyzer = FeatureEngineeringAnalyzer(str(raw_path), str(eng_path), args.output)
    analyzer.generate_comprehensive_report()
    
    print("\n🎓 Feature engineering analysis complete!")


if __name__ == "__main__":
    main() 