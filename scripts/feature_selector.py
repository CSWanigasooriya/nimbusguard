#!/usr/bin/env python3
"""
Simplified Consumer Pod DQN Feature Selector
==========================================

Focuses exclusively on CPU and Memory metrics for scaling decisions:
1. process_cpu_seconds_total_rate: CPU usage rate (seconds/second)
2. process_resident_memory_bytes: Current memory usage (bytes)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Set
import json
import re
from sklearn.preprocessing import RobustScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SimplifiedConsumerFeatureSelector:
    """Simplified feature selector focusing on CPU and Memory metrics only."""
    
    def __init__(self, data_dir: Path, target_features: int = 2):
        self.data_dir = Path(data_dir)
        self.target_features = target_features  # Only need 2 features now
        self.consumer_metrics = []
        self.metric_analysis = {}
        
        # Resource targets matching Kubernetes configuration
        self.cpu_limit_cores = 0.5  # 500m
        self.memory_limit_mb = 1024  # 1Gi
        self.cpu_target_utilization = 0.7  # 70%
        self.memory_target_utilization = 0.8  # 80%
        
    def discover_consumer_metrics(self) -> List[str]:
        """Discover CPU and Memory metrics for consumer pods."""
        logger.info("🔍 Discovering consumer pod CPU and Memory metrics...")
        
        consumer_metrics = []
        csv_files = list(self.data_dir.glob("*.csv"))
        
        # Only look for CPU and memory related metrics
        target_metrics = [
            'process_cpu_seconds_total',
            'process_resident_memory_bytes'
        ]
        
        for csv_file in csv_files:
            metric_name = csv_file.stem
            
            # Only include our target metrics
            if any(target in metric_name for target in target_metrics):
                try:
                    # Quick sample to check for consumer data
                    df_sample = pd.read_csv(csv_file, nrows=10)
                    if 'instance' in df_sample.columns:
                        if df_sample['instance'].str.contains(':8000', na=False).any():
                            consumer_metrics.append(metric_name)
                            logger.info(f"  ✓ Found consumer metric: {metric_name}")
                except Exception as e:
                    logger.warning(f"  ⚠️ Could not analyze {csv_file.name}: {e}")
        
        logger.info(f"📊 Discovered {len(consumer_metrics)} relevant consumer pod metrics")
        self.consumer_metrics = consumer_metrics
        return consumer_metrics

    def analyze_metric_characteristics(self, metric_name: str, series: pd.Series) -> Dict:
        """Analyze CPU and Memory metric characteristics."""
        
        # Basic statistical analysis
        values = series.dropna()
        if len(values) < 2:
            return {'type': 'invalid', 'category': 'unknown', 'scaling_relevance': 0.0}
        
        mean_val = values.mean()
        std_val = values.std()
        min_val = values.min()
        max_val = values.max()
        
        # Detect metric type
        metric_type = self._detect_metric_type(metric_name, values)
        
        # Categorize as CPU or Memory
        category = self._categorize_metric(metric_name)
        
        # Assess scaling relevance
        scaling_relevance = self._assess_scaling_relevance(metric_name, values, category)
        
        # Calculate variability score
        coeff_var = std_val / (mean_val + 1e-10) if mean_val != 0 else 0
        range_score = (max_val - min_val) / (mean_val + 1e-10) if mean_val != 0 else 0
        
        return {
            'type': metric_type,
            'category': category,
            'scaling_relevance': scaling_relevance,
            'variability': coeff_var + range_score,
            'is_core_metric': True,  # All our metrics are core now
            'stats': {
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'samples': len(values)
            }
        }
    
    def _detect_metric_type(self, name: str, values: pd.Series) -> str:
        """Detect if metric is counter or gauge for CPU and Memory."""
        
        name_lower = name.lower()
        
        # CPU seconds total is a counter
        if 'cpu_seconds_total' in name_lower:
            return 'counter'
        
        # Memory bytes is a gauge (current state)
        if 'memory_bytes' in name_lower:
            return 'gauge'
        
        # Data behavior analysis for edge cases
        if len(values) > 10:
            clean_values = values.dropna()
            if len(clean_values) > 5:
                # Check for monotonic increase (counter indicator)
                diffs = clean_values.diff().dropna()
                non_decreasing_ratio = (diffs >= -0.01).sum() / len(diffs)
                positive_trend = clean_values.iloc[-1] > clean_values.iloc[0]
                
                if non_decreasing_ratio > 0.85 and positive_trend:
                    logger.info(f"  📊 Detected COUNTER from data: {name}")
                    return 'counter'
        
        return 'gauge'
    
    def _categorize_metric(self, name: str) -> str:
        """Categorize as CPU or Memory."""
        name_lower = name.lower()
        
        if 'cpu' in name_lower:
            return 'cpu'
        elif 'memory' in name_lower:
            return 'memory'
        else:
            return 'other'
    
    def _assess_scaling_relevance(self, name: str, values: pd.Series, category: str) -> float:
        """Assess scaling relevance for CPU and Memory metrics."""
        name_lower = name.lower()
        
        # Both CPU and Memory are critical for scaling
        if 'process_resident_memory_bytes' in name_lower:
            return 100.0  # Memory usage is critical
        elif 'process_cpu_seconds_total' in name_lower:
            return 95.0   # CPU usage is critical
        
        return 0.0  # No other metrics should have relevance

    def calculate_rate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rate metrics from CPU counter only."""
        logger.info("🧮 Calculating rate metrics from CPU counter...")
        
        rate_df = df.copy()
        
        # Only calculate rate for CPU counter
        cpu_counter = 'process_cpu_seconds_total'
        
        for column in df.columns:
            if column == cpu_counter and column in self.metric_analysis:
                analysis = self.metric_analysis[column]
                
                if analysis['type'] == 'counter':
                    # Calculate per-second rate
                    rate_series = df[column].diff() / 60  # 60-second intervals
                    rate_series = rate_series.fillna(0).clip(lower=0)  # Remove negative values
                    
                    rate_column = f"{column}_rate"
                    rate_df[rate_column] = rate_series
                    
                    # Update analysis for rate metric
                    self.metric_analysis[rate_column] = {
                        'type': 'gauge',
                        'category': 'cpu',
                        'scaling_relevance': 100.0,  # CPU rate is critical
                        'variability': rate_series.std() / (rate_series.mean() + 1e-10),
                        'is_core_metric': True,
                        'derived_from': column
                    }
                    
                    logger.info(f"  ✓ Created CPU rate metric: {rate_column}")
        
        return rate_df
    
    def select_core_features(self, ranked_features: List[str]) -> List[str]:
        """Select the two core features required for DQN."""
        logger.info("🎯 Selecting core CPU and Memory features...")
        
        # Required features for DQN
        required_features = [
            'process_cpu_seconds_total_rate',  # CPU usage rate
            'process_resident_memory_bytes'    # Memory usage
        ]
        
        selected = []
        
        # Select required features
        for required_feature in required_features:
            found = False
            
            # Try exact match first
            for feature in ranked_features:
                if feature == required_feature:
                    selected.append(feature)
                    logger.info(f"  ✓ Selected required: {feature}")
                    found = True
                    break
            
            if not found:
                # Try variations
                if 'cpu_seconds_total_rate' in required_feature:
                    variations = ['process_cpu_seconds_total_rate', 'cpu_seconds_total_rate']
                elif 'memory_bytes' in required_feature:
                    variations = ['process_resident_memory_bytes', 'resident_memory_bytes']
                else:
                    variations = []
                
                for variation in variations:
                    for feature in ranked_features:
                        if variation in feature and feature not in selected:
                            selected.append(feature)
                            logger.info(f"  ✓ Selected variation: {feature} (for {required_feature})")
                            found = True
                            break
                    if found:
                        break
            
            if not found:
                logger.error(f"  ❌ Could not find required feature: {required_feature}")
                raise ValueError(f"Missing critical feature: {required_feature}")
        
        logger.info(f"📊 Selected {len(selected)} core features: {selected}")
        return selected
    
    def load_and_analyze_data(self) -> pd.DataFrame:
        """Load consumer data and perform analysis."""
        
        # Discover metrics
        self.discover_consumer_metrics()
        
        if not self.consumer_metrics:
            raise ValueError("No relevant CPU/Memory consumer data found!")
        
        # Load all consumer metric data
        all_data = []
        for metric in self.consumer_metrics:
            file_path = self.data_dir / f"{metric}.csv"
            try:
                df = pd.read_csv(file_path)
                if 'instance' in df.columns:
                    consumer_data = df[df['instance'].str.contains(':8000', na=False)].copy()
                    if len(consumer_data) > 0:
                        consumer_data['metric'] = metric
                        all_data.append(consumer_data)
            except Exception as e:
                logger.warning(f"Failed to load {metric}: {e}")
        
        if not all_data:
            raise ValueError("No consumer data found!")
        
        # Combine and pivot
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        
        feature_df = combined_df.pivot_table(
            index='timestamp',
            columns='metric',
            values='value',
            aggfunc='mean'
        ).fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Analyze each metric
        logger.info("🔬 Analyzing CPU and Memory metric characteristics...")
        for column in feature_df.columns:
            self.metric_analysis[column] = self.analyze_metric_characteristics(column, feature_df[column])
        
        # Calculate rate metrics (only for CPU)
        feature_df = self.calculate_rate_metrics(feature_df)
        
        return feature_df
    
    def rank_features_intelligently(self, df: pd.DataFrame) -> List[str]:
        """Rank CPU and Memory features by importance."""
        logger.info("🧠 Ranking CPU and Memory features...")
        
        feature_scores = {}
        
        for column in df.columns:
            if column not in self.metric_analysis:
                continue
                
            analysis = self.metric_analysis[column]
            
            # Base score from scaling relevance
            base_score = analysis['scaling_relevance']
            
            # Variability bonus (metrics that change are more useful)
            variability_score = analysis['variability'] * 10
            
            # Type-based scoring
            type_bonus = 0.0
            if analysis['type'] == 'gauge':
                type_bonus = 30.0  # Current state is important
            elif analysis['type'] == 'counter':
                type_bonus = -20.0  # Raw counters less useful
                
            # Rate bonus
            rate_bonus = 20.0 if 'rate' in column else 0.0
            
            # Final score
            final_score = base_score + variability_score + type_bonus + rate_bonus
            feature_scores[column] = max(final_score, 0.0)
            
            logger.info(f"  📊 {column:<40} (score: {final_score:.1f}, type: {analysis['type']}, category: {analysis['category']})")
        
        # Sort by score
        ranked = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("🏆 Ranked features:")
        for i, (feature, score) in enumerate(ranked, 1):
            analysis = self.metric_analysis.get(feature, {})
            logger.info(f"  {i}. {feature:<40} (score: {score:.1f})")
        
        return [feature for feature, score in ranked]
    
    def generate_scaling_actions(self, df: pd.DataFrame) -> pd.Series:
        """Generate realistic scaling actions based on CPU and Memory usage."""
        actions = []
        
        # Find CPU and Memory metrics
        cpu_rate_col = None
        memory_col = None
        
        for col in df.columns:
            if 'cpu_seconds_total_rate' in col:
                cpu_rate_col = col
            elif 'resident_memory_bytes' in col:
                memory_col = col
        
        logger.info(f"Using metrics for scaling decisions: CPU={cpu_rate_col}, Memory={memory_col}")
        
        for i, row in df.iterrows():
            pressure_score = 0.0
            
            # CPU pressure (per-replica basis)
            if cpu_rate_col and cpu_rate_col in row:
                cpu_rate = row[cpu_rate_col]
                # Assume 1 replica for simplicity, adjust pressure based on target
                cpu_target = self.cpu_limit_cores * self.cpu_target_utilization  # 0.5 * 0.7 = 0.35
                cpu_pressure = min(cpu_rate / cpu_target, 2.0)  # Cap at 2x target
                pressure_score += cpu_pressure * 0.6  # CPU weighted 60%
            
            # Memory pressure (per-replica basis)
            if memory_col and memory_col in row:
                memory_bytes = row[memory_col]
                # Target memory in bytes
                memory_target = self.memory_limit_mb * 1024 * 1024 * self.memory_target_utilization  # 1024MB * 0.8
                memory_pressure = min(memory_bytes / memory_target, 2.0)  # Cap at 2x target
                pressure_score += memory_pressure * 0.4  # Memory weighted 40%
            
            # Scaling decisions based on combined pressure
            if pressure_score > 1.2:  # 120% of target capacity
                actions.append('scale_up')
            elif pressure_score < 0.5:  # 50% of target capacity
                actions.append('scale_down')
            else:
                actions.append('keep_same')
        
        action_counts = pd.Series(actions).value_counts()
        logger.info(f"📊 Generated scaling actions: {action_counts.to_dict()}")
        
        return pd.Series(actions, index=df.index)
    
    def create_dqn_scaler(self, feature_df: pd.DataFrame, output_dir: Path):
        """Create a scaler for the two core features."""
        logger.info("🎯 Creating DQN-compatible scaler for CPU and Memory...")
        
        # DQN expects these exact features
        dqn_expected_features = [
            'process_cpu_seconds_total_rate',
            'process_resident_memory_bytes'
        ]
        
        # Find available features in the data
        available_features = []
        feature_mapping = {}
        
        for expected_feature in dqn_expected_features:
            found = False
            
            # Try exact match first
            if expected_feature in feature_df.columns:
                available_features.append(expected_feature)
                feature_mapping[expected_feature] = expected_feature
                found = True
                logger.info(f"  ✓ Found exact match: {expected_feature}")
            else:
                # Try variations
                if 'cpu_seconds_total_rate' in expected_feature:
                    variations = ['cpu_seconds_total_rate', 'process_cpu_seconds_total_rate']
                elif 'memory_bytes' in expected_feature:
                    variations = ['resident_memory_bytes', 'process_resident_memory_bytes']
                else:
                    variations = []
                
                for variation in variations:
                    if variation in feature_df.columns:
                        available_features.append(variation)
                        feature_mapping[expected_feature] = variation
                        found = True
                        logger.info(f"  ✓ Found variation: {expected_feature} -> {variation}")
                        break
            
            if not found:
                logger.error(f"  ❌ Missing critical feature: {expected_feature}")
                raise ValueError(f"Missing critical feature: {expected_feature}")
        
        # Create dataframe with exact DQN feature names and order
        dqn_feature_df = pd.DataFrame()
        for expected_feature in dqn_expected_features:
            source_feature = feature_mapping[expected_feature]
            dqn_feature_df[expected_feature] = feature_df[source_feature]
        
        # Create and fit the scaler
        dqn_scaler = RobustScaler()
        dqn_scaler.fit(dqn_feature_df)
        
        # Save as .pkl for DQN compatibility
        scaler_path = output_dir / 'feature_scaler.pkl'
        joblib.dump(dqn_scaler, scaler_path)
        
        logger.info(f"✅ DQN-compatible scaler saved to {scaler_path}")
        logger.info(f"   Scaler fitted on {len(dqn_expected_features)} features: {dqn_expected_features}")
        
        # Create feature mapping log
        mapping_path = output_dir / 'dqn_feature_mapping.json'
        with open(mapping_path, 'w') as f:
            json.dump(feature_mapping, f, indent=2)
        logger.info(f"   Feature mapping saved to {mapping_path}")
        
        return dqn_scaler
    
    def validate_dqn_compatibility(self, output_dir: Path) -> bool:
        """Validate that created files are compatible with DQN agent."""
        logger.info("🔍 Validating DQN compatibility...")
        
        try:
            # Expected DQN features
            expected_features = [
                'process_cpu_seconds_total_rate',
                'process_resident_memory_bytes'
            ]
            
            # 1. Check if scaler file exists
            scaler_path = output_dir / 'feature_scaler.pkl'
            if not scaler_path.exists():
                logger.error(f"❌ Scaler file not found: {scaler_path}")
                return False
            
            # 2. Load and validate scaler
            try:
                scaler = joblib.load(scaler_path)
                logger.info(f"✅ Scaler loaded successfully: {type(scaler).__name__}")
                
                if not hasattr(scaler, 'transform') or not hasattr(scaler, 'fit'):
                    logger.error("❌ Invalid scaler: missing transform/fit methods")
                    return False
                
            except Exception as e:
                logger.error(f"❌ Failed to load scaler: {e}")
                return False
            
            # 3. Test scaler with expected features
            try:
                # Create test data with realistic values
                test_data = pd.DataFrame({
                    'process_cpu_seconds_total_rate': [0.8],  # 80% of one core
                    'process_resident_memory_bytes': [200_000_000.0]  # 200MB
                })
                
                # Test transform
                scaled_data = scaler.transform(test_data)
                
                if scaled_data.shape != (1, len(expected_features)):
                    logger.error(f"❌ Scaler output shape mismatch: expected (1, {len(expected_features)}), got {scaled_data.shape}")
                    return False
                
                # Check for reasonable scaling
                if np.any(np.abs(scaled_data) > 10):
                    logger.warning(f"⚠️ Some scaled values are extreme: {scaled_data[0]}")
                else:
                    logger.info(f"✅ Scaled values look reasonable: {scaled_data[0]}")
                
                # Check for NaN or infinite values
                if np.any(~np.isfinite(scaled_data)):
                    logger.error("❌ Scaled values contain NaN or infinite values")
                    return False
                
                logger.info(f"✅ Scaler transform works correctly")
                
            except Exception as e:
                logger.error(f"❌ Scaler transform failed: {e}")
                return False
            
            # 4. Test with various realistic scenarios
            test_scenarios = [
                {
                    'name': 'Low Load',
                    'process_cpu_seconds_total_rate': 0.1,
                    'process_resident_memory_bytes': 100_000_000.0  # 100MB
                },
                {
                    'name': 'Medium Load',
                    'process_cpu_seconds_total_rate': 0.35,
                    'process_resident_memory_bytes': 350_000_000.0  # 350MB
                },
                {
                    'name': 'High Load',
                    'process_cpu_seconds_total_rate': 0.6,
                    'process_resident_memory_bytes': 800_000_000.0  # 800MB
                }
            ]
            
            logger.info("🧪 Testing with realistic scenarios:")
            for scenario in test_scenarios:
                try:
                    test_df = pd.DataFrame([{
                        'process_cpu_seconds_total_rate': scenario['process_cpu_seconds_total_rate'],
                        'process_resident_memory_bytes': scenario['process_resident_memory_bytes']
                    }])
                    
                    scaled_result = scaler.transform(test_df)
                    logger.info(f"  ✅ {scenario['name']}: CPU={scenario['process_cpu_seconds_total_rate']}, "
                              f"Memory={scenario['process_resident_memory_bytes']/1e6:.0f}MB -> "
                              f"Scaled=[{scaled_result[0][0]:.2f}, {scaled_result[0][1]:.2f}]")
                    
                except Exception as e:
                    logger.error(f"  ❌ {scenario['name']} failed: {e}")
                    return False
            
            logger.info("🎉 All DQN compatibility checks PASSED!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Validation failed with error: {e}")
            return False
    
    def run(self, output_dir: Path):
        """Execute simplified feature selection pipeline."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        logger.info("🚀 Starting simplified CPU and Memory feature selection...")
        
        # Load and analyze data
        feature_df = self.load_and_analyze_data()
        
        # Rank features
        ranked_features = self.rank_features_intelligently(feature_df)
        
        # Select core features
        selected_features = self.select_core_features(ranked_features)
        
        # Generate scaling actions
        scaling_actions = self.generate_scaling_actions(feature_df)
        
        # Create final dataset with only the selected features
        final_df = feature_df[selected_features].copy()
        final_df['scaling_action'] = scaling_actions
        
        # Create DQN-compatible scaler
        dqn_scaler = self.create_dqn_scaler(feature_df, output_dir)
        
        # Scale features using DQN scaler
        final_df[selected_features] = dqn_scaler.transform(final_df[selected_features])
        
        # Save outputs
        final_df.to_parquet(output_dir / 'dqn_features.parquet')
        
        # Create comprehensive metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'methodology': 'Simplified CPU and Memory focused feature selection',
            'discovered_metrics': len(self.consumer_metrics),
            'analyzed_metrics': len(self.metric_analysis),
            'selected_features': selected_features,
            'metric_analysis': self.metric_analysis,
            'dataset_shape': list(final_df.shape),
            'scaling_action_distribution': scaling_actions.value_counts().to_dict(),
            'resource_targets': {
                'cpu_limit_cores': self.cpu_limit_cores,
                'memory_limit_mb': self.memory_limit_mb,
                'cpu_target_utilization': self.cpu_target_utilization,
                'memory_target_utilization': self.memory_target_utilization
            }
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create summary
        summary = f"""# Simplified Consumer Pod Feature Selection Results

## Methodology
- **Focus**: CPU and Memory metrics only
- **Auto-discovered**: {len(self.consumer_metrics)} relevant consumer pod metrics
- **Auto-analyzed**: {len(self.metric_analysis)} metric characteristics  
- **Auto-calculated**: CPU rate from counter
- **Resource Targets**: {self.cpu_limit_cores} CPU cores, {self.memory_limit_mb}MB memory

## Dataset Summary
- **Samples**: {len(final_df)}
- **Features**: {len(selected_features)}
- **Scaling Actions**: {scaling_actions.value_counts().to_dict()}

## Selected Features
"""
        
        for i, feature in enumerate(selected_features, 1):
            analysis = self.metric_analysis.get(feature, {})
            category = analysis.get('category', 'unknown')
            metric_type = analysis.get('type', 'unknown')
            relevance = analysis.get('scaling_relevance', 0)
            summary += f"{i}. **`{feature}`** (category: {category}, type: {metric_type}, relevance: {relevance:.1f})\n"
        
        summary += f"""
## Resource Configuration
- **CPU Limit**: {self.cpu_limit_cores} cores (500m)
- **Memory Limit**: {self.memory_limit_mb}MB (1Gi)
- **CPU Target**: {self.cpu_target_utilization*100}% utilization
- **Memory Target**: {self.memory_target_utilization*100}% utilization

## Scaling Logic
- **Scale Up**: When pressure score > 120% of target capacity
- **Scale Down**: When pressure score < 50% of target capacity
- **Keep Same**: When pressure score is between 50-120% of target
"""
        
        with open(output_dir / 'summary.md', 'w') as f:
            f.write(summary)
        
        logger.info(f"✅ Simplified feature selection complete! Output saved to {output_dir}")
        logger.info(f"Selected features: {selected_features}")
        
        # Validate DQN compatibility
        logger.info("\n" + "="*60)
        if self.validate_dqn_compatibility(output_dir):
            logger.info("🎉 SUCCESS: All files are DQN-compatible!")
        else:
            logger.error("❌ FAILURE: DQN compatibility issues found!")
        logger.info("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simplified Consumer Pod DQN Feature Selector")
    parser.add_argument("--data-dir", required=True, help="Directory containing CSV files")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--target-features", type=int, default=2, help="Number of features to select (2 for CPU+Memory)")
    
    args = parser.parse_args()
    
    selector = SimplifiedConsumerFeatureSelector(
        data_dir=args.data_dir,
        target_features=args.target_features
    )
    
    selector.run(output_dir=args.output_dir)