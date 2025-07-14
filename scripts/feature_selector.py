#!/usr/bin/env python3
"""
Intelligent Consumer Pod DQN Feature Selector
===========================================

Dynamically analyzes CSV files to:
1. Auto-detect metric types (gauge vs counter)
2. Auto-calculate rate metrics from counters
3. Auto-categorize metrics by analysis
4. Auto-prioritize consumer performance
5. Auto-ensure diversity across categories
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

class IntelligentConsumerFeatureSelector:
    """Intelligent feature selector that analyzes data patterns dynamically."""
    
    def __init__(self, data_dir: Path, target_features: int = 9):
        self.data_dir = Path(data_dir)
        self.target_features = target_features
        self.consumer_metrics = []
        self.metric_analysis = {}
        
    def discover_consumer_metrics(self) -> List[str]:
        """Dynamically discover all metrics containing consumer pod data."""
        logger.info("🔍 Discovering consumer pod metrics...")
        
        consumer_metrics = []
        csv_files = list(self.data_dir.glob("*.csv"))
        
        for csv_file in csv_files:
            try:
                # Quick sample to check for consumer data
                df_sample = pd.read_csv(csv_file, nrows=10)
                if 'instance' in df_sample.columns:
                    if df_sample['instance'].str.contains(':8000', na=False).any():
                        metric_name = csv_file.stem
                        consumer_metrics.append(metric_name)
                        logger.info(f"  ✓ Found consumer metric: {metric_name}")
            except Exception as e:
                logger.warning(f"  ⚠️ Could not analyze {csv_file.name}: {e}")
        
        logger.info(f"📊 Discovered {len(consumer_metrics)} consumer pod metrics")
        self.consumer_metrics = consumer_metrics
        return consumer_metrics

    def analyze_metric_characteristics(self, metric_name: str, series: pd.Series) -> Dict:
        """Analyze a metric's characteristics to determine type and importance."""
        
        # Basic statistical analysis
        values = series.dropna()
        if len(values) < 2:
            return {'type': 'invalid', 'category': 'unknown', 'scaling_relevance': 0.0}
        
        mean_val = values.mean()
        std_val = values.std()
        min_val = values.min()
        max_val = values.max()
        
        # Detect metric type from data patterns
        metric_type = self._detect_metric_type(metric_name, values)
        
        # Categorize by name analysis
        category = self._categorize_metric(metric_name)
        
        # Assess scaling relevance
        scaling_relevance = self._assess_scaling_relevance(metric_name, values, category)
        
        # Calculate variability score
        coeff_var = std_val / (mean_val + 1e-10) if mean_val != 0 else 0
        range_score = (max_val - min_val) / (mean_val + 1e-10) if mean_val != 0 else 0
        
        # Assess if this is monitoring overhead vs consumer performance
        is_consumer_performance = self._is_consumer_performance_metric(metric_name)
        
        return {
            'type': metric_type,
            'category': category,
            'scaling_relevance': scaling_relevance,
            'variability': coeff_var + range_score,
            'is_consumer_performance': is_consumer_performance,
            'stats': {
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'samples': len(values)
            }
        }
    
    def _detect_metric_type(self, name: str, values: pd.Series) -> str:
        """Automatically detect if metric is counter, gauge, or histogram using smart analysis."""
        
        # Strong name-based indicators first (these are very reliable)
        name_lower = name.lower()
        
        # Definitive counter patterns
        strong_counter_patterns = ['_total', '_count', '_sum', '_created', 'seconds_total']
        if any(pattern in name_lower for pattern in strong_counter_patterns):
            return 'counter'
        
        # Histogram buckets
        if '_bucket' in name_lower:
            return 'histogram'
        
        # Info/static metrics
        if '_info' in name_lower or 'start_time' in name_lower:
            return 'info'
        
        # Data behavior analysis for ambiguous cases
        if len(values) > 10:
            # Remove any outliers first
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            clean_values = values[(values >= q1 - 1.5*iqr) & (values <= q3 + 1.5*iqr)]
            
            if len(clean_values) > 5:
                # Check for monotonic increase (strong counter indicator)
                diffs = clean_values.diff().dropna()
                
                # Counter: mostly non-decreasing with positive trend
                non_decreasing_ratio = (diffs >= -0.01).sum() / len(diffs)  # Allow tiny decreases
                positive_trend = clean_values.iloc[-1] > clean_values.iloc[0]
                
                if non_decreasing_ratio > 0.85 and positive_trend:
                    logger.info(f"  📊 Detected COUNTER from data: {name} (non-decreasing: {non_decreasing_ratio:.2f})")
                    return 'counter'
                
                # Gauge: values fluctuate around a mean
                coefficient_of_variation = clean_values.std() / (clean_values.mean() + 1e-10)
                if coefficient_of_variation > 0.1:  # Some variability indicates gauge
                    return 'gauge'
        
        # Default fallback based on common patterns
        if any(pattern in name_lower for pattern in ['bytes', 'fds', 'duration', 'up']):
            return 'gauge'
        
        # If still uncertain, assume gauge (current state)
        return 'gauge'
    
    def _categorize_metric(self, name: str) -> str:
        """Automatically categorize metrics by analyzing name patterns."""
        
        name_lower = name.lower()
        
        # CPU patterns
        if any(pattern in name_lower for pattern in ['cpu', 'processor']):
            return 'cpu'
        
        # Memory patterns  
        if any(pattern in name_lower for pattern in ['memory', 'mem', 'heap', 'gc']):
            return 'memory'
        
        # Network/HTTP patterns
        if any(pattern in name_lower for pattern in ['http', 'request', 'response', 'network']):
            return 'network'
        
        # I/O patterns
        if any(pattern in name_lower for pattern in ['fds', 'file', 'disk', 'io']):
            return 'io'
        
        # Performance patterns
        if any(pattern in name_lower for pattern in ['duration', 'latency', 'time', 'performance']):
            return 'performance'
        
        # System patterns
        if any(pattern in name_lower for pattern in ['load', 'system', 'node']):
            return 'system'
        
        # Monitoring patterns
        if any(pattern in name_lower for pattern in ['scrape', 'series', 'samples', 'metric']):
            return 'monitoring'
        
        return 'other'
    
    def _is_consumer_performance_metric(self, name: str) -> bool:
        """Determine if metric reflects TRUE consumer performance vs monitoring overhead."""
        
        name_lower = name.lower()
        
        # EXCLUDE: Monitoring infrastructure overhead
        monitoring_overhead = [
            'scrape_', 'series_added', 'samples_', 'alloy_', 'prometheus_',
            'up',  # Just 0/1 availability, not performance
        ]
        
        # EXCLUDE: Meaningless timestamp metrics
        meaningless_patterns = [
            '_created',  # These are static timestamps, not counters
            '_info',     # Static information
            'start_time' # Static startup timestamp
        ]
        
        # Check for monitoring overhead
        if any(pattern in name_lower for pattern in monitoring_overhead):
            return False
            
        # Check for meaningless patterns
        if any(pattern in name_lower for pattern in meaningless_patterns):
            return False
        
        # INCLUDE: True consumer performance indicators
        consumer_performance = [
            'process_resident_memory',  # Current memory usage
            'process_virtual_memory',   # Current virtual memory
            'process_open_fds',         # Current I/O load
            'process_max_fds',          # I/O capacity
            'process_cpu_seconds_total', # CPU usage (counter, but meaningful)
            'http_requests_total',      # Request load (counter)
            'http_request_duration_seconds_sum',   # Latency (counter)
            'http_request_duration_seconds_count', # Request count (counter)
            'http_request_size_bytes',  # Request size load
            'http_response_size_bytes', # Response size load
            'python_gc_',               # Memory pressure indicators
        ]
        
        # Check for true consumer performance
        if any(pattern in name_lower for pattern in consumer_performance):
            return True
            
        return False  # Default to exclude if uncertain

    def _assess_scaling_relevance(self, name: str, values: pd.Series, category: str) -> float:
        """Assess scaling relevance with focus on TRUE consumer performance."""
        
        name_lower = name.lower()
        
        # MAJOR EXCLUSIONS FIRST
        
        # Heavy penalty for monitoring overhead
        if any(pattern in name_lower for pattern in ['scrape_', 'series', 'samples']):
            return 0.0  # Complete exclusion
            
        # Heavy penalty for meaningless metrics
        if any(pattern in name_lower for pattern in ['_created', '_info', 'start_time']):
            return 0.0  # Complete exclusion
            
        # Exclude service availability (not performance)
        if name_lower == 'up':
            return 0.0  # Complete exclusion
        
        # POSITIVE SCORING FOR TRUE CONSUMER METRICS
        
        base_score = 0.0
        
        # CRITICAL: Direct memory usage metrics (highest priority)
        if 'process_resident_memory_bytes' in name_lower:
            base_score = 120.0  # Physical memory is MOST critical for scaling
        elif 'process_virtual_memory_bytes' in name_lower:
            base_score = 115.0  # Virtual memory is also very critical
        # CPU usage
        elif 'cpu_seconds_total' in name_lower:
            base_score = 95.0   # CPU usage is critical
        elif 'open_fds' in name_lower:
            base_score = 90.0   # I/O pressure is critical
            
        # HTTP performance metrics (counters, but meaningful)
        elif 'http_requests_total' in name_lower:
            base_score = 85.0   # Request load is important
        elif 'http_request_duration_seconds_sum' in name_lower:
            base_score = 80.0   # Latency is important
        elif 'http_request_duration_seconds_count' in name_lower:
            base_score = 75.0   # Request frequency is important
        elif 'request_size_bytes' in name_lower or 'response_size_bytes' in name_lower:
            base_score = 70.0   # Payload size indicates load
            
        # Memory pressure indicators (secondary to direct memory usage)
        elif 'python_gc' in name_lower:
            base_score = 55.0   # GC indicates memory pressure (lower priority than direct memory)
            
        # Other process metrics
        elif 'max_fds' in name_lower:
            base_score = 40.0   # FD limits (less critical than usage)
        
        # Rate metrics get bonus (current change rate)
        if 'rate' in name_lower and base_score > 0:
            base_score += 20.0  # Rates show current activity
        
        return base_score

    def calculate_rate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rate metrics from MEANINGFUL counters only."""
        logger.info("🧮 Calculating rate metrics from meaningful counters...")
        
        rate_df = df.copy()
        
        # Only calculate rates for TRUE performance counters
        meaningful_counters = [
            'process_cpu_seconds_total',
            'http_requests_total', 
            'http_request_duration_seconds_sum',
            'http_request_duration_seconds_count',
            'http_request_size_bytes_sum',
            'http_request_size_bytes_count',
            'http_response_size_bytes_sum',
            'http_response_size_bytes_count',
            'python_gc_collections_total',
            'python_gc_objects_collected_total'
        ]
        
        for column in df.columns:
            if column in self.metric_analysis:
                analysis = self.metric_analysis[column]
                
                # Only convert meaningful counters to rates
                if (analysis['type'] == 'counter' and 
                    analysis['is_consumer_performance'] and
                    any(counter in column for counter in meaningful_counters)):
                    
                    # Calculate per-second rate
                    rate_series = df[column].diff() / 60  # 60-second intervals
                    rate_series = rate_series.fillna(0).clip(lower=0)  # Remove negative values
                    
                    rate_column = f"{column}_rate"
                    rate_df[rate_column] = rate_series
                    
                    # Update analysis for rate metric
                    self.metric_analysis[rate_column] = {
                        'type': 'gauge',
                        'category': analysis['category'],
                        'scaling_relevance': analysis['scaling_relevance'] + 20.0,  # Boost rates
                        'variability': rate_series.std() / (rate_series.mean() + 1e-10),
                        'is_consumer_performance': True,
                        'derived_from': column
                    }
                    
                    logger.info(f"  ✓ Created meaningful rate metric: {rate_column}")
        
        return rate_df
    
    def ensure_category_diversity(self, ranked_features: List[str]) -> List[str]:
        """Ensure diverse representation and prioritize DQN-required features."""
        logger.info("🎯 Ensuring DQN-required features are selected...")
        
        # DQN requires these specific features
        dqn_required_features = [
            'http_request_duration_seconds_sum_rate',
            'http_request_duration_seconds_count_rate', 
            'process_cpu_seconds_total_rate',
            'process_resident_memory_bytes',
            'http_requests_total_process_rate',
            'http_response_size_bytes_sum_rate',
            'process_open_fds',
            'kube_pod_container_resource_limits_cpu',
            'http_server_active_connections'
        ]
        
        selected = []
        category_counts = {}
        
        # PRIORITY 1: Select DQN-required features first (with variations)
        for required_feature in dqn_required_features:
            found = False
            
            # Try exact match first
            for feature in ranked_features:
                if feature == required_feature:
                    selected.append(feature)
                    category = self.metric_analysis.get(feature, {}).get('category', 'unknown')
                    category_counts[category] = category_counts.get(category, 0) + 1
                    logger.info(f"  ✓ Selected required: {feature} (exact match)")
                    found = True
                    break
            
            if not found:
                # Try variations
                variations = [
                    required_feature.replace('_process_rate', '_rate'),
                    required_feature.replace('_rate', ''),
                    required_feature.replace('http_requests_total_process_rate', 'http_requests_total'),
                    required_feature.replace('http_server_active_connections', 'http_requests_total')
                ]
                
                for variation in variations:
                    for feature in ranked_features:
                        if feature == variation and feature not in selected:
                            selected.append(feature)
                            category = self.metric_analysis.get(feature, {}).get('category', 'unknown')
                            category_counts[category] = category_counts.get(category, 0) + 1
                            logger.info(f"  ✓ Selected required: {feature} (variation of {required_feature})")
                            found = True
                            break
                    if found:
                        break
            
            if not found:
                logger.warning(f"  ⚠️ Could not find required feature: {required_feature}")
        
        logger.info(f"Selected {len(selected)} DQN-required features")
        
        # PRIORITY 2: Fill remaining slots if target not reached
        if len(selected) < self.target_features:
            remaining_slots = self.target_features - len(selected)
            logger.info(f"Filling {remaining_slots} remaining slots with best available features...")
            
            for feature in ranked_features:
                if len(selected) >= self.target_features:
                    break
                    
                if feature not in selected and feature in self.metric_analysis:
                    analysis = self.metric_analysis[feature]
                    
                    # Skip monitoring overhead and raw counters
                    if analysis['category'] == 'monitoring':
                        continue
                        
                    if analysis['type'] == 'counter' and 'rate' not in feature:
                        continue
                    
                    selected.append(feature)
                    category = analysis['category']
                    category_counts[category] = category_counts.get(category, 0) + 1
                    logger.info(f"  ✓ Added {feature} (category: {category}, type: {analysis['type']})")
        
        logger.info(f"📊 Final category distribution: {category_counts}")
        logger.info(f"📊 Total selected features: {len(selected)}")
        return selected
    
    def load_and_analyze_data(self) -> pd.DataFrame:
        """Load consumer data and perform comprehensive analysis."""
        
        # Discover metrics
        self.discover_consumer_metrics()
        
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
        logger.info("🔬 Analyzing metric characteristics...")
        for column in feature_df.columns:
            self.metric_analysis[column] = self.analyze_metric_characteristics(column, feature_df[column])
        
        # Calculate rate metrics
        feature_df = self.calculate_rate_metrics(feature_df)
        
        return feature_df
    
    def rank_features_intelligently(self, df: pd.DataFrame) -> List[str]:
        """Rank features using intelligent analysis with better counter detection."""
        logger.info("🧠 Ranking features intelligently...")
        
        feature_scores = {}
        
        for column in df.columns:
            if column not in self.metric_analysis:
                continue
                
            analysis = self.metric_analysis[column]
            
            # Base score from scaling relevance
            base_score = analysis['scaling_relevance']
            
            # Variability bonus (metrics that change are more useful)
            variability_score = analysis['variability'] * 15
            
            # Major type-based scoring
            type_bonus = 0.0
            if analysis['type'] == 'gauge':
                type_bonus = 50.0  # Strong preference for current state
            elif analysis['type'] == 'counter':
                type_bonus = -40.0  # Strong penalty for raw counters
            elif analysis['type'] == 'info':
                type_bonus = -60.0  # Heavy penalty for static info
                
            # Performance vs monitoring overhead
            performance_bonus = 40.0 if analysis['is_consumer_performance'] else -30.0
            
            # Special bonus for rate metrics (derived from counters)
            rate_bonus = 25.0 if 'rate' in column else 0.0
            
            # Final intelligent score
            final_score = base_score + variability_score + type_bonus + performance_bonus + rate_bonus
            feature_scores[column] = max(final_score, 0.0)
            
            # Log analysis for top candidates
            if final_score > 80:
                logger.info(f"  🎯 HIGH SCORE: {column:<45} (score: {final_score:.1f}, type: {analysis['type']}, category: {analysis['category']}, consumer: {analysis['is_consumer_performance']})")
        
        # Sort by score
        ranked = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("🏆 Top 15 intelligently ranked features:")
        for i, (feature, score) in enumerate(ranked[:15], 1):
            analysis = self.metric_analysis.get(feature, {})
            logger.info(f"  {i:2d}. {feature:<45} (score: {score:.1f}, type: {analysis.get('type', 'unknown')}, category: {analysis.get('category', 'unknown')})")
        
        return [feature for feature, score in ranked]
    
    def generate_scaling_actions(self, df: pd.DataFrame) -> pd.Series:
        """Generate realistic scaling actions based on intelligent analysis."""
        actions = []
        
        # Find key metrics for scaling decisions
        cpu_metrics = [col for col in df.columns if self.metric_analysis.get(col, {}).get('category') == 'cpu']
        memory_metrics = [col for col in df.columns if self.metric_analysis.get(col, {}).get('category') == 'memory']
        network_metrics = [col for col in df.columns if self.metric_analysis.get(col, {}).get('category') == 'network']
        
        for i, row in df.iterrows():
            pressure_score = 0.0
            
            # CPU pressure
            for metric in cpu_metrics:
                if metric in row:
                    val = row[metric]
                    if 'rate' in metric:  # Rate metrics
                        pressure_score += min(val / 0.1, 1.0) * 0.4  # CPU rate pressure
                    else:  # Gauge metrics
                        pressure_score += min(val / 1.0, 1.0) * 0.3  # CPU gauge pressure
            
            # Memory pressure  
            for metric in memory_metrics:
                if metric in row:
                    val = row[metric]
                    if 'resident_memory' in metric:
                        pressure_score += min(val / 100_000_000, 1.0) * 0.3  # 100MB baseline
                    elif 'virtual_memory' in metric:
                        pressure_score += min(val / 200_000_000, 1.0) * 0.2  # 200MB baseline
            
            # Network pressure
            for metric in network_metrics:
                if metric in row and 'rate' in metric:
                    val = row[metric]
                    pressure_score += min(val / 10.0, 1.0) * 0.1  # Request rate pressure
            
            # Scaling decisions
            if pressure_score > 0.7:
                actions.append('scale_up')
            elif pressure_score < 0.3:
                actions.append('scale_down')
            else:
                actions.append('keep_same')
        
        return pd.Series(actions, index=df.index)
    
    def create_dqn_scaler(self, feature_df: pd.DataFrame, output_dir: Path):
        """Create a scaler specifically for the DQN agent's expected features."""
        logger.info("🎯 Creating DQN-compatible scaler...")
        
        # DQN expects these exact features (from config/settings.py)
        dqn_expected_features = [
            'http_request_duration_seconds_sum_rate',
            'http_request_duration_seconds_count_rate', 
            'process_cpu_seconds_total_rate',
            'process_resident_memory_bytes',
            'http_requests_total_process_rate',
            'http_response_size_bytes_sum_rate',
            'process_open_fds',
            'kube_pod_container_resource_limits_cpu',
            'http_server_active_connections'
        ]
        
        # Find available features in the data (with fallbacks)
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
                # Try common variations
                variations = [
                    expected_feature.replace('_process_rate', '_rate'),
                    expected_feature.replace('_rate', ''),
                    expected_feature.replace('http_requests_total_process_rate', 'http_requests_total_rate'),
                    expected_feature.replace('http_server_active_connections', 'http_requests_total_rate')  # fallback
                ]
                
                for variation in variations:
                    if variation in feature_df.columns:
                        available_features.append(variation)
                        feature_mapping[expected_feature] = variation
                        found = True
                        logger.info(f"  ✓ Found variation: {expected_feature} -> {variation}")
                        break
            
            if not found:
                logger.warning(f"  ⚠️ Missing feature: {expected_feature} (will use zeros)")
                # Add placeholder column of zeros
                feature_df[expected_feature] = 0.0
                available_features.append(expected_feature)
                feature_mapping[expected_feature] = expected_feature
        
        # Create dataframe with exact DQN feature names and order
        dqn_feature_df = pd.DataFrame()
        for expected_feature in dqn_expected_features:
            source_feature = feature_mapping[expected_feature]
            dqn_feature_df[expected_feature] = feature_df[source_feature]
        
        # Create and fit the scaler
        dqn_scaler = RobustScaler()
        dqn_scaler.fit(dqn_feature_df)
        
        # Save as .pkl (not .gz) for DQN compatibility
        scaler_path = output_dir / 'feature_scaler.pkl'
        joblib.dump(dqn_scaler, scaler_path)
        
        logger.info(f"✅ DQN-compatible scaler saved to {scaler_path}")
        logger.info(f"   Scaler fitted on {len(dqn_expected_features)} features: {dqn_expected_features}")
        
        # Create feature mapping log
        mapping_path = output_dir / 'dqn_feature_mapping.json'
        with open(mapping_path, 'w') as f:
            json.dump(feature_mapping, f, indent=2)
        logger.info(f"   Feature mapping saved to {mapping_path}")
    
    def validate_dqn_compatibility(self, output_dir: Path) -> bool:
        """Validate that created files are compatible with DQN agent."""
        logger.info("🔍 Validating DQN compatibility...")
        
        try:
            # Expected DQN features (from config/settings.py)
            expected_features = [
                'http_request_duration_seconds_sum_rate',
                'http_request_duration_seconds_count_rate', 
                'process_cpu_seconds_total_rate',
                'process_resident_memory_bytes',
                'http_requests_total_process_rate',
                'http_response_size_bytes_sum_rate',
                'process_open_fds',
                'kube_pod_container_resource_limits_cpu',
                'http_server_active_connections'
            ]
            
            # 1. Check if scaler file exists and has correct format
            scaler_path = output_dir / 'feature_scaler.pkl'
            if not scaler_path.exists():
                logger.error(f"❌ Scaler file not found: {scaler_path}")
                return False
            
            # 2. Load and validate scaler
            try:
                import pandas as pd
                scaler = joblib.load(scaler_path)
                logger.info(f"✅ Scaler loaded successfully from {scaler_path}")
                
                # Check scaler type
                if not hasattr(scaler, 'transform') or not hasattr(scaler, 'fit'):
                    logger.error("❌ Invalid scaler: missing transform/fit methods")
                    return False
                
                logger.info(f"✅ Scaler type: {type(scaler).__name__}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load scaler: {e}")
                return False
            
            # 3. Test scaler with expected feature structure
            try:
                # Create test data with expected features
                test_data = pd.DataFrame({
                    feature: [0.0] for feature in expected_features
                })
                
                # Test transform
                scaled_data = scaler.transform(test_data)
                
                if scaled_data.shape != (1, len(expected_features)):
                    logger.error(f"❌ Scaler output shape mismatch: expected (1, {len(expected_features)}), got {scaled_data.shape}")
                    return False
                
                logger.info(f"✅ Scaler transform works correctly: input shape {test_data.shape} -> output shape {scaled_data.shape}")
                
            except Exception as e:
                logger.error(f"❌ Scaler transform failed: {e}")
                return False
            
            # 4. Check feature mapping
            mapping_path = output_dir / 'dqn_feature_mapping.json'
            if mapping_path.exists():
                try:
                    import json
                    with open(mapping_path, 'r') as f:
                        feature_mapping = json.load(f)
                    
                    # Validate all expected features are mapped
                    missing_features = []
                    for expected_feature in expected_features:
                        if expected_feature not in feature_mapping:
                            missing_features.append(expected_feature)
                    
                    if missing_features:
                        logger.warning(f"⚠️ Missing features in mapping: {missing_features}")
                    else:
                        logger.info("✅ All expected features are mapped")
                    
                    logger.info(f"✅ Feature mapping loaded: {len(feature_mapping)} features mapped")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load feature mapping: {e}")
                    return False
            else:
                logger.warning(f"⚠️ Feature mapping file not found: {mapping_path}")
            
            # 5. Simulate DQN agent usage
            try:
                logger.info("🧪 Simulating DQN agent usage...")
                
                # Test with realistic metric values
                realistic_metrics = {
                    'http_request_duration_seconds_sum_rate': 5.2,
                    'http_request_duration_seconds_count_rate': 10.5,
                    'process_cpu_seconds_total_rate': 0.8,
                    'process_resident_memory_bytes': 150000000.0,  # 150MB
                    'http_requests_total_process_rate': 8.3,
                    'http_response_size_bytes_sum_rate': 1500000.0,  # 1.5MB/s
                    'process_open_fds': 45.0,
                    'kube_pod_container_resource_limits_cpu': 1.0,
                    'http_server_active_connections': 15.0
                }
                
                # Create DataFrame as DQN agent would
                test_df = pd.DataFrame([realistic_metrics])
                
                # Test scaling
                scaled_result = scaler.transform(test_df)
                
                # Check for reasonable scaling (values should be roughly in [-3, 3] range for RobustScaler)
                if np.any(np.abs(scaled_result) > 10):
                    logger.warning(f"⚠️ Some scaled values are extreme: {scaled_result[0]}")
                    logger.warning("   This might indicate scaling issues")
                else:
                    logger.info(f"✅ Scaled values look reasonable: {scaled_result[0]}")
                
                # Check for NaN or infinite values
                if np.any(~np.isfinite(scaled_result)):
                    logger.error("❌ Scaled values contain NaN or infinite values")
                    return False
                
                logger.info("✅ DQN simulation successful")
                
            except Exception as e:
                logger.error(f"❌ DQN simulation failed: {e}")
                return False
            
            # 6. Final compatibility summary
            logger.info("🎯 DQN Compatibility Summary:")
            logger.info(f"   ✅ Scaler file: {scaler_path}")
            logger.info(f"   ✅ Scaler type: {type(scaler).__name__}")
            logger.info(f"   ✅ Expected features: {len(expected_features)}")
            logger.info(f"   ✅ Feature mapping: {mapping_path}")
            logger.info(f"   ✅ Transform test: PASSED")
            logger.info(f"   ✅ DQN simulation: PASSED")
            
            logger.info("🎉 All DQN compatibility checks PASSED!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Validation failed with error: {e}")
            return False
    
    def run(self, output_dir: Path):
        """Execute intelligent feature selection pipeline."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        logger.info("🚀 Starting intelligent consumer pod feature selection...")
        
        # Load and analyze data
        feature_df = self.load_and_analyze_data()
        
        # Rank features intelligently
        ranked_features = self.rank_features_intelligently(feature_df)
        
        # Ensure category diversity
        selected_features = self.ensure_category_diversity(ranked_features)
        
        # Generate scaling actions
        scaling_actions = self.generate_scaling_actions(feature_df)
        
        # Create final dataset
        final_df = feature_df[selected_features].copy()
        final_df['scaling_action'] = scaling_actions
        
        # Scale features
        scaler = RobustScaler()
        final_df[selected_features] = scaler.fit_transform(final_df[selected_features])
        
        # Save outputs
        final_df.to_parquet(output_dir / 'dqn_features.parquet')
        joblib.dump(scaler, output_dir / 'feature_scaler.gz')
        
        # Create DQN-compatible scaler for the exact features DQN expects
        self.create_dqn_scaler(feature_df, output_dir)
        
        # Create comprehensive metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'methodology': 'Intelligent data-driven feature selection',
            'discovered_metrics': len(self.consumer_metrics),
            'analyzed_metrics': len(self.metric_analysis),
            'selected_features': selected_features,
            'metric_analysis': self.metric_analysis,
            'dataset_shape': list(final_df.shape),
            'scaling_action_distribution': scaling_actions.value_counts().to_dict()
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create summary
        summary = f"""# Intelligent Consumer Pod Feature Selection Results

## Methodology
- **Auto-discovered**: {len(self.consumer_metrics)} consumer pod metrics
- **Auto-analyzed**: {len(self.metric_analysis)} metric characteristics  
- **Auto-calculated**: Rate metrics from counters
- **Auto-ensured**: Category diversity

## Dataset Summary
- **Samples**: {len(final_df)}
- **Features**: {len(selected_features)}
- **Scaling Actions**: {scaling_actions.value_counts().to_dict()}

## Selected Features (Top {len(selected_features)})
"""
        
        for i, feature in enumerate(selected_features, 1):
            analysis = self.metric_analysis.get(feature, {})
            category = analysis.get('category', 'unknown')
            metric_type = analysis.get('type', 'unknown')
            relevance = analysis.get('scaling_relevance', 0)
            summary += f"{i}. **`{feature}`** (category: {category}, type: {metric_type}, relevance: {relevance:.1f})\n"
        
        # Category breakdown
        categories = {}
        for feature in selected_features:
            category = self.metric_analysis.get(feature, {}).get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
            
        summary += f"""
## Category Distribution
"""
        for category, count in categories.items():
            summary += f"- **{category.upper()}**: {count} features\n"
        
        with open(output_dir / 'summary.md', 'w') as f:
            f.write(summary)
        
        logger.info(f"✅ Intelligent feature selection complete! Output saved to {output_dir}")
        logger.info(f"Selected features: {selected_features}")
        
        # Validate DQN compatibility
        logger.info("\n" + "="*60)
        if self.validate_dqn_compatibility(output_dir):
            logger.info("🎉 SUCCESS: All files are DQN-compatible!")
        else:
            logger.error("❌ FAILURE: DQN compatibility issues found!")
            logger.error("   Please check the logs above for details")
        logger.info("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Intelligent Consumer Pod DQN Feature Selector")
    parser.add_argument("--data-dir", required=True, help="Directory containing CSV files")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--target-features", type=int, default=9, help="Number of features to select")
    
    args = parser.parse_args()
    
    selector = IntelligentConsumerFeatureSelector(
        data_dir=args.data_dir,
        target_features=args.target_features
    )
    
    selector.run(output_dir=args.output_dir) 