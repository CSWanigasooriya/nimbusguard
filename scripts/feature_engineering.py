import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


class InfrastructureFeatureEngineer:
    """Advanced feature engineering for infrastructure monitoring metrics."""
    
    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.df = None
        self.logger = logging.getLogger(__name__)
        
        # Define metric types for proper handling
        self.metric_types = {
            'counter': [
                'node_cpu_seconds_total',
                'process_cpu_seconds_total',
                'http_requests_total',
                'node_disk_reads_completed_total',
                'node_disk_writes_completed_total'
            ],
            'gauge': [
                'node_memory_MemAvailable_bytes',
                'node_memory_MemTotal_bytes',
                'go_memstats_alloc_bytes',
                'go_goroutines'
            ],
            'histogram': [
                'http_request_duration_seconds',
                'prometheus_http_request_duration_seconds'
            ]
        }
        
        # Define metric categories based on your list
        self.metric_categories = {
            'cpu': ['node_cpu_seconds_total', 'process_cpu_seconds_total', 'go_sched_gomaxprocs_threads'],
            'memory': ['node_memory_.*', 'go_memstats_.*', 'process_.*memory.*'],
            'disk': ['node_disk_.*', 'node_filesystem_.*'],
            'network': ['node_network_.*', 'node_netstat_.*', 'net_conntrack_.*'],
            'kubernetes': ['kube_.*'],
            'prometheus': ['prometheus_.*'],
            'http': ['http_.*', 'promhttp_.*'],
            'go_runtime': ['go_gc_.*', 'go_goroutines'],
            'system_load': ['node_load.*', 'node_procs_.*'],
            'loki': ['loki_.*'],
            'alloy': ['alloy_.*'],
            'postgres': ['postgres_.*'],
            'otel': ['otel.*'],
            'python': ['python_.*'],
            'rpc': ['rpc_.*']
        }
        
        # Define metric units for conversion
        self.metric_units = {
            'bytes': ['_bytes', '_bytes_total'],
            'seconds': ['_seconds', '_seconds_total'],
            'ratio': ['_ratio', '_utilization'],
            'count': ['_total', '_count']
        }
        
        # Define metric groupings for aggregation
        self.metric_groups = {
            'cpu': ['node_cpu', 'process_cpu', 'container_cpu'],
            'memory': ['node_memory', 'process_memory', 'container_memory'],
            'network': ['node_network', 'container_network'],
            'disk': ['node_disk', 'node_filesystem'],
            'http': ['http_request', 'http_response'],
            'kubernetes': ['kube_pod', 'kube_deployment']
        }
        
        # Aggregation functions to apply
        self.agg_functions = ['mean', 'max', 'min', 'std', 'p95', 'p99']
        
    def load_data(self) -> pd.DataFrame:
        """Load the prepared dataset."""
        self.logger.info(f"Loading data from {self.input_path}")
        
        if self.input_path.suffix == '.parquet':
            self.df = pd.read_parquet(self.input_path)
        elif self.input_path.suffix == '.csv':
            self.df = pd.read_csv(self.input_path)
        else:
            raise ValueError(f"Unsupported file format: {self.input_path.suffix}")
            
        self.logger.info(f"Loaded {len(self.df)} rows with {len(self.df.columns)} columns")
        return self.df
    
    def categorize_metrics(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Categorize metrics based on their names."""
        categorized = {category: [] for category in self.metric_categories.keys()}
        categorized['other'] = []
        
        metric_columns = [col for col in df.columns 
                         if col not in ['timestamp', 'hour', 'day_of_week', 'day_of_month', 
                                       'month', 'is_weekend', 'is_business_hours', 'time_since_start']]
        
        for col in metric_columns:
            categorized_flag = False
            for category, patterns in self.metric_categories.items():
                for pattern in patterns:
                    if any(p in col for p in pattern.split('.*')):
                        categorized[category].append(col)
                        categorized_flag = True
                        break
                if categorized_flag:
                    break
            
            if not categorized_flag:
                categorized['other'].append(col)
        
        # Log categorization results
        for category, metrics in categorized.items():
            if metrics:
                self.logger.info(f"{category.upper()}: {len(metrics)} metrics")
        
        return categorized
    
    def _calculate_rate(self, df: pd.DataFrame, col: str, window: int = 5) -> pd.Series:
        """Calculate rate of change for counter metrics."""
        # Ensure timestamp is datetime
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Calculate time difference in seconds
        time_diff = df['timestamp'].diff().dt.total_seconds()
        
        # Calculate value difference
        value_diff = df[col].diff()
        
        # Calculate rate (handle division by zero)
        rate = value_diff / (time_diff + 1e-6)
        
        # Apply rolling window to smooth the rate
        if window > 1:
            rate = rate.rolling(window=window, min_periods=1).mean()
        
        return rate
    
    def _normalize_units(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Normalize metric units to standard form."""
        # Convert bytes to megabytes
        if any(unit in col.lower() for unit in self.metric_units['bytes']):
            return df[col] / (1024 * 1024)  # Convert to MB
        
        # Convert nanoseconds to milliseconds
        if '_nanoseconds' in col.lower():
            return df[col] / 1_000_000  # Convert to ms
        
        return df[col]
    
    def _is_counter(self, metric_name: str) -> bool:
        """Check if metric is a counter type."""
        return any(counter in metric_name for counter in self.metric_types['counter'])
    
    def _is_gauge(self, metric_name: str) -> bool:
        """Check if metric is a gauge type."""
        return any(gauge in metric_name for gauge in self.metric_types['gauge'])
    
    def _is_histogram(self, metric_name: str) -> bool:
        """Check if metric is a histogram type."""
        return any(hist in metric_name for hist in self.metric_types['histogram'])
    
    def _process_histogram_metric(self, df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        """Process histogram metrics and their quantiles."""
        # Find all quantile columns for this histogram
        quantile_cols = [col for col in df.columns 
                        if metric_name in col 
                        and ('_bucket' in col or '_sum' in col or '_count' in col)]
        
        if not quantile_cols:
            return pd.DataFrame()
        
        result = pd.DataFrame()
        
        # Calculate key percentiles if bucket information is available
        bucket_cols = [col for col in quantile_cols if '_bucket' in col]
        if bucket_cols:
            # Extract bucket boundaries from column names
            buckets = []
            for col in bucket_cols:
                try:
                    # Extract the le="X" value from the column name
                    bucket = float(col.split('le="')[1].split('"')[0])
                    buckets.append((bucket, col))
                except:
                    continue
            
            if buckets:
                # Sort buckets by boundary
                buckets.sort(key=lambda x: x[0])
                
                # Calculate approximate percentiles
                for p in [50, 75, 90, 95, 99]:
                    target_value = df[bucket_cols].sum(axis=1) * (p/100)
                    for bucket_value, bucket_col in buckets:
                        result[f'{metric_name}_p{p}'] = bucket_value
                        if df[bucket_col] >= target_value:
                            break
        
        # Add sum and count if available
        sum_col = next((col for col in quantile_cols if '_sum' in col), None)
        count_col = next((col for col in quantile_cols if '_count' in col), None)
        
        if sum_col and count_col:
            # Calculate average
            result[f'{metric_name}_avg'] = df[sum_col] / df[count_col].clip(lower=1)
        
        return result
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from timestamp."""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & 
                                 (df['day_of_week'] < 5)).astype(int)
        return df
    
    def _create_health_scores(self, df: pd.DataFrame, metrics: list) -> list:
        """Create system health score features."""
        health_scores = []
        
        # CPU health score
        cpu_metrics = [m for m in metrics if 'cpu' in m.lower()]
        if cpu_metrics:
            df['cpu_health_score'] = 1 - df[cpu_metrics].mean(axis=1)
            health_scores.append('cpu_health_score')
        
        # Memory health score
        memory_metrics = [m for m in metrics if 'memory' in m.lower()]
        if memory_metrics:
            df['memory_health_score'] = 1 - df[memory_metrics].mean(axis=1)
            health_scores.append('memory_health_score')
        
        # Overall system health score
        if health_scores:
            df['system_health_score'] = df[health_scores].mean(axis=1)
            health_scores.append('system_health_score')
        
        return health_scores
    
    def _create_utilization_features(self, df: pd.DataFrame, metrics: list) -> list:
        """Create resource utilization features."""
        utilization_features = []
        
        # CPU utilization features
        cpu_metrics = [m for m in metrics if 'cpu' in m.lower()]
        if cpu_metrics:
            df['cpu_utilization_mean'] = df[cpu_metrics].mean(axis=1)
            df['cpu_utilization_max'] = df[cpu_metrics].max(axis=1)
            utilization_features.extend(['cpu_utilization_mean', 'cpu_utilization_max'])
        
        # Memory utilization features
        memory_metrics = [m for m in metrics if 'memory' in m.lower()]
        if memory_metrics:
            df['memory_utilization_mean'] = df[memory_metrics].mean(axis=1)
            df['memory_utilization_max'] = df[memory_metrics].max(axis=1)
            utilization_features.extend(['memory_utilization_mean', 'memory_utilization_max'])
        
        return utilization_features
    
    def _create_performance_features(self, df: pd.DataFrame, metrics: list) -> list:
        """Create performance indicator features."""
        performance_features = []
        
        # Latency features
        latency_metrics = [m for m in metrics if 'latency' in m.lower()]
        if latency_metrics:
            df['latency_mean'] = df[latency_metrics].mean(axis=1)
            df['latency_p95'] = df[latency_metrics].quantile(0.95, axis=1)
            df['latency_p99'] = df[latency_metrics].quantile(0.99, axis=1)
            performance_features.extend(['latency_mean', 'latency_p95', 'latency_p99'])
        
        # Error rate features
        error_metrics = [m for m in metrics if 'error' in m.lower()]
        if error_metrics:
            df['error_rate'] = df[error_metrics].mean(axis=1)
            performance_features.append('error_rate')
        
        return performance_features
    
    def _create_anomaly_features(self, df: pd.DataFrame, metrics: list) -> list:
        """Create anomaly detection features (limited for performance)."""
        anomaly_features = []
        
        # Limit to top 100 most important metrics to avoid feature explosion
        metrics_variance = {metric: df[metric].var() for metric in metrics}
        top_metrics = sorted(metrics_variance.keys(), key=metrics_variance.get, reverse=True)[:100]
        
        self.logger.info(f"Creating anomaly features for top {len(top_metrics)} most variable metrics")
        
        for metric in top_metrics:
            # Z-score based anomaly score
            mean = df[metric].mean()
            std = df[metric].std()
            if std > 0:
                score_name = f'{metric}_zscore'
                df[score_name] = abs(df[metric] - mean) / std
                anomaly_features.append(score_name)
            
            # Rolling statistics
            window = 5  # 5-minute window assuming 1-minute data
            rolling_mean = df[metric].rolling(window=window, min_periods=1).mean()
            rolling_std = df[metric].rolling(window=window, min_periods=1).std()
            
            # Deviation from rolling mean
            deviation_name = f'{metric}_deviation'
            df[deviation_name] = abs(df[metric] - rolling_mean) / (rolling_std + 1e-6)
            anomaly_features.append(deviation_name)
        
        return anomaly_features
    
    def _create_correlation_features(self, df: pd.DataFrame, metrics: list) -> list:
        """Create correlation-based features (limited for performance)."""
        correlation_features = []
        
        # Limit to top 50 most variable metrics to avoid explosion
        metrics_variance = {metric: df[metric].var() for metric in metrics}
        top_metrics = sorted(metrics_variance.keys(), key=metrics_variance.get, reverse=True)[:50]
        
        self.logger.info(f"Creating correlations for top {len(top_metrics)} most variable metrics")
        
        # Calculate correlations between top metrics only
        for i, metric1 in enumerate(top_metrics[:-1]):
            for metric2 in top_metrics[i+1:i+6]:  # Limit to next 5 metrics
                corr_name = f'corr_{metric1}_{metric2}'
                window = 5  # 5-minute window
                df[corr_name] = df[metric1].rolling(window=window, min_periods=2).corr(df[metric2])
                correlation_features.append(corr_name)
        
        return correlation_features
    
    def _create_time_series_features(self, df: pd.DataFrame, metrics: list) -> list:
        """Create time series features."""
        time_series_features = []
        
        for metric in metrics:
            # Rate of change
            rate_name = f'{metric}_rate'
            df[rate_name] = df[metric].diff()
            time_series_features.append(rate_name)
            
            # Rolling statistics
            window = 5  # 5-minute window
            mean_name = f'{metric}_rolling_mean'
            std_name = f'{metric}_rolling_std'
            df[mean_name] = df[metric].rolling(window=window, min_periods=1).mean()
            df[std_name] = df[metric].rolling(window=window, min_periods=1).std()
            time_series_features.extend([mean_name, std_name])
            
            # Trend features
            trend_name = f'{metric}_trend'
            df[trend_name] = df[metric].diff(periods=5)  # 5-minute trend
            time_series_features.append(trend_name)
        
        return time_series_features
    
    def _create_dimensionality_reduction_features(self, df: pd.DataFrame) -> list:
        """Create dimensionality reduction features using PCA."""
        features = []
        
        try:
            # Get numeric columns for PCA
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col not in ['timestamp']]
            
            if len(numeric_cols) < 2:
                return features
            
            # Sample data if too large
            if len(df) > 10000:
                sample_df = df.sample(n=10000, random_state=42)
            else:
                sample_df = df.copy()
            
            # Standardize features for PCA
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(sample_df[numeric_cols].fillna(0))
            
            # Apply PCA for top 10 components
            pca = PCA(n_components=min(10, len(numeric_cols)))
            pca_features = pca.fit_transform(scaled_data)
            
            # Add PCA features to the original dataframe
            for i in range(pca_features.shape[1]):
                feature_name = f'pca_component_{i+1}'
                features.append(feature_name)
            
                # Apply same transformation to full dataset
                full_scaled = scaler.transform(df[numeric_cols].fillna(0))
                full_pca = pca.transform(full_scaled)
                df[feature_name] = full_pca[:, i]
            
        except Exception as e:
            logging.warning(f"Could not create PCA features: {e}")
        
        return features
    
    def _create_dqn_utilization_features(self, df: pd.DataFrame, metrics: list) -> list:
        """Create DQN-specific utilization features for pod scaling decisions."""
        features = []
        
        # Group metrics by resource type
        cpu_metrics = [m for m in metrics if 'cpu' in m.lower()]
        memory_metrics = [m for m in metrics if 'memory' in m.lower() or 'mem' in m.lower()]
        disk_metrics = [m for m in metrics if 'disk' in m.lower() or 'filesystem' in m.lower()]
        network_metrics = [m for m in metrics if 'network' in m.lower() or 'net' in m.lower()]
        
        # CPU utilization features
        if cpu_metrics:
            df['cpu_utilization_mean'] = df[cpu_metrics].mean(axis=1)
            df['cpu_utilization_max'] = df[cpu_metrics].max(axis=1)
            df['cpu_utilization_p95'] = df[cpu_metrics].quantile(0.95, axis=1)
            df['cpu_saturation_count'] = (df[cpu_metrics] > df[cpu_metrics].quantile(0.9)).sum(axis=1)
            features.extend(['cpu_utilization_mean', 'cpu_utilization_max', 'cpu_utilization_p95', 'cpu_saturation_count'])
        
        # Memory utilization features
        if memory_metrics:
            df['memory_utilization_mean'] = df[memory_metrics].mean(axis=1)
            df['memory_utilization_max'] = df[memory_metrics].max(axis=1)
            df['memory_pressure_score'] = (df[memory_metrics] > df[memory_metrics].quantile(0.85)).sum(axis=1)
            features.extend(['memory_utilization_mean', 'memory_utilization_max', 'memory_pressure_score'])
        
        # Resource contention indicators
        if cpu_metrics and memory_metrics:
            df['resource_contention'] = df['cpu_utilization_mean'] * df['memory_utilization_mean']
            features.append('resource_contention')
        
        return features
    
    def _create_dqn_performance_features(self, df: pd.DataFrame, metrics: list) -> list:
        """Create DQN-specific performance features."""
        features = []
        
        # Response time/latency metrics
        latency_metrics = [m for m in metrics if any(term in m.lower() for term in ['latency', 'duration', 'time'])]
        if latency_metrics:
            df['avg_response_time'] = df[latency_metrics].mean(axis=1)
            df['max_response_time'] = df[latency_metrics].max(axis=1)
            df['response_time_variance'] = df[latency_metrics].var(axis=1)
            features.extend(['avg_response_time', 'max_response_time', 'response_time_variance'])
        
        # Throughput metrics
        throughput_metrics = [m for m in metrics if any(term in m.lower() for term in ['requests', 'rps', 'qps', 'rate'])]
        if throughput_metrics:
            df['total_throughput'] = df[throughput_metrics].sum(axis=1)
            df['avg_throughput'] = df[throughput_metrics].mean(axis=1)
            features.extend(['total_throughput', 'avg_throughput'])
        
        # Error rate metrics
        error_metrics = [m for m in metrics if any(term in m.lower() for term in ['error', 'fail', 'timeout'])]
        if error_metrics:
            df['total_errors'] = df[error_metrics].sum(axis=1)
            df['error_rate'] = df['total_errors'] / (df['total_throughput'] + 1e-6)
            features.extend(['total_errors', 'error_rate'])
        
        return features
    
    def _create_dqn_anomaly_features(self, df: pd.DataFrame, metrics: list) -> list:
        """Create DQN-specific anomaly detection features."""
        features = []
        
        for metric in metrics:
            try:
                # Z-score based anomaly detection
                mean_val = df[metric].mean()
                std_val = df[metric].std()
                if std_val > 0:
                    df[f'{metric}_zscore'] = (df[metric] - mean_val) / std_val
                    df[f'{metric}_is_anomaly'] = (abs(df[f'{metric}_zscore']) > 3).astype(int)
                    features.extend([f'{metric}_zscore', f'{metric}_is_anomaly'])
            except Exception as e:
                logging.warning(f"Could not create anomaly features for {metric}: {e}")
                continue
        
        # Overall anomaly score
        anomaly_cols = [f for f in features if f.endswith('_is_anomaly')]
        if anomaly_cols:
            df['total_anomaly_score'] = df[anomaly_cols].sum(axis=1)
            features.append('total_anomaly_score')
        
        return features
    
    def _create_dqn_time_series_features(self, df: pd.DataFrame, metrics: list) -> list:
        """Create DQN-specific time series features."""
        features = []
        
        # Ensure data is sorted by timestamp
        df = df.sort_values('timestamp')
        
        for metric in metrics:
            try:
                # Rate of change
                df[f'{metric}_rate'] = df[metric].diff()
                
                # Moving averages
                df[f'{metric}_ma_5'] = df[metric].rolling(window=5, min_periods=1).mean()
                df[f'{metric}_ma_15'] = df[metric].rolling(window=15, min_periods=1).mean()
                
                # Trend indicator
                df[f'{metric}_trend'] = (df[f'{metric}_ma_5'] - df[f'{metric}_ma_15']).fillna(0)
                
                features.extend([f'{metric}_rate', f'{metric}_ma_5', f'{metric}_ma_15', f'{metric}_trend'])
                
            except Exception as e:
                logging.warning(f"Could not create time series features for {metric}: {e}")
                continue
        
        return features
    
    def _add_dqn_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add target variable for DQN training (optimal pod scaling decision)."""
        
        # FIX: The original 'current_pod_count' had corrupted values (46 quintillion!)
        # This was caused by summing metric VALUES (memory bytes, CPU nanoseconds) instead of counting pods
        # Create realistic pod counts for DQN training instead
        
        self.logger.info("🔧 Creating realistic pod counts for DQN training...")
        
        # Get resource utilization metrics to estimate realistic pod counts
        cpu_cols = [col for col in df.columns if 'cpu' in col.lower() and 'utilization' in col.lower()]
        memory_cols = [col for col in df.columns if 'memory' in col.lower() and 'utilization' in col.lower()]
        
        # Calculate overall system load
        system_load = 0.5  # Default moderate load
        if cpu_cols and len(cpu_cols) > 0:
            # Use existing DQN features if available, otherwise create simple estimates
            if 'dqn_cpu_utilization_mean' in df.columns:
                cpu_load = df['dqn_cpu_utilization_mean'] / 100.0  # Normalize to 0-1
            else:
                cpu_load = df[cpu_cols].mean(axis=1) / df[cpu_cols].mean(axis=1).max()
            system_load = np.maximum(system_load, cpu_load)
        
        if memory_cols and len(memory_cols) > 0:
            if 'dqn_memory_utilization_mean' in df.columns:
                memory_load = df['dqn_memory_utilization_mean'] / 100.0  # Normalize to 0-1
            else:
                memory_load = df[memory_cols].mean(axis=1) / df[memory_cols].mean(axis=1).max()
            system_load = np.maximum(system_load, memory_load)
        
        # Create realistic pod counts based on system load
        # Base pod count: 1-3 pods for light load, up to 8-12 for heavy load
        base_pods = 2  # Start with 2 pods
        current_pods = base_pods + np.round(system_load * 6).astype(int)  # Scale 2-8 pods
        current_pods = np.clip(current_pods, 1, 12)  # Realistic range: 1-12 pods
        
        # Add some time-based variation to simulate real scaling events
        if 'hour' in df.columns:
            # Higher load during business hours
            business_hours_boost = np.where(
                (df['hour'] >= 9) & (df['hour'] <= 17), 1, 0
            )
            current_pods = current_pods + business_hours_boost
        
        # Add some random variation to create scaling events
        np.random.seed(42)  # Reproducible
        variation = np.random.choice([-1, 0, 0, 0, 1], size=len(df))  # Occasional +1/-1
        current_pods = np.clip(current_pods + variation, 1, 12)
        
        # Calculate optimal pod count based on utilization
        optimal_pods = current_pods.copy()
        
        # Scale up if high utilization (>70th percentile)
        high_load_threshold = np.percentile(system_load, 70)
        scale_up_mask = system_load > high_load_threshold
        optimal_pods[scale_up_mask] = np.clip(optimal_pods[scale_up_mask] + 1, 1, 12)
        
        # Scale down if low utilization (<30th percentile)
        low_load_threshold = np.percentile(system_load, 30)
        scale_down_mask = system_load < low_load_threshold
        optimal_pods[scale_down_mask] = np.clip(optimal_pods[scale_down_mask] - 1, 1, 12)
        
        # Add scaling action (0: scale down, 1: keep, 2: scale up)
        df['scaling_action'] = np.where(
            optimal_pods > current_pods, 2,  # Scale up
            np.where(optimal_pods < current_pods, 0, 1)  # Scale down or keep
        )
        
        df['optimal_pod_count'] = optimal_pods
        df['current_pod_count'] = current_pods
        
        # Log scaling action distribution
        action_counts = df['scaling_action'].value_counts().sort_index()
        self.logger.info(f"Scaling actions: Scale Down={action_counts.get(0, 0)}, Keep={action_counts.get(1, 0)}, Scale Up={action_counts.get(2, 0)}")
        self.logger.info(f"Pod count range: {current_pods.min()}-{current_pods.max()} current, {optimal_pods.min()}-{optimal_pods.max()} optimal")
        
        return df
    
    def _save_dqn_metadata(self, df: pd.DataFrame, output_path: Path, 
                          original_metrics: int, engineered_features: int, config: dict):
        """Save detailed metadata for DQN training."""
        metadata_path = output_path.parent / f"{output_path.stem}_metadata.txt"
        
        with open(metadata_path, 'w') as f:
            f.write("DQN Infrastructure Monitoring Dataset\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("📊 DATASET OVERVIEW:\n")
            f.write(f"   Shape: {df.shape}\n")
            f.write(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
            f.write(f"   Duration: {df['timestamp'].max() - df['timestamp'].min()}\n")
            f.write(f"   Missing values: {df.isnull().sum().sum()}\n\n")
            
            f.write("🎯 DQN FEATURES:\n")
            f.write(f"   Original metrics: {original_metrics}\n")
            f.write(f"   Engineered features: {engineered_features}\n")
            f.write(f"   Total features: {len(df.columns)}\n\n")
            
            f.write("🎮 ACTION SPACE:\n")
            if 'scaling_action' in df.columns:
                action_counts = df['scaling_action'].value_counts().sort_index()
                f.write(f"   0 (Scale Down): {action_counts.get(0, 0)} samples\n")
                f.write(f"   1 (Keep Same): {action_counts.get(1, 0)} samples\n")
                f.write(f"   2 (Scale Up): {action_counts.get(2, 0)} samples\n\n")
            
            f.write("🔧 FEATURE CATEGORIES:\n")
            if not config['skip_utilization']:
                f.write("   ✓ Resource Utilization Features\n")
            if not config['skip_performance']:
                f.write("   ✓ Performance Features\n")
            if not config['skip_anomaly']:
                f.write("   ✓ Anomaly Detection Features\n")
            if not config['skip_time_series']:
                f.write("   ✓ Time Series Features\n")
            
            f.write("\n📈 KEY METRICS FOR DQN:\n")
            key_metrics = ['cpu_utilization_mean', 'memory_utilization_mean', 'avg_response_time', 
                          'total_throughput', 'error_rate', 'total_anomaly_score']
            for metric in key_metrics:
                if metric in df.columns:
                    f.write(f"   {metric}: min={df[metric].min():.2f}, max={df[metric].max():.2f}, mean={df[metric].mean():.2f}\n")
    
    def engineer_features(self, config: dict) -> None:
        """Engineer features from the input dataset."""
        try:
            self.logger.info("Starting infrastructure feature engineering for DQN...")
            self.logger.info(f"Loading data from {self.input_path}")
            
            # Load data
            df = pd.read_parquet(self.input_path)
            self.logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
            self.logger.info(f"Loaded data: {df.shape}")
            
            # Check for missing values in input
            missing_before = df.isnull().sum().sum()
            self.logger.info(f"Missing values in input: {missing_before}")
            
            # Get list of metrics (exclude time and DQN features)
            exclude_cols = ['timestamp', 'hour', 'day_of_week', 'is_weekend', 'hour_sin', 'hour_cos', 
                           'day_sin', 'day_cos', 'day_of_month', 'month', 'is_business_hours',
                           'avg_cpu_utilization', 'max_cpu_utilization', 'cpu_pressure',
                           'avg_memory_utilization', 'max_memory_utilization', 'memory_pressure',
                           'total_request_rate', 'avg_request_rate', 'total_error_rate',
                           'current_pod_count', 'system_load', 'network_traffic']
            
            all_columns = [col for col in df.columns if col not in exclude_cols]
            
            # First filter to only numeric columns
            numeric_metrics = []
            for col in all_columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_metrics.append(col)
                else:
                    self.logger.info(f"Skipping non-numeric column: {col}")
            
            self.logger.info(f"Found {len(numeric_metrics)} numeric metric columns out of {len(all_columns)} total")
            
            # Filter out static metrics and limit for computational efficiency
            metrics = []
            metric_variance = {}
            
            for metric in numeric_metrics:
                try:
                    metric_std = df[metric].std()
                    metric_variance[metric] = metric_std
                    if metric_std > 1e-6:  # Keep metrics with some variation
                        metrics.append(metric)
                    else:
                        self.logger.debug(f"Filtering out static metric: {metric}")
                except Exception as e:
                    self.logger.warning(f"Error processing metric {metric}: {e}")
                    continue
            
            # Limit to top N most variable metrics for computational efficiency
            if len(metrics) > 200:
                # Sort by variance and keep top 200
                sorted_metrics = sorted(metrics, key=lambda x: metric_variance[x], reverse=True)
                metrics = sorted_metrics[:200]
                self.logger.info(f"Limited to top 200 most variable metrics from {len(sorted_metrics)} total")
            
            self.logger.info(f"Using {len(metrics)} metrics after filtering")
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create time features if not already present
            if 'hour' not in df.columns:
                df = self._create_time_features(df)
            
            # Initialize feature lists
            engineered_features = []
            
            # DQN-specific feature engineering
            if not config['skip_utilization']:
                self.logger.info("Creating DQN utilization features...")
                utilization_features = self._create_dqn_utilization_features(df, metrics)
                engineered_features.extend(utilization_features)
                self.logger.info(f"Created {len(utilization_features)} utilization features")
            
            if not config['skip_performance']:
                self.logger.info("Creating DQN performance features...")
                performance_features = self._create_dqn_performance_features(df, metrics)
                engineered_features.extend(performance_features)
                self.logger.info(f"Created {len(performance_features)} performance features")
            
            if not config['skip_anomaly']:
                self.logger.info("Creating DQN anomaly features...")
                anomaly_features = self._create_dqn_anomaly_features(df, metrics[:50])  # Limit for speed
                engineered_features.extend(anomaly_features)
                self.logger.info(f"Created {len(anomaly_features)} anomaly features")
            
            if not config['skip_time_series']:
                self.logger.info("Creating DQN time series features...")
                time_series_features = self._create_dqn_time_series_features(df, metrics[:30])  # Limit for speed
                engineered_features.extend(time_series_features)
                self.logger.info(f"Created {len(time_series_features)} time series features")
            
            # Skip correlation features by default (broken due to 1,180+ constant columns causing 48% NaN correlations)
            if not config.get('skip_correlation', True):  # Default to skip=True
                self.logger.warning("⚠️  Correlation features are DISABLED by default due to data quality issues:")
                self.logger.warning("   - 1,180+ constant columns cause correlation matrix to be 48% NaN values")
                self.logger.warning("   - Enable with --correlation flag only if you've filtered constant columns")
                correlation_features = self._create_correlation_features(df, metrics[:50])
                engineered_features.extend(correlation_features)
                self.logger.info(f"Created {len(correlation_features)} correlation features")
            else:
                self.logger.info("⏭️  Skipping correlation features (recommended - broken with current data)")
            
            # Skip dimensionality reduction features by default (broken due to NaN-filled correlation matrix)
            if not config.get('skip_dimensionality_reduction', True):  # Default to skip=True
                self.logger.warning("⚠️  Dimensionality reduction features are DISABLED by default:")
                self.logger.warning("   - PCA fails on correlation matrix with 8.5M+ NaN values")
                self.logger.warning("   - Enable with --dimensionality-reduction flag only if correlation works")
                dim_reduction_features = self._create_dimensionality_reduction_features(df)
                engineered_features.extend(dim_reduction_features)
                self.logger.info(f"Created {len(dim_reduction_features)} dimensionality reduction features")
            else:
                self.logger.info("⏭️  Skipping dimensionality reduction features (recommended - broken with current data)")
            
            # Add target variable for DQN (pod scaling decision)
            df = self._add_dqn_target(df)
            
            # Add all engineered features to dataframe with proper handling
            for feature_name in engineered_features:
                if feature_name not in df.columns:
                    # Initialize with zeros instead of leaving as missing
                    df[feature_name] = 0.0
            
            # Fill any remaining missing values
            missing_after = df.isnull().sum().sum()
            if missing_after > 0:
                self.logger.warning(f"Found {missing_after} missing values, filling with forward fill then zeros")
                df = df.fillna(method='ffill').fillna(0)
            
            # Final check
            final_missing = df.isnull().sum().sum()
            self.logger.info(f"Final missing values: {final_missing}")
            
            self.logger.info(f"Added {len(engineered_features)} new features")
            
            # Save the engineered features
            output_path = self.output_path
            if not str(output_path).endswith(config['output_format']):
                output_path = output_path.with_suffix(f".{config['output_format']}")
            
            if config['output_format'] == 'parquet':
                df.to_parquet(output_path, index=False)
            else:
                df.to_csv(output_path, index=False)
            
            # Save detailed metadata for DQN
            self._save_dqn_metadata(df, output_path, len(metrics), len(engineered_features), config)
            
            self.logger.info("✅ DQN feature engineering completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            raise


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Engineer features from consolidated metrics dataset"
    )
    
    # Input/Output options
    parser.add_argument('input_path', nargs='?', type=Path,
                      help='Input dataset path')
    parser.add_argument('output_path', nargs='?', type=Path,
                      help='Output path for engineered features')
    parser.add_argument('--output-format', choices=['parquet', 'csv'],
                      default='parquet',
                      help='Output format for engineered features')
    
    # Feature selection options
    parser.add_argument('--skip-health-scores', action='store_true',
                      help='Skip health score features')
    parser.add_argument('--skip-utilization', action='store_true',
                      help='Skip utilization features')
    parser.add_argument('--skip-performance', action='store_true',
                      help='Skip performance features')
    parser.add_argument('--skip-anomaly', action='store_true',
                      help='Skip anomaly detection features')
    parser.add_argument('--skip-correlation', action='store_true',
                      help='Skip correlation features')
    parser.add_argument('--skip-time-series', action='store_true',
                      help='Skip time series features')
    parser.add_argument('--skip-dimensionality-reduction', action='store_true',
                      help='Skip dimensionality reduction features')
    
    args = parser.parse_args()
    
    try:
        # Get the scripts directory path (where this file is located)
        scripts_dir = Path(__file__).parent.resolve()
        processed_dir = scripts_dir / 'processed_data'
        
        # Set default paths if not provided
        if args.input_path is None:
            args.input_path = processed_dir / 'dataset.parquet'
        else:
            # If input_path is relative, make it relative to scripts_dir
            if not args.input_path.is_absolute():
                args.input_path = scripts_dir / args.input_path
            args.input_path = args.input_path.resolve()
            
        if args.output_path is None:
            args.output_path = processed_dir / 'engineered_features'
        else:
            # If output_path is relative, make it relative to scripts_dir
            if not args.output_path.is_absolute():
                args.output_path = scripts_dir / args.output_path
            args.output_path = args.output_path.resolve()
        
        # Create output directory
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize feature engineer
        engineer = InfrastructureFeatureEngineer(
            input_path=args.input_path,
            output_path=args.output_path
        )
        
        # Configure feature selection
        config = {
            'skip_health_scores': args.skip_health_scores,
            'skip_utilization': args.skip_utilization,
            'skip_performance': args.skip_performance,
            'skip_anomaly': args.skip_anomaly,
            'skip_correlation': args.skip_correlation,
            'skip_time_series': args.skip_time_series,
            'skip_dimensionality_reduction': args.skip_dimensionality_reduction,
            'output_format': args.output_format
        }
        
        # Run feature engineering
        engineer.engineer_features(config)
        return 0
        
    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())