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
        
    def load_data(self) -> pd.DataFrame:
        """Load the prepared dataset."""
        logging.info(f"Loading data from {self.input_path}")
        
        if self.input_path.suffix == '.parquet':
            self.df = pd.read_parquet(self.input_path)
        elif self.input_path.suffix == '.csv':
            self.df = pd.read_csv(self.input_path)
        else:
            raise ValueError(f"Unsupported file format: {self.input_path.suffix}")
            
        logging.info(f"Loaded {len(self.df)} rows with {len(self.df.columns)} columns")
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
                logging.info(f"{category.upper()}: {len(metrics)} metrics")
        
        return categorized
    
    def create_system_health_score(self, df: pd.DataFrame, categorized_metrics: Dict[str, List[str]]) -> pd.DataFrame:
        """Create composite system health scores."""
        logging.info("Creating system health scores...")
        
        df_copy = df.copy()
        
        # CPU Health Score
        cpu_metrics = categorized_metrics.get('cpu', [])
        if cpu_metrics:
            cpu_cols = [col for col in cpu_metrics if col in df_copy.columns]
            if cpu_cols:
                # Normalize CPU metrics (higher values generally mean more load)
                cpu_normalized = df_copy[cpu_cols].fillna(0)
                for col in cpu_cols:
                    if df_copy[col].std() > 0:
                        cpu_normalized[col] = (df_copy[col] - df_copy[col].min()) / (df_copy[col].max() - df_copy[col].min())
                
                df_copy['cpu_health_score'] = 1 - cpu_normalized.mean(axis=1)  # Invert: lower usage = better health
        
        # Memory Health Score
        memory_metrics = categorized_metrics.get('memory', [])
        if memory_metrics:
            mem_cols = [col for col in memory_metrics if col in df_copy.columns and 'available' not in col.lower()]
            available_cols = [col for col in memory_metrics if col in df_copy.columns and 'available' in col.lower()]
            
            if mem_cols:
                mem_normalized = df_copy[mem_cols].fillna(0)
                for col in mem_cols:
                    if df_copy[col].std() > 0:
                        mem_normalized[col] = (df_copy[col] - df_copy[col].min()) / (df_copy[col].max() - df_copy[col].min())
                
                memory_usage_score = mem_normalized.mean(axis=1)
                
                # Factor in available memory if present
                if available_cols:
                    available_normalized = df_copy[available_cols].fillna(0)
                    for col in available_cols:
                        if df_copy[col].std() > 0:
                            available_normalized[col] = (df_copy[col] - df_copy[col].min()) / (df_copy[col].max() - df_copy[col].min())
                    
                    df_copy['memory_health_score'] = (available_normalized.mean(axis=1) + (1 - memory_usage_score)) / 2
                else:
                    df_copy['memory_health_score'] = 1 - memory_usage_score
        
        # Disk Health Score
        disk_metrics = categorized_metrics.get('disk', [])
        if disk_metrics:
            disk_cols = [col for col in disk_metrics if col in df_copy.columns]
            if disk_cols:
                disk_available_cols = [col for col in disk_cols if 'avail' in col.lower() or 'free' in col.lower()]
                disk_usage_cols = [col for col in disk_cols if col not in disk_available_cols and 'io' not in col.lower()]
                
                if disk_available_cols and disk_usage_cols:
                    # Normalize available space (higher = better)
                    available_normalized = df_copy[disk_available_cols].fillna(0)
                    for col in disk_available_cols:
                        if df_copy[col].std() > 0:
                            available_normalized[col] = (df_copy[col] - df_copy[col].min()) / (df_copy[col].max() - df_copy[col].min())
                    
                    # Normalize usage metrics (lower = better)
                    usage_normalized = df_copy[disk_usage_cols].fillna(0)
                    for col in disk_usage_cols:
                        if df_copy[col].std() > 0:
                            usage_normalized[col] = (df_copy[col] - df_copy[col].min()) / (df_copy[col].max() - df_copy[col].min())
                    
                    df_copy['disk_health_score'] = (available_normalized.mean(axis=1) + (1 - usage_normalized.mean(axis=1))) / 2
        
        # Network Health Score
        network_metrics = categorized_metrics.get('network', [])
        if network_metrics:
            net_cols = [col for col in network_metrics if col in df_copy.columns]
            error_cols = [col for col in net_cols if 'err' in col.lower() or 'drop' in col.lower() or 'fail' in col.lower()]
            throughput_cols = [col for col in net_cols if 'bytes' in col.lower() or 'packets' in col.lower() and col not in error_cols]
            
            if error_cols:
                error_normalized = df_copy[error_cols].fillna(0)
                for col in error_cols:
                    if df_copy[col].std() > 0:
                        error_normalized[col] = (df_copy[col] - df_copy[col].min()) / (df_copy[col].max() - df_copy[col].min())
                
                df_copy['network_error_rate'] = error_normalized.mean(axis=1)
                df_copy['network_health_score'] = 1 - df_copy['network_error_rate']
            elif throughput_cols:
                # Use throughput stability as a proxy for health
                throughput_data = df_copy[throughput_cols].fillna(0)
                df_copy['network_health_score'] = 1 - throughput_data.std(axis=1) / (throughput_data.mean(axis=1) + 1e-6)
        
        # Overall System Health Score
        health_scores = [col for col in df_copy.columns if 'health_score' in col]
        if health_scores:
            df_copy['overall_system_health'] = df_copy[health_scores].mean(axis=1)
        
        return df_copy
    
    def create_resource_utilization_features(self, df: pd.DataFrame, categorized_metrics: Dict[str, List[str]]) -> pd.DataFrame:
        """Create resource utilization and efficiency features."""
        logging.info("Creating resource utilization features...")
        
        df_copy = df.copy()
        
        # CPU Utilization Patterns
        cpu_metrics = categorized_metrics.get('cpu', [])
        if cpu_metrics:
            cpu_cols = [col for col in cpu_metrics if col in df_copy.columns]
            if cpu_cols:
                cpu_data = df_copy[cpu_cols].fillna(0)
                
                # CPU utilization variance (higher = more volatile)
                df_copy['cpu_utilization_variance'] = cpu_data.var(axis=1)
                
                # Peak to average ratio
                df_copy['cpu_peak_avg_ratio'] = cpu_data.max(axis=1) / (cpu_data.mean(axis=1) + 1e-6)
                
                # CPU efficiency (how evenly distributed the load is)
                df_copy['cpu_efficiency'] = 1 / (df_copy['cpu_peak_avg_ratio'] + 1e-6)
        
        # Memory Pressure Indicators
        memory_metrics = categorized_metrics.get('memory', [])
        if memory_metrics:
            mem_cols = [col for col in memory_metrics if col in df_copy.columns]
            if mem_cols:
                total_mem_cols = [col for col in mem_cols if 'total' in col.lower()]
                used_mem_cols = [col for col in mem_cols if 'alloc' in col.lower() or 'inuse' in col.lower()]
                available_mem_cols = [col for col in mem_cols if 'available' in col.lower() or 'free' in col.lower()]
                
                if total_mem_cols and (used_mem_cols or available_mem_cols):
                    total_memory = df_copy[total_mem_cols].sum(axis=1)
                    
                    if used_mem_cols:
                        used_memory = df_copy[used_mem_cols].sum(axis=1)
                        df_copy['memory_utilization_pct'] = (used_memory / (total_memory + 1e-6)) * 100
                    
                    if available_mem_cols:
                        available_memory = df_copy[available_mem_cols].sum(axis=1)
                        df_copy['memory_pressure'] = 1 - (available_memory / (total_memory + 1e-6))
        
        # Disk I/O Patterns
        disk_metrics = categorized_metrics.get('disk', [])
        if disk_metrics:
            disk_cols = [col for col in disk_metrics if col in df_copy.columns]
            read_cols = [col for col in disk_cols if 'read' in col.lower()]
            write_cols = [col for col in disk_cols if 'write' in col.lower()]
            
            if read_cols and write_cols:
                read_activity = df_copy[read_cols].sum(axis=1)
                write_activity = df_copy[write_cols].sum(axis=1)
                
                df_copy['disk_total_activity'] = read_activity + write_activity
                df_copy['disk_read_write_ratio'] = read_activity / (write_activity + 1e-6)
                
                # Disk I/O burstiness
                if len(df_copy) > 10:
                    df_copy['disk_io_burstiness'] = df_copy['disk_total_activity'].rolling(10, min_periods=1).std()
        
        return df_copy
    
    def create_performance_indicators(self, df: pd.DataFrame, categorized_metrics: Dict[str, List[str]]) -> pd.DataFrame:
        """Create performance and latency indicators."""
        logging.info("Creating performance indicators...")
        
        df_copy = df.copy()
        
        # HTTP Performance Metrics
        http_metrics = categorized_metrics.get('http', [])
        if http_metrics:
            http_cols = [col for col in http_metrics if col in df_copy.columns]
            duration_cols = [col for col in http_cols if 'duration' in col.lower()]
            request_cols = [col for col in http_cols if 'request' in col.lower() and 'total' in col.lower()]
            
            if duration_cols:
                # Average response time
                df_copy['avg_response_time'] = df_copy[duration_cols].mean(axis=1)
                
                # Response time variability
                df_copy['response_time_variability'] = df_copy[duration_cols].std(axis=1)
            
            if request_cols:
                # Request rate
                df_copy['request_rate'] = df_copy[request_cols].sum(axis=1)
                
                # Request rate change
                if len(df_copy) > 1:
                    df_copy['request_rate_change'] = df_copy['request_rate'].pct_change()
        
        # Kubernetes Performance
        k8s_metrics = categorized_metrics.get('kubernetes', [])
        if k8s_metrics:
            k8s_cols = [col for col in k8s_metrics if col in df_copy.columns]
            
            # Pod health indicators
            pod_ready_cols = [col for col in k8s_cols if 'pod' in col and 'ready' in col]
            pod_restart_cols = [col for col in k8s_cols if 'restart' in col.lower()]
            
            if pod_ready_cols:
                df_copy['pod_readiness_ratio'] = df_copy[pod_ready_cols].mean(axis=1)
            
            if pod_restart_cols:
                df_copy['pod_restart_rate'] = df_copy[pod_restart_cols].sum(axis=1)
            
            # Resource requests vs limits
            resource_request_cols = [col for col in k8s_cols if 'request' in col and 'resource' in col]
            resource_limit_cols = [col for col in k8s_cols if 'limit' in col and 'resource' in col]
            
            if resource_request_cols and resource_limit_cols:
                requests = df_copy[resource_request_cols].sum(axis=1)
                limits = df_copy[resource_limit_cols].sum(axis=1)
                df_copy['resource_request_limit_ratio'] = requests / (limits + 1e-6)
        
        # Go Runtime Performance
        go_metrics = categorized_metrics.get('go_runtime', [])
        if go_metrics:
            go_cols = [col for col in go_metrics if col in df_copy.columns]
            gc_cols = [col for col in go_cols if 'gc' in col.lower()]
            goroutine_cols = [col for col in go_cols if 'goroutine' in col.lower()]
            
            if gc_cols:
                # GC pressure
                df_copy['gc_pressure'] = df_copy[gc_cols].sum(axis=1)
            
            if goroutine_cols:
                # Goroutine count (concurrency indicator)
                df_copy['goroutine_count'] = df_copy[goroutine_cols].sum(axis=1)
        
        return df_copy
    
    def create_anomaly_detection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for anomaly detection."""
        logging.info("Creating anomaly detection features...")
        
        df_copy = df.copy()
        
        # Get numeric columns only
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        metric_cols = [col for col in numeric_cols 
                      if col not in ['timestamp', 'hour', 'day_of_week', 'day_of_month', 
                                    'month', 'is_weekend', 'is_business_hours', 'time_since_start']]
        
        if not metric_cols:
            return df_copy
        
        # Statistical anomaly features (limit to avoid too many features)
        for col in metric_cols[:50]:  # Limit to avoid too many features
            if df_copy[col].std() > 0:
                # Z-score
                df_copy[f'{col}_zscore'] = (df_copy[col] - df_copy[col].mean()) / df_copy[col].std()
                
                # Modified z-score using median
                median = df_copy[col].median()
                mad = np.median(np.abs(df_copy[col] - median))
                if mad > 0:
                    df_copy[f'{col}_modified_zscore'] = 0.6745 * (df_copy[col] - median) / mad
                
                # Percentile-based features
                df_copy[f'{col}_percentile'] = df_copy[col].rank(pct=True)
                
                # Rate of change
                if len(df_copy) > 1:
                    df_copy[f'{col}_rate_of_change'] = df_copy[col].pct_change().abs()
        
        # Time-based anomaly features
        if 'hour' in df_copy.columns:
            # Create hourly baselines
            hourly_stats = df_copy.groupby('hour')[metric_cols[:20]].agg(['mean', 'std']).fillna(0)
            
            for col in metric_cols[:20]:
                if (col, 'mean') in hourly_stats.columns and (col, 'std') in hourly_stats.columns:
                    # Deviation from hourly normal
                    hour_means = df_copy['hour'].map(hourly_stats[(col, 'mean')])
                    hour_stds = df_copy['hour'].map(hourly_stats[(col, 'std')])
                    
                    df_copy[f'{col}_hourly_deviation'] = np.abs(df_copy[col] - hour_means) / (hour_stds + 1e-6)
        
        return df_copy
    
    def create_correlation_features(self, df: pd.DataFrame, categorized_metrics: Dict[str, List[str]]) -> pd.DataFrame:
        """Create cross-metric correlation features."""
        logging.info("Creating correlation features...")
        
        df_copy = df.copy()
        
        # CPU-Memory correlation
        cpu_cols = [col for col in categorized_metrics.get('cpu', []) if col in df_copy.columns]
        memory_cols = [col for col in categorized_metrics.get('memory', []) if col in df_copy.columns]
        
        if cpu_cols and memory_cols:
            cpu_sum = df_copy[cpu_cols].sum(axis=1)
            memory_sum = df_copy[memory_cols].sum(axis=1)
            
            # Rolling correlation
            window = min(50, len(df_copy) // 4) if len(df_copy) > 10 else len(df_copy)
            if window > 1:
                df_copy['cpu_memory_correlation'] = cpu_sum.rolling(window, min_periods=2).corr(memory_sum)
        
        # Network-Disk correlation (often related in I/O intensive apps)
        network_cols = [col for col in categorized_metrics.get('network', []) if col in df_copy.columns]
        disk_cols = [col for col in categorized_metrics.get('disk', []) if col in df_copy.columns]
        
        if network_cols and disk_cols:
            network_sum = df_copy[network_cols].sum(axis=1)
            disk_sum = df_copy[disk_cols].sum(axis=1)
            
            window = min(50, len(df_copy) // 4) if len(df_copy) > 10 else len(df_copy)
            if window > 1:
                df_copy['network_disk_correlation'] = network_sum.rolling(window, min_periods=2).corr(disk_sum)
        
        # HTTP requests vs system resources
        http_cols = [col for col in categorized_metrics.get('http', []) if col in df_copy.columns]
        if http_cols and cpu_cols:
            http_sum = df_copy[http_cols].sum(axis=1)
            
            window = min(50, len(df_copy) // 4) if len(df_copy) > 10 else len(df_copy)
            if window > 1:
                df_copy['http_cpu_correlation'] = http_sum.rolling(window, min_periods=2).corr(cpu_sum)
        
        return df_copy
    
    def create_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced time series features."""
        logging.info("Creating time series features...")
        
        df_copy = df.copy()
        
        # Get numeric columns
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        metric_cols = [col for col in numeric_cols 
                      if col not in ['timestamp', 'hour', 'day_of_week', 'day_of_month', 
                                    'month', 'is_weekend', 'is_business_hours', 'time_since_start']]
        
        # Seasonal patterns
        if 'hour' in df_copy.columns and 'day_of_week' in df_copy.columns:
            # Create time-of-week feature (0-167 hours in a week)
            df_copy['hour_of_week'] = df_copy['day_of_week'] * 24 + df_copy['hour']
            
            # Cyclical encoding for better ML model performance
            df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy['hour'] / 24)
            df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy['hour'] / 24)
            df_copy['day_sin'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7)
            df_copy['day_cos'] = np.cos(2 * np.pi * df_copy['day_of_week'] / 7)
        
        # Trend analysis for key metrics
        key_metrics = metric_cols[:20]  # Limit to avoid too many features
        
        for col in key_metrics:
            if len(df_copy) > 10:
                # Linear trend (slope over time)
                x = np.arange(len(df_copy))
                y = df_copy[col].ffill().fillna(0)  # Fixed deprecated method usage
                
                if len(y) > 1 and y.std() > 0:
                    # Simple linear regression slope
                    slope = np.corrcoef(x, y)[0, 1] * (y.std() / x.std()) if x.std() > 0 else 0
                    df_copy[f'{col}_trend'] = slope
                    
                    # Detrended values
                    trend_line = slope * x + y.mean()
                    df_copy[f'{col}_detrended'] = y - trend_line
        
        return df_copy
    
    def create_dimensionality_reduction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create PCA and clustering features."""
        logging.info("Creating dimensionality reduction features...")
        
        # Try importing sklearn, skip if not available
        try:
            from sklearn.preprocessing import RobustScaler
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans
        except ImportError:
            logging.warning("scikit-learn not available, skipping dimensionality reduction features")
            return df
        
        df_copy = df.copy()
        
        # Get numeric metric columns
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        metric_cols = [col for col in numeric_cols 
                      if col not in ['timestamp', 'hour', 'day_of_week', 'day_of_month', 
                                    'month', 'is_weekend', 'is_business_hours', 'time_since_start']]
        
        if len(metric_cols) < 3:
            return df_copy
        
        # Prepare data for dimensionality reduction
        metric_data = df_copy[metric_cols].fillna(0)
        
        # Remove columns with zero variance
        numeric_cols_filtered = [col for col in metric_cols if metric_data[col].std() > 0]
        
        if len(numeric_cols_filtered) < 3:
            return df_copy
        
        metric_data_filtered = metric_data[numeric_cols_filtered]
        
        # Standardize the data
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(metric_data_filtered)
        
        # PCA
        try:
            n_components = min(5, len(numeric_cols_filtered), len(df_copy) - 1)
            if n_components > 0:
                pca = PCA(n_components=n_components)
                pca_components = pca.fit_transform(scaled_data)
                
                for i in range(n_components):
                    df_copy[f'pca_component_{i+1}'] = pca_components[:, i]
                
                # Explained variance ratio
                df_copy['pca_explained_variance'] = pca.explained_variance_ratio_.sum()
        except Exception as e:
            logging.warning(f"PCA failed: {e}")
        
        # K-means clustering for anomaly detection
        try:
            if len(df_copy) > 10:
                n_clusters = min(5, len(df_copy) // 3)
                if n_clusters > 1:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(scaled_data)
                    
                    df_copy['cluster_id'] = clusters
                    
                    # Distance to cluster center (anomaly score)
                    distances = np.min(kmeans.transform(scaled_data), axis=1)
                    df_copy['cluster_distance'] = distances
                    
                    # Normalize distance to 0-1 scale
                    if distances.std() > 0:
                        df_copy['cluster_anomaly_score'] = (distances - distances.min()) / (distances.max() - distances.min())
        except Exception as e:
            logging.warning(f"Clustering failed: {e}")
        
        return df_copy
    
    def engineer_features(self, 
                         include_health_scores: bool = True,
                         include_utilization: bool = True,
                         include_performance: bool = True,
                         include_anomaly: bool = True,
                         include_correlation: bool = True,
                         include_time_series: bool = True,
                         include_dimensionality_reduction: bool = True) -> pd.DataFrame:
        """Complete feature engineering pipeline."""
        logging.info("Starting infrastructure feature engineering...")
        
        # Load data
        self.load_data()
        df_engineered = self.df.copy()
        
        # Categorize metrics based on current dataframe
        categorized_metrics = self.categorize_metrics(df_engineered)
        
        if include_health_scores:
            df_engineered = self.create_system_health_score(df_engineered, categorized_metrics)
        
        if include_utilization:
            df_engineered = self.create_resource_utilization_features(df_engineered, categorized_metrics)
        
        if include_performance:
            df_engineered = self.create_performance_indicators(df_engineered, categorized_metrics)
        
        if include_anomaly:
            df_engineered = self.create_anomaly_detection_features(df_engineered)
        
        if include_correlation:
            df_engineered = self.create_correlation_features(df_engineered, categorized_metrics)
        
        if include_time_series:
            df_engineered = self.create_time_series_features(df_engineered)
        
        if include_dimensionality_reduction:
            df_engineered = self.create_dimensionality_reduction_features(df_engineered)
        
        # Fill any remaining NaN values - Fixed deprecated method usage
        df_engineered = df_engineered.ffill().bfill().fillna(0)
        
        logging.info(f"Feature engineering complete. Final shape: {df_engineered.shape}")
        logging.info(f"Original columns: {len(self.df.columns)}, New columns: {len(df_engineered.columns)}")
        
        return df_engineered
    
    def save_engineered_features(self, df: pd.DataFrame, format: str = 'parquet'):
        """Save the engineered features with metadata."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'parquet':
            output_file = self.output_path.with_suffix('.parquet')
            df.to_parquet(output_file, index=False)
        elif format == 'csv':
            output_file = self.output_path.with_suffix('.csv')
            df.to_csv(output_file, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logging.info(f"Engineered features saved to {output_file}")
        
        # Save feature metadata
        feature_info = {
            'original_features': len(self.df.columns),
            'engineered_features': len(df.columns),
            'new_features': len(df.columns) - len(self.df.columns),
            'time_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
            'total_rows': len(df)
        }
        
        metadata_file = self.output_path.parent / f"{self.output_path.name}_metadata.txt"
        with open(metadata_file, 'w') as f:
            f.write("Infrastructure Feature Engineering Summary\n")
            f.write("=" * 50 + "\n\n")
            for key, value in feature_info.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            
            f.write("\nNew Feature Categories:\n")
            new_features = [col for col in df.columns if col not in self.df.columns]
            
            categories = {
                'Health Scores': [f for f in new_features if 'health' in f],
                'Utilization': [f for f in new_features if any(x in f for x in ['utilization', 'pressure', 'efficiency'])],
                'Performance': [f for f in new_features if any(x in f for x in ['performance', 'latency', 'response', 'rate'])],
                'Anomaly Detection': [f for f in new_features if any(x in f for x in ['zscore', 'anomaly', 'deviation', 'percentile'])],
                'Correlation': [f for f in new_features if 'correlation' in f],
                'Time Series': [f for f in new_features if any(x in f for x in ['trend', 'seasonal', 'sin', 'cos'])],
                'Dimensionality Reduction': [f for f in new_features if any(x in f for x in ['pca', 'cluster'])]
            }
            
            for category, features in categories.items():
                if features:
                    f.write(f"\n{category}: {len(features)} features\n")
                    for feature in features[:10]:  # Show first 10
                        f.write(f"  - {feature}\n")
                    if len(features) > 10:
                        f.write(f"  ... and {len(features) - 10} more\n")
        
        logging.info(f"Feature metadata saved to {metadata_file}")


def main():
    """Main function to parse arguments and engineer features."""
    parser = argparse.ArgumentParser(
        description="Advanced feature engineering for infrastructure monitoring metrics."
    )
    parser.add_argument(
        "input_path",
        nargs='?',  # Make optional
        default="consolidated_dataset.parquet",
        help="Path to prepared dataset (default: consolidated_dataset.parquet)"
    )
    parser.add_argument(
        "output_path", 
        nargs='?',  # Make optional
        default="engineered_features",
        help="Path for output engineered features (default: engineered_features)"
    )
    parser.add_argument(
        "--output-format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output file format (default: parquet)"
    )
    parser.add_argument(
        "--skip-health-scores",
        action="store_true",
        help="Skip system health score features"
    )
    parser.add_argument(
        "--skip-utilization",
        action="store_true",
        help="Skip resource utilization features"
    )
    parser.add_argument(
        "--skip-performance",
        action="store_true",
        help="Skip performance indicator features"
    )
    parser.add_argument(
        "--skip-anomaly",
        action="store_true",
        help="Skip anomaly detection features"
    )
    parser.add_argument(
        "--skip-correlation",
        action="store_true",
        help="Skip correlation features"
    )
    parser.add_argument(
        "--skip-time-series",
        action="store_true",
        help="Skip time series features"
    )
    parser.add_argument(
        "--skip-dimensionality-reduction",
        action="store_true",
        help="Skip PCA and clustering features"
    )
    
    args = parser.parse_args()
    
    # Show what we're doing
    print(f"🔧 Engineering features...")
    print(f"   Input: {args.input_path}")
    print(f"   Output: {args.output_path}.{args.output_format}")
    
    # Show which features will be created
    features_to_create = []
    if not args.skip_health_scores: features_to_create.append("Health Scores")
    if not args.skip_utilization: features_to_create.append("Utilization")
    if not args.skip_performance: features_to_create.append("Performance")
    if not args.skip_anomaly: features_to_create.append("Anomaly Detection")
    if not args.skip_correlation: features_to_create.append("Correlation")
    if not args.skip_time_series: features_to_create.append("Time Series")
    if not args.skip_dimensionality_reduction: features_to_create.append("Dimensionality Reduction")
    
    print(f"   Features: {', '.join(features_to_create)}")
    print()
    
    try:
        engineer = InfrastructureFeatureEngineer(args.input_path, args.output_path)
        
        engineered_df = engineer.engineer_features(
            include_health_scores=not args.skip_health_scores,
            include_utilization=not args.skip_utilization,
            include_performance=not args.skip_performance,
            include_anomaly=not args.skip_anomaly,
            include_correlation=not args.skip_correlation,
            include_time_series=not args.skip_time_series,
            include_dimensionality_reduction=not args.skip_dimensionality_reduction
        )
        
        engineer.save_engineered_features(engineered_df, args.output_format)
        
        logging.info("Infrastructure feature engineering completed successfully!")
        print(f"\n✅ Success! Engineered features saved to {args.output_path}.{args.output_format}")
        print(f"   Metadata: {args.output_path}_metadata.txt")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()