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
        """Create anomaly detection features."""
        anomaly_features = []
        
        for metric in metrics:
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
        """Create correlation-based features."""
        correlation_features = []
        
        # Calculate correlations between metrics
        for i, metric1 in enumerate(metrics[:-1]):
            for metric2 in metrics[i+1:]:
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
        """Create dimensionality reduction features."""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            numeric_cols = [col for col in numeric_cols if col not in ['timestamp', 'hour', 'day_of_week', 'day_of_month', 'month', 'is_business_hours']]
            
            if len(numeric_cols) < 2:
                return []
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[numeric_cols])
            
            # Apply PCA
            n_components = min(3, len(numeric_cols))  # Use at most 3 components
            pca = PCA(n_components=n_components)
            pca_data = pca.fit_transform(scaled_data)
            
            # Create PCA features
            pca_features = []
            for i in range(n_components):
                feature_name = f'pca_component_{i+1}'
                df[feature_name] = pca_data[:, i]
                pca_features.append(feature_name)
            
            return pca_features
            
        except Exception as e:
            self.logger.warning(f"Failed to create dimensionality reduction features: {e}")
            return []
    
    def engineer_features(self, config: dict) -> None:
        """Engineer features from the input dataset."""
        try:
            self.logger.info("Starting infrastructure feature engineering...")
            self.logger.info(f"Loading data from {self.input_path}")
            
            # Load data
            df = pd.read_parquet(self.input_path)
            self.logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
            self.logger.info(f"Loaded data: {df.shape}")
            
            # Get list of metrics
            metrics = [col for col in df.columns if col != 'timestamp']
            self.logger.info(f"MEMORY: {len(metrics)} metrics")
            
            # Create time features first - this is critical
            df = self._create_time_features(df)
            
            # Store original feature count
            original_features = len(df.columns)
            
            # Initialize feature list
            engineered_features = []
            
            if not config['skip_health_scores']:
                self.logger.info("Creating system health scores...")
                health_scores = self._create_health_scores(df, metrics)
                engineered_features.extend(health_scores)
                self.logger.info("Health scores created")
            
            if not config['skip_utilization']:
                self.logger.info("Creating resource utilization features...")
                utilization_features = self._create_utilization_features(df, metrics)
                engineered_features.extend(utilization_features)
                self.logger.info("Resource utilization features created")
            
            if not config['skip_performance']:
                self.logger.info("Creating performance indicators...")
                performance_features = self._create_performance_features(df, metrics)
                engineered_features.extend(performance_features)
                self.logger.info("Performance indicators created")
            
            if not config['skip_anomaly']:
                self.logger.info("Creating anomaly detection features...")
                anomaly_features = self._create_anomaly_features(df, metrics)
                engineered_features.extend(anomaly_features)
                self.logger.info("Anomaly detection features created")
            
            if not config['skip_correlation']:
                self.logger.info("Creating correlation features...")
                correlation_features = self._create_correlation_features(df, metrics)
                engineered_features.extend(correlation_features)
                self.logger.info("Correlation features created")
            
            if not config['skip_time_series']:
                self.logger.info("Creating time series features...")
                time_series_features = self._create_time_series_features(df, metrics)
                engineered_features.extend(time_series_features)
                self.logger.info("Time series features created")
            
            if not config['skip_dimensionality_reduction']:
                self.logger.info("Creating dimensionality reduction features...")
                dim_reduction_features = self._create_dimensionality_reduction_features(df)
                engineered_features.extend(dim_reduction_features)
                self.logger.info("Dimensionality reduction features created")
            
            # Add engineered features to dataframe
            for feature in engineered_features:
                if feature not in df.columns:
                    df[feature] = 0  # Initialize with default value
            
            self.logger.info(f"Added {len(engineered_features)} new features")
            
            # Save the engineered features
            output_path = self.output_path
            if not str(output_path).endswith(config['output_format']):
                output_path = output_path.with_suffix(f".{config['output_format']}")
            
            if config['output_format'] == 'parquet':
                df.to_parquet(output_path, index=False)
            else:
                df.to_csv(output_path, index=False)
            
            # Save metadata
            metadata_path = self.output_path.with_suffix('.txt')
            with open(metadata_path, 'w') as f:
                f.write("Infrastructure Monitoring Features\n")
                f.write("=" * 30 + "\n\n")
                f.write(f"Total features: {len(df.columns)}\n")
                f.write(f"Original metrics: {len(metrics)}\n")
                f.write(f"Engineered features: {len(engineered_features)}\n\n")
                f.write("Feature Categories:\n")
                f.write("-" * 20 + "\n")
                if not config['skip_health_scores']:
                    f.write("- Health Scores\n")
                if not config['skip_utilization']:
                    f.write("- Resource Utilization\n")
                if not config['skip_performance']:
                    f.write("- Performance Indicators\n")
                if not config['skip_anomaly']:
                    f.write("- Anomaly Detection\n")
                if not config['skip_correlation']:
                    f.write("- Correlation Features\n")
                if not config['skip_time_series']:
                    f.write("- Time Series Features\n")
                if not config['skip_dimensionality_reduction']:
                    f.write("- Dimensionality Reduction\n")
            
            self.logger.info("Feature engineering completed successfully!")
            
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