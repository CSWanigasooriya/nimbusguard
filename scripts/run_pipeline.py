#!/usr/bin/env python3
"""
Unified Infrastructure Monitoring Pipeline
Complete end-to-end pipeline: Prometheus extraction → Data preparation → Feature engineering → Analysis

This single script replaces all the wrapper scripts and provides a unified interface
for the entire infrastructure monitoring data pipeline.
"""

import argparse
import logging
import re
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import glob
import shutil

import pandas as pd
import numpy as np
from prometheus_api_client import PrometheusConnect

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

class UnifiedInfrastructurePipeline:
    """Complete infrastructure monitoring pipeline in a single class."""
    
    def __init__(self, config: dict):
        self.config = config
        self.work_dir = Path(config.get('work_dir', Path.cwd()))
        self.prometheus_data_dir = self.work_dir / 'prometheus_data'
        self.consolidated_file = self.work_dir / 'consolidated_dataset.parquet'
        self.engineered_file = self.work_dir / 'engineered_features.parquet'
        
        # Metric categories for feature engineering
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
    
    def test_prometheus_connection(self) -> bool:
        """Test connection to Prometheus server."""
        url = self.config['prometheus_url']
        logging.info(f"Testing connection to {url}...")
        
        try:
            prom = PrometheusConnect(url=url, disable_ssl=True)
            if prom.check_prometheus_connection():
                logging.info("✅ Connection successful!")
                return True
            else:
                logging.error("❌ Connection failed. Check URL and Prometheus status.")
                return False
        except Exception as e:
            logging.error(f"❌ Connection test failed: {e}")
            return False
    
    def extract_prometheus_data(self) -> bool:
        """Extract data from Prometheus and save as individual CSV files."""
        logging.info("🚀 Extracting data from Prometheus...")
        
        config = self.config
        url = config['prometheus_url']
        days = config['days']
        step = config['step']
        workers = config['workers']
        
        # Test connection first
        if not self.test_prometheus_connection():
            return False
        
        # Create output directory
        self.prometheus_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up Prometheus client
        prom = PrometheusConnect(url=url, disable_ssl=True)
        
        # Get metrics list
        try:
            metric_names = prom.all_metrics()
        except Exception as e:
            logging.error(f"Failed to get metrics list: {e}")
            return False
        
        # Apply filters if specified
        if config.get('metric_filter'):
            try:
                regex = re.compile(config['metric_filter'])
                metric_names = [m for m in metric_names if regex.search(m)]
            except re.error as e:
                logging.error(f"Invalid regex filter '{config['metric_filter']}': {e}")
                return False
        
        logging.info(f"Found {len(metric_names)} metrics to extract")
        if not metric_names:
            logging.warning("No metrics found. Check your Prometheus server.")
            return False
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        logging.info(f"Time range: {start_time} to {end_time}")
        
        # Extract metrics in parallel
        successful_extractions = 0
        
        def extract_metric(metric_name):
            try:
                metric_data = prom.custom_query_range(
                    query=metric_name,
                    start_time=start_time,
                    end_time=end_time,
                    step=step,
                )
                
                if not metric_data:
                    return False
                
                df_list = []
                for d in metric_data:
                    metric_labels = d["metric"]
                    df = pd.DataFrame(d["values"], columns=["timestamp", "value"])
                    df["metric_name"] = metric_labels.pop("__name__", metric_name)
                    
                    for label, value in metric_labels.items():
                        df[label] = value
                    
                    df_list.append(df)
                
                if not df_list:
                    return False
                
                metric_df = pd.concat(df_list, ignore_index=True)
                metric_df["timestamp"] = pd.to_datetime(metric_df["timestamp"], unit="s")
                metric_df["value"] = pd.to_numeric(metric_df["value"], errors="coerce")
                metric_df = metric_df.dropna(subset=["value"])
                
                # Save to CSV
                safe_metric_name = re.sub(r'[^0-9a-zA-Z_]', '_', metric_name)
                output_path = self.prometheus_data_dir / f"{safe_metric_name}.csv"
                metric_df.to_csv(output_path, index=False)
                
                return True
            except Exception as e:
                logging.error(f"Failed to extract {metric_name}: {e}")
                return False
        
        # Extract with thread pool
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(extract_metric, metric_names))
            successful_extractions = sum(results)
        
        logging.info(f"✅ Successfully extracted {successful_extractions}/{len(metric_names)} metrics")
        return successful_extractions > 0
    
    def consolidate_data(self) -> bool:
        """Consolidate individual CSV files into a unified dataset."""
        logging.info("🔄 Consolidating CSV files...")
        
        # Get CSV files
        csv_files = list(self.prometheus_data_dir.glob("*.csv"))
        if not csv_files:
            logging.error("No CSV files found for consolidation")
            return False
        
        logging.info(f"Found {len(csv_files)} CSV files to consolidate")
        
        # Process files in batches
        batch_size = self.config.get('batch_size', 50)
        all_dataframes = []
        
        for i in range(0, len(csv_files), batch_size):
            batch_files = csv_files[i:i + batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}/{(len(csv_files)-1)//batch_size + 1}")
            
            batch_dfs = []
            for file in batch_files:
                try:
                    df = pd.read_csv(file)
                    
                    # Ensure required columns
                    if 'metric_name' not in df.columns:
                        df['metric_name'] = file.stem
                    
                    if 'timestamp' not in df.columns or 'value' not in df.columns:
                        continue
                    
                    # Standardize columns
                    df.columns = df.columns.str.lower().str.strip()
                    batch_dfs.append(df)
                    
                except Exception as e:
                    logging.warning(f"Error processing {file.name}: {e}")
                    continue
            
            if batch_dfs:
                batch_combined = pd.concat([df for df in batch_dfs if not df.empty], 
                                         ignore_index=True)
                if not batch_combined.empty:
                    all_dataframes.append(batch_combined)
        
        if not all_dataframes:
            logging.error("No valid data found in CSV files")
            return False
        
        # Combine all batches
        logging.info("Combining all batches...")
        df = pd.concat([df for df in all_dataframes if not df.empty], ignore_index=True)
        
        # Clean and standardize data...
        logging.info("Cleaning and standardizing data...")
        initial_rows = len(df)
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Handle mixed-type object columns (fix for parquet compatibility)
        object_columns = df.select_dtypes(include=['object']).columns
        for col in object_columns:
            if col not in ['timestamp', 'metric_name']:
                # Convert mixed types to strings to avoid parquet issues
                df[col] = df[col].astype(str)
                # Replace 'nan' strings with actual NaN
                df[col] = df[col].replace('nan', None)
        
        # Remove invalid data
        df = df.dropna(subset=['timestamp', 'value'])
        logging.info(f"Removed {initial_rows - len(df)} rows with invalid data")
        
        # Sort and deduplicate
        df = df.sort_values(['metric_name', 'timestamp'])
        df = df.drop_duplicates()
        
        # Add basic time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['is_business_hours'] = df['hour'].between(9, 17)
        
        # Convert to wide format for feature engineering
        logging.info("Converting to wide format...")
        
        # Create metric keys (simplified)
        df['metric_key'] = df['metric_name']
        if 'instance' in df.columns:
            df['metric_key'] += '_' + df['instance'].fillna('default').astype(str)
        
        # Handle ALL metrics intelligently
        unique_metrics = df['metric_key'].nunique()
        logging.info(f"Processing ALL {unique_metrics} metrics...")
        
        # For large numbers of metrics, use category-based aggregation instead of individual metrics
        if unique_metrics > 5000:
            logging.info("Large number of metrics detected - using category-based aggregation")
            
            # Group by timestamp and create category summaries
            time_cols = ['hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend', 'is_business_hours']
            timestamp_groups = df.groupby(['timestamp'] + time_cols)
            summary_data = []
            
            for timestamp_vals, group in timestamp_groups:
                row_data = dict(zip(['timestamp'] + time_cols, timestamp_vals))
                
                # Category-based aggregations that preserve all metric information
                categories = {
                    'cpu': group[group['metric_name'].str.contains('cpu', case=False, na=False)],
                    'memory': group[group['metric_name'].str.contains('memory|mem', case=False, na=False)],
                    'disk': group[group['metric_name'].str.contains('disk|filesystem', case=False, na=False)],
                    'network': group[group['metric_name'].str.contains('network|net', case=False, na=False)],
                    'http': group[group['metric_name'].str.contains('http', case=False, na=False)],
                    'k8s': group[group['metric_name'].str.startswith('kube_')],
                    'go_runtime': group[group['metric_name'].str.startswith('go_')],
                    'prometheus': group[group['metric_name'].str.startswith('prometheus_')],
                    'alloy': group[group['metric_name'].str.startswith('alloy_')],
                    'loki': group[group['metric_name'].str.startswith('loki_')]
                }
                
                # Add comprehensive statistics for each category
                for cat_name, cat_data in categories.items():
                    if not cat_data.empty:
                        values = cat_data['value']
                        row_data[f'{cat_name}_mean'] = values.mean()
                        row_data[f'{cat_name}_max'] = values.max()
                        row_data[f'{cat_name}_min'] = values.min()
                        row_data[f'{cat_name}_std'] = values.std()
                        row_data[f'{cat_name}_count'] = len(values)
                        row_data[f'{cat_name}_sum'] = values.sum()
                        row_data[f'{cat_name}_median'] = values.median()
                        
                        # Add percentiles for better distribution understanding
                        row_data[f'{cat_name}_p95'] = values.quantile(0.95)
                        row_data[f'{cat_name}_p99'] = values.quantile(0.99)
                
                # Also include top individual metrics for detailed analysis
                top_individual = group['metric_key'].value_counts().head(100)  # Top 100 per timestamp
                for metric_name, _ in top_individual.items():
                    metric_data = group[group['metric_key'] == metric_name]
                    if not metric_data.empty:
                        # Create safe column name
                        safe_name = re.sub(r'[^0-9a-zA-Z_]', '_', metric_name)[:50]
                        row_data[f'top_metric_{safe_name}'] = metric_data['value'].mean()
                
                summary_data.append(row_data)
            
            wide_df = pd.DataFrame(summary_data).fillna(0)
            
            # Clean data types before saving to parquet
            logging.info("Cleaning data types for parquet compatibility...")
            
            # Convert object columns with mixed types to strings
            object_columns = df.select_dtypes(include=['object']).columns
            for col in object_columns:
                if col not in ['metric_name', 'metric_key']:  # Keep these as they are
                    df[col] = df[col].astype(str)
            
            # Save the complete long format for reference
            complete_file = self.work_dir / 'all_metrics_complete.parquet'
            try:
                df.to_parquet(complete_file, index=False)
                logging.info(f"✅ Complete dataset with ALL {unique_metrics} metrics saved to: {complete_file}")
            except Exception as e:
                logging.warning(f"Failed to save parquet format: {e}")
                logging.info("Falling back to CSV format...")
                complete_file_csv = self.work_dir / 'all_metrics_complete.csv'
                df.to_csv(complete_file_csv, index=False)
                logging.info(f"✅ Complete dataset saved as CSV: {complete_file_csv}")
            
            # Save metrics catalog for reference
            metrics_catalog = df.groupby('metric_key').agg({
                'value': ['count', 'mean', 'std', 'min', 'max'],
                'timestamp': ['min', 'max', 'nunique']
            }).round(4)
            metrics_catalog.columns = ['_'.join(col) for col in metrics_catalog.columns]
            catalog_file = self.work_dir / 'metrics_catalog.csv'
            metrics_catalog.to_csv(catalog_file, index=False)
            logging.info(f"✅ Metrics catalog saved to: {catalog_file}")
            
        else:
            # For smaller datasets, use standard wide format with all metrics
            logging.info(f"Using wide format with all {unique_metrics} metrics")
        
        # Pivot to wide format
        time_cols = ['hour', 'day_of_week', 'day_of_month', 'month', 
                    'is_weekend', 'is_business_hours']
        
        wide_df = df.pivot_table(
            index=['timestamp'] + time_cols,
            columns='metric_key',
            values='value',
            aggfunc='mean'
        ).reset_index()
        
        wide_df.columns.name = None
        
        # Save consolidated dataset
        self.consolidated_file.parent.mkdir(parents=True, exist_ok=True)
        wide_df.to_parquet(self.consolidated_file, index=False)
        
        logging.info(f"✅ Consolidated dataset saved: {wide_df.shape}")
        return True
    
    def categorize_metrics(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Categorize metrics based on their names."""
        categorized = {category: [] for category in self.metric_categories.keys()}
        categorized['other'] = []
        
        metric_columns = [col for col in df.columns 
                         if col not in ['timestamp', 'hour', 'day_of_week', 'day_of_month', 
                                       'month', 'is_weekend', 'is_business_hours']]
        
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
        
        return categorized
    
    def engineer_features(self) -> bool:
        """Engineer features from consolidated dataset."""
        logging.info("🔧 Engineering features...")
        
        if not self.consolidated_file.exists():
            logging.error("Consolidated dataset not found")
            return False
        
        # Load data
        df = pd.read_parquet(self.consolidated_file)
        logging.info(f"Loaded data: {df.shape}")
        
        # Categorize metrics
        categorized_metrics = self.categorize_metrics(df)
        
        # Original feature count
        original_features = len(df.columns)
        
        # 1. System Health Scores
        if not self.config.get('skip_health_scores', False):
            logging.info("Creating system health scores...")
            df = self._create_health_scores(df, categorized_metrics)
        
        # 2. Resource Utilization Features  
        if not self.config.get('skip_utilization', False):
            logging.info("Creating utilization features...")
            df = self._create_utilization_features(df, categorized_metrics)
        
        # 3. Performance Indicators
        if not self.config.get('skip_performance', False):
            logging.info("Creating performance indicators...")
            df = self._create_performance_features(df, categorized_metrics)
        
        # 4. Anomaly Detection Features
        if not self.config.get('skip_anomaly', False):
            logging.info("Creating anomaly detection features...")
            df = self._create_anomaly_features(df)
        
        # 5. Time Series Features
        if not self.config.get('skip_time_series', False):
            logging.info("Creating time series features...")
            df = self._create_time_series_features(df)
        
        # 6. Correlation Features
        if not self.config.get('skip_correlation', False):
            logging.info("Creating correlation features...")
            df = self._create_correlation_features(df, categorized_metrics)
        
        # 7. Advanced ML Features (PCA, clustering)
        if not self.config.get('skip_advanced', False):
            logging.info("Creating advanced ML features...")
            df = self._create_advanced_features(df)
        
        # Fill any remaining NaN values
        df = df.ffill().bfill().fillna(0)
        
        # Save engineered features
        self.engineered_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.engineered_file, index=False)
        
        # Save metadata
        self._save_feature_metadata(df, original_features)
        
        logging.info(f"✅ Feature engineering complete: {df.shape}")
        logging.info(f"Added {len(df.columns) - original_features} new features")
        
        return True
    
    def _create_health_scores(self, df: pd.DataFrame, categorized: Dict) -> pd.DataFrame:
        """Create system health scores."""
        df_copy = df.copy()
        
        # CPU Health Score
        cpu_metrics = [col for col in categorized.get('cpu', []) if col in df_copy.columns]
        if cpu_metrics:
            cpu_data = df_copy[cpu_metrics].fillna(0)
            cpu_normalized = cpu_data.copy()
            
            for col in cpu_metrics:
                if df_copy[col].std() > 0:
                    cpu_normalized[col] = (df_copy[col] - df_copy[col].min()) / (df_copy[col].max() - df_copy[col].min())
            
            df_copy['cpu_health_score'] = 1 - cpu_normalized.mean(axis=1)
        
        # Memory Health Score
        memory_metrics = [col for col in categorized.get('memory', []) if col in df_copy.columns]
        if memory_metrics:
            available_cols = [col for col in memory_metrics if 'available' in col.lower()]
            used_cols = [col for col in memory_metrics if col not in available_cols]
            
            if available_cols:
                available_data = df_copy[available_cols].fillna(0)
                df_copy['memory_health_score'] = available_data.mean(axis=1) / (available_data.mean(axis=1).max() + 1e-6)
            elif used_cols:
                used_data = df_copy[used_cols].fillna(0)
                used_normalized = used_data.copy()
                for col in used_cols:
                    if df_copy[col].std() > 0:
                        used_normalized[col] = (df_copy[col] - df_copy[col].min()) / (df_copy[col].max() - df_copy[col].min())
                df_copy['memory_health_score'] = 1 - used_normalized.mean(axis=1)
        
        # Overall system health
        health_scores = [col for col in df_copy.columns if 'health_score' in col]
        if health_scores:
            df_copy['overall_system_health'] = df_copy[health_scores].mean(axis=1)
        
        return df_copy
    
    def _create_utilization_features(self, df: pd.DataFrame, categorized: Dict) -> pd.DataFrame:
        """Create resource utilization features."""
        df_copy = df.copy()
        
        # CPU utilization patterns
        cpu_metrics = [col for col in categorized.get('cpu', []) if col in df_copy.columns]
        if cpu_metrics:
            cpu_data = df_copy[cpu_metrics].fillna(0)
            df_copy['cpu_utilization_variance'] = cpu_data.var(axis=1)
            df_copy['cpu_peak_avg_ratio'] = cpu_data.max(axis=1) / (cpu_data.mean(axis=1) + 1e-6)
        
        # Memory pressure
        memory_metrics = [col for col in categorized.get('memory', []) if col in df_copy.columns]
        if memory_metrics:
            mem_data = df_copy[memory_metrics].fillna(0)
            df_copy['memory_total_usage'] = mem_data.sum(axis=1)
            df_copy['memory_usage_variance'] = mem_data.var(axis=1)
        
        return df_copy
    
    def _create_performance_features(self, df: pd.DataFrame, categorized: Dict) -> pd.DataFrame:
        """Create performance indicators."""
        df_copy = df.copy()
        
        # HTTP performance
        http_metrics = [col for col in categorized.get('http', []) if col in df_copy.columns]
        if http_metrics:
            duration_cols = [col for col in http_metrics if 'duration' in col.lower()]
            if duration_cols:
                df_copy['avg_response_time'] = df_copy[duration_cols].mean(axis=1)
                df_copy['response_time_variability'] = df_copy[duration_cols].std(axis=1)
        
        # Kubernetes performance
        k8s_metrics = [col for col in categorized.get('kubernetes', []) if col in df_copy.columns]
        if k8s_metrics:
            pod_cols = [col for col in k8s_metrics if 'pod' in col.lower()]
            if pod_cols:
                df_copy['k8s_pod_activity'] = df_copy[pod_cols].sum(axis=1)
        
        return df_copy
    
    def _create_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create anomaly detection features."""
        df_copy = df.copy()
        
        # Get numeric columns for anomaly detection
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        metric_cols = [col for col in numeric_cols 
                      if col not in ['timestamp', 'hour', 'day_of_week', 'day_of_month', 
                                    'month', 'is_weekend', 'is_business_hours']]
        
        # Limit to avoid too many features
        for col in metric_cols[:20]:
            if df_copy[col].std() > 0:
                # Z-score
                df_copy[f'{col}_zscore'] = (df_copy[col] - df_copy[col].mean()) / df_copy[col].std()
                
                # Percentile rank
                df_copy[f'{col}_percentile'] = df_copy[col].rank(pct=True)
        
        return df_copy
    
    def _create_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time series features."""
        df_copy = df.copy()
        
        # Cyclical encoding
        if 'hour' in df_copy.columns:
            df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy['hour'] / 24)
            df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy['hour'] / 24)
        
        if 'day_of_week' in df_copy.columns:
            df_copy['day_sin'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7)
            df_copy['day_cos'] = np.cos(2 * np.pi * df_copy['day_of_week'] / 7)
        
        return df_copy
    
    def _create_correlation_features(self, df: pd.DataFrame, categorized: Dict) -> pd.DataFrame:
        """Create correlation features between metric groups."""
        df_copy = df.copy()
        
        # CPU-Memory correlation
        cpu_cols = [col for col in categorized.get('cpu', []) if col in df_copy.columns]
        memory_cols = [col for col in categorized.get('memory', []) if col in df_copy.columns]
        
        if cpu_cols and memory_cols and len(df_copy) > 10:
            cpu_sum = df_copy[cpu_cols].sum(axis=1)
            memory_sum = df_copy[memory_cols].sum(axis=1)
            
            window = min(50, len(df_copy) // 4)
            if window > 1:
                df_copy['cpu_memory_correlation'] = cpu_sum.rolling(window, min_periods=2).corr(memory_sum)
        
        return df_copy
    
    def _create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced ML features like PCA."""
        try:
            from sklearn.preprocessing import RobustScaler
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans
        except ImportError:
            logging.warning("scikit-learn not available, skipping advanced features")
            return df
        
        df_copy = df.copy()
        
        # Get numeric metric columns
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        metric_cols = [col for col in numeric_cols 
                      if col not in ['timestamp', 'hour', 'day_of_week', 'day_of_month', 
                                    'month', 'is_weekend', 'is_business_hours']]
        
        # Filter columns with variance
        metric_cols_filtered = [col for col in metric_cols if df_copy[col].std() > 0]
        
        if len(metric_cols_filtered) >= 3:
            # Prepare data
            metric_data = df_copy[metric_cols_filtered].fillna(0)
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(metric_data)
            
            # PCA
            try:
                n_components = min(5, len(metric_cols_filtered), len(df_copy) - 1)
                if n_components > 0:
                    pca = PCA(n_components=n_components)
                    pca_components = pca.fit_transform(scaled_data)
                    
                    for i in range(n_components):
                        df_copy[f'pca_component_{i+1}'] = pca_components[:, i]
            except Exception as e:
                logging.warning(f"PCA failed: {e}")
            
            # K-means clustering
            try:
                if len(df_copy) > 10:
                    n_clusters = min(5, len(df_copy) // 3)
                    if n_clusters > 1:
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(scaled_data)
                        df_copy['cluster_id'] = clusters
                        
                        # Distance to cluster center
                        distances = np.min(kmeans.transform(scaled_data), axis=1)
                        df_copy['cluster_anomaly_score'] = distances
            except Exception as e:
                logging.warning(f"Clustering failed: {e}")
        
        return df_copy
    
    def _save_feature_metadata(self, df: pd.DataFrame, original_features: int):
        """Save feature engineering metadata."""
        metadata_file = self.engineered_file.parent / "engineered_features_metadata.txt"
        
        with open(metadata_file, 'w') as f:
            f.write("Infrastructure Feature Engineering Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Original features: {original_features}\n")
            f.write(f"Engineered features: {len(df.columns)}\n")
            f.write(f"New features added: {len(df.columns) - original_features}\n")
            f.write(f"Total rows: {len(df)}\n")
            f.write(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}\n\n")
            
            # Categorize new features
            new_features = [col for col in df.columns if col not in ['timestamp', 'hour', 'day_of_week']]
            
            categories = {
                'Health Scores': [f for f in new_features if 'health' in f],
                'Utilization': [f for f in new_features if any(x in f for x in ['utilization', 'pressure', 'variance'])],
                'Performance': [f for f in new_features if any(x in f for x in ['response', 'activity', 'avg_'])],
                'Anomaly': [f for f in new_features if any(x in f for x in ['zscore', 'percentile', 'anomaly'])],
                'Time Series': [f for f in new_features if any(x in f for x in ['sin', 'cos'])],
                'Correlation': [f for f in new_features if 'correlation' in f],
                'Advanced ML': [f for f in new_features if any(x in f for x in ['pca', 'cluster'])]
            }
            
            for category, features in categories.items():
                if features:
                    f.write(f"{category}: {len(features)} features\n")
                    for feature in features[:5]:
                        f.write(f"  - {feature}\n")
                    if len(features) > 5:
                        f.write(f"  ... and {len(features) - 5} more\n")
                    f.write("\n")
        
        logging.info(f"Feature metadata saved to {metadata_file}")
    
    def analyze_final_dataset(self) -> bool:
        """Analyze the final engineered dataset."""
        logging.info("🔍 Analyzing final dataset...")
        
        if not self.engineered_file.exists():
            logging.error("Engineered features file not found")
            return False
        
        df = pd.read_parquet(self.engineered_file)
        
        # Analysis
        print(f"\n{'='*60}")
        print(f"FINAL DATASET ANALYSIS")
        print(f"{'='*60}")
        print(f"Shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Duration: {df['timestamp'].max() - df['timestamp'].min()}")
        
        # Feature categories
        feature_categories = {
            'Time': [col for col in df.columns if col in ['timestamp', 'hour', 'day_of_week', 'month']],
            'Health Scores': [col for col in df.columns if 'health' in col],
            'Utilization': [col for col in df.columns if any(x in col for x in ['utilization', 'pressure'])],
            'Performance': [col for col in df.columns if any(x in col for x in ['response', 'activity'])],
            'Anomaly': [col for col in df.columns if any(x in col for x in ['zscore', 'anomaly'])],
            'ML Features': [col for col in df.columns if any(x in col for x in ['pca', 'cluster'])],
            'Original Metrics': [col for col in df.columns if not any(x in col for x in 
                               ['health', 'utilization', 'pressure', 'response', 'activity', 
                                'zscore', 'anomaly', 'pca', 'cluster', 'timestamp', 'hour', 'day'])]
        }
        
        print(f"\n📊 FEATURE BREAKDOWN:")
        for category, features in feature_categories.items():
            if features:
                print(f"   {category}: {len(features)} features")
        
        # Missing values
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        
        print(f"\n❌ MISSING VALUES:")
        if len(missing_cols) > 0:
            print(f"   Columns with missing: {len(missing_cols)}")
            print(f"   Total missing: {missing_counts.sum()}")
        else:
            print("   ✅ No missing values!")
        
        # Save analysis
        analysis_file = self.engineered_file.parent / "final_analysis.txt"
        with open(analysis_file, 'w') as f:
            f.write("Final Dataset Analysis\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Shape: {df.shape}\n")
            f.write(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
            f.write(f"Missing values: {missing_counts.sum()}\n\n")
            
            f.write("Feature Categories:\n")
            for category, features in feature_categories.items():
                if features:
                    f.write(f"  {category}: {len(features)}\n")
        
        print(f"\n✅ Analysis complete! Summary saved to {analysis_file}")
        return True
    
    def cleanup_intermediate_files(self):
        """Clean up intermediate files."""
        if self.config.get('cleanup', False):
            logging.info("🧹 Cleaning up intermediate files...")
            
            files_to_remove = [
                self.consolidated_file,
                self.work_dir / "consolidated_dataset_summary.txt"
            ]
            
            for file in files_to_remove:
                if file.exists():
                    file.unlink()
                    logging.info(f"Removed {file.name}")
    
    def run_full_pipeline(self) -> bool:
        """Run the complete pipeline."""
        start_time = datetime.now()
        
        print("🏗️  Unified Infrastructure Monitoring Pipeline")
        print("=" * 60)
        print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Show configuration
        print("📋 Configuration:")
        print(f"   Prometheus URL: {self.config['prometheus_url']}")
        print(f"   Days to extract: {self.config['days']}")
        print(f"   Time resolution: {self.config['step']}")
        print(f"   Workers: {self.config['workers']}")
        print(f"   Quick mode: {self.config.get('quick_mode', False)}")
        print()
        
        success = True
        
        # Step 1: Extract data (unless skipped)
        if not self.config.get('skip_extraction', False):
            # Clean existing data unless keeping
            if not self.config.get('keep_existing', False):
                if self.prometheus_data_dir.exists():
                    logging.info("Cleaning existing data...")
                    shutil.rmtree(self.prometheus_data_dir)
                
                # Clean old outputs
                for file in [self.consolidated_file, self.engineered_file]:
                    if file.exists():
                        file.unlink()
            
            print("🚀 STEP 1: Extracting data from Prometheus")
            print("-" * 45)
            if not self.extract_prometheus_data():
                print("❌ Data extraction failed")
                return False
            print()
        else:
            print("⏭️  STEP 1: Skipping extraction (using existing data)")
            print()
        
        # Step 2: Consolidate data
        print("🔄 STEP 2: Consolidating data")
        print("-" * 30)
        if not self.consolidate_data():
            print("❌ Data consolidation failed")
            return False
        print()
        
        # Step 3: Engineer features
        print("🔧 STEP 3: Engineering features")
        print("-" * 30)
        if not self.engineer_features():
            print("❌ Feature engineering failed")
            return False
        print()
        
        # Step 4: Analyze results
        print("🔍 STEP 4: Analyzing results")
        print("-" * 25)
        if not self.analyze_final_dataset():
            print("❌ Analysis failed")
            return False
        print()
        
        # Step 5: Cleanup
        self.cleanup_intermediate_files()
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 45)
        print(f"Duration: {duration}")
        print()
        
        print("📁 Output files:")
        print(f"   • {self.engineered_file.name} - ML-ready dataset")
        print(f"   • engineered_features_metadata.txt - Feature descriptions")
        print(f"   • final_analysis.txt - Dataset analysis")
        print()
        
        print("🎯 Your data is ready for:")
        print("   • Machine Learning models")
        print("   • Anomaly detection")
        print("   • Performance monitoring")
        print("   • Capacity planning")
        print()
        
        return True


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Unified Infrastructure Monitoring Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python unified_pipeline.py                         # Full pipeline with defaults
  python unified_pipeline.py --test-connection       # Test Prometheus connection
  python unified_pipeline.py --quick                 # Quick mode (1 day, skip advanced)
  python unified_pipeline.py --skip-extraction       # Use existing data
  python unified_pipeline.py --days 7 --cleanup     # 7 days with cleanup
        """
    )
    
    # Prometheus settings
    parser.add_argument('--url', default='http://localhost:9090',
                      help='Prometheus server URL')
    parser.add_argument('--days', type=int, default=1,
                      help='Number of days to extract')
    parser.add_argument('--step', default='1m',
                      help='Query resolution step width')
    parser.add_argument('--workers', type=int, default=4,
                      help='Number of parallel workers')
    
    # Pipeline control
    parser.add_argument('--test-connection', action='store_true',
                      help='Only test Prometheus connection')
    parser.add_argument('--skip-extraction', action='store_true',
                      help='Skip extraction, use existing data')
    parser.add_argument('--keep-existing', action='store_true',
                      help='Keep existing data during extraction')
    parser.add_argument('--quick', action='store_true',
                      help='Quick mode: 1 day, skip advanced features')
    parser.add_argument('--cleanup', action='store_true',
                      help='Clean up intermediate files')
    
    # Feature engineering control
    parser.add_argument('--skip-health-scores', action='store_true',
                      help='Skip health score features')
    parser.add_argument('--skip-utilization', action='store_true',
                      help='Skip utilization features')
    parser.add_argument('--skip-performance', action='store_true',
                      help='Skip performance features')
    parser.add_argument('--skip-anomaly', action='store_true',
                      help='Skip anomaly detection features')
    parser.add_argument('--skip-time-series', action='store_true',
                      help='Skip time series features')
    parser.add_argument('--skip-correlation', action='store_true',
                      help='Skip correlation features')
    parser.add_argument('--skip-advanced', action='store_true',
                      help='Skip advanced ML features (PCA, clustering)')
    
    # Filters
    parser.add_argument('--metric-filter',
                      help='Regex filter for metric names')
    parser.add_argument('--batch-size', type=int, default=50,
                      help='Batch size for processing CSV files')
    
    args = parser.parse_args()
    
    # Handle quick mode
    if args.quick:
        args.days = 1
        args.skip_advanced = True
        args.skip_correlation = True
        args.cleanup = True
    
    # Build configuration
    config = {
        'prometheus_url': args.url,
        'days': args.days,
        'step': args.step,
        'workers': args.workers,
        'skip_extraction': args.skip_extraction,
        'keep_existing': args.keep_existing,
        'quick_mode': args.quick,
        'cleanup': args.cleanup,
        'skip_health_scores': args.skip_health_scores,
        'skip_utilization': args.skip_utilization,
        'skip_performance': args.skip_performance,
        'skip_anomaly': args.skip_anomaly,
        'skip_time_series': args.skip_time_series,
        'skip_correlation': args.skip_correlation,
        'skip_advanced': args.skip_advanced,
        'metric_filter': args.metric_filter,
        'batch_size': args.batch_size
    }
    
    # Create and run pipeline
    pipeline = UnifiedInfrastructurePipeline(config)
    
    # Test connection only
    if args.test_connection:
        success = pipeline.test_prometheus_connection()
        sys.exit(0 if success else 1)
    
    # Run full pipeline
    try:
        success = pipeline.run_full_pipeline()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
