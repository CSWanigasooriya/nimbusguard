"""
Consolidate individual metric CSV files into a unified dataset.
Enhanced version with better handling of Prometheus data specifics.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import glob

import pandas as pd
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

class PrometheusMetricProcessor:
    """Process Prometheus metrics with proper handling of labels and instances."""
    
    def __init__(self, safe_mode: bool = False):
        self.safe_mode = safe_mode
        
    def clean_instance_label(self, instance: str) -> str:
        """Clean instance label to extract meaningful identifier."""
        if pd.isna(instance):
            return 'unknown'
        instance = str(instance)
        # Extract pod number or last part of hostname
        if ':' in instance:
            instance = instance.split(':')[0]
        return instance.split('.')[-1]
    
    def standardize_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize label columns and values."""
        # Handle known label columns
        if 'instance' in df.columns:
            df['instance'] = df['instance'].apply(self.clean_instance_label)
        
        # Convert all label values to strings
        label_cols = [col for col in df.columns if col not in ['timestamp', 'value', 'metric_name']]
        for col in label_cols:
            df[col] = df[col].fillna('unknown').astype(str)
            
        return df
    
    def create_metric_key(self, row: pd.Series) -> str:
        """Create a clean metric key from metric name and essential labels."""
        key = str(row['metric_name'])
        
        # Add instance identifier if available
        if 'instance' in row and row['instance'] != 'unknown':
            key += f"_i{row['instance']}"
            
        # Add job identifier if available and meaningful
        if 'job' in row and row['job'] != 'unknown':
            job = str(row['job']).split('.')[-1]  # Take last part of job name
            key += f"_{job}"
            
        return key
    
    def aggregate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate metrics with proper handling of instances."""
        # Group by timestamp and metric key
        df['metric_key'] = df.apply(self.create_metric_key, axis=1)
        
        # Calculate aggregations
        agg_df = df.groupby(['timestamp', 'metric_key']).agg({
            'value': ['mean', 'min', 'max', 'count']
        }).reset_index()
        
        # Flatten column names
        agg_df.columns = [
            col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
            for col in agg_df.columns
        ]
        
        return agg_df

class CSVConsolidator:
    """Efficiently consolidate individual metric CSV files into a unified dataset."""
    
    def __init__(self, input_dir: Path, output_path: Path, safe_mode: bool = False):
        self.input_dir = Path(input_dir)
        self.output_path = Path(output_path)
        self.safe_mode = safe_mode
        self.metric_processor = PrometheusMetricProcessor(safe_mode)
        
    def get_csv_files(self) -> List[Path]:
        """Get all CSV files in the input directory."""
        csv_files = list(self.input_dir.glob("*.csv"))
        logging.info(f"Found {len(csv_files)} CSV files")
        return csv_files
    
    def analyze_file_structure(self, csv_files: List[Path]) -> dict:
        """Analyze the structure of CSV files to understand the data."""
        logging.info("Analyzing file structure...")
        
        file_info = {}
        for file in csv_files:
            try:
                df = pd.read_csv(file, nrows=5)
                file_info[file.name] = {
                    'columns': list(df.columns),
                    'metric_name': file.stem
                }
                
                # Get row count efficiently
                with open(file, 'r') as f:
                    row_count = sum(1 for line in f) - 1
                file_info[file.name]['row_count'] = row_count
                
            except Exception as e:
                logging.warning(f"Could not analyze {file.name}: {e}")
                
        return file_info
    
    def validate_dataframe(self, df: pd.DataFrame, source_file: str) -> bool:
        """Validate dataframe structure and content."""
        if self.safe_mode:
            try:
                # Check required columns
                required_cols = {'timestamp', 'value'}
                missing_cols = required_cols - set(df.columns)
                if missing_cols:
                    logging.error(f"Missing required columns in {source_file}: {missing_cols}")
                    return False
                
                # Check for empty DataFrame
                if df.empty:
                    logging.warning(f"Empty DataFrame from {source_file}")
                    return False
                
                # Check column types
                if not pd.api.types.is_numeric_dtype(df['value']):
                    logging.warning(f"Non-numeric values in 'value' column in {source_file}")
                    try:
                        df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    except:
                        return False
                
                # Remove infinite values
                inf_mask = np.isinf(df['value'])
                if inf_mask.any():
                    logging.warning(f"Found {inf_mask.sum()} infinite values in {source_file}")
                    df = df[~inf_mask]
                
                # Check for reasonable value range
                abs_values = df['value'].abs()
                if abs_values.max() > 1e12:
                    logging.warning(f"Very large values in {source_file}: max abs value = {abs_values.max()}")
                
                # Check for too many unique categories in label columns
                label_cols = [col for col in df.columns if col not in ['timestamp', 'value', 'metric_name']]
                for col in label_cols:
                    unique_count = df[col].nunique()
                    if unique_count > 1000:
                        logging.warning(f"High cardinality in column '{col}' ({unique_count} unique values)")
                
                return len(df) > 0  # Return True only if we have valid rows
                
            except Exception as e:
                logging.error(f"Validation error in {source_file}: {e}")
                return False
        return True
    
    def consolidate_files(self, 
                         csv_files: List[Path], 
                         batch_size: int = 50,
                         filter_metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """Consolidate CSV files with proper validation and error handling."""
        if filter_metrics:
            csv_files = [f for f in csv_files if any(pattern in f.stem for pattern in filter_metrics)]
            logging.info(f"Filtered to {len(csv_files)} files matching criteria")
        
        all_dataframes = []
        
        for i in range(0, len(csv_files), batch_size):
            batch_files = csv_files[i:i + batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}/{(len(csv_files)-1)//batch_size + 1}")
            
            batch_dfs = []
            for file in batch_files:
                try:
                    # Read CSV with proper data type handling
                    df = pd.read_csv(
                        file,
                        low_memory=False,
                        dtype={
                            'timestamp': str,  # Read timestamp as string first
                            'value': float,    # Values should be numeric
                            'metric_name': str,
                            'instance': str,
                            'job': str
                        }
                    )
                    
                    if not self.validate_dataframe(df, file.name):
                        continue
                    
                    # Add metric name if needed
                    if 'metric_name' not in df.columns:
                        df['metric_name'] = file.stem
                    
                    # Convert timestamp to datetime safely
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    except Exception as e:
                        logging.error(f"Failed to convert timestamps in {file.name}: {e}")
                        continue
                    
                    # Standardize labels
                    df = self.metric_processor.standardize_labels(df)
                    
                    # Drop any rows with invalid timestamps
                    df = df.dropna(subset=['timestamp'])
                    
                    if not df.empty:
                        batch_dfs.append(df)
                    
                except Exception as e:
                    logging.error(f"Error processing {file.name}: {e}")
                    continue
            
            if batch_dfs:
                try:
                    # Ensure all DataFrames have compatible dtypes before concat
                    common_cols = set.intersection(*[set(df.columns) for df in batch_dfs])
                    for col in common_cols:
                        dtype = None
                        if col == 'timestamp':
                            dtype = 'datetime64[ns]'
                        elif col == 'value':
                            dtype = float
                        elif col in ['metric_name', 'instance', 'job']:
                            dtype = str
                        
                        if dtype:
                            batch_dfs = [df.astype({col: dtype}, errors='ignore') for df in batch_dfs]
                    
                    batch_df = pd.concat(batch_dfs, ignore_index=True)
                    batch_df = self.metric_processor.aggregate_metrics(batch_df)
                    all_dataframes.append(batch_df)
                except Exception as e:
                    logging.error(f"Failed to process batch: {e}")
                    continue
        
        if not all_dataframes:
            raise ValueError("No valid data could be processed")
        
        try:
            # Final concatenation with explicit dtypes
            final_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Ensure timestamp is datetime
            final_df['timestamp'] = pd.to_datetime(final_df['timestamp'])
            
            # Sort by timestamp
            final_df = final_df.sort_values('timestamp')
            
            return final_df
            
        except Exception as e:
            logging.error(f"Failed to combine all batches: {e}")
            raise
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive time-based features."""
        # Basic time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Cyclical encoding of time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, format: str = 'parquet'):
        """Save the dataset with proper error handling."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == 'parquet':
                output_file = self.output_path.with_suffix('.parquet')
                df.to_parquet(output_file, index=False)
            else:
                output_file = self.output_path.with_suffix('.csv')
                df.to_csv(output_file, index=False)
            
            # Save summary
            self.save_summary(df, output_file)
            
        except Exception as e:
            logging.error(f"Error saving dataset: {e}")
            raise
    
    def save_summary(self, df: pd.DataFrame, output_file: Path):
        """Save detailed summary of the dataset."""
        summary_file = output_file.parent / f"{output_file.stem}_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("Dataset Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Time Range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
            f.write(f"Total Samples: {len(df):,}\n")
            f.write(f"Unique Metrics: {df['metric_key'].nunique():,}\n\n")
            
            f.write("Sample Metrics:\n")
            for metric in sorted(df['metric_key'].unique())[:20]:
                metric_stats = df[df['metric_key'] == metric]['value_mean'].describe()
                f.write(f"  {metric}:\n")
                f.write(f"    Mean: {metric_stats['mean']:.2f}\n")
                f.write(f"    Min: {metric_stats['min']:.2f}\n")
                f.write(f"    Max: {metric_stats['max']:.2f}\n")
            
            if df['metric_key'].nunique() > 20:
                f.write(f"\n... and {df['metric_key'].nunique() - 20} more metrics\n")
    
    def consolidate(self,
                   batch_size: int = 50,
                   format: str = 'parquet',
                   filter_metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """Full consolidation pipeline with proper error handling."""
        try:
            csv_files = self.get_csv_files()
            if not csv_files:
                raise ValueError(f"No CSV files found in {self.input_dir}")
            
            file_info = self.analyze_file_structure(csv_files)
            df = self.consolidate_files(csv_files, batch_size, filter_metrics)
            df = self.add_time_features(df)
            
            self.save_dataset(df, format)
            return df
            
        except Exception as e:
            logging.error(f"Consolidation failed: {e}")
            raise

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Consolidate individual metric CSV files into a unified dataset"
    )
    
    parser.add_argument('--batch-size', type=int, default=50,
                      help='Batch size for processing CSV files')
    parser.add_argument('--format', choices=['parquet', 'csv'], default='parquet',
                      help='Output format')
    parser.add_argument('--filter-metrics', nargs='+',
                      help='List of metric patterns to include')
    parser.add_argument('--safe-mode', action='store_true',
                      help='Enable additional data validation')
    
    args = parser.parse_args()
    
    try:
        # Set up paths - get script directory automatically
        scripts_dir = Path(__file__).parent.resolve()
        input_dir = scripts_dir / 'prometheus_data'
        output_dir = scripts_dir / 'processed_data'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize consolidator
        consolidator = CSVConsolidator(
            input_dir=input_dir,
            output_path=output_dir / 'dataset',
            safe_mode=args.safe_mode
        )
        
        # Run consolidation
        df = consolidator.consolidate(
            batch_size=args.batch_size,
            format=args.format,
            filter_metrics=args.filter_metrics
        )
        
        logging.info("✅ Dataset consolidation complete!")
        return 0
        
    except Exception as e:
        logging.error(f"Failed to consolidate dataset: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())