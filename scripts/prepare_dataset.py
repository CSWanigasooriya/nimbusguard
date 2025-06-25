import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
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


class CSVConsolidator:
    """Efficiently consolidate individual metric CSV files into a unified dataset."""
    
    def __init__(self, input_dir: Path, output_path: Path):
        self.input_dir = Path(input_dir)
        self.output_path = Path(output_path)
        
    def get_csv_files(self) -> List[Path]:
        """Get all CSV files in the input directory."""
        csv_files = list(self.input_dir.glob("*.csv"))
        logging.info(f"Found {len(csv_files)} CSV files")
        return csv_files
    
    def analyze_file_structure(self, csv_files: List[Path], sample_size: int = 5) -> dict:
        """Analyze the structure of CSV files to understand the data."""
        logging.info("Analyzing file structure...")
        
        sample_files = csv_files[:min(sample_size, len(csv_files))]
        file_info = {}
        
        for file in sample_files:
            try:
                df = pd.read_csv(file, nrows=5)  # Just read first 5 rows
                file_info[file.name] = {
                    'columns': list(df.columns),
                    'shape_estimate': 'unknown',
                    'sample_data': df.head(2).to_dict()
                }
                
                # Try to get full row count efficiently
                with open(file, 'r') as f:
                    row_count = sum(1 for line in f) - 1  # Subtract header
                file_info[file.name]['shape_estimate'] = (row_count, len(df.columns))
                
            except Exception as e:
                logging.warning(f"Could not analyze {file.name}: {e}")
                file_info[file.name] = {'error': str(e)}
        
        # Print analysis
        for filename, info in file_info.items():
            if 'error' not in info:
                logging.info(f"{filename}: {info['shape_estimate']} - Columns: {info['columns']}")
        
        return file_info
    
    def consolidate_files(self, 
                         csv_files: List[Path], 
                         batch_size: int = 50,
                         filter_metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """Consolidate CSV files in batches to manage memory."""
        logging.info(f"Consolidating {len(csv_files)} files in batches of {batch_size}...")
        
        # Filter files if requested
        if filter_metrics:
            filtered_files = []
            for file in csv_files:
                metric_name = file.stem  # filename without extension
                if any(pattern in metric_name for pattern in filter_metrics):
                    filtered_files.append(file)
            csv_files = filtered_files
            logging.info(f"Filtered to {len(csv_files)} files matching criteria")
        
        all_dataframes = []
        
        # Process files in batches
        for i in range(0, len(csv_files), batch_size):
            batch_files = csv_files[i:i + batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}/{(len(csv_files)-1)//batch_size + 1}")
            
            batch_dfs = []
            for file in batch_files:
                try:
                    df = pd.read_csv(file)
                    
                    # Ensure metric_name column exists
                    if 'metric_name' not in df.columns:
                        # Use filename as metric name
                        metric_name = file.stem
                        df['metric_name'] = metric_name
                    
                    # Standardize column names
                    df.columns = df.columns.str.lower().str.strip()
                    
                    # Ensure required columns exist
                    if 'timestamp' not in df.columns:
                        logging.warning(f"No timestamp column in {file.name}")
                        continue
                    
                    if 'value' not in df.columns:
                        logging.warning(f"No value column in {file.name}")
                        continue
                    
                    batch_dfs.append(df)
                    
                except Exception as e:
                    logging.error(f"Error processing {file.name}: {e}")
                    continue
            
            if batch_dfs:
                # Filter out empty dataframes to avoid FutureWarning
                non_empty_dfs = [df for df in batch_dfs if not df.empty and len(df.dropna(how='all')) > 0]
                if non_empty_dfs:
                    batch_combined = pd.concat(non_empty_dfs, ignore_index=True)
                    all_dataframes.append(batch_combined)
        
        if not all_dataframes:
            raise ValueError("No valid CSV files could be processed")
        
        # Combine all batches
        logging.info("Combining all batches...")
        if all_dataframes:
            # Filter out any remaining empty dataframes
            final_dataframes = [df for df in all_dataframes if not df.empty]
            if final_dataframes:
                final_df = pd.concat(final_dataframes, ignore_index=True)
            else:
                raise ValueError("No valid data found after processing all batches")
        else:
            raise ValueError("No valid data found in any CSV files")
        
        return final_df
    
    def clean_and_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the consolidated dataset."""
        logging.info("Cleaning and standardizing data...")
        
        initial_rows = len(df)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Convert value to numeric
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Remove rows with invalid timestamps or values
        df = df.dropna(subset=['timestamp', 'value'])
        
        logging.info(f"Removed {initial_rows - len(df)} rows with invalid data")
        
        # Sort by metric and timestamp
        df = df.sort_values(['metric_name', 'timestamp'])
        
        # Remove exact duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        logging.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # Handle duplicate timestamps for same metric (take mean)
        duplicate_mask = df.duplicated(subset=['metric_name', 'timestamp'], keep=False)
        if duplicate_mask.any():
            logging.info(f"Found {duplicate_mask.sum()} rows with duplicate timestamps, averaging values")
            
            # Group by metric and timestamp, take mean of values and first of other columns
            agg_dict = {'value': 'mean'}
            other_cols = [col for col in df.columns if col not in ['metric_name', 'timestamp', 'value']]
            for col in other_cols:
                agg_dict[col] = 'first'
            
            df = df.groupby(['metric_name', 'timestamp']).agg(agg_dict).reset_index()
        
        return df
    
    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic time-based features."""
        logging.info("Adding basic features...")
        
        # Time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['is_business_hours'] = df['hour'].between(9, 17)
        
        return df
    
    def create_wide_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert to wide format with metrics as columns."""
        logging.info("Converting to wide format...")
        
        # Only use essential label columns for pivot (avoid explosion of columns)
        essential_labels = ['instance', 'job']  # Most important for distinguishing metrics
        label_columns = [col for col in df.columns 
                        if col in essential_labels and col in df.columns]
        
        # Create simplified metric identifiers
        if label_columns:
            # Only use non-null, meaningful labels
            df_clean = df.copy()
            
            # Clean up label values - replace NaN/None with empty string
            for col in label_columns:
                df_clean[col] = df_clean[col].fillna('').astype(str)
                df_clean[col] = df_clean[col].replace(['nan', 'None', 'none'], '')
            
            # Create cleaner metric keys
            df_clean['metric_key'] = df_clean['metric_name'].astype(str)
            
            for col in label_columns:
                # Only add label if it's not empty and adds meaningful distinction
                label_values = df_clean[col].str.strip()
                non_empty_labels = label_values[label_values != '']
                
                if len(non_empty_labels) > 0 and len(non_empty_labels.unique()) > 1:
                    df_clean['metric_key'] += '_' + label_values.replace('', 'default')
        else:
            df_clean = df.copy()
            df_clean['metric_key'] = df_clean['metric_name']
        
        # Limit the number of unique metrics to prevent column explosion
        unique_metrics = df_clean['metric_key'].nunique()
        logging.info(f"Found {unique_metrics} unique metric keys")
        
        if unique_metrics > 1000:
            logging.warning(f"Too many unique metrics ({unique_metrics}). Consider filtering or using long format.")
            # Keep only the most frequently occurring metrics
            metric_counts = df_clean['metric_key'].value_counts()
            top_metrics = metric_counts.head(1000).index
            df_clean = df_clean[df_clean['metric_key'].isin(top_metrics)]
            logging.info(f"Reduced to top 1000 most frequent metrics")
        
        # Pivot to wide format
        time_cols = ['hour', 'day_of_week', 'day_of_month', 'month', 
                    'is_weekend', 'is_business_hours']
        
        wide_df = df_clean.pivot_table(
            index=['timestamp'] + time_cols,
            columns='metric_key',
            values='value',
            aggfunc='mean'  # Handle any remaining duplicates
        ).reset_index()
        
        # Flatten column names
        wide_df.columns.name = None
        
        logging.info(f"Wide format shape: {wide_df.shape}")
        return wide_df
    
    def save_dataset(self, df: pd.DataFrame, format: str = 'parquet'):
        """Save the consolidated dataset."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'parquet':
            output_file = self.output_path.with_suffix('.parquet')
            df.to_parquet(output_file, index=False)
        elif format == 'csv':
            output_file = self.output_path.with_suffix('.csv')
            df.to_csv(output_file, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logging.info(f"Dataset saved to {output_file}")
        
        # Save summary
        summary_file = self.output_path.parent / f"{self.output_path.name}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("CSV Consolidation Summary\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Final dataset shape: {df.shape}\n")
            f.write(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
            
            # Count metric columns more safely
            metric_cols = [col for col in df.columns if col not in ['timestamp', 'hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend', 'is_business_hours']]
            f.write(f"Number of unique metrics: {len(metric_cols)}\n\n")
            
            f.write("Sample metrics:\n")
            for metric in metric_cols[:20]:
                f.write(f"  - {metric}\n")
            if len(metric_cols) > 20:
                f.write(f"  ... and {len(metric_cols) - 20} more metrics\n")
        
        logging.info(f"Summary saved to {summary_file}")
    
    def consolidate(self, 
                   batch_size: int = 50,
                   format: str = 'parquet',
                   wide_format: bool = True,
                   filter_metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """Full consolidation pipeline."""
        
        # Get CSV files
        csv_files = self.get_csv_files()
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.input_dir}")
        
        # Analyze structure
        self.analyze_file_structure(csv_files)
        
        # Consolidate files
        df = self.consolidate_files(csv_files, batch_size, filter_metrics)
        
        # Clean and standardize
        df = self.clean_and_standardize(df)
        
        # Add basic features
        df = self.add_basic_features(df)
        
        # Convert to wide format if requested
        if wide_format:
            df = self.create_wide_format(df)
        
        # Save dataset
        self.save_dataset(df, format)
        
        return df


def main():
    """Main function to consolidate CSV files."""
    parser = argparse.ArgumentParser(
        description="Consolidate individual metric CSV files into a unified dataset."
    )
    parser.add_argument(
        "input_dir",
        nargs='?',  # Make optional
        default="prometheus_data",
        help="Directory containing individual metric CSV files (default: prometheus_data)"
    )
    parser.add_argument(
        "output_path",
        nargs='?',  # Make optional
        default="consolidated_dataset",
        help="Path for output consolidated dataset (default: consolidated_dataset)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of CSV files to process in each batch (default: 50)"
    )
    parser.add_argument(
        "--output-format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output file format (default: parquet)"
    )
    parser.add_argument(
        "--long-format",
        action="store_true",
        help="Keep long format instead of pivoting to wide format"
    )
    parser.add_argument(
        "--filter-metrics",
        nargs="+",
        help="Only include metrics containing these patterns (e.g., 'node_cpu' 'kube_pod')"
    )
    
    args = parser.parse_args()
    
    # Show what we're doing
    print(f"🔄 Consolidating CSV files...")
    print(f"   Input: {args.input_dir}")
    print(f"   Output: {args.output_path}.{args.output_format}")
    print(f"   Format: {'Long' if args.long_format else 'Wide'} format")
    if args.filter_metrics:
        print(f"   Filter: {', '.join(args.filter_metrics)}")
    print()
    
    try:
        consolidator = CSVConsolidator(args.input_dir, args.output_path)
        
        df = consolidator.consolidate(
            batch_size=args.batch_size,
            format=args.output_format,
            wide_format=not args.long_format,
            filter_metrics=args.filter_metrics
        )
        
        logging.info("CSV consolidation completed successfully!")
        print(f"\n✅ Success! Dataset saved to {args.output_path}.{args.output_format}")
        print(f"   Summary: {args.output_path}_summary.txt")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()