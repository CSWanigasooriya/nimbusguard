import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def inspect_dataset(file_path: Path, output_summary: bool = True):
    """Inspect and analyze a dataset."""
    logging.info(f"Inspecting dataset: {file_path}")
    
    try:
        # Read the dataset
        if file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Basic info
        print(f"\n{'='*60}")
        print(f"DATASET ANALYSIS: {file_path.name}")
        print(f"{'='*60}")
        
        print(f"\n📊 BASIC INFORMATION:")
        print(f"   Shape: {df.shape}")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"   File size: {file_path.stat().st_size / 1024**2:.2f} MB")
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"   Duration: {df['timestamp'].max() - df['timestamp'].min()}")
        
        # Column analysis
        print(f"\n📈 COLUMN ANALYSIS:")
        print(f"   Total columns: {len(df.columns)}")
        
        # Categorize columns
        time_cols = [col for col in df.columns if col in ['timestamp', 'hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend', 'is_business_hours']]
        feature_cols = [col for col in df.columns if any(x in col for x in ['health', 'utilization', 'pressure', 'efficiency', 'zscore', 'anomaly', 'correlation', 'pca', 'cluster'])]
        metric_cols = [col for col in df.columns if col not in time_cols and col not in feature_cols]
        
        print(f"   Time columns: {len(time_cols)}")
        print(f"   Engineered features: {len(feature_cols)}")
        print(f"   Original metrics: {len(metric_cols)}")
        
        # Data types
        print(f"\n📋 DATA TYPES:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   {dtype}: {count} columns")
        
        # Missing values
        print(f"\n❌ MISSING VALUES:")
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        if len(missing_cols) > 0:
            print(f"   Columns with missing values: {len(missing_cols)}")
            print(f"   Total missing values: {missing_counts.sum()}")
            print(f"   Percentage missing: {(missing_counts.sum() / df.size) * 100:.2f}%")
            if len(missing_cols) <= 10:
                for col, count in missing_cols.items():
                    print(f"     {col}: {count} ({(count/len(df))*100:.1f}%)")
        else:
            print("   No missing values found! ✅")
        
        # Numeric statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\n📊 NUMERIC STATISTICS:")
            print(f"   Numeric columns: {len(numeric_cols)}")
            
            # Basic stats for first few numeric columns
            sample_cols = numeric_cols[:5]
            print(f"\n   Sample statistics (first 5 numeric columns):")
            print(df[sample_cols].describe())
        
        # Feature categories (if engineered features exist)
        if feature_cols:
            print(f"\n🔧 ENGINEERED FEATURES:")
            
            categories = {
                'Health Scores': [f for f in feature_cols if 'health' in f],
                'Utilization': [f for f in feature_cols if any(x in f for x in ['utilization', 'pressure', 'efficiency'])],
                'Anomaly Detection': [f for f in feature_cols if any(x in f for x in ['zscore', 'anomaly', 'deviation', 'percentile'])],
                'Correlation': [f for f in feature_cols if 'correlation' in f],
                'Time Series': [f for f in feature_cols if any(x in f for x in ['trend', 'seasonal', 'sin', 'cos'])],
                'Dimensionality Reduction': [f for f in feature_cols if any(x in f for x in ['pca', 'cluster'])]
            }
            
            for category, features in categories.items():
                if features:
                    print(f"   {category}: {len(features)} features")
                    for feature in features[:3]:  # Show first 3
                        print(f"     - {feature}")
                    if len(features) > 3:
                        print(f"     ... and {len(features) - 3} more")
        
        # Metric categories (if original metrics exist)
        if metric_cols:
            print(f"\n📏 ORIGINAL METRICS:")
            
            metric_categories = {
                'Node System': [col for col in metric_cols if col.startswith('node_')],
                'Kubernetes': [col for col in metric_cols if col.startswith('kube_')],
                'Prometheus': [col for col in metric_cols if col.startswith('prometheus_')],
                'HTTP': [col for col in metric_cols if col.startswith('http_')],
                'Go Runtime': [col for col in metric_cols if col.startswith('go_')],
                'Alloy': [col for col in metric_cols if col.startswith('alloy_')],
                'Loki': [col for col in metric_cols if col.startswith('loki_')],
                'OpenTelemetry': [col for col in metric_cols if col.startswith('otel')],
                'Other': [col for col in metric_cols if not any(col.startswith(prefix) for prefix in ['node_', 'kube_', 'prometheus_', 'http_', 'go_', 'alloy_', 'loki_', 'otel'])]
            }
            
            for category, metrics in metric_categories.items():
                if metrics:
                    print(f"   {category}: {len(metrics)} metrics")
        
        # Sample data
        print(f"\n🔍 SAMPLE DATA:")
        print("   First 3 rows:")
        sample_cols = df.columns[:10] if len(df.columns) > 10 else df.columns
        print(df[sample_cols].head(3))
        
        if len(df.columns) > 10:
            print(f"   ... showing {len(sample_cols)} of {len(df.columns)} columns")
        
        # Save summary if requested
        if output_summary:
            summary_file = file_path.parent / f"{file_path.stem}_inspection_summary.txt"
            
            with open(summary_file, 'w') as f:
                f.write(f"Dataset Inspection Summary\n")
                f.write(f"="*50 + "\n\n")
                f.write(f"File: {file_path}\n")
                f.write(f"Generated: {pd.Timestamp.now()}\n\n")
                
                f.write(f"Basic Information:\n")
                f.write(f"  Shape: {df.shape}\n")
                f.write(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
                f.write(f"  File size: {file_path.stat().st_size / 1024**2:.2f} MB\n")
                
                if 'timestamp' in df.columns:
                    f.write(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
                
                f.write(f"\nColumn Breakdown:\n")
                f.write(f"  Total columns: {len(df.columns)}\n")
                f.write(f"  Time columns: {len(time_cols)}\n")
                f.write(f"  Engineered features: {len(feature_cols)}\n")
                f.write(f"  Original metrics: {len(metric_cols)}\n")
                
                f.write(f"\nData Types:\n")
                for dtype, count in dtype_counts.items():
                    f.write(f"  {dtype}: {count} columns\n")
                
                f.write(f"\nMissing Values:\n")
                if len(missing_cols) > 0:
                    f.write(f"  Columns with missing values: {len(missing_cols)}\n")
                    f.write(f"  Total missing: {missing_counts.sum()} ({(missing_counts.sum() / df.size) * 100:.2f}%)\n")
                else:
                    f.write(f"  No missing values found\n")
                
                f.write(f"\nAll Columns:\n")
                for col in df.columns:
                    f.write(f"  - {col}\n")
            
            logging.info(f"Summary saved to {summary_file}")
        
        print(f"\n✅ Dataset inspection complete!")
        
    except Exception as e:
        logging.error(f"Error inspecting dataset: {e}")
        raise


def main():
    """Main function to parse arguments and inspect dataset."""
    parser = argparse.ArgumentParser(
        description="Inspect and analyze a dataset (CSV or Parquet)."
    )
    parser.add_argument(
        "input_file",
        nargs='?',  # Make optional
        default="engineered_features.parquet",
        help="Path to the dataset file (default: engineered_features.parquet)"
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Don't save a summary file"
    )
    
    args = parser.parse_args()
    
    # Show what we're doing
    print(f"🔍 Inspecting dataset: {args.input_file}")
    print()
    
    try:
        file_path = Path(args.input_file)
        if not file_path.exists():
            logging.error(f"File not found: {file_path}")
            sys.exit(1)
        
        inspect_dataset(file_path, output_summary=not args.no_summary)
        
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()