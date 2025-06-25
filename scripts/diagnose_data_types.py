#!/usr/bin/env python3
"""
Data Type Diagnostic Tool
Helps identify and fix data type issues in your CSV files before processing.
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def diagnose_data_types():
    """Diagnose data type issues in prometheus data."""
    
    prometheus_data_dir = Path('prometheus_data')
    
    if not prometheus_data_dir.exists():
        print("❌ prometheus_data directory not found!")
        return
    
    csv_files = list(prometheus_data_dir.glob("*.csv"))
    if not csv_files:
        print("❌ No CSV files found in prometheus_data/")
        return
    
    print(f"🔍 Analyzing {len(csv_files)} CSV files for data type issues...")
    
    problematic_files = []
    mixed_type_columns = {}
    
    # Sample a few files to check for issues
    sample_files = csv_files[:20]  # Check first 20 files
    
    for file in sample_files:
        try:
            df = pd.read_csv(file)
            
            # Check for mixed types in object columns
            object_cols = df.select_dtypes(include=['object']).columns
            
            for col in object_cols:
                if col in ['timestamp', 'metric_name']:
                    continue
                    
                # Check if column has mixed types
                unique_types = set(type(x).__name__ for x in df[col].dropna().iloc[:100])
                if len(unique_types) > 1:
                    if col not in mixed_type_columns:
                        mixed_type_columns[col] = set()
                    mixed_type_columns[col].update(unique_types)
                    problematic_files.append(file.name)
                    
        except Exception as e:
            print(f"⚠️  Error reading {file.name}: {e}")
            problematic_files.append(file.name)
    
    # Report findings
    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"   Files analyzed: {len(sample_files)}")
    print(f"   Problematic files: {len(set(problematic_files))}")
    
    if mixed_type_columns:
        print(f"\n⚠️  MIXED TYPE COLUMNS FOUND:")
        for col, types in mixed_type_columns.items():
            print(f"   {col}: {', '.join(types)}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if mixed_type_columns:
        print("   1. The pipeline now handles mixed types automatically")
        print("   2. Run with the updated run_pipeline.py")
        print("   3. Mixed type columns will be converted to strings")
    else:
        print("   ✅ No obvious data type issues found")
    
    return mixed_type_columns

def quick_fix_sample():
    """Show how the data type fix works."""
    
    print(f"\n🔧 HOW THE FIX WORKS:")
    print("Before fix:")
    print("  device column: ['sda', 'nvme0n1', 8, 'loop0', 259]  ❌ Mixed types")
    print()
    print("After fix:")
    print("  device column: ['sda', 'nvme0n1', '8', 'loop0', '259']  ✅ All strings")
    print()
    print("This prevents the PyArrow parquet conversion error!")

if __name__ == "__main__":
    print("🩺 Data Type Diagnostic Tool")
    print("=" * 40)
    
    diagnose_data_types()
    quick_fix_sample()
    
    print(f"\n🚀 TO RUN YOUR FIXED PIPELINE:")
    print("   python run_pipeline.py --skip-extraction")
    print("   python run_pipeline.py --quick")
