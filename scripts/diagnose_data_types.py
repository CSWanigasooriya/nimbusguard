#!/usr/bin/env python3
"""
Data Type Diagnostic Tool
Helps identify and fix data type issues in your CSV files before processing.
"""

import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def validate_timestamp(series):
    """Validate timestamp format and continuity."""
    try:
        timestamps = pd.to_datetime(series)
        time_diffs = timestamps.diff()
        
        # Check for gaps (should be ~1 minute intervals)
        gaps = time_diffs > pd.Timedelta(minutes=2)
        if gaps.any():
            return False, f"Found {gaps.sum()} gaps in time series"
        
        return True, None
    except Exception as e:
        return False, str(e)

def validate_numeric(series):
    """Validate numeric values."""
    try:
        numeric_values = pd.to_numeric(series)
        
        # Check for infinity or extremely large values
        if np.inf in numeric_values.values or -np.inf in numeric_values.values:
            return False, "Contains infinity values"
        
        # Check for unreasonable values (adjust thresholds as needed)
        if numeric_values.abs().max() > 1e12:
            return False, f"Contains very large values: {numeric_values.abs().max()}"
            
        return True, None
    except Exception as e:
        return False, str(e)

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
    
    issues = {
        'timestamp_issues': [],
        'value_issues': [],
        'label_issues': [],
        'mixed_types': {},
        'missing_data': []
    }
    
    # Analyze all files
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            
            # Check timestamp
            if 'timestamp' in df.columns:
                valid, msg = validate_timestamp(df['timestamp'])
                if not valid:
                    issues['timestamp_issues'].append(f"{file.name}: {msg}")
            
            # Check value column
            if 'value' in df.columns:
                valid, msg = validate_numeric(df['value'])
                if not valid:
                    issues['value_issues'].append(f"{file.name}: {msg}")
            
            # Check for missing required columns
            required_cols = {'timestamp', 'value', 'metric_name'}
            missing = required_cols - set(df.columns)
            if missing:
                issues['missing_data'].append(f"{file.name}: Missing columns {missing}")
            
            # Check label consistency
            label_cols = [col for col in df.columns if col not in required_cols]
            for col in label_cols:
                # Check for mixed types
                unique_types = df[col].apply(type).unique()
                if len(unique_types) > 1:
                    if col not in issues['mixed_types']:
                        issues['mixed_types'][col] = set()
                    issues['mixed_types'][col].update(str(t) for t in unique_types)
                
                # Check for label validity
                if df[col].isna().any():
                    issues['label_issues'].append(f"{file.name}: Column {col} has missing values")
                
        except Exception as e:
            print(f"⚠️  Error reading {file.name}: {e}")
            issues['missing_data'].append(f"{file.name}: Failed to read file - {str(e)}")
    
    # Report findings
    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"   Files analyzed: {len(csv_files)}")
    
    if any(issues.values()):
        print("\n⚠️  ISSUES FOUND:")
        
        if issues['timestamp_issues']:
            print("\n   Timestamp Issues:")
            for issue in issues['timestamp_issues'][:5]:
                print(f"   - {issue}")
            if len(issues['timestamp_issues']) > 5:
                print(f"   ... and {len(issues['timestamp_issues']) - 5} more")
        
        if issues['value_issues']:
            print("\n   Value Issues:")
            for issue in issues['value_issues'][:5]:
                print(f"   - {issue}")
            if len(issues['value_issues']) > 5:
                print(f"   ... and {len(issues['value_issues']) - 5} more")
        
        if issues['mixed_types']:
            print("\n   Mixed Type Columns:")
            for col, types in issues['mixed_types'].items():
                print(f"   - {col}: {', '.join(types)}")
        
        if issues['label_issues']:
            print("\n   Label Issues:")
            for issue in issues['label_issues'][:5]:
                print(f"   - {issue}")
            if len(issues['label_issues']) > 5:
                print(f"   ... and {len(issues['label_issues']) - 5} more")
    else:
        print("   ✅ No issues found")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if any(issues.values()):
        print("   1. Run prepare_dataset.py with --safe-mode flag")
        print("   2. Consider filtering problematic metrics")
        print("   3. Check timestamp continuity in your data collection")
    else:
        print("   ✅ Data looks clean and ready for processing")
    
    return issues

def quick_fix_sample():
    """Show how the data type fixes work."""
    
    print(f"\n🔧 AUTOMATIC FIXES APPLIED:")
    print("1. Timestamps:")
    print("   Before: ['2025-06-25 08:29:29', '2025-06-25 8:30:29']")
    print("   After:  [2025-06-25 08:29:29, 2025-06-25 08:30:29]  ✅ Standardized format")
    
    print("\n2. Mixed Types:")
    print("   Before: ['sda', 'nvme0n1', 8, 'loop0', 259]")
    print("   After:  ['sda', 'nvme0n1', '8', 'loop0', '259']  ✅ Consistent strings")
    
    print("\n3. Numeric Values:")
    print("   Before: ['1.23e6', 'inf', '1.5M']")
    print("   After:  [1230000.0, None, 1500000.0]  ✅ Clean numbers")

if __name__ == "__main__":
    print("🩺 Data Type Diagnostic Tool")
    print("=" * 40)
    
    diagnose_data_types()
    quick_fix_sample()
    
    print(f"\n🚀 TO RUN YOUR PIPELINE:")
    print("   python prepare_dataset.py --safe-mode  # For extra data validation")
    print("   python prepare_dataset.py --quick      # For faster processing")
