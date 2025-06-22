#!/usr/bin/env python3
"""
Test script to verify MinIO dataset loading
"""

import os
import sys
from pathlib import Path

# Add the src directory to path
sys.path.append('/Users/chamathwanigasooriya/Documents/FYP/nimbusguard/src')

try:
    import boto3
    from botocore.exceptions import ClientError
    print("✅ boto3 imported successfully")
except ImportError:
    print("❌ boto3 not available")
    sys.exit(1)

def test_minio_connection():
    """Test MinIO connection"""
    try:
        # MinIO configuration
        s3_client = boto3.client(
            's3',
            endpoint_url="http://localhost:9000",
            aws_access_key_id="nimbusguard",
            aws_secret_access_key="nimbusguard123",
            region_name='us-east-1'
        )
        
        # List buckets
        response = s3_client.list_buckets()
        print(f"✅ Connected to MinIO. Buckets: {[b['Name'] for b in response['Buckets']]}")
        
        # Check datasets bucket
        try:
            response = s3_client.list_objects_v2(Bucket='datasets')
            if 'Contents' in response:
                print(f"✅ Found {len(response['Contents'])} objects in datasets bucket:")
                for obj in response['Contents']:
                    print(f"  - {obj['Key']} ({obj['Size']} bytes)")
            else:
                print("❌ No objects found in datasets bucket")
                return False
        except ClientError as e:
            print(f"❌ Error accessing datasets bucket: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ MinIO connection failed: {e}")
        return False

def test_dataset_processing():
    """Test dataset processing"""
    try:
        # Import the data processor
        from kubeflow.data_preprocessing import KubernetesDatasetProcessor
        
        # MinIO configuration
        s3_config = {
            'endpoint_url': "http://localhost:9000",
            'access_key': "nimbusguard",
            'secret_key': "nimbusguard123",
            'bucket_name': "datasets",
            'region': 'us-east-1'
        }
        
        processor = KubernetesDatasetProcessor(
            dataset_path="", 
            use_s3=True, 
            s3_config=s3_config
        )
        
        print("✅ Data processor initialized")
        
        # Load TrainData.csv
        print("📦 Loading TrainData.csv from MinIO...")
        train_df = processor.load_training_data("TrainData.csv")
        print(f"✅ Loaded dataset with {len(train_df)} rows and columns: {list(train_df.columns)}")
        
        # Process for DQN training
        print("🔄 Processing dataset for DQN training...")
        data = processor.process_for_dqn_training(train_df)
        
        print(f"✅ Processed {len(data['states'])} training samples")
        print(f"   States shape: {data['states'].shape}")
        print(f"   Actions shape: {data['actions'].shape}")
        print(f"   Rewards shape: {data['rewards'].shape}")
        
        # Show sample data
        print("\n📊 Sample data:")
        print(f"   First state: {data['states'][0]}")
        print(f"   First action: {data['actions'][0]}")
        print(f"   First reward: {data['rewards'][0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 Testing MinIO Dataset Loading")
    print("=" * 40)
    
    # Test MinIO connection
    print("\n1. Testing MinIO connection...")
    if not test_minio_connection():
        print("❌ MinIO connection test failed")
        return
    
    # Test dataset processing
    print("\n2. Testing dataset processing...")
    if not test_dataset_processing():
        print("❌ Dataset processing test failed")
        return
    
    print("\n✅ All tests passed!")
    print("💡 The dataset loading should work in the training job")

if __name__ == "__main__":
    main() 