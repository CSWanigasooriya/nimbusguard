#!/usr/bin/env python3
"""
Upload TrainData.csv to MinIO for model training
"""

import os
import sys
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("❌ boto3 not installed. Installing...")
    os.system("pip3 install boto3")
    import boto3
    from botocore.exceptions import ClientError

def upload_dataset():
    """Upload TrainData.csv to MinIO"""
    
    # MinIO configuration
    endpoint_url = "http://localhost:9000"  # Port forwarded MinIO
    access_key = "nimbusguard"
    secret_key = "nimbusguard123"
    bucket_name = "datasets"
    
    # Initialize S3 client
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='us-east-1'
        )
        
        print(f"✅ Connected to MinIO at {endpoint_url}")
        
        # Check if bucket exists, create if not
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"✅ Bucket '{bucket_name}' exists")
        except ClientError:
            s3_client.create_bucket(Bucket=bucket_name)
            print(f"✅ Created bucket '{bucket_name}'")
        
        # Upload TrainData.csv
        dataset_path = Path("datasets/TrainData.csv")
        if not dataset_path.exists():
            print(f"❌ Dataset not found: {dataset_path}")
            return False
        
        print(f"📤 Uploading {dataset_path} to s3://{bucket_name}/TrainData.csv")
        s3_client.upload_file(str(dataset_path), bucket_name, "TrainData.csv")
        
        # Verify upload
        try:
            response = s3_client.head_object(Bucket=bucket_name, Key="TrainData.csv")
            size = response['ContentLength']
            print(f"✅ Upload successful! File size: {size} bytes")
            return True
        except ClientError as e:
            print(f"❌ Upload verification failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ MinIO connection failed: {e}")
        return False

if __name__ == "__main__":
    print("📦 Uploading TrainData.csv to MinIO...")
    if upload_dataset():
        print("🎉 Dataset upload completed!")
    else:
        print("❌ Dataset upload failed!")
        sys.exit(1) 