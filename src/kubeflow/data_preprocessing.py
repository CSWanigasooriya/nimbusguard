#!/usr/bin/env python3
"""
Data Preprocessing for External Datasets
Processes real-world Kubernetes scaling datasets for DQN training
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from io import StringIO, BytesIO

# MinIO/S3 support
try:
    import boto3
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    logging.warning("boto3 not available - S3/MinIO support disabled")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KubernetesDatasetProcessor:
    """
    Processes external Kubernetes scaling datasets for DQN training
    Supports the format described with TrainData.csv, TestK8sData.csv, TestJMeterData.csv
    """
    
    def __init__(self, dataset_path: str, use_s3: bool = False, s3_config: Dict[str, str] = None):
        self.dataset_path = Path(dataset_path) if not use_s3 else dataset_path
        self.use_s3 = use_s3 and S3_AVAILABLE
        self.state_dim = 11  # Match current DQN architecture
        self.action_dim = 5   # 5 scaling actions
        
        # Action mapping: SCALE_DOWN_2, SCALE_DOWN_1, NO_ACTION, SCALE_UP_1, SCALE_UP_2
        self.action_mapping = {
            'SCALE_DOWN_2': 0,
            'SCALE_DOWN_1': 1, 
            'NO_ACTION': 2,
            'SCALE_UP_1': 3,
            'SCALE_UP_2': 4
        }
        
        # Initialize S3/MinIO client if needed
        self.s3_client = None
        self.bucket_name = None
        if self.use_s3 and s3_config:
            self._init_s3_client(s3_config)
        
        logger.info(f"Initialized dataset processor for: {self.dataset_path} (S3: {self.use_s3})")
    
    def _init_s3_client(self, s3_config: Dict[str, str]):
        """Initialize S3/MinIO client"""
        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=s3_config.get('endpoint_url'),
                aws_access_key_id=s3_config.get('access_key'),
                aws_secret_access_key=s3_config.get('secret_key'),
                region_name=s3_config.get('region', 'us-east-1')
            )
            self.bucket_name = s3_config.get('bucket_name', 'datasets')
            logger.info(f"S3 client initialized for bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self.use_s3 = False
    
    def load_training_data(self, filename: str = "TrainData.csv") -> pd.DataFrame:
        """Load the main training dataset"""
        if self.use_s3:
            return self._load_from_s3(filename)
        else:
            return self._load_from_filesystem(filename)
    
    def _load_from_filesystem(self, filename: str) -> pd.DataFrame:
        """Load dataset from local filesystem"""
        file_path = self.dataset_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Training data file not found: {file_path}")
        
        logger.info(f"Loading training data from: {file_path}")
        df = pd.read_csv(file_path)
        
        logger.info(f"Loaded {len(df)} training samples with columns: {list(df.columns)}")
        return df
    
    def _load_from_s3(self, filename: str) -> pd.DataFrame:
        """Load dataset from S3/MinIO"""
        try:
            logger.info(f"Loading training data from S3: s3://{self.bucket_name}/{filename}")
            
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=filename)
            csv_content = response['Body'].read().decode('utf-8')
            
            df = pd.read_csv(StringIO(csv_content))
            logger.info(f"Loaded {len(df)} training samples with columns: {list(df.columns)}")
            return df
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"Training data file not found in S3: s3://{self.bucket_name}/{filename}")
            else:
                raise Exception(f"Error loading from S3: {e}")
        except Exception as e:
            raise Exception(f"Error loading training data from S3: {e}")
    
    def load_test_data(self, k8s_filename: str = "TestK8sData.csv", 
                      jmeter_filename: str = "TestJMeterData.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load test datasets"""
        k8s_path = self.dataset_path / k8s_filename
        jmeter_path = self.dataset_path / jmeter_filename
        
        k8s_df = None
        jmeter_df = None
        
        if k8s_path.exists():
            logger.info(f"Loading K8s test data from: {k8s_path}")
            k8s_df = pd.read_csv(k8s_path)
            logger.info(f"Loaded {len(k8s_df)} K8s test samples")
        
        if jmeter_path.exists():
            logger.info(f"Loading JMeter test data from: {jmeter_path}")
            jmeter_df = pd.read_csv(jmeter_path)
            logger.info(f"Loaded {len(jmeter_df)} JMeter test samples")
        
        return k8s_df, jmeter_df
    
    def process_for_dqn_training(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Process dataset into DQN training format
        
        Expected columns (adapt based on your actual dataset):
        - cpu_load, cpu_usage, mean_cpu_load
        - packet_reception_rate, network_io
        - pod_count, replicas, mean_pods
        - response_time, latency
        - experiment_type (HPA/ANN)
        - Any additional performance metrics
        """
        logger.info("Processing dataset for DQN training...")
        
        # Identify available columns (adapt to your actual dataset)
        available_cols = df.columns.tolist()
        logger.info(f"Available columns: {available_cols}")
        
        states = []
        actions = []
        rewards = []
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                # Create 11-dimensional state vector
                state_vector = self._extract_state_vector(row, available_cols)
                
                # Extract or infer action
                action = self._extract_action(row, idx, df)
                
                # Calculate reward
                reward = self._calculate_reward(row, idx, df)
                
                states.append(state_vector)
                actions.append(action)
                rewards.append(reward)
                
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                continue
        
        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        
        logger.info(f"Processed {len(states)} samples successfully")
        logger.info(f"State shape: {states.shape}, Actions shape: {actions.shape}, Rewards shape: {rewards.shape}")
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'metadata': {
                'source_dataset': str(self.dataset_path),
                'processed_samples': len(states),
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'processing_timestamp': datetime.now().isoformat()
            }
        }
    
    def _extract_state_vector(self, row: pd.Series, available_cols: List[str]) -> List[float]:
        """Extract 11-dimensional state vector from dataset row - optimized for TrainData.csv"""
        
        # Check if this is the TrainData.csv format (4 columns)
        if 'CPU_Mean' in available_cols and 'PackRecv_Mean' in available_cols:
            return self._extract_traindata_state_vector(row)
        
        # Fallback to original complex extraction for other datasets
        return self._extract_complex_state_vector(row, available_cols)
    
    def _extract_traindata_state_vector(self, row: pd.Series) -> List[float]:
        """Extract state vector specifically for TrainData.csv format"""
        
        # Get the 4 core metrics from your dataset
        cpu_mean = self._get_column_value(row, ['CPU_Mean'])  # Already normalized (1-2 range)
        pack_recv_mean = self._get_column_value(row, ['PackRecv_Mean'])  # Network metric (0-1 range)
        pods_number_mean = self._get_column_value(row, ['PodsNumber_Mean'])  # Pod count (10-13 range)
        stress_rate_mean = self._get_column_value(row, ['StressRate_Mean'])  # Load indicator (always 1)
        
        # Normalize CPU (assuming range 1-3 based on your data)
        cpu_normalized = min(max((cpu_mean - 1.0) / 2.0, 0.0), 1.0)
        
        # PackRecv is already in 0-1 range
        network_normalized = min(max(pack_recv_mean, 0.0), 1.0)
        
        # Normalize pod count (assuming range 1-20)
        pods_normalized = min(max(pods_number_mean / 20.0, 0.0), 1.0)
        
        # Stress rate (binary indicator)
        stress_normalized = min(max(stress_rate_mean, 0.0), 1.0)
        
        # Create 11-dimensional state vector using your 4 core metrics
        # Map them intelligently to the expected 11 dimensions
        state_vector = [
            cpu_normalized,                    # [0] CPU utilization (primary metric)
            cpu_normalized * 0.8,             # [1] Memory utilization (correlated with CPU)
            network_normalized,               # [2] Network I/O rate (from PackRecv_Mean)
            stress_normalized * 100.0 / 1000, # [3] Request rate (derived from stress)
            pods_normalized,                  # [4] Pod count (from PodsNumber_Mean)
            self._calculate_response_time(cpu_normalized, pods_normalized), # [5] Response time (derived)
            self._calculate_error_rate(cpu_normalized, stress_normalized),  # [6] Error rate (derived)
            stress_normalized * 0.5,          # [7] Queue depth (derived from stress)
            max(0.0, cpu_normalized - 0.7),  # [8] CPU throttling (when CPU > 70%)
            cpu_normalized * 0.9,             # [9] Memory pressure (correlated with CPU)
            (cpu_normalized + network_normalized) / 2.0  # [10] Node utilization (combined metric)
        ]
        
        # Ensure all values are in [0, 1] range
        state_vector = [min(max(val, 0.0), 1.0) for val in state_vector]
        
        return state_vector
    
    def _calculate_response_time(self, cpu_util: float, pod_ratio: float) -> float:
        """Calculate estimated response time based on CPU and pod count"""
        # Higher CPU and lower pod count = higher response time
        base_response = cpu_util
        pod_factor = max(0.1, 1.0 - pod_ratio)  # Fewer pods = higher response time
        return min(base_response * pod_factor, 1.0)
    
    def _calculate_error_rate(self, cpu_util: float, stress: float) -> float:
        """Calculate estimated error rate based on CPU and stress"""
        # High CPU + high stress = higher error rate
        if cpu_util > 0.8 and stress > 0.5:
            return min(cpu_util * stress * 0.1, 1.0)
        return 0.0
    
    def _extract_complex_state_vector(self, row: pd.Series, available_cols: List[str]) -> List[float]:
        """Original complex extraction for other dataset formats"""
        
        # Define feature extractors with fallback names
        feature_extractors = [
            # CPU utilization (0-1)
            lambda r: self._safe_normalize(self._get_column_value(r, ['cpu_load', 'mean_cpu_load', 'cpu_usage', 'cpu_utilization']), 0, 100) / 100,
            
            # Memory utilization (0-1) 
            lambda r: self._safe_normalize(self._get_column_value(r, ['memory_usage', 'mean_memory', 'memory_utilization']), 0, 100) / 100,
            
            # Network I/O rate (normalized)
            lambda r: self._safe_normalize(self._get_column_value(r, ['packet_reception_rate', 'network_io', 'packets_per_sec']), 0, 10000) / 10000,
            
            # Request rate (normalized)
            lambda r: self._safe_normalize(self._get_column_value(r, ['request_rate', 'requests_per_sec', 'throughput']), 0, 1000) / 1000,
            
            # Pod count (normalized to 0-1, assuming max 20 pods)
            lambda r: self._safe_normalize(self._get_column_value(r, ['pod_count', 'replicas', 'mean_pods', 'num_pods']), 1, 20) / 20,
            
            # Response time (normalized, assuming max 5 seconds)
            lambda r: min(self._safe_normalize(self._get_column_value(r, ['response_time', 'latency', 'avg_response_time']), 0, 5000) / 5000, 1.0),
            
            # Error rate (0-1)
            lambda r: min(self._safe_normalize(self._get_column_value(r, ['error_rate', 'failure_rate', 'errors_per_sec']), 0, 100) / 100, 1.0),
            
            # Queue depth or load factor
            lambda r: self._safe_normalize(self._get_column_value(r, ['queue_depth', 'load_factor', 'concurrent_users']), 0, 1000) / 1000,
            
            # CPU throttling or pressure indicator
            lambda r: min(self._safe_normalize(self._get_column_value(r, ['cpu_throttling', 'cpu_pressure', 'cpu_wait']), 0, 100) / 100, 1.0),
            
            # System stability indicator (derived from variance or jitter)
            lambda r: 1.0 - min(self._safe_normalize(self._get_column_value(r, ['jitter', 'variance', 'std_dev']), 0, 1000) / 1000, 1.0),
            
            # Overall system health (0-1)
            lambda r: self._calculate_health_score(r)
        ]
        
        # Extract features
        state_vector = []
        for extractor in feature_extractors:
            try:
                value = extractor(row)
                state_vector.append(float(value))
            except Exception as e:
                logger.debug(f"Feature extraction error: {e}")
                state_vector.append(0.0)
        
        # Ensure exactly 11 dimensions
        while len(state_vector) < 11:
            state_vector.append(0.0)
        
        return state_vector[:11]
    
    def _get_column_value(self, row: pd.Series, column_names: List[str]) -> float:
        """Get value from first available column"""
        for col in column_names:
            if col in row.index and pd.notna(row[col]):
                return float(row[col])
        return 0.0
    
    def _safe_normalize(self, value: float, min_val: float, max_val: float) -> float:
        """Safely normalize value to 0-max_val range"""
        return max(min_val, min(value, max_val))
    
    def _calculate_health_score(self, row: pd.Series) -> float:
        """Calculate overall system health score (0-1)"""
        # Simple health score based on available metrics
        cpu = self._get_column_value(row, ['cpu_load', 'cpu_usage']) / 100
        response_time = self._get_column_value(row, ['response_time', 'latency']) / 1000  # ms to s
        error_rate = self._get_column_value(row, ['error_rate', 'failure_rate']) / 100
        
        # Health decreases with high CPU, high response time, high errors
        health = 1.0 - (cpu * 0.4 + min(response_time, 1.0) * 0.3 + error_rate * 0.3)
        return max(0.0, min(1.0, health))
    
    def _extract_action(self, row: pd.Series, idx: int, df: pd.DataFrame) -> int:
        """Extract or infer scaling action from dataset"""
        
        # Method 1: Direct action column
        if 'action' in row.index:
            action_str = str(row['action']).upper()
            return self.action_mapping.get(action_str, 2)  # Default to NO_ACTION
        
        # Method 2: TrainData.csv format - infer from pod count changes
        if 'PodsNumber_Mean' in row.index and idx > 0:
            current_pods = self._get_column_value(row, ['PodsNumber_Mean'])
            prev_pods = self._get_column_value(df.iloc[idx-1], ['PodsNumber_Mean'])
            
            pod_diff = current_pods - prev_pods
            if pod_diff >= 2:
                return 4  # SCALE_UP_2
            elif pod_diff == 1:
                return 3  # SCALE_UP_1
            elif pod_diff <= -2:
                return 0  # SCALE_DOWN_2
            elif pod_diff == -1:
                return 1  # SCALE_DOWN_1
            else:
                return 2  # NO_ACTION
        
        # Method 3: TrainData.csv format - infer from CPU and stress patterns
        if 'CPU_Mean' in row.index and 'StressRate_Mean' in row.index:
            cpu_mean = self._get_column_value(row, ['CPU_Mean'])
            stress_rate = self._get_column_value(row, ['StressRate_Mean'])
            pods_count = self._get_column_value(row, ['PodsNumber_Mean'])
            
            # Normalize CPU (1-3 range to 0-1)
            cpu_normalized = (cpu_mean - 1.0) / 2.0
            
            # Smart action inference based on CPU load and pod count
            if cpu_normalized > 0.8:  # High CPU load
                if pods_count < 15:  # Room to scale up
                    return 3 if cpu_normalized < 0.9 else 4  # SCALE_UP_1 or SCALE_UP_2
                else:
                    return 2  # NO_ACTION (already many pods)
            elif cpu_normalized < 0.3:  # Low CPU load
                if pods_count > 12:  # Can scale down
                    return 1 if pods_count < 15 else 0  # SCALE_DOWN_1 or SCALE_DOWN_2
                else:
                    return 2  # NO_ACTION (already few pods)
            else:
                return 2  # NO_ACTION (CPU in good range)
        
        # Method 4: Experiment type (HPA vs ANN)
        if 'experiment_type' in row.index:
            exp_type = str(row['experiment_type']).upper()
            if 'HPA' in exp_type:
                # Simulate HPA-like decisions based on CPU
                cpu = self._get_column_value(row, ['cpu_load', 'cpu_usage'])
                if cpu > 80:
                    return 3  # SCALE_UP_1
                elif cpu < 30:
                    return 1  # SCALE_DOWN_1
                else:
                    return 2  # NO_ACTION
        
        # Method 5: Generic pod count changes
        if idx > 0 and 'pod_count' in row.index:
            current_pods = self._get_column_value(row, ['pod_count', 'replicas'])
            prev_pods = self._get_column_value(df.iloc[idx-1], ['pod_count', 'replicas'])
            
            pod_diff = current_pods - prev_pods
            if pod_diff >= 2:
                return 4  # SCALE_UP_2
            elif pod_diff == 1:
                return 3  # SCALE_UP_1
            elif pod_diff <= -2:
                return 0  # SCALE_DOWN_2
            elif pod_diff == -1:
                return 1  # SCALE_DOWN_1
            else:
                return 2  # NO_ACTION
        
        # Default: weighted random action (favor moderate actions)
        weights = [0.1, 0.2, 0.4, 0.2, 0.1]  # Favor NO_ACTION and small scaling
        return np.random.choice(5, p=weights)
    
    def _calculate_reward(self, row: pd.Series, idx: int, df: pd.DataFrame) -> float:
        """Calculate reward for the state-action pair"""
        
        # Check if this is TrainData.csv format
        if 'CPU_Mean' in row.index and 'PodsNumber_Mean' in row.index:
            return self._calculate_traindata_reward(row, idx, df)
        
        # Fallback to original reward calculation
        return self._calculate_complex_reward(row, idx, df)
    
    def _calculate_traindata_reward(self, row: pd.Series, idx: int, df: pd.DataFrame) -> float:
        """Calculate reward specifically for TrainData.csv format"""
        
        cpu_mean = self._get_column_value(row, ['CPU_Mean'])
        pack_recv = self._get_column_value(row, ['PackRecv_Mean'])
        pods_count = self._get_column_value(row, ['PodsNumber_Mean'])
        stress_rate = self._get_column_value(row, ['StressRate_Mean'])
        
        # Normalize metrics
        cpu_normalized = (cpu_mean - 1.0) / 2.0  # 1-3 range to 0-1
        network_normalized = pack_recv  # Already 0-1
        pod_ratio = pods_count / 20.0  # Normalize pod count
        
        # Performance reward - balance CPU utilization
        # Target CPU around 60-70% (0.6-0.7 normalized)
        optimal_cpu = 0.65
        cpu_distance = abs(cpu_normalized - optimal_cpu)
        performance_reward = 1.0 - (cpu_distance * 2.0)  # Penalize deviation from optimal
        
        # Efficiency reward - penalize over-provisioning
        if cpu_normalized < 0.3 and pod_ratio > 0.6:  # Low CPU but many pods
            efficiency_reward = -0.5
        elif cpu_normalized > 0.8 and pod_ratio < 0.5:  # High CPU but few pods
            efficiency_reward = -0.3
        else:
            efficiency_reward = 0.3  # Good balance
        
        # Network performance reward
        network_reward = 0.2 if network_normalized > 0.3 else -0.1  # Reward good network activity
        
        # Stability reward (penalize large pod changes)
        stability_reward = 0.0
        if idx > 0:
            prev_pods = self._get_column_value(df.iloc[idx-1], ['PodsNumber_Mean'])
            pod_change = abs(pods_count - prev_pods)
            if pod_change == 0:
                stability_reward = 0.1  # Small reward for stability
            elif pod_change == 1:
                stability_reward = 0.0  # Neutral for small changes
            else:
                stability_reward = -0.2 * (pod_change - 1)  # Penalty for large changes
        
        # Combined reward with weights
        total_reward = (performance_reward * 0.5 +  # Most important: good CPU utilization
                       efficiency_reward * 0.3 +   # Important: resource efficiency
                       network_reward * 0.1 +      # Minor: network activity
                       stability_reward * 0.1)     # Minor: stability
        
        # Normalize to [-1, 1] range
        return max(-1.0, min(1.0, total_reward))
    
    def _calculate_complex_reward(self, row: pd.Series, idx: int, df: pd.DataFrame) -> float:
        """Original complex reward calculation for other dataset formats"""
        
        # Performance component
        response_time = self._get_column_value(row, ['response_time', 'latency'])
        error_rate = self._get_column_value(row, ['error_rate', 'failure_rate'])
        
        performance_reward = 1.0 - (response_time / 5000 + error_rate / 100) * 0.5
        
        # Efficiency component  
        cpu_usage = self._get_column_value(row, ['cpu_load', 'cpu_usage']) / 100
        pod_count = self._get_column_value(row, ['pod_count', 'replicas'])
        
        # Reward efficient resource usage
        efficiency_reward = 1.0 - abs(cpu_usage - 0.7) - (pod_count / 20) * 0.2
        
        # Stability component (penalize rapid changes)
        stability_reward = 0.0
        if idx > 0:
            prev_pods = self._get_column_value(df.iloc[idx-1], ['pod_count', 'replicas'])
            current_pods = self._get_column_value(row, ['pod_count', 'replicas'])
            pod_change = abs(current_pods - prev_pods)
            stability_reward = 1.0 - (pod_change / 5)  # Penalize large changes
        
        # Combined reward
        total_reward = (performance_reward * 0.5 + 
                       efficiency_reward * 0.3 + 
                       stability_reward * 0.2)
        
        # Normalize to [-1, 1] range
        return max(-1.0, min(1.0, total_reward))
    
    def save_processed_data(self, processed_data: Dict[str, Any], output_path: str):
        """Save processed data for training"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as compressed numpy format
        np.savez_compressed(
            output_file,
            states=processed_data['states'],
            actions=processed_data['actions'],
            rewards=processed_data['rewards'],
            metadata=json.dumps(processed_data['metadata'])
        )
        
        logger.info(f"Saved processed data to: {output_file}")
    
    def load_processed_data(self, input_path: str) -> Dict[str, Any]:
        """Load previously processed data"""
        data = np.load(input_path)
        return {
            'states': data['states'],
            'actions': data['actions'], 
            'rewards': data['rewards'],
            'metadata': json.loads(str(data['metadata']))
        }


def main():
    """Example usage of the dataset processor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process external Kubernetes dataset for DQN training')
    parser.add_argument('--dataset-path', required=True, help='Path to dataset directory')
    parser.add_argument('--output-path', default='/models/processed_training_data.npz', help='Output path for processed data')
    parser.add_argument('--train-file', default='TrainData.csv', help='Training data filename')
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = KubernetesDatasetProcessor(args.dataset_path)
        
        # Load and process training data
        train_df = processor.load_training_data(args.train_file)
        processed_data = processor.process_for_dqn_training(train_df)
        
        # Save processed data
        processor.save_processed_data(processed_data, args.output_path)
        
        logger.info("✅ Dataset processing completed successfully!")
        logger.info(f"📊 Processed {len(processed_data['states'])} training samples")
        logger.info(f"💾 Saved to: {args.output_path}")
        
    except Exception as e:
        logger.error(f"❌ Dataset processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 