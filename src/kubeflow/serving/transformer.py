#!/usr/bin/env python3
"""
Enhanced Model Transformer for NimbusGuard DQN inference
Automatically loads the latest trained model and serves predictions
"""

import asyncio
import json
import logging
import os
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional

import aiohttp
import numpy as np
import torch
import torch.nn as nn
from aiohttp import web
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# MinIO/S3 support
try:
    import boto3
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    logging.warning("boto3 not available - S3/MinIO model loading disabled")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
model_predictions = Counter('nimbusguard_model_predictions_total', 'Total model predictions', ['action'])
prediction_confidence = Histogram('nimbusguard_prediction_confidence', 'Prediction confidence scores')
model_load_time = Histogram('nimbusguard_model_load_seconds', 'Time to load model')
model_version = Gauge('nimbusguard_model_version', 'Current model version')

class DQNModel(nn.Module):
    """DQN Model architecture matching the training script"""
    def __init__(self, state_dim=11, action_dim=5, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class NimbusGuardTransformer:
    """Enhanced transformer with automatic model loading and real-time predictions"""
    
    def __init__(self):
        self.state_dim = 11
        self.action_dim = 5
        self.model_path = Path(os.getenv('MODEL_PATH', '/models'))
        self.model_name = os.getenv('MODEL_NAME', 'nimbusguard-dqn')
        self.prometheus_endpoint = os.getenv('PROMETHEUS_ENDPOINT', 
                                           'http://prometheus.monitoring.svc.cluster.local:9090')
        
        # MinIO configuration
        self.use_minio = os.getenv('USE_MINIO', 'false').lower() == 'true' and S3_AVAILABLE
        self.s3_client = None
        self.bucket_name = None
        
        if self.use_minio:
            self._init_s3_client()
        
        # Model management
        self.model = None
        self.model_metadata = {}
        self.last_model_check = 0
        self.model_check_interval = 300  # Check for new model every 5 minutes
        
        # Action mapping
        self.action_mapping = {
            0: "SCALE_DOWN_2",
            1: "SCALE_DOWN_1", 
            2: "NO_ACTION",
            3: "SCALE_UP_1",
            4: "SCALE_UP_2"
        }
        
        # Load initial model
        self.load_latest_model()
        
        logger.info(f"NimbusGuard Transformer initialized (MinIO: {self.use_minio})")
    
    def _init_s3_client(self):
        """Initialize S3/MinIO client"""
        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=f"http://{os.getenv('MINIO_ENDPOINT', 'minio-api.minio.svc.cluster.local:9000')}",
                aws_access_key_id=os.getenv('MINIO_ACCESS_KEY', 'nimbusguard'),
                aws_secret_access_key=os.getenv('MINIO_SECRET_KEY', 'nimbusguard123'),
                region_name=os.getenv('MINIO_REGION', 'us-east-1')
            )
            self.bucket_name = os.getenv('MINIO_BUCKET', 'models')
            logger.info(f"S3 client initialized for bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self.use_minio = False
    
    def load_latest_model(self) -> bool:
        """Load the latest trained model from MinIO (no fallback)"""
        try:
            start_time = time.time()
            
            if self.use_minio:
                # Use MinIO only - fail if not available
                success = self._load_model_from_minio()
                if success:
                    load_time = time.time() - start_time
                    model_load_time.observe(load_time)
                    self.last_model_check = time.time()
                    return True
                else:
                    logger.error("Failed to load from MinIO - no fallback configured")
                    raise Exception("MinIO model loading failed and fallback is disabled")
            else:
                # Local storage mode
                return self._load_model_from_local()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            if self.model is None:
                logger.info("Creating mock model as fallback")
                self.model = self.create_mock_model()
                self.model_metadata = {'version': 0, 'type': 'mock'}
            return False
    
    def _load_model_from_minio(self) -> bool:
        """Load model from MinIO S3 storage"""
        try:
            model_key = f"{self.model_name}/latest/model.pth"
            
            logger.info(f"Loading model from MinIO: s3://{self.bucket_name}/{model_key}")
            
            # Download model to temporary file
            with tempfile.NamedTemporaryFile(suffix='.pth') as temp_file:
                self.s3_client.download_file(self.bucket_name, model_key, temp_file.name)
                
                # Load model checkpoint
                checkpoint = torch.load(temp_file.name, map_location='cpu')
                
                # Create and load model
                self.model = DQNModel(
                    state_dim=checkpoint.get('state_dim', self.state_dim),
                    action_dim=checkpoint.get('action_dim', self.action_dim)
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                # Update metadata
                self.model_metadata = checkpoint.get('metadata', {})
                current_version = self.model_metadata.get('model_version', 0)
                model_version.set(current_version)
                
                logger.info(f"Loaded model version {current_version} from MinIO")
                logger.info(f"Model trained on {self.model_metadata.get('total_samples', 0)} samples")
                
                return True
                
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"No model found in MinIO: s3://{self.bucket_name}/{model_key}")
            else:
                logger.error(f"MinIO error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading model from MinIO: {e}")
            return False
    
    def _load_model_from_local(self) -> bool:
        """Load model from local file system"""
        try:
            model_dir = self.model_path / self.model_name
            latest_path = model_dir / 'latest' / 'model.pth'
            
            if not latest_path.exists():
                logger.warning(f"No trained model found at {latest_path}, using mock model")
                self.model = self.create_mock_model()
                self.model_metadata = {'version': 0, 'type': 'mock'}
                return False
            
            # Load model checkpoint
            checkpoint = torch.load(latest_path, map_location='cpu')
            
            # Create and load model
            self.model = DQNModel(
                state_dim=checkpoint.get('state_dim', self.state_dim),
                action_dim=checkpoint.get('action_dim', self.action_dim)
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Update metadata
            self.model_metadata = checkpoint.get('metadata', {})
            current_version = self.model_metadata.get('model_version', 0)
            model_version.set(current_version)
            
            logger.info(f"Loaded model version {current_version} from local storage")
            logger.info(f"Model trained on {self.model_metadata.get('total_samples', 0)} samples")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model from local storage: {e}")
            if self.model is None:
                logger.info("Creating mock model as fallback")
                self.model = self.create_mock_model()
                self.model_metadata = {'version': 0, 'type': 'mock'}
            return False
    
    def create_mock_model(self) -> DQNModel:
        """Create a mock model for testing when no trained model is available"""
        model = DQNModel()
        model.eval()
        
        # Initialize with reasonable weights for testing
        with torch.no_grad():
            for param in model.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.constant_(param, 0.0)
        
        return model
    
    def check_for_model_updates(self):
        """Check if a new model version is available"""
        if time.time() - self.last_model_check > self.model_check_interval:
            try:
                new_version = None
                
                if self.use_minio:
                    # Check MinIO only - fail if not available
                    new_version = self._check_minio_model_version()
                    if new_version is None:
                        logger.error("Failed to check MinIO for model updates - no fallback configured")
                        raise Exception("MinIO model version check failed and fallback is disabled")
                else:
                    # Local storage mode
                    new_version = self._check_local_model_version()
                
                # Load new model if version is newer
                if new_version is not None:
                    current_version = self.model_metadata.get('model_version', 0)
                    if new_version > current_version:
                        logger.info(f"New model version {new_version} available, loading...")
                        self.load_latest_model()
                
            except Exception as e:
                logger.error(f"Error checking for model updates: {e}")
            
            self.last_model_check = time.time()
    
    def _check_minio_model_version(self) -> Optional[int]:
        """Check model version in MinIO"""
        try:
            model_key = f"{self.model_name}/latest/model.pth"
            
            # Download just the metadata to check version
            with tempfile.NamedTemporaryFile(suffix='.pth') as temp_file:
                self.s3_client.download_file(self.bucket_name, model_key, temp_file.name)
                checkpoint = torch.load(temp_file.name, map_location='cpu')
                return checkpoint.get('metadata', {}).get('model_version', 0)
                
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchKey':
                logger.warning(f"MinIO version check error: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error checking MinIO model version: {e}")
            return None
    
    def _check_local_model_version(self) -> Optional[int]:
        """Check model version in local storage"""
        try:
            model_dir = self.model_path / self.model_name
            latest_path = model_dir / 'latest' / 'model.pth'
            
            if latest_path.exists():
                checkpoint = torch.load(latest_path, map_location='cpu')
                return checkpoint.get('metadata', {}).get('model_version', 0)
            
        except Exception as e:
            logger.warning(f"Error checking local model version: {e}")
        
        return None
    
    async def collect_current_state(self) -> List[float]:
        """Collect current cluster state from Prometheus and create compatible state vector"""
        try:
            async with aiohttp.ClientSession() as session:
                metrics = {}
                
                # Real-time queries for current state - focused on the 4 core metrics
                queries = {
                    'cpu_utilization': 'avg(rate(container_cpu_usage_seconds_total{namespace="nimbusguard"}[5m]))',
                    'memory_utilization': 'avg(container_memory_working_set_bytes{namespace="nimbusguard"}) / avg(container_spec_memory_limit_bytes{namespace="nimbusguard"})',
                    'network_io_rate': 'avg(rate(container_network_transmit_bytes_total{namespace="nimbusguard"}[5m]))',
                    'request_rate': 'avg(rate(http_requests_total{namespace="nimbusguard"}[5m]))',
                    'pod_count': 'count(kube_pod_info{namespace="nimbusguard"})',
                    'response_time_p95': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{namespace="nimbusguard"}[5m]))',
                    'error_rate': 'avg(rate(http_requests_total{namespace="nimbusguard",status=~"5.."}[5m]))',
                    'queue_depth': 'avg(kafka_consumer_lag_sum{namespace="nimbusguard"})',
                    'cpu_throttling': 'avg(rate(container_cpu_cfs_throttled_seconds_total{namespace="nimbusguard"}[5m]))',
                    'memory_pressure': 'avg(container_memory_working_set_bytes{namespace="nimbusguard"} / container_spec_memory_limit_bytes{namespace="nimbusguard"})',
                    'node_utilization': 'avg(1 - rate(node_cpu_seconds_total{mode="idle"}[5m]))'
                }
                
                for metric_name, query in queries.items():
                    try:
                        async with session.get(
                            f"{self.prometheus_endpoint}/api/v1/query",
                            params={'query': query},
                            timeout=5
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                result = data.get('data', {}).get('result', [])
                                if result:
                                    metrics[metric_name] = float(result[0]['value'][1])
                                else:
                                    metrics[metric_name] = 0.0
                            else:
                                metrics[metric_name] = 0.0
                    except Exception as e:
                        logger.warning(f"Failed to collect {metric_name}: {e}")
                        metrics[metric_name] = 0.0
                
                # Create state vector using the same logic as TrainData.csv processing
                return self._create_traindata_compatible_state_vector(metrics)
                
        except Exception as e:
            logger.error(f"State collection error: {e}")
            # Return default state if collection fails
            return [0.5] * self.state_dim
    
    def _create_traindata_compatible_state_vector(self, metrics: Dict[str, float]) -> List[float]:
        """Create state vector compatible with TrainData.csv training format"""
        
        # Map Prometheus metrics to TrainData.csv equivalent
        # CPU: Use actual CPU utilization (0-1 range, convert to 1-3 range equivalent)
        cpu_raw = metrics.get('cpu_utilization', 0.0)
        cpu_traindata_equiv = 1.0 + (cpu_raw * 2.0)  # Map 0-1 to 1-3 range
        cpu_normalized = min(max((cpu_traindata_equiv - 1.0) / 2.0, 0.0), 1.0)
        
        # Network: Use network I/O rate (normalize to 0-1 range like PackRecv_Mean)
        network_raw = metrics.get('network_io_rate', 0.0) / 1e6  # Convert to MB/s
        network_normalized = min(max(network_raw, 0.0), 1.0)
        
        # Pod count: Use actual pod count (normalize to 0-1 range)
        pods_raw = metrics.get('pod_count', 1.0)
        pods_normalized = min(max(pods_raw / 20.0, 0.0), 1.0)
        
        # Stress rate: Derive from request rate and error rate
        request_rate = metrics.get('request_rate', 0.0)
        error_rate = metrics.get('error_rate', 0.0)
        stress_normalized = min(max((request_rate / 100.0) + error_rate, 0.0), 1.0)
        
        # Create 11-dimensional state vector using the same logic as training
        # This matches exactly with _extract_traindata_state_vector in data_preprocessing.py
        state_vector = [
            cpu_normalized,                                    # [0] CPU utilization (primary metric)
            cpu_normalized * 0.8,                             # [1] Memory utilization (correlated with CPU)
            network_normalized,                               # [2] Network I/O rate (from network metrics)
            stress_normalized * 100.0 / 1000,                # [3] Request rate (derived from stress)
            pods_normalized,                                  # [4] Pod count (from pod metrics)
            self._calculate_response_time_inference(cpu_normalized, pods_normalized), # [5] Response time (derived)
            self._calculate_error_rate_inference(cpu_normalized, stress_normalized),  # [6] Error rate (derived)
            stress_normalized * 0.5,                          # [7] Queue depth (derived from stress)
            max(0.0, cpu_normalized - 0.7),                  # [8] CPU throttling (when CPU > 70%)
            cpu_normalized * 0.9,                             # [9] Memory pressure (correlated with CPU)
            (cpu_normalized + network_normalized) / 2.0       # [10] Node utilization (combined metric)
        ]
        
        # Ensure all values are in [0, 1] range
        state_vector = [min(max(val, 0.0), 1.0) for val in state_vector]
        
        logger.debug(f"TrainData-compatible state vector: {state_vector}")
        logger.debug(f"Core metrics - CPU: {cpu_normalized:.3f}, Network: {network_normalized:.3f}, Pods: {pods_normalized:.3f}, Stress: {stress_normalized:.3f}")
        
        return state_vector
    
    def _calculate_response_time_inference(self, cpu_util: float, pod_ratio: float) -> float:
        """Calculate estimated response time - matches training logic"""
        # Higher CPU and lower pod count = higher response time
        base_response = cpu_util
        pod_factor = max(0.1, 1.0 - pod_ratio)  # Fewer pods = higher response time
        return min(base_response * pod_factor, 1.0)
    
    def _calculate_error_rate_inference(self, cpu_util: float, stress: float) -> float:
        """Calculate estimated error rate - matches training logic"""
        # High CPU + high stress = higher error rate
        if cpu_util > 0.8 and stress > 0.5:
            return min(cpu_util * stress * 0.1, 1.0)
        return 0.0
    
    def create_state_vector(self, cluster_state: Dict[str, Any]) -> List[float]:
        """Create DQN state vector from provided cluster state"""
        try:
            state_vector = [
                float(cluster_state.get('cpu_utilization', 0.0)),
                float(cluster_state.get('memory_utilization', 0.0)),
                float(cluster_state.get('network_io_rate', 0.0)),
                float(cluster_state.get('request_rate', 0.0)),
                float(cluster_state.get('pod_count', 1.0)),
                float(cluster_state.get('response_time_p95', 0.0)),
                float(cluster_state.get('error_rate', 0.0)),
                float(cluster_state.get('queue_depth', 0.0)),
                float(cluster_state.get('cpu_throttling', 0.0)),
                float(cluster_state.get('memory_pressure', 0.0)),
                float(cluster_state.get('node_utilization', 0.0))
            ]
            
            # Normalize values to [0, 1] range
            state_vector = [min(max(val, 0.0), 1.0) for val in state_vector]
            
            # Pad or truncate to ensure exactly 11 dimensions
            if len(state_vector) < self.state_dim:
                state_vector.extend([0.0] * (self.state_dim - len(state_vector)))
            elif len(state_vector) > self.state_dim:
                state_vector = state_vector[:self.state_dim]
            
            return state_vector
            
        except Exception as e:
            logger.error(f"State vector creation error: {e}")
            return [0.0] * self.state_dim
    
    async def predict(self, state_vector: List[float]) -> Dict[str, Any]:
        """Make prediction using the loaded model"""
        try:
            # Check for model updates
            self.check_for_model_updates()
            
            # Ensure state vector is correct dimension
            if len(state_vector) != self.state_dim:
                logger.warning(f"State vector dimension mismatch: {len(state_vector)} != {self.state_dim}")
                state_vector = state_vector[:self.state_dim] + [0.0] * max(0, self.state_dim - len(state_vector))
            
            # Convert to tensor and predict
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
            
            with torch.no_grad():
                q_values = self.model(state_tensor)
                q_values_np = q_values.squeeze().numpy()
                
                # Get best action
                best_action_idx = int(np.argmax(q_values_np))
                best_action = self.action_mapping[best_action_idx]
                confidence = float(np.max(q_values_np))
                
                # Calculate replica changes based on action
                replica_changes = {
                    "SCALE_DOWN_2": -2,
                    "SCALE_DOWN_1": -1,
                    "NO_ACTION": 0,
                    "SCALE_UP_1": 1,
                    "SCALE_UP_2": 2
                }
                
                current_replicas = int(state_vector[4] * 10)  # Denormalize pod count
                target_replicas = max(1, current_replicas + replica_changes[best_action])
                
                response = {
                    "action": best_action,
                    "action_index": best_action_idx,
                    "q_values": q_values_np.tolist(),
                    "confidence": confidence,
                    "current_replicas": current_replicas,
                    "target_replicas": target_replicas,
                    "replica_change": replica_changes[best_action],
                    "model_version": self.model_metadata.get('model_version', 0),
                    "timestamp": time.time()
                }
                
                # Update metrics
                model_predictions.labels(action=best_action).inc()
                prediction_confidence.observe(confidence)
                
                logger.info(f"Action prediction: {best_action} (confidence: {confidence:.3f})")
                return response
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "action": "NO_ACTION",
                "action_index": 2,
                "q_values": [0.0] * self.action_dim,
                "confidence": 0.0,
                "error": str(e),
                "timestamp": time.time()
            }

# Global transformer instance
transformer = NimbusGuardTransformer()

async def health_check(request):
    """Health check endpoint"""
    return web.json_response({
        "status": "healthy",
        "model_version": transformer.model_metadata.get('model_version', 0),
        "model_type": transformer.model_metadata.get('type', 'trained'),
        "timestamp": time.time()
    })

async def predict_endpoint(request):
    """Main prediction endpoint"""
    try:
        # Parse request
        if request.content_type == 'application/json':
            data = await request.json()
        else:
            data = {}
        
        # Get state vector
        if 'state' in data:
            state_vector = data['state']
        elif 'cluster_state' in data:
            state_vector = transformer.create_state_vector(data['cluster_state'])
        else:
            # Collect current state from Prometheus
            state_vector = await transformer.collect_current_state()
        
        # Make prediction
        prediction = await transformer.predict(state_vector)
        
        return web.json_response(prediction)
        
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return web.json_response({
            "error": str(e),
            "action": "NO_ACTION"
        }, status=500)

async def model_info(request):
    """Model information endpoint"""
    return web.json_response({
        "model_metadata": transformer.model_metadata,
        "state_dim": transformer.state_dim,
        "action_dim": transformer.action_dim,
        "action_mapping": transformer.action_mapping,
        "model_path": str(transformer.model_path),
        "last_model_check": transformer.last_model_check
    })

def create_app():
    """Create the web application"""
    app = web.Application()
    
    # Add routes
    app.router.add_get('/health', health_check)
    app.router.add_post('/predict', predict_endpoint)
    app.router.add_get('/v1/models/{model_name}', model_info)
    app.router.add_get('/v1/models/{model_name}/metadata', model_info)
    
    return app

async def main():
    """Main function"""
    # Start Prometheus metrics server
    start_http_server(8090)
    logger.info("Prometheus metrics server started on port 8090")
    
    # Create and start web server
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()
    
    logger.info("Transformer initialized successfully")
    logger.info("NimbusGuard Model Transformer started on port 8080")
    
    # Keep the server running
    try:
        while True:
            await asyncio.sleep(3600)  # Sleep for 1 hour
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
