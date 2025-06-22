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
from pathlib import Path
from typing import Dict, List, Any, Optional

import aiohttp
import numpy as np
import torch
import torch.nn as nn
from aiohttp import web
from prometheus_client import Counter, Histogram, Gauge, start_http_server

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
        
        logger.info("NimbusGuard Transformer initialized")
    
    def load_latest_model(self) -> bool:
        """Load the latest trained model"""
        try:
            start_time = time.time()
            
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
            
            # Update metrics
            load_time = time.time() - start_time
            model_load_time.observe(load_time)
            model_version.set(current_version)
            
            logger.info(f"Loaded model version {current_version} in {load_time:.2f}s")
            logger.info(f"Model trained on {self.model_metadata.get('total_samples', 0)} samples")
            
            self.last_model_check = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
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
                model_dir = self.model_path / self.model_name
                latest_path = model_dir / 'latest' / 'model.pth'
                
                if latest_path.exists():
                    # Check if model has been updated
                    checkpoint = torch.load(latest_path, map_location='cpu')
                    new_version = checkpoint.get('metadata', {}).get('model_version', 0)
                    current_version = self.model_metadata.get('model_version', 0)
                    
                    if new_version > current_version:
                        logger.info(f"New model version {new_version} available, loading...")
                        self.load_latest_model()
                
            except Exception as e:
                logger.error(f"Error checking for model updates: {e}")
            
            self.last_model_check = time.time()
    
    async def collect_current_state(self) -> List[float]:
        """Collect current cluster state from Prometheus"""
        try:
            async with aiohttp.ClientSession() as session:
                metrics = {}
                
                # Real-time queries for current state
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
                
                # Create 11-dimensional state vector
                state_vector = [
                    min(max(metrics.get('cpu_utilization', 0.0), 0.0), 1.0),
                    min(max(metrics.get('memory_utilization', 0.0), 0.0), 1.0),
                    min(max(metrics.get('network_io_rate', 0.0) / 1e6, 0.0), 1.0),
                    min(max(metrics.get('request_rate', 0.0) / 100, 0.0), 1.0),
                    min(max(metrics.get('pod_count', 1.0) / 10, 0.0), 1.0),
                    min(max(metrics.get('response_time_p95', 0.0), 0.0), 1.0),
                    min(max(metrics.get('error_rate', 0.0), 0.0), 1.0),
                    min(max(metrics.get('queue_depth', 0.0) / 1000, 0.0), 1.0),
                    min(max(metrics.get('cpu_throttling', 0.0), 0.0), 1.0),
                    min(max(metrics.get('memory_pressure', 0.0), 0.0), 1.0),
                    min(max(metrics.get('node_utilization', 0.0), 0.0), 1.0)
                ]
                
                logger.debug(f"Collected state vector: {state_vector}")
                return state_vector
                
        except Exception as e:
            logger.error(f"State collection error: {e}")
            # Return default state if collection fails
            return [0.5] * self.state_dim
    
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
