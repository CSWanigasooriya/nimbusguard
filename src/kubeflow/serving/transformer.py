#!/usr/bin/env python3
"""
KServe Model Transformer for NimbusGuard DQN
Handles preprocessing and postprocessing for model inference
"""

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

import aiohttp
import numpy as np
import pandas as pd
from aiohttp import web, ClientSession
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
transform_requests = Counter('nimbusguard_transformer_requests_total', 'Transform requests', ['type', 'status'])
transform_duration = Histogram('nimbusguard_transformer_duration_seconds', 'Transform duration', ['type'])
prometheus_query_duration = Histogram('nimbusguard_prometheus_query_duration_seconds', 'Prometheus query duration')
model_predictions = Counter('nimbusguard_model_predictions_total', 'Model predictions', ['action'])
state_vector_values = Histogram('nimbusguard_state_vector_values', 'State vector feature values', ['feature'])


class PrometheusClient:
    """Client for collecting metrics from Prometheus"""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip('/')
        self.session: Optional[ClientSession] = None
        
    async def __aenter__(self):
        self.session = ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def query_metric(self, query: str) -> Optional[float]:
        """Query single metric value from Prometheus"""
        if not self.session:
            raise RuntimeError("Client not initialized")
            
        try:
            params = {'query': query}
            async with self.session.get(
                f"{self.endpoint}/api/v1/query",
                params=params
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                if data['status'] == 'success' and data['data']['result']:
                    value = float(data['data']['result'][0]['value'][1])
                    return value
                return 0.0
                
        except Exception as e:
            logger.warning(f"Failed to query metric '{query}': {e}")
            return 0.0


class StateVectorBuilder:
    """Builds DQN state vectors from cluster metrics"""
    
    def __init__(self, prometheus_client: PrometheusClient):
        self.prometheus = prometheus_client
        
        # Define the 11-dimensional state vector mapping
        self.metric_queries = {
            'cpu_utilization': 'avg(rate(container_cpu_usage_seconds_total{namespace="nimbusguard"}[5m]))',
            'memory_utilization': 'avg(container_memory_working_set_bytes{namespace="nimbusguard"}) / avg(container_spec_memory_limit_bytes{namespace="nimbusguard"})',
            'network_io_rate': 'rate(container_network_transmit_bytes_total{namespace="nimbusguard"}[5m])',
            'request_rate': 'rate(http_requests_total{service="consumer-workload"}[5m])',
            'pod_count': 'count(kube_pod_info{namespace="nimbusguard"})',
            'response_time_p95': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{service="consumer-workload"}[5m]))',
            'error_rate': 'rate(http_requests_total{service="consumer-workload",status=~"5.."}[5m]) / rate(http_requests_total{service="consumer-workload"}[5m])',
            'queue_depth': 'avg(http_requests_in_flight{service="consumer-workload"})',
            'cpu_throttling': 'rate(container_cpu_cfs_throttled_seconds_total{namespace="nimbusguard"}[5m])',
            'memory_pressure': 'avg(container_memory_working_set_bytes{namespace="nimbusguard"}) / avg(node_memory_MemTotal_bytes)',
            'node_utilization': 'avg(1 - rate(node_cpu_seconds_total{mode="idle"}[5m]))'
        }
        
        # Feature normalization parameters (learned from training data)
        self.normalization_params = {
            'cpu_utilization': {'min': 0.0, 'max': 1.0},
            'memory_utilization': {'min': 0.0, 'max': 1.0}, 
            'network_io_rate': {'min': 0.0, 'max': 1000000.0},  # bytes/sec
            'request_rate': {'min': 0.0, 'max': 1000.0},        # requests/sec
            'pod_count': {'min': 1.0, 'max': 50.0},
            'response_time_p95': {'min': 0.0, 'max': 5.0},      # seconds
            'error_rate': {'min': 0.0, 'max': 0.1},
            'queue_depth': {'min': 0.0, 'max': 100.0},
            'cpu_throttling': {'min': 0.0, 'max': 1.0},
            'memory_pressure': {'min': 0.0, 'max': 1.0},
            'node_utilization': {'min': 0.0, 'max': 1.0}
        }
        
    async def build_state_vector(self, custom_state: Dict[str, Any] = None) -> List[float]:
        """Build normalized state vector from current cluster metrics"""
        with prometheus_query_duration.time():
            if custom_state:
                # Use provided state data
                state_values = custom_state
            else:
                # Collect current metrics from Prometheus
                state_values = {}
                for feature, query in self.metric_queries.items():
                    value = await self.prometheus.query_metric(query)
                    state_values[feature] = value if value is not None else 0.0
        
        # Build and normalize state vector
        state_vector = []
        for feature in self.metric_queries.keys():
            raw_value = state_values.get(feature, 0.0)
            normalized_value = self._normalize_feature(feature, raw_value)
            state_vector.append(normalized_value)
            
            # Record metrics for monitoring
            state_vector_values.labels(feature=feature).observe(normalized_value)
            
        logger.debug(f"Built state vector: {state_vector}")
        return state_vector
        
    def _normalize_feature(self, feature: str, value: float) -> float:
        """Normalize feature value to [0, 1] range"""
        params = self.normalization_params.get(feature, {'min': 0.0, 'max': 1.0})
        min_val, max_val = params['min'], params['max']
        
        # Clamp and normalize
        clamped = max(min_val, min(value, max_val))
        if max_val > min_val:
            normalized = (clamped - min_val) / (max_val - min_val)
        else:
            normalized = 0.0
            
        return float(normalized)


class ActionMapper:
    """Maps model outputs to NimbusGuard actions"""
    
    def __init__(self):
        self.action_mapping = {
            0: "SCALE_DOWN_2",
            1: "SCALE_DOWN_1", 
            2: "NO_ACTION",
            3: "SCALE_UP_1",
            4: "SCALE_UP_2"
        }
        
        self.replica_changes = {
            0: -2,  # SCALE_DOWN_2
            1: -1,  # SCALE_DOWN_1
            2: 0,   # NO_ACTION
            3: 1,   # SCALE_UP_1
            4: 2    # SCALE_UP_2
        }
        
    def map_prediction(self, q_values: List[float], current_replicas: int = 3) -> Dict[str, Any]:
        """Map Q-values to scaling action"""
        best_action_idx = int(np.argmax(q_values))
        action_name = self.action_mapping[best_action_idx]
        replica_change = self.replica_changes[best_action_idx]
        confidence = float(max(q_values))
        
        # Calculate target replicas with bounds checking
        target_replicas = max(1, min(50, current_replicas + replica_change))
        
        # Record prediction metrics
        model_predictions.labels(action=action_name).inc()
        
        return {
            "action": action_name,
            "action_index": best_action_idx,
            "q_values": q_values,
            "confidence": confidence,
            "current_replicas": current_replicas,
            "target_replicas": target_replicas,
            "replica_change": replica_change,
            "timestamp": datetime.now().isoformat()
        }


class NimbusGuardTransformer:
    """Main transformer class for KServe integration"""
    
    def __init__(self):
        self.prometheus_endpoint = os.getenv('PROMETHEUS_ENDPOINT', 
                                           'http://prometheus.nimbusguard.svc.cluster.local:9090')
        self.action_mapper = ActionMapper()
        
    async def initialize(self):
        """Initialize transformer components"""
        self.prometheus_client = PrometheusClient(self.prometheus_endpoint)
        self.state_builder = StateVectorBuilder(self.prometheus_client)
        logger.info("Transformer initialized successfully")
        
    async def preprocess(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess request for model inference"""
        with transform_duration.labels(type='preprocess').time():
            try:
                # Check if cluster state is provided
                if 'cluster_state' in request_data:
                    cluster_state = request_data['cluster_state']
                    async with self.prometheus_client:
                        state_vector = await self.state_builder.build_state_vector(cluster_state)
                elif 'instances' in request_data:
                    # Direct state vector provided
                    state_vector = request_data['instances'][0]
                else:
                    # Collect current cluster state
                    async with self.prometheus_client:
                        state_vector = await self.state_builder.build_state_vector()
                
                # Validate state vector dimensions
                if len(state_vector) != 11:
                    logger.warning(f"State vector dimension mismatch: {len(state_vector)} != 11")
                    state_vector = state_vector[:11] + [0.0] * max(0, 11 - len(state_vector))
                
                model_input = {"instances": [state_vector]}
                
                transform_requests.labels(type='preprocess', status='success').inc()
                logger.debug(f"Preprocessed request: {len(state_vector)} features")
                
                return model_input
                
            except Exception as e:
                transform_requests.labels(type='preprocess', status='error').inc()
                logger.error(f"Preprocessing failed: {e}")
                # Return safe default state
                return {"instances": [[0.5] * 11]}
                
    async def postprocess(self, model_output: Dict[str, Any], 
                         original_request: Dict[str, Any] = None) -> Dict[str, Any]:
        """Postprocess model output for NimbusGuard"""
        with transform_duration.labels(type='postprocess').time():
            try:
                # Extract predictions from model output
                if 'predictions' in model_output:
                    q_values = model_output['predictions'][0]
                elif 'outputs' in model_output:
                    # Handle different model output formats
                    q_values = model_output['outputs'][0]
                else:
                    raise ValueError(f"Unexpected model output format: {model_output}")
                
                # Get current replica count if provided
                current_replicas = 3  # Default
                if original_request and 'current_replicas' in original_request:
                    current_replicas = original_request['current_replicas']
                
                # Map to action
                result = self.action_mapper.map_prediction(q_values, current_replicas)
                
                transform_requests.labels(type='postprocess', status='success').inc()
                logger.info(f"Action prediction: {result['action']} (confidence: {result['confidence']:.3f})")
                
                return result
                
            except Exception as e:
                transform_requests.labels(type='postprocess', status='error').inc()
                logger.error(f"Postprocessing failed: {e}")
                return {
                    "action": "NO_ACTION",
                    "action_index": 2,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }


# HTTP Server for KServe integration
async def create_app() -> web.Application:
    """Create aiohttp application with transformer endpoints"""
    transformer = NimbusGuardTransformer()
    await transformer.initialize()
    
    async def health_handler(request):
        """Health check endpoint"""
        return web.json_response({"status": "healthy", "timestamp": datetime.now().isoformat()})
    
    async def preprocess_handler(request):
        """Preprocessing endpoint"""
        try:
            request_data = await request.json()
            result = await transformer.preprocess(request_data)
            return web.json_response(result)
        except Exception as e:
            logger.error(f"Preprocess handler error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def postprocess_handler(request):
        """Postprocessing endpoint"""
        try:
            request_data = await request.json()
            model_output = request_data.get('model_output', {})
            original_request = request_data.get('original_request', {})
            
            result = await transformer.postprocess(model_output, original_request)
            return web.json_response(result)
        except Exception as e:
            logger.error(f"Postprocess handler error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def predict_handler(request):
        """Full prediction pipeline endpoint"""
        try:
            request_data = await request.json()
            
            # Preprocess
            model_input = await transformer.preprocess(request_data)
            
            # For testing, simulate model prediction
            # In production, this would call the actual model
            state_vector = model_input['instances'][0]
            simulated_q_values = [0.1 + 0.1 * i + 0.05 * sum(state_vector) for i in range(5)]
            model_output = {"predictions": [simulated_q_values]}
            
            # Postprocess
            result = await transformer.postprocess(model_output, request_data)
            
            return web.json_response(result)
            
        except Exception as e:
            logger.error(f"Predict handler error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    # Create application
    app = web.Application()
    app.router.add_get('/health', health_handler)
    app.router.add_post('/preprocess', preprocess_handler)
    app.router.add_post('/postprocess', postprocess_handler)
    app.router.add_post('/predict', predict_handler)  # For testing
    
    return app


async def main():
    """Main function to run the transformer server"""
    # Start Prometheus metrics server
    try:
        start_http_server(8090)
        logger.info("Prometheus metrics server started on port 8090")
    except Exception as e:
        logger.warning(f"Failed to start metrics server: {e}")
    
    # Create and start HTTP server
    app = await create_app()
    
    # Graceful shutdown handling
    async def shutdown_handler():
        logger.info("Shutting down transformer server...")
    
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown_handler()))
    
    # Start server
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()
    
    logger.info("NimbusGuard Model Transformer started on port 8080")
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logger.info("Server shutdown requested")
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
