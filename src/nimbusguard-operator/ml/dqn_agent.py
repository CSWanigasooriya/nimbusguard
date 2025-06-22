# DQN Agent for Kubernetes Autoscaling
# ============================================================================
# KServe-based implementation for production use
# ============================================================================

import asyncio
import aiohttp
import json
import logging
import time
from collections import deque
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from .reward_system import RewardSystem, RewardComponents
from .state_representation import EnvironmentState, ScalingActions

LOG = logging.getLogger(__name__)

# DQN Metrics for Grafana visualization
dqn_decisions = Counter('nimbusguard_dqn_decisions_total', 'Total decisions made', ['action', 'decision_type'])
dqn_rewards = Histogram('nimbusguard_dqn_rewards', 'Reward values', ['reward_type'], 
                       buckets=(-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 5.0))
dqn_q_values = Histogram('nimbusguard_dqn_q_values', 'Q-values from decisions', 
                        buckets=(-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0, 20.0, 50.0))
kserve_requests = Counter('nimbusguard_kserve_requests_total', 'Total KServe requests', ['status'])
kserve_latency = Histogram('nimbusguard_kserve_latency_seconds', 'KServe request latency')


class KServeModelClient:
    """Async client for KServe model inference"""
    
    def __init__(self, endpoint_url: str, timeout: int = 30):
        self.endpoint_url = endpoint_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def predict(self, state_vector: List[float]) -> Dict[str, Any]:
        """Make prediction using KServe endpoint"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
            
        # Ensure state_vector is JSON serializable
        if hasattr(state_vector, 'tolist'):
            state_vector = state_vector.tolist()
        elif isinstance(state_vector, np.ndarray):
            state_vector = state_vector.tolist()
            
        payload = {
            "instances": [state_vector]
        }
        
        start_time = time.time()
        try:
            # Ensure we're hitting the predict endpoint
            predict_url = f"{self.endpoint_url}/predict" if not self.endpoint_url.endswith('/predict') else self.endpoint_url
            async with self.session.post(
                predict_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                response.raise_for_status()
                result = await response.json()
                
                # Record metrics
                latency = time.time() - start_time
                kserve_latency.observe(latency)
                kserve_requests.labels(status='success').inc()
                
                return result
                
        except aiohttp.ClientError as e:
            kserve_requests.labels(status='error').inc()
            LOG.error(f"KServe prediction failed: {e}")
            raise
            
    async def health_check(self) -> bool:
        """Check if KServe endpoint is healthy"""
        if not self.session:
            return False
            
        try:
            # Check the health endpoint
            health_url = f"{self.endpoint_url.replace('/predict', '')}/health"
            async with self.session.get(health_url) as response:
                return response.status == 200
        except:
            # Fallback: try the root endpoint
            try:
                async with self.session.get(f"{self.endpoint_url.replace(':predict', '')}") as response:
                    return response.status in [200, 404]  # 404 is OK for root endpoint
            except:
                return False


class DQNAgent:
    """
    DQN Agent for Kubernetes autoscaling using KServe model serving
    """

    def __init__(self,
                 kserve_endpoint: str,
                 model_name: str = "nimbusguard-dqn",
                 state_dim: int = 11,
                 action_dim: int = 5,
                 confidence_threshold: float = 0.7,
                 health_check_interval: int = 300):
        """
        Initialize DQN Agent
        
        Args:
            kserve_endpoint: KServe model serving endpoint URL (required)
            model_name: Name of the model in KServe
            state_dim: Dimension of state vector
            action_dim: Number of possible actions
            confidence_threshold: Minimum confidence for taking action
            health_check_interval: Seconds between health checks
        """
        if not kserve_endpoint:
            raise ValueError("KServe endpoint is required")
            
        self.kserve_endpoint = kserve_endpoint
        self.model_name = model_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.confidence_threshold = confidence_threshold
        self.health_check_interval = health_check_interval
        
        # KServe state tracking
        self.kserve_available = False
        self.last_health_check = 0
        self.total_requests = 0
        self.failed_requests = 0
        self.total_decisions = 0
        
        # Reward system for experience evaluation
        self.reward_system = RewardSystem()
        
        # Performance tracking
        self.recent_latencies = deque(maxlen=100)
        self.recent_confidences = deque(maxlen=100)
        
        # Async lock for concurrent access
        self._kserve_lock = asyncio.Lock()
        
        LOG.info(f"DQN Agent initialized with KServe")
        LOG.info(f"Endpoint: {self.kserve_endpoint}")
        LOG.info(f"Model: {self.model_name}")
        LOG.info(f"Confidence threshold: {self.confidence_threshold}")
        
        # Start metrics server on port 8090 (only start once)
        if not hasattr(DQNAgent, '_metrics_server_started'):
            try:
                start_http_server(8090)
                DQNAgent._metrics_server_started = True
                LOG.info("Started Prometheus metrics server on port 8090")
            except Exception as e:
                LOG.warning(f"Failed to start metrics server: {e}")

    async def select_action(self,
                           state: EnvironmentState,
                           force_valid: bool = True) -> Tuple[ScalingActions, Dict[str, Any]]:
        """
        Select action using KServe model inference
        """
        state_vector = state.to_dqn_input()

        # Validate state dimensions
        if len(state_vector) != self.state_dim:
            LOG.error(f"State dimension mismatch: Expected {self.state_dim}, got {len(state_vector)}")
            return ScalingActions.NO_ACTION, {"error": "state_dimension_mismatch", "prediction_source": "error"}

        async with self._kserve_lock:
            # Check if we should validate KServe health
            current_time = time.time()
            if current_time - self.last_health_check > self.health_check_interval:
                await self._validate_kserve_health()
                
            # Make prediction using KServe
            if not self.kserve_available:
                LOG.error("KServe endpoint not available")
                return ScalingActions.NO_ACTION, {
                    "error": "kserve_unavailable", 
                    "prediction_source": "error",
                    "endpoint": self.kserve_endpoint
                }
                
            try:
                action, metadata = await self._predict_with_kserve(state, force_valid)
                metadata["prediction_source"] = "kserve"
                metadata["endpoint"] = self.kserve_endpoint
                
                # Track confidence
                confidence = metadata.get("confidence", metadata.get("max_q_value", 0.0))
                self.recent_confidences.append(confidence)
                
                # Check confidence threshold
                if confidence < self.confidence_threshold:
                    LOG.warning(f"Confidence {confidence:.3f} below threshold {self.confidence_threshold}")
                    return ScalingActions.NO_ACTION, {
                        **metadata,
                        "reason": f"low_confidence_{confidence:.3f}",
                        "action_overridden": action.name
                    }
                
                self.total_requests += 1
                self.total_decisions += 1
                
                return action, metadata
                
            except Exception as e:
                self.failed_requests += 1
                LOG.error(f"KServe prediction failed: {e}")
                return ScalingActions.NO_ACTION, {
                    "error": str(e), 
                    "prediction_source": "kserve_error",
                    "endpoint": self.kserve_endpoint
                }

    async def _validate_kserve_health(self):
        """Validate KServe endpoint health"""
        try:
            async with KServeModelClient(self.kserve_endpoint) as client:
                self.kserve_available = await client.health_check()
                self.last_health_check = time.time()
                
                if self.kserve_available:
                    LOG.debug("KServe endpoint health check passed")
                else:
                    LOG.warning("KServe endpoint health check failed")
                    
        except Exception as e:
            LOG.error(f"KServe health check error: {e}")
            self.kserve_available = False

    async def _predict_with_kserve(self, 
                                   state: EnvironmentState, 
                                   force_valid: bool) -> Tuple[ScalingActions, Dict[str, Any]]:
        """Make prediction using KServe endpoint"""
        state_vector = state.to_dqn_input()
        
        start_time = time.time()
        
        async with KServeModelClient(self.kserve_endpoint) as client:
            result = await client.predict(state_vector)
            
            latency = time.time() - start_time
            self.recent_latencies.append(latency)
            
            # Parse KServe response - handle different response formats
            if 'predictions' in result:
                # Standard KServe format with Q-values
                q_values = result['predictions'][0]
                action, metadata = self._process_q_values(q_values, state, force_valid)
                metadata["kserve_response"] = result
                metadata["latency"] = latency
                return action, metadata
                
            elif 'action' in result:
                # Custom transformer response format
                action_name = result['action']
                action_mapping = {
                    "SCALE_DOWN_2": ScalingActions.SCALE_DOWN_2,
                    "SCALE_DOWN_1": ScalingActions.SCALE_DOWN_1,
                    "NO_ACTION": ScalingActions.NO_ACTION,
                    "SCALE_UP_1": ScalingActions.SCALE_UP_1,
                    "SCALE_UP_2": ScalingActions.SCALE_UP_2
                }
                action = action_mapping.get(action_name, ScalingActions.NO_ACTION)
                
                metadata = {
                    "timestamp": time.time(),
                    "q_values": result.get('q_values', []),
                    "confidence": result.get('confidence', 0.0),
                    "kserve_response": result,
                    "latency": latency
                }
                
                return action, metadata
            else:
                raise ValueError(f"Unexpected KServe response format: {result}")

    def _process_q_values(self, q_values: List[float], state: EnvironmentState, force_valid: bool) -> Tuple[ScalingActions, Dict[str, Any]]:
        """Process Q-values to select action"""
        valid_actions = state.get_valid_actions() if force_valid else list(ScalingActions)
        
        if force_valid:
            valid_indices = [action.value for action in valid_actions]
            masked_q_values = np.full(len(q_values), float('-inf'))
            for idx in valid_indices:
                if idx < len(q_values):
                    masked_q_values[idx] = q_values[idx]
            action_idx = np.argmax(masked_q_values)
        else:
            action_idx = np.argmax(q_values)
            
        action = ScalingActions(action_idx)
        
        metadata = {
            "timestamp": time.time(),
            "q_values": q_values,
            "max_q_value": float(np.max(q_values)),
            "confidence": float(np.max(q_values)),  # Use max Q-value as confidence
            "valid_actions": [a.name for a in valid_actions],
            "selected_action_q_value": float(q_values[action_idx]) if action_idx < len(q_values) else 0.0
        }
        
        # Record metrics
        dqn_decisions.labels(action=action.name, decision_type="kserve_inference").inc()
        dqn_q_values.observe(metadata["max_q_value"])
        
        return action, metadata

    def evaluate_decision(self, *args, **kwargs) -> RewardComponents:
        """Evaluate a scaling decision by calculating its reward"""
        return self.reward_system.calculate_reward(*args, **kwargs)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        success_rate = ((self.total_requests - self.failed_requests) / self.total_requests) if self.total_requests > 0 else 0.0
        avg_latency = np.mean(self.recent_latencies) if self.recent_latencies else 0.0
        avg_confidence = np.mean(self.recent_confidences) if self.recent_confidences else 0.0
        
        return {
            "total_decisions": self.total_decisions,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "avg_latency_seconds": avg_latency,
            "avg_confidence": avg_confidence,
            "confidence_threshold": self.confidence_threshold,
            "kserve_available": self.kserve_available,
            "endpoint": self.kserve_endpoint,
            "model_name": self.model_name
        }

    def get_kserve_metrics(self) -> Dict[str, Any]:
        """Get KServe-specific metrics"""
        return {
            "kserve_available": self.kserve_available,
            "endpoint": self.kserve_endpoint,
            "model_name": self.model_name,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": ((self.total_requests - self.failed_requests) / self.total_requests) if self.total_requests > 0 else 0.0,
            "last_health_check": self.last_health_check,
            "avg_latency": np.mean(self.recent_latencies) if self.recent_latencies else 0.0
        }

    def update_kserve_endpoint(self, new_endpoint: str):
        """Update KServe endpoint and reset health status"""
        self.kserve_endpoint = new_endpoint
        self.kserve_available = False
        self.last_health_check = 0
        LOG.info(f"Updated KServe endpoint to: {new_endpoint}")

    def update_confidence_threshold(self, new_threshold: float):
        """Update confidence threshold for decision making"""
        self.confidence_threshold = new_threshold
        LOG.info(f"Updated confidence threshold to: {new_threshold}")


# Factory function for creating the agent
async def create_dqn_agent(config: Dict[str, Any]) -> DQNAgent:
    """
    Factory function to create DQN Agent
    
    Args:
        config: Configuration dictionary with KServe settings
        
    Returns:
        Configured DQNAgent instance
        
    Raises:
        ValueError: If required KServe configuration is missing
    """
    
    # Extract required KServe configuration
    kserve_endpoint = config.get('kserve_endpoint')
    if not kserve_endpoint:
        raise ValueError("kserve_endpoint is required")
    
    agent = DQNAgent(
        kserve_endpoint=kserve_endpoint,
        model_name=config.get('model_name', 'nimbusguard-dqn'),
        state_dim=config.get('state_dim', 11),
        action_dim=config.get('action_dim', 5),
        confidence_threshold=config.get('confidence_threshold', 0.7),
        health_check_interval=config.get('health_check_interval', 300)
    )
    
    # Validate KServe connection
    await agent._validate_kserve_health()
    
    if agent.kserve_available:
        LOG.info("DQN agent initialized successfully - KServe endpoint is healthy")
    else:
        LOG.warning("DQN agent initialized but KServe endpoint is not available")
        
    return agent
