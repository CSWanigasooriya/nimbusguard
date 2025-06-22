"""
Enhanced DQN Agent with Kubeflow KServe Integration
This module extends the existing DQN agent to work with KServe model serving
"""

import asyncio
import aiohttp
import json
import logging
import os
from typing import Dict, Any, Optional, Tuple, List
import torch
import numpy as np

from dqn_agent import DQNAgent
from state_representation import EnvironmentState, ScalingActions

LOG = logging.getLogger(__name__)


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
            
        payload = {
            "instances": [state_vector]
        }
        
        try:
            async with self.session.post(
                f"{self.endpoint_url}:predict",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result
                
        except aiohttp.ClientError as e:
            LOG.error(f"KServe prediction failed: {e}")
            raise
            
    async def health_check(self) -> bool:
        """Check if KServe endpoint is healthy"""
        if not self.session:
            return False
            
        try:
            async with self.session.get(f"{self.endpoint_url}") as response:
                return response.status == 200
        except:
            return False


class KubeflowDQNAgent(DQNAgent):
    """Enhanced DQN Agent with Kubeflow KServe integration"""
    
    def __init__(self, 
                 kserve_endpoint: Optional[str] = None,
                 fallback_to_local: bool = True,
                 model_validation_interval: int = 300,  # 5 minutes
                 *args, **kwargs):
        """
        Initialize Kubeflow-integrated DQN Agent
        
        Args:
            kserve_endpoint: KServe model serving endpoint URL
            fallback_to_local: Whether to fallback to local model if KServe fails
            model_validation_interval: Seconds between KServe health checks
        """
        super().__init__(*args, **kwargs)
        
        self.kserve_endpoint = kserve_endpoint
        self.fallback_to_local = fallback_to_local
        self.model_validation_interval = model_validation_interval
        
        # KServe state tracking
        self.kserve_available = False
        self.last_health_check = 0
        self.kserve_requests = 0
        self.kserve_failures = 0
        
        # Async lock for concurrent access
        self._kserve_lock = asyncio.Lock()
        
        LOG.info(f"KubeflowDQNAgent initialized with KServe endpoint: {kserve_endpoint}")
        
    async def select_action_async(self, 
                                  state: EnvironmentState,
                                  training: bool = False,
                                  force_valid: bool = True) -> Tuple[ScalingActions, Dict[str, Any]]:
        """
        Async action selection with KServe integration
        
        Tries KServe first, falls back to local model if configured
        """
        if not self.kserve_endpoint or training:
            # Use local model for training or when KServe not configured
            return self.select_action(state, training=training, force_valid=force_valid)
            
        async with self._kserve_lock:
            # Check if we should validate KServe health
            current_time = asyncio.get_event_loop().time()
            if current_time - self.last_health_check > self.model_validation_interval:
                await self._validate_kserve_health()
                
            # Try KServe prediction if available
            if self.kserve_available:
                try:
                    action, metadata = await self._predict_with_kserve(state, force_valid)
                    metadata["prediction_source"] = "kserve"
                    self.kserve_requests += 1
                    return action, metadata
                    
                except Exception as e:
                    LOG.warning(f"KServe prediction failed: {e}")
                    self.kserve_failures += 1
                    self.kserve_available = False
                    
                    if not self.fallback_to_local:
                        raise
                        
            # Fallback to local model
            if self.fallback_to_local:
                action, metadata = self.select_action(state, training=training, force_valid=force_valid)
                metadata["prediction_source"] = "local_fallback"
                metadata["kserve_failure_reason"] = "endpoint_unavailable"
                return action, metadata
            else:
                raise RuntimeError("KServe unavailable and local fallback disabled")
                
    async def _validate_kserve_health(self):
        """Validate KServe endpoint health"""
        try:
            async with KServeModelClient(self.kserve_endpoint) as client:
                self.kserve_available = await client.health_check()
                self.last_health_check = asyncio.get_event_loop().time()
                
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
        
        async with KServeModelClient(self.kserve_endpoint) as client:
            result = await client.predict(state_vector)
            
            # Parse KServe response
            if 'predictions' in result:
                q_values = result['predictions'][0]
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
                    "timestamp": asyncio.get_event_loop().time(),
                    "q_values": result.get('q_values', []),
                    "confidence": result.get('confidence', 0.0),
                    "kserve_response": result
                }
                
                return action, metadata
            else:
                raise ValueError(f"Unexpected KServe response format: {result}")
                
            # Handle raw Q-values response
            valid_actions = state.get_valid_actions() if force_valid else list(ScalingActions)
            
            if force_valid:
                valid_indices = [action.value for action in valid_actions]
                masked_q_values = np.full(len(q_values), float('-inf'))
                for idx in valid_indices:
                    masked_q_values[idx] = q_values[idx]
                action_idx = np.argmax(masked_q_values)
            else:
                action_idx = np.argmax(q_values)
                
            action = ScalingActions(action_idx)
            
            metadata = {
                "timestamp": asyncio.get_event_loop().time(),
                "q_values": q_values,
                "max_q_value": float(np.max(q_values)),
                "valid_actions": [a.name for a in valid_actions],
                "kserve_response": result
            }
            
            return action, metadata
            
    def get_kserve_metrics(self) -> Dict[str, Any]:
        """Get KServe integration metrics"""
        total_requests = self.kserve_requests + self.kserve_failures
        success_rate = (self.kserve_requests / total_requests) if total_requests > 0 else 0.0
        
        return {
            "kserve_available": self.kserve_available,
            "kserve_requests": self.kserve_requests,
            "kserve_failures": self.kserve_failures,
            "kserve_success_rate": success_rate,
            "last_health_check": self.last_health_check,
            "endpoint": self.kserve_endpoint
        }
        
    def update_kserve_endpoint(self, new_endpoint: str):
        """Update KServe endpoint and reset health status"""
        self.kserve_endpoint = new_endpoint
        self.kserve_available = False
        self.last_health_check = 0
        LOG.info(f"Updated KServe endpoint to: {new_endpoint}")


class KubeflowModelManager:
    """Manages model lifecycle in Kubeflow environment"""
    
    def __init__(self, 
                 pipeline_client_endpoint: str,
                 model_registry_endpoint: str,
                 namespace: str = "nimbusguard-ml"):
        self.pipeline_client_endpoint = pipeline_client_endpoint
        self.model_registry_endpoint = model_registry_endpoint
        self.namespace = namespace
        
    async def trigger_training_pipeline(self, 
                                        hyperparameters: Dict[str, Any] = None,
                                        experiment_name: str = "nimbusguard-dqn-training") -> str:
        """Trigger DQN training pipeline execution"""
        # This would integrate with Kubeflow Pipelines client
        # Implementation depends on your pipeline setup
        pass
        
    async def get_best_model_version(self) -> Dict[str, Any]:
        """Get the best performing model from experiments"""
        # This would query Katib for best hyperparameters
        # and model registry for best model version
        pass
        
    async def deploy_model_to_kserve(self, 
                                     model_version: str,
                                     serving_config: Dict[str, Any] = None) -> str:
        """Deploy model version to KServe"""
        # This would create/update KServe InferenceService
        pass


# Integration with existing operator
async def create_kubeflow_agent(config: Dict[str, Any]) -> KubeflowDQNAgent:
    """Factory function to create KubeflowDQNAgent with proper configuration"""
    
    # Extract KServe configuration from environment or config
    kserve_endpoint = config.get('kserve_endpoint') or os.getenv('KSERVE_ENDPOINT')
    
    if not kserve_endpoint:
        LOG.warning("No KServe endpoint configured, falling back to local-only mode")
        
    agent = KubeflowDQNAgent(
        kserve_endpoint=kserve_endpoint,
        fallback_to_local=config.get('fallback_to_local', True),
        model_validation_interval=config.get('model_validation_interval', 300),
        **config.get('dqn_params', {})
    )
    
    # Validate KServe connection if endpoint provided
    if kserve_endpoint:
        await agent._validate_kserve_health()
        if agent.kserve_available:
            LOG.info("KServe integration active and healthy")
        else:
            LOG.warning("KServe endpoint configured but not available")
            
    return agent
