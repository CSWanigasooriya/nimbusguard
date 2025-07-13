"""
Direct Kubernetes deployment scaler.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Tuple

from k8s.client import KubernetesClient

logger = logging.getLogger(__name__)

class DirectScaler:
    """Direct deployment scaler that bypasses KEDA/HPA."""
    
    def __init__(self, k8s_client: KubernetesClient, scaling_config):
        self.k8s_client = k8s_client
        self.config = scaling_config
        self.logger = logger
        
        # Scaling constraints
        self.min_replicas = scaling_config.min_replicas
        self.max_replicas = scaling_config.max_replicas
        
        # Target deployment
        self.target_deployment = scaling_config.target_deployment
        self.target_namespace = scaling_config.target_namespace
        
        self.logger.info(f"DirectScaler initialized: target={self.target_deployment}, "
                        f"range=[{self.min_replicas}, {self.max_replicas}]")
    
    async def scale_to(self, desired_replicas: int, reason: str = "DQN decision") -> Tuple[bool, Dict[str, Any]]:
        """
        Scale deployment to desired replica count with safety checks.
        
        Returns:
            Tuple of (success, result_info)
        """
        try:
            # Apply safety constraints
            constrained_replicas = max(self.min_replicas, min(self.max_replicas, desired_replicas))
            
            if constrained_replicas != desired_replicas:
                self.logger.warning(f"Replica count constrained: {desired_replicas} → {constrained_replicas}")
            
            # Get current deployment state
            current_deployment = await self.k8s_client.get_deployment(
                self.target_deployment, 
                self.target_namespace
            )
            
            if not current_deployment:
                return False, {"error": "Target deployment not found"}
            
            current_replicas = current_deployment['replicas']
            
            # Check if scaling is needed
            if current_replicas == constrained_replicas:
                self.logger.debug(f"No scaling needed: already at {current_replicas} replicas")
                return True, {
                    "action": "no_change",
                    "current_replicas": current_replicas,
                    "desired_replicas": constrained_replicas,
                    "reason": reason
                }
            
            # Perform scaling
            self.logger.info(f"Scaling {self.target_deployment}: {current_replicas} → {constrained_replicas} ({reason})")
            
            success = await self.k8s_client.scale_deployment(
                self.target_deployment,
                self.target_namespace,
                constrained_replicas
            )
            
            if success:
                action = "scale_up" if constrained_replicas > current_replicas else "scale_down"
                
                result = {
                    "action": action,
                    "previous_replicas": current_replicas,
                    "current_replicas": constrained_replicas,
                    "desired_replicas": desired_replicas,
                    "constrained": constrained_replicas != desired_replicas,
                    "reason": reason,
                    "timestamp": asyncio.get_event_loop().time()
                }
                
                self.logger.info(f"✅ Scaling successful: {action} to {constrained_replicas} replicas")
                return True, result
            else:
                return False, {"error": "Kubernetes API scaling failed"}
                
        except Exception as e:
            self.logger.error(f"❌ Scaling failed: {e}")
            return False, {"error": str(e)}
    
    async def get_current_state(self) -> Optional[Dict[str, Any]]:
        """Get current deployment state and metrics."""
        try:
            metrics = await self.k8s_client.get_deployment_metrics(
                self.target_deployment,
                self.target_namespace
            )
            
            if not metrics:
                return None
            
            deployment = metrics['deployment']
            pods = metrics['pods']
            
            return {
                'replicas': deployment['replicas'],
                'ready_replicas': deployment['ready_replicas'],
                'available_replicas': deployment['available_replicas'],
                'unavailable_replicas': deployment['unavailable_replicas'],
                'health_score': metrics['health_score'],
                'pods': pods,
                'generation': deployment['generation'],
                'observed_generation': deployment['observed_generation'],
                'rollout_complete': deployment['generation'] == deployment['observed_generation']
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get current state: {e}")
            return None
    
    async def wait_for_stability(self, timeout: int = 60) -> bool:
        """Wait for deployment to reach stable state after scaling."""
        try:
            return await self.k8s_client.wait_for_rollout(
                self.target_deployment,
                self.target_namespace,
                timeout
            )
        except Exception as e:
            self.logger.error(f"Error waiting for stability: {e}")
            return False
    
    def validate_scaling_decision(self, current_replicas: int, desired_replicas: int, 
                                metrics: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate scaling decision based on current state and metrics.
        
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check replica bounds
        if desired_replicas < self.min_replicas:
            return False, f"Below minimum replicas ({self.min_replicas})"
        
        if desired_replicas > self.max_replicas:
            return False, f"Above maximum replicas ({self.max_replicas})"
        
        # Check for extreme scaling
        replica_change = abs(desired_replicas - current_replicas)
        if replica_change > 5:
            return False, f"Extreme scaling detected: {replica_change} replicas"
        
        # Check deployment health
        if metrics and metrics.get('health_score', 1.0) < 0.5:
            if desired_replicas < current_replicas:
                return False, "Cannot scale down: deployment unhealthy"
        
        # Check for rapid scaling (prevent oscillation)
        # This could be enhanced with a scaling history tracker
        
        return True, "Scaling decision validated"
    
    async def emergency_scale(self, replicas: int, reason: str = "Emergency scaling") -> bool:
        """Emergency scaling that bypasses some safety checks."""
        try:
            self.logger.warning(f"🚨 Emergency scaling to {replicas} replicas: {reason}")
            
            # Apply only hard limits
            constrained_replicas = max(1, min(self.max_replicas * 2, replicas))  # Allow 2x max in emergency
            
            success = await self.k8s_client.scale_deployment(
                self.target_deployment,
                self.target_namespace,
                constrained_replicas
            )
            
            if success:
                self.logger.info(f"✅ Emergency scaling completed: {constrained_replicas} replicas")
            else:
                self.logger.error("❌ Emergency scaling failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Emergency scaling error: {e}")
            return False
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling configuration and statistics."""
        return {
            'target_deployment': self.target_deployment,
            'target_namespace': self.target_namespace,
            'min_replicas': self.min_replicas,
            'max_replicas': self.max_replicas,
            'scaling_method': 'direct_k8s_api'
        } 