"""
Deployment context manager for storing and tracking deployment information.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from kubernetes import client

logger = logging.getLogger(__name__)


class DeploymentContextManager:
    """
    Manages deployment context information for DQN decision making.
    """
    
    def __init__(self):
        """Initialize deployment context manager."""
        self.deployment_contexts = {}  # deployment_name -> context
        self.k8s_apps_v1 = client.AppsV1Api()
        
        logger.info("Initialized DeploymentContextManager")
    
    async def get_deployment_context(self, 
                                   deployment_name: str, 
                                   namespace: str) -> Dict[str, Any]:
        """
        Get or create deployment context for the specified deployment.
        
        Args:
            deployment_name: Name of the deployment
            namespace: Kubernetes namespace
            
        Returns:
            Deployment context dictionary
        """
        context_key = f"{namespace}/{deployment_name}"
        
        # Get existing context or create new one
        context = self.deployment_contexts.get(context_key, {})
        
        try:
            # Fetch current deployment info from Kubernetes
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            # Update context with current deployment info
            context.update({
                'deployment_name': deployment_name,
                'namespace': namespace,
                'current_replicas': deployment.status.replicas or 0,
                'ready_replicas': deployment.status.ready_replicas or 0,
                'available_replicas': deployment.status.available_replicas or 0,
                'updated_replicas': deployment.status.updated_replicas or 0,
                'last_updated': datetime.now().isoformat(),
            })
            
            # Extract resource requests from deployment spec
            resource_requests = self._extract_resource_requests(deployment)
            if resource_requests:
                context['resource_requests'] = resource_requests
            
            # Extract resource limits
            resource_limits = self._extract_resource_limits(deployment)
            if resource_limits:
                context['resource_limits'] = resource_limits
            
            # Extract other deployment metadata
            context.update({
                'labels': deployment.metadata.labels or {},
                'annotations': deployment.metadata.annotations or {},
                'creation_timestamp': deployment.metadata.creation_timestamp.isoformat() if deployment.metadata.creation_timestamp else None,
                'generation': deployment.metadata.generation,
                'observed_generation': deployment.status.observed_generation or 0,
            })
            
            # Initialize scaling history if not present
            if 'scaling_history' not in context:
                context['scaling_history'] = []
            
            # Initialize min/max replicas from HPA or defaults
            if 'min_replicas' not in context:
                context['min_replicas'] = self._get_min_replicas(deployment)
            if 'max_replicas' not in context:
                context['max_replicas'] = self._get_max_replicas(deployment)
            
            # Store updated context
            self.deployment_contexts[context_key] = context
            
            logger.debug(f"Updated context for {context_key}: "
                        f"replicas={context['current_replicas']}, "
                        f"ready={context['ready_replicas']}")
            
        except Exception as e:
            logger.error(f"Failed to get deployment context for {context_key}: {e}")
            
            # Return existing context or minimal default
            if not context:
                context = {
                    'deployment_name': deployment_name,
                    'namespace': namespace,
                    'current_replicas': 1,
                    'ready_replicas': 1,
                    'min_replicas': 1,
                    'max_replicas': 10,
                    'scaling_history': [],
                    'last_updated': datetime.now().isoformat(),
                    'error': str(e)
                }
                self.deployment_contexts[context_key] = context
        
        return context
    
    def _extract_resource_requests(self, deployment) -> Optional[Dict[str, float]]:
        """Extract resource requests from deployment spec."""
        try:
            containers = deployment.spec.template.spec.containers
            if not containers:
                return None
            
            # Get resources from first container (assuming single container pods)
            container = containers[0]
            if not container.resources or not container.resources.requests:
                return None
            
            requests = container.resources.requests
            resource_requests = {}
            
            # Parse CPU request
            if 'cpu' in requests:
                cpu_str = requests['cpu']
                resource_requests['cpu'] = self._parse_cpu_quantity(cpu_str)
            
            # Parse memory request
            if 'memory' in requests:
                memory_str = requests['memory']
                resource_requests['memory'] = self._parse_memory_quantity(memory_str)
            
            return resource_requests if resource_requests else None
            
        except Exception as e:
            logger.warning(f"Failed to extract resource requests: {e}")
            return None
    
    def _extract_resource_limits(self, deployment) -> Optional[Dict[str, float]]:
        """Extract resource limits from deployment spec."""
        try:
            containers = deployment.spec.template.spec.containers
            if not containers:
                return None
            
            container = containers[0]
            if not container.resources or not container.resources.limits:
                return None
            
            limits = container.resources.limits
            resource_limits = {}
            
            # Parse CPU limit
            if 'cpu' in limits:
                cpu_str = limits['cpu']
                resource_limits['cpu'] = self._parse_cpu_quantity(cpu_str)
            
            # Parse memory limit
            if 'memory' in limits:
                memory_str = limits['memory']
                resource_limits['memory'] = self._parse_memory_quantity(memory_str)
            
            return resource_limits if resource_limits else None
            
        except Exception as e:
            logger.warning(f"Failed to extract resource limits: {e}")
            return None
    
    def _parse_cpu_quantity(self, cpu_str: str) -> float:
        """Parse Kubernetes CPU quantity to float (cores)."""
        cpu_str = cpu_str.strip()
        
        if cpu_str.endswith('m'):
            # Millicores
            return float(cpu_str[:-1]) / 1000.0
        elif cpu_str.endswith('u'):
            # Microcores
            return float(cpu_str[:-1]) / 1000000.0
        elif cpu_str.endswith('n'):
            # Nanocores
            return float(cpu_str[:-1]) / 1000000000.0
        else:
            # Assume cores
            return float(cpu_str)
    
    def _parse_memory_quantity(self, memory_str: str) -> float:
        """Parse Kubernetes memory quantity to float (bytes)."""
        memory_str = memory_str.strip()
        
        # Binary suffixes
        if memory_str.endswith('Ki'):
            return float(memory_str[:-2]) * 1024
        elif memory_str.endswith('Mi'):
            return float(memory_str[:-2]) * 1024 * 1024
        elif memory_str.endswith('Gi'):
            return float(memory_str[:-2]) * 1024 * 1024 * 1024
        elif memory_str.endswith('Ti'):
            return float(memory_str[:-2]) * 1024 * 1024 * 1024 * 1024
        
        # Decimal suffixes
        elif memory_str.endswith('k') or memory_str.endswith('K'):
            return float(memory_str[:-1]) * 1000
        elif memory_str.endswith('M'):
            return float(memory_str[:-1]) * 1000 * 1000
        elif memory_str.endswith('G'):
            return float(memory_str[:-1]) * 1000 * 1000 * 1000
        elif memory_str.endswith('T'):
            return float(memory_str[:-1]) * 1000 * 1000 * 1000 * 1000
        
        # Assume bytes
        else:
            return float(memory_str)
    
    def _get_min_replicas(self, deployment) -> int:
        """Get minimum replicas from NimbusGuard annotations."""
        try:
            # Check NimbusGuard annotations for min replicas
            annotations = deployment.metadata.annotations or {}
            min_replicas_annotation = annotations.get('nimbusguard.io/min-replicas')
            if min_replicas_annotation:
                return int(min_replicas_annotation)
            
            # Fallback to HPA annotations if NimbusGuard not set
            hpa_min_annotation = annotations.get('autoscaling.alpha.kubernetes.io/min-replicas')
            if hpa_min_annotation:
                return int(hpa_min_annotation)
            
            # Default minimum
            return 1
            
        except Exception:
            return 1
    
    def _get_max_replicas(self, deployment) -> int:
        """Get maximum replicas from NimbusGuard annotations."""
        try:
            # Check NimbusGuard annotations for max replicas
            annotations = deployment.metadata.annotations or {}
            max_replicas_annotation = annotations.get('nimbusguard.io/max-replicas')
            if max_replicas_annotation:
                return int(max_replicas_annotation)
            
            # Fallback to HPA annotations if NimbusGuard not set
            hpa_max_annotation = annotations.get('autoscaling.alpha.kubernetes.io/max-replicas')
            if hpa_max_annotation:
                return int(hpa_max_annotation)
            
            # Default maximum
            return 10
            
        except Exception:
            return 10
    
    def update_scaling_history(self, 
                             deployment_name: str, 
                             namespace: str, 
                             action: str, 
                             from_replicas: int, 
                             to_replicas: int):
        """
        Update scaling history for a deployment.
        
        Args:
            deployment_name: Name of the deployment
            namespace: Kubernetes namespace
            action: Scaling action taken
            from_replicas: Previous replica count
            to_replicas: New replica count
        """
        context_key = f"{namespace}/{deployment_name}"
        context = self.deployment_contexts.get(context_key, {})
        
        if 'scaling_history' not in context:
            context['scaling_history'] = []
        
        # Add scaling event
        scaling_event = {
            'action': action,
            'from_replicas': from_replicas,
            'to_replicas': to_replicas,
            'timestamp': datetime.now().isoformat()
        }
        
        context['scaling_history'].append(scaling_event)
        
        # Keep only recent history (last 20 events)
        if len(context['scaling_history']) > 20:
            context['scaling_history'] = context['scaling_history'][-20:]
        
        self.deployment_contexts[context_key] = context
        
        logger.debug(f"Updated scaling history for {context_key}: {action} "
                    f"({from_replicas} → {to_replicas})")
    
    def get_recent_scaling_actions(self, 
                                 deployment_name: str, 
                                 namespace: str, 
                                 minutes: int = 10) -> list:
        """
        Get recent scaling actions for a deployment.
        
        Args:
            deployment_name: Name of the deployment
            namespace: Kubernetes namespace
            minutes: How many minutes back to look
            
        Returns:
            List of recent scaling actions
        """
        context_key = f"{namespace}/{deployment_name}"
        context = self.deployment_contexts.get(context_key, {})
        
        scaling_history = context.get('scaling_history', [])
        if not scaling_history:
            return []
        
        # Filter by time
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_actions = []
        
        for event in scaling_history:
            try:
                event_time = datetime.fromisoformat(event['timestamp'])
                if event_time >= cutoff_time:
                    recent_actions.append(event['action'])
            except Exception:
                continue
        
        return recent_actions
    
    def cleanup_old_contexts(self, hours: int = 24):
        """
        Clean up old deployment contexts.
        
        Args:
            hours: Remove contexts older than this many hours
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        to_remove = []
        for context_key, context in self.deployment_contexts.items():
            try:
                last_updated = datetime.fromisoformat(context.get('last_updated', ''))
                if last_updated < cutoff_time:
                    to_remove.append(context_key)
            except Exception:
                continue
        
        for context_key in to_remove:
            del self.deployment_contexts[context_key]
            logger.debug(f"Cleaned up old context: {context_key}")
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old deployment contexts")
    
    def get_all_contexts(self) -> Dict[str, Dict[str, Any]]:
        """Get all stored deployment contexts."""
        return self.deployment_contexts.copy() 