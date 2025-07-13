"""
Kubernetes client for cluster interactions.
"""

import logging
from typing import Dict, Any, Optional

from kubernetes import client, config
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)

class KubernetesClient:
    """Client for Kubernetes API interactions."""
    
    def __init__(self, scaling_config):
        self.config = scaling_config
        self.logger = logger
        
        # Load Kubernetes configuration
        try:
            # Try in-cluster config first (for production)
            config.load_incluster_config()
            self.logger.info("Loaded in-cluster Kubernetes configuration")
        except Exception:
            try:
                # Fall back to local kubeconfig (for development)
                config.load_kube_config()
                self.logger.info("Loaded local Kubernetes configuration")
            except Exception as e:
                self.logger.error(f"Failed to load Kubernetes configuration: {e}")
                raise
        
        # Initialize API clients
        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
        
        self.logger.info("Kubernetes client initialized")
    
    async def get_deployment(self, name: str, namespace: str) -> Optional[Dict[str, Any]]:
        """Get deployment information including replica limits from annotations."""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=name,
                namespace=namespace
            )
            
            # Extract min/max replicas from annotations (common HPA pattern)
            annotations = deployment.metadata.annotations or {}
            min_replicas = self._extract_replica_limit(annotations, 'min', default=1)
            max_replicas = self._extract_replica_limit(annotations, 'max', default=10)
            
            return {
                'name': deployment.metadata.name,
                'namespace': deployment.metadata.namespace,
                'replicas': deployment.spec.replicas,
                'min_replicas': min_replicas,
                'max_replicas': max_replicas,
                'ready_replicas': deployment.status.ready_replicas or 0,
                'available_replicas': deployment.status.available_replicas or 0,
                'unavailable_replicas': deployment.status.unavailable_replicas or 0,
                'generation': deployment.metadata.generation,
                'observed_generation': deployment.status.observed_generation
            }
        except ApiException as e:
            self.logger.error(f"Failed to get deployment {name}: {e}")
            return None
    
    def _extract_replica_limit(self, annotations: Dict[str, str], limit_type: str, default: int) -> int:
        """Extract min/max replica limits from deployment annotations."""
        # Check various annotation formats commonly used for replica limits
        possible_keys = [
            f"nimbusguard.io/{limit_type}-replicas",
            f"autoscaling.nimbusguard.io/{limit_type}-replicas", 
            f"deployment.kubernetes.io/{limit_type}-replicas",
            f"autoscaling.alpha.kubernetes.io/{limit_type}-replicas",
            f"{limit_type}-replicas"
        ]
        
        for key in possible_keys:
            if key in annotations:
                try:
                    value = int(annotations[key])
                    self.logger.debug(f"Found {limit_type}_replicas={value} from annotation {key}")
                    return value
                except ValueError:
                    self.logger.warning(f"Invalid {limit_type}_replicas value in annotation {key}: {annotations[key]}")
                    continue
        
        self.logger.debug(f"No {limit_type}_replicas annotation found, using default: {default}")
        return default
    
    async def scale_deployment(self, name: str, namespace: str, replicas: int) -> bool:
        """Scale deployment to specified replica count."""
        try:
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=name,
                namespace=namespace
            )
            
            current_replicas = deployment.spec.replicas
            
            # Only scale if there's a difference
            if current_replicas != replicas:
                # Update replica count
                deployment.spec.replicas = replicas
                
                # Apply the change
                self.apps_v1.patch_namespaced_deployment(
                    name=name,
                    namespace=namespace,
                    body=deployment
                )
                
                self.logger.info(f"Scaled deployment {name}: {current_replicas} → {replicas}")
                return True
            else:
                self.logger.debug(f"Deployment {name} already at {replicas} replicas")
                return True
                
        except ApiException as e:
            self.logger.error(f"Failed to scale deployment {name}: {e}")
            return False
    
    async def get_pods(self, namespace: str, label_selector: str = None) -> list:
        """Get pods in namespace with optional label selector."""
        try:
            pods = self.core_v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=label_selector
            )
            
            pod_list = []
            for pod in pods.items:
                pod_info = {
                    'name': pod.metadata.name,
                    'namespace': pod.metadata.namespace,
                    'phase': pod.status.phase,
                    'ready': self._is_pod_ready(pod),
                    'restart_count': sum(
                        container.restart_count for container in pod.status.container_statuses or []
                    ),
                    'node_name': pod.spec.node_name,
                    'creation_timestamp': pod.metadata.creation_timestamp
                }
                pod_list.append(pod_info)
            
            return pod_list
            
        except ApiException as e:
            self.logger.error(f"Failed to get pods: {e}")
            return []
    
    def _is_pod_ready(self, pod) -> bool:
        """Check if pod is ready."""
        if not pod.status.conditions:
            return False
        
        for condition in pod.status.conditions:
            if condition.type == "Ready":
                return condition.status == "True"
        
        return False
    
    async def get_deployment_metrics(self, name: str, namespace: str) -> Dict[str, Any]:
        """Get deployment metrics and status."""
        try:
            deployment_info = await self.get_deployment(name, namespace)
            if not deployment_info:
                return {}
            
            # Get pods for this deployment
            label_selector = f"app={name}"  # Assuming standard labeling
            pods = await self.get_pods(namespace, label_selector)
            
            # Calculate metrics
            total_pods = len(pods)
            ready_pods = sum(1 for pod in pods if pod['ready'])
            running_pods = sum(1 for pod in pods if pod['phase'] == 'Running')
            total_restarts = sum(pod['restart_count'] for pod in pods)
            
            return {
                'deployment': deployment_info,
                'pods': {
                    'total': total_pods,
                    'ready': ready_pods,
                    'running': running_pods,
                    'total_restarts': total_restarts
                },
                'health_score': ready_pods / max(total_pods, 1)  # 0-1 health score
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get deployment metrics: {e}")
            return {}
    
    async def wait_for_rollout(self, name: str, namespace: str, timeout: int = 300) -> bool:
        """Wait for deployment rollout to complete."""
        import asyncio
        
        start_time = asyncio.get_event_loop().time()
        
        while True:
            try:
                deployment = self.apps_v1.read_namespaced_deployment(
                    name=name,
                    namespace=namespace
                )
                
                # Check if rollout is complete
                if (deployment.status.ready_replicas == deployment.spec.replicas and
                    deployment.status.observed_generation == deployment.metadata.generation):
                    self.logger.info(f"Deployment {name} rollout completed")
                    return True
                
                # Check timeout
                if asyncio.get_event_loop().time() - start_time > timeout:
                    self.logger.warning(f"Deployment {name} rollout timed out")
                    return False
                
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except ApiException as e:
                self.logger.error(f"Error waiting for rollout: {e}")
                return False 