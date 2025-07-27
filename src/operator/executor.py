import logging
from datetime import datetime
from kubernetes import client
from state_manager import state

class ClusterExecutor:
    """
    Handles Kubernetes cluster operations including deployment scaling.
    Updates shared state with operation results.
    """
    
    def __init__(self, k8s_client=None):
        self.k8s_apps_v1 = k8s_client or client.AppsV1Api()
        
    def parse_resource_value(self, resource_str):
        """Parse Kubernetes resource strings (CPU/Memory) to numeric values."""
        if not resource_str: 
            return 0.0
        resource_str = resource_str.lower()
        if resource_str.endswith('m'): 
            return float(resource_str[:-1]) / 1000.0
        if resource_str.endswith('gi'): 
            return float(resource_str[:-2]) * (1024**3)
        if resource_str.endswith('mi'): 
            return float(resource_str[:-2]) * (1024**2)
        if resource_str.endswith('ki'): 
            return float(resource_str[:-2]) * 1024
        return float(resource_str)
    
    def get_deployment_info(self, name, namespace):
        """
        Get comprehensive deployment information.
        
        Returns:
            Dictionary with deployment details or None if failed
        """
        try:
            deployment = self.k8s_apps_v1.read_namespaced_deployment(name=name, namespace=namespace)
            
            # Basic deployment info
            current_replicas = deployment.spec.replicas
            available_replicas = deployment.status.available_replicas or 0
            ready_replicas = deployment.status.ready_replicas or 0
            
            # Resource limits and requests
            container = deployment.spec.template.spec.containers[0]
            resources = container.resources
            
            cpu_limit = self.parse_resource_value(
                resources.limits.get('cpu') if resources.limits else None or
                resources.requests.get('cpu') if resources.requests else None
            )
            
            memory_limit = self.parse_resource_value(
                resources.limits.get('memory') if resources.limits else None or
                resources.requests.get('memory') if resources.requests else None
            )
            
            cpu_request = self.parse_resource_value(
                resources.requests.get('cpu') if resources.requests else None
            )
            
            memory_request = self.parse_resource_value(
                resources.requests.get('memory') if resources.requests else None
            )
            
            # Scaling constraints from annotations
            annotations = deployment.metadata.annotations or {}
            min_replicas = int(annotations.get('nimbusguard.io/min-replicas', '1'))
            max_replicas = int(annotations.get('nimbusguard.io/max-replicas', '10'))
            
            # Additional metadata
            labels = deployment.metadata.labels or {}
            creation_timestamp = deployment.metadata.creation_timestamp
            
            deployment_info = {
                'name': name,
                'namespace': namespace,
                'current_replicas': current_replicas,
                'available_replicas': available_replicas,
                'ready_replicas': ready_replicas,
                'min_replicas': min_replicas,
                'max_replicas': max_replicas,
                'cpu_limit': cpu_limit,
                'memory_limit': memory_limit,
                'cpu_request': cpu_request,
                'memory_request': memory_request,
                'labels': labels,
                'annotations': annotations,
                'creation_timestamp': creation_timestamp.isoformat() if creation_timestamp else None,
                'container_name': container.name,
                'image': container.image
            }
            
            logging.info(f"--- Deployment Info: {name} | Replicas: {current_replicas} | Constraints: {min_replicas}-{max_replicas} ---")
            logging.info(f"    CPU Limit: {cpu_limit:.2f} cores, Memory Limit: {memory_limit / (1024*1024):.0f} MB")
            
            return deployment_info
            
        except Exception as e:
            logging.error(f"Failed to get deployment info for {namespace}/{name}: {e}")
            return None
    

    
    def scale_deployment(self, name, namespace, new_replicas, dry_run=False, reason="Scaling requested"):
        """
        Execute deployment scaling without validation - assumes validation is done externally.
        
        Args:
            name: Deployment name
            namespace: Deployment namespace
            new_replicas: Target replica count (should already be validated)
            dry_run: If True, validate but don't execute
            reason: Reason for scaling (for logging and history)
            
        Returns:
            Dictionary with scaling result
        """
        try:
            # Get current deployment info
            deployment_info = self.get_deployment_info(name, namespace)
            if not deployment_info:
                return {'success': False, 'error': 'Could not get deployment info'}
            
            current_replicas = deployment_info['current_replicas']
            
            # Check if scaling is needed
            if new_replicas == current_replicas:
                logging.info("No scaling action required.")
                return {
                    'success': True,
                    'action': 'none',
                    'current_replicas': current_replicas,
                    'target_replicas': new_replicas,
                    'reason': 'No change needed'
                }
            
            # Determine action type
            action = 'scale_up' if new_replicas > current_replicas else 'scale_down'
            
            if dry_run:
                logging.info(f"DRY RUN: Would scale {name} from {current_replicas} to {new_replicas} replicas")
                return {
                    'success': True,
                    'action': action,
                    'current_replicas': current_replicas,
                    'target_replicas': new_replicas,
                    'dry_run': True,
                    'reason': reason
                }
            
            # Execute scaling
            logging.info(f"SCALING: Changing replicas from {current_replicas} to {new_replicas}. Reason: {reason}")
            
            patch = {'spec': {'replicas': new_replicas}}
            result = self.k8s_apps_v1.patch_namespaced_deployment_scale(
                name=name, 
                namespace=namespace, 
                body=patch
            )
            
            # Update shared state with scaling action
            scaling_record = {
                'timestamp': datetime.now().isoformat(),
                'deployment': f"{namespace}/{name}",
                'action': action,
                'from_replicas': current_replicas,
                'to_replicas': new_replicas,
                'reason': reason,
                'success': True
            }
            
            # Store in state
            if not hasattr(state, 'scaling_history'):
                state.scaling_history = []
            state.scaling_history.append(scaling_record)
            
            # Keep only recent history
            if len(state.scaling_history) > 100:
                state.scaling_history = state.scaling_history[-100:]
            
            logging.info(f"✅ Successfully scaled {name} to {new_replicas} replicas")
            
            return {
                'success': True,
                'action': action,
                'current_replicas': current_replicas,
                'target_replicas': new_replicas,
                'kubernetes_response': result.metadata.resource_version,
                'reason': reason
            }
            
        except Exception as e:
            error_msg = f"Failed to scale deployment {namespace}/{name}: {e}"
            logging.error(error_msg)
            
            # Record failed scaling attempt
            scaling_record = {
                'timestamp': datetime.now().isoformat(),
                'deployment': f"{namespace}/{name}",
                'action': 'scale_failed',
                'from_replicas': current_replicas if 'current_replicas' in locals() else 'unknown',
                'to_replicas': new_replicas,
                'reason': str(e),
                'success': False
            }
            
            if not hasattr(state, 'scaling_history'):
                state.scaling_history = []
            state.scaling_history.append(scaling_record)
            
            return {'success': False, 'error': error_msg}
    
    def calculate_new_replicas(self, action, current_replicas, min_replicas, max_replicas):
        """
        Calculate new replica count based on DQN action.
        
        Args:
            action: DQN action (0=none, 1=scale_up, 2=scale_down)
            current_replicas: Current replica count
            min_replicas: Minimum allowed replicas
            max_replicas: Maximum allowed replicas
            
        Returns:
            New replica count
        """
        if action == 0:  # No action
            return current_replicas
        elif action == 1:  # Scale up
            return min(current_replicas + 1, max_replicas)
        elif action == 2:  # Scale down
            return max(current_replicas - 1, min_replicas)
        else:
            logging.warning(f"Unknown action {action}, defaulting to no change")
            return current_replicas
    
    def get_scaling_history(self, limit=10):
        """Get recent scaling history."""
        if not hasattr(state, 'scaling_history'):
            return []
        return state.scaling_history[-limit:] if state.scaling_history else []
    
    def get_deployment_status(self, name, namespace):
        """Get deployment status summary."""
        deployment_info = self.get_deployment_info(name, namespace)
        if not deployment_info:
            return None
            
        return {
            'name': deployment_info['name'],
            'namespace': deployment_info['namespace'],
            'replicas': {
                'current': deployment_info['current_replicas'],
                'available': deployment_info['available_replicas'],
                'ready': deployment_info['ready_replicas'],
                'min': deployment_info['min_replicas'],
                'max': deployment_info['max_replicas']
            },
            'resources': {
                'cpu_limit_cores': deployment_info['cpu_limit'],
                'memory_limit_mb': deployment_info['memory_limit'] / (1024*1024),
                'cpu_request_cores': deployment_info['cpu_request'],
                'memory_request_mb': deployment_info['memory_request'] / (1024*1024)
            },
            'recent_scaling': self.get_scaling_history(3)
        }
    
    def check_deployment_health(self, name, namespace):
        """Check if deployment is healthy and ready for scaling (basic health check)."""
        try:
            deployment_info = self.get_deployment_info(name, namespace)
            if not deployment_info:
                return False, "Could not get deployment info"
            
            current = deployment_info['current_replicas']
            available = deployment_info['available_replicas']
            ready = deployment_info['ready_replicas']
            
            if available < current:
                return False, f"Not all replicas available: {available}/{current}"
            
            if ready < current:
                return False, f"Not all replicas ready: {ready}/{current}"
            
            return True, "Deployment is healthy"
            
        except Exception as e:
            return False, f"Health check failed: {e}" 