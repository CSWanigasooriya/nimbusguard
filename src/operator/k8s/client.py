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
        """Get deployment information including replica limits from annotations and resource limits from specs."""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=name,
                namespace=namespace
            )

            # Extract min/max replicas from annotations (common HPA pattern)
            annotations = deployment.metadata.annotations or {}
            min_replicas = self._extract_replica_limit(annotations, 'min', default=1)
            max_replicas = self._extract_replica_limit(annotations, 'max', default=50)

            # Extract resource limits and requests from container specs
            resource_info = self._extract_resource_limits(deployment)

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
                'observed_generation': deployment.status.observed_generation,
                # NEW: Resource limits and requests from container specs
                'resource_limits': resource_info['limits'],
                'resource_requests': resource_info['requests'],
                'resource_info': resource_info
            }
        except ApiException as e:
            self.logger.error(f"Failed to get deployment {name}: {e}")
            return None

    def _extract_replica_limit(self, annotations: Dict[str, str], limit_type: str, default: int) -> int:
        """Extract min/max replica limits from deployment annotations."""
        self.logger.debug(f"Extracting {limit_type}_replicas from annotations: {list(annotations.keys())}")
        
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
                    self.logger.info(f"✅ Found {limit_type}_replicas={value} from annotation '{key}'")
                    return value
                except ValueError:
                    self.logger.warning(f"Invalid {limit_type}_replicas value in annotation {key}: {annotations[key]}")
                    continue

        self.logger.warning(f"❌ No {limit_type}_replicas annotation found in {list(annotations.keys())}, using default: {default}")
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

    def _extract_resource_limits(self, deployment) -> Dict[str, Any]:
        """Extract resource limits and requests from deployment container specs."""
        try:
            containers = deployment.spec.template.spec.containers

            # Aggregate resource limits and requests across all containers
            total_cpu_limit = 0.0
            total_memory_limit = 0
            total_cpu_request = 0.0
            total_memory_request = 0

            container_resources = []

            for container in containers:
                container_info = {
                    'name': container.name,
                    'limits': {},
                    'requests': {}
                }

                # Extract limits
                if container.resources and container.resources.limits:
                    limits = container.resources.limits

                    # CPU limit (convert to float - can be "500m" or "1.5")
                    if 'cpu' in limits:
                        cpu_limit = self._parse_cpu_resource(limits['cpu'])
                        container_info['limits']['cpu'] = cpu_limit
                        total_cpu_limit += cpu_limit

                    # Memory limit (convert to bytes)
                    if 'memory' in limits:
                        memory_limit = self._parse_memory_resource(limits['memory'])
                        container_info['limits']['memory'] = memory_limit
                        total_memory_limit += memory_limit

                # Extract requests
                if container.resources and container.resources.requests:
                    requests = container.resources.requests

                    # CPU request
                    if 'cpu' in requests:
                        cpu_request = self._parse_cpu_resource(requests['cpu'])
                        container_info['requests']['cpu'] = cpu_request
                        total_cpu_request += cpu_request

                    # Memory request
                    if 'memory' in requests:
                        memory_request = self._parse_memory_resource(requests['memory'])
                        container_info['requests']['memory'] = memory_request
                        total_memory_request += memory_request

                container_resources.append(container_info)
                self.logger.debug(f"Container {container.name} resources: {container_info}")

            resource_summary = {
                'limits': {
                    'cpu': total_cpu_limit,  # Total CPU cores
                    'memory': total_memory_limit  # Total memory in bytes
                },
                'requests': {
                    'cpu': total_cpu_request,  # Total CPU cores
                    'memory': total_memory_request  # Total memory in bytes
                },
                'containers': container_resources,
                'container_count': len(containers)
            }

            self.logger.debug(f"Deployment resource summary: {resource_summary}")
            return resource_summary

        except Exception as e:
            self.logger.warning(f"Failed to extract resource limits: {e}")
            return {
                'limits': {'cpu': 0.0, 'memory': 0},
                'requests': {'cpu': 0.0, 'memory': 0},
                'containers': [],
                'container_count': 0
            }

    def _parse_cpu_resource(self, cpu_str: str) -> float:
        """Parse CPU resource string to float (cores)."""
        try:
            cpu_str = str(cpu_str).strip()

            if cpu_str.endswith('m'):
                # Millicores: "500m" = 0.5 cores
                return float(cpu_str[:-1]) / 1000.0
            else:
                # Direct cores: "1.5" = 1.5 cores
                return float(cpu_str)
        except (ValueError, AttributeError):
            self.logger.warning(f"Failed to parse CPU resource: {cpu_str}")
            return 0.0

    def _parse_memory_resource(self, memory_str: str) -> int:
        """Parse memory resource string to bytes."""
        try:
            memory_str = str(memory_str).strip().upper()

            # Remove 'i' suffix if present (Ki, Mi, Gi -> K, M, G)
            if memory_str.endswith('I'):
                memory_str = memory_str[:-1]

            # Parse different units
            if memory_str.endswith('K'):
                return int(float(memory_str[:-1]) * 1024)
            elif memory_str.endswith('M'):
                return int(float(memory_str[:-1]) * 1024 * 1024)
            elif memory_str.endswith('G'):
                return int(float(memory_str[:-1]) * 1024 * 1024 * 1024)
            elif memory_str.endswith('T'):
                return int(float(memory_str[:-1]) * 1024 * 1024 * 1024 * 1024)
            else:
                # Assume bytes
                return int(memory_str)
        except (ValueError, AttributeError):
            self.logger.warning(f"Failed to parse memory resource: {memory_str}")
            return 0
