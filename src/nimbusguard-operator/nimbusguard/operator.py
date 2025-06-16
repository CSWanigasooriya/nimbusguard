"""
Main NimbusGuard operator implementation.
"""

import kubernetes
import logging
from typing import Dict, Any
from datetime import datetime
from kubernetes.client.rest import ApiException

from .config import Config
from .health import set_component_health
from .metrics import CURRENT_REPLICAS, SCALING_OPERATIONS, DECISIONS_MADE
from .clients.prometheus import PrometheusClient
from .engines.decision import DecisionEngine
from .crd_manager import crd_manager
from .models.nimbusguard_hpa_crd import NimbusGuardHPA

LOG = logging.getLogger(__name__)

# ============================================================================
# Main Operator
# ============================================================================

class NimbusGuardOperator:
    """Main operator class"""
    
    def __init__(self):
        self.decision_engine = DecisionEngine()
        self.k8s_client = None
        self.custom_api = None
        self.apps_api = None
    
    async def initialize(self):
        """Initialize operator"""
        try:
            # Load Kubernetes config
            try:
                kubernetes.config.load_incluster_config()
                LOG.info("Loaded in-cluster Kubernetes config")
            except:
                kubernetes.config.load_kube_config()
                LOG.info("Loaded local Kubernetes config")
            
            # Initialize API clients
            self.k8s_client = kubernetes.client.ApiClient()
            self.custom_api = kubernetes.client.CustomObjectsApi(self.k8s_client)
            self.apps_api = kubernetes.client.AppsV1Api(self.k8s_client)
            
            # Initialize CRD manager
            crd_manager.initialize(self.k8s_client)
            
            # Install CRDs using kubecrd
            LOG.info("Installing Custom Resource Definitions using kubecrd...")
            success = await crd_manager.install_crds()
            if not success:
                raise Exception("Failed to install CRDs")
            
            # Wait for CRDs to be ready
            await crd_manager.wait_for_crd_ready("nimbusguardhpa.nimbusguard.io")
            
            set_component_health("kubernetes", True)
            LOG.info("Kubernetes client ready")
            LOG.info("✅ CRDs installed and ready via kubecrd")
        except Exception as e:
            LOG.error(f"K8s init failed: {e}")
            set_component_health("kubernetes", False)
            raise
    
    async def evaluate_scaling(self, body: Dict[str, Any], name: str, namespace: str) -> Dict[str, Any]:
        """Main scaling evaluation logic"""
        try:
            # Parse the NimbusGuardHPA resource using kubecrd format
            from .utils import parse_nimbus_guard_hpa
            config = parse_nimbus_guard_hpa(body)
            
            # Extract configuration 
            target_namespace = config.get("namespace", namespace)
            target_labels = config.get("target_labels", {})
            metrics_config = config.get("metrics_config", {})
            
            # Fetch metrics from Prometheus
            metrics = await self._fetch_metrics(metrics_config)
            
            # Get current replica count
            current_replicas = await self._get_current_replicas(target_labels, target_namespace)
            
            # Make scaling decision
            decision = self.decision_engine.make_decision(
                metrics=metrics,
                current_replicas=current_replicas,
                metric_configs=metrics_config.get("metrics", []),
                min_replicas=config.get("min_replicas", Config.DEFAULT_MIN_REPLICAS),
                max_replicas=config.get("max_replicas", Config.DEFAULT_MAX_REPLICAS)
            )
            
            # Execute scaling action if needed
            if decision.get("action") != "none":
                await self._execute_scaling(decision, target_labels, target_namespace)
                DECISIONS_MADE.labels(action=decision.get("action")).inc()
            
            return {
                "current_replicas": current_replicas,
                "target_replicas": decision.get("target_replicas", current_replicas),
                "decision_reason": decision.get("reason", ""),
                "last_evaluation": datetime.now().isoformat(),
                "action": decision.get("action", "none")
            }
            
        except Exception as e:
            LOG.error(f"Evaluation failed: {e}")
            raise
    
    async def _fetch_metrics(self, metrics_config: Dict[str, Any]) -> Dict[str, float]:
        """Fetch metrics from Prometheus"""
        prometheus_url = metrics_config.get("prometheus_url", Config.DEFAULT_PROMETHEUS_URL)
        client = PrometheusClient(prometheus_url)
        metrics = {}
        
        for metric_config in metrics_config.get("metrics", []):
            query = metric_config.get("query", "")
            if query:
                value = await client.query(query)
                if value is not None:
                    name = self.decision_engine._extract_metric_name(query)
                    metrics[name] = value
        
        return metrics
    
    async def _get_current_replicas(self, target_labels: Dict[str, str], namespace: str) -> int:
        """Get current replica count from target deployment"""
        try:
            if not target_labels:
                LOG.warning("No target labels specified")
                return 1
            
            # Create label selector
            selector = ",".join([f"{k}={v}" for k, v in target_labels.items()])
            
            # Get deployments matching the labels
            deployments = self.apps_api.list_namespaced_deployment(
                namespace=namespace,
                label_selector=selector
            )
            
            if deployments.items:
                current = deployments.items[0].status.replicas or 1
                CURRENT_REPLICAS.labels(
                    name=deployments.items[0].metadata.name, 
                    namespace=namespace
                ).set(current)
                return current
                
        except Exception as e:
            LOG.warning(f"Failed to get current replicas: {e}")
        
        return 1
    
    async def _execute_scaling(self, decision: Dict[str, Any], target_labels: Dict[str, str], namespace: str):
        """Execute scaling action on target deployment"""
        try:
            target_replicas = decision.get("target_replicas", 1)
            action = decision.get("action", "none")
            
            if not target_labels:
                LOG.warning("No target labels - cannot scale")
                return
            
            # Create label selector
            selector = ",".join([f"{k}={v}" for k, v in target_labels.items()])
            
            # Get deployments matching the labels
            deployments = self.apps_api.list_namespaced_deployment(
                namespace=namespace,
                label_selector=selector
            )
            
            for deployment in deployments.items:
                # Update replica count
                deployment.spec.replicas = target_replicas
                
                # Patch the deployment
                self.apps_api.patch_namespaced_deployment(
                    name=deployment.metadata.name,
                    namespace=namespace,
                    body=deployment
                )
                
                LOG.info(f"Scaled deployment {deployment.metadata.name} to {target_replicas} replicas")
                SCALING_OPERATIONS.labels(operation_type=action, namespace=namespace).inc()
                
        except Exception as e:
            LOG.error(f"Scaling execution failed: {e}")
            raise
    
    async def get_status(self, name: str, namespace: str) -> Dict[str, Any]:
        """Get current status of a NimbusGuardHPA resource"""
        try:
            resource = self.custom_api.get_namespaced_custom_object(
                group=Config.CRD_GROUP,
                version=Config.CRD_VERSION,
                namespace=namespace,
                plural=Config.CRD_PLURAL,
                name=name
            )
            return resource.get("status", {})
        except ApiException as e:
            LOG.error(f"Failed to get status for {name}: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup resources"""
        if self.k8s_client:
            try:
                # Cleanup CRD manager
                crd_manager.cleanup()
                
                # Kubernetes ApiClient doesn't have async close, so we don't need to await
                # Just set it to None to indicate cleanup
                self.k8s_client = None
                self.custom_api = None
                self.apps_api = None
                LOG.info("Kubernetes clients cleaned up")
            except Exception as e:
                LOG.warning(f"Error during cleanup: {e}")
