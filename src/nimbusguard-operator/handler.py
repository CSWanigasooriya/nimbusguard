# engine/handler.py
# ============================================================================
# Kubernetes Interaction and Orchestration
# ============================================================================

import logging
from typing import Dict, Any, Optional

import kopf
import kubernetes

from config import health_status
from prometheus import PrometheusClient
from rules import make_decision, extract_metric_name

LOG = logging.getLogger(__name__)


async def _fetch_metrics(metrics_config: Dict[str, Any]) -> Dict[str, float]:
    """Fetches all required metrics from Prometheus."""
    prometheus_url = metrics_config.get("prometheus_url", "http://prometheus-operated.default.svc:9090")
    client = PrometheusClient(prometheus_url)
    metrics = {}
    for metric_conf in metrics_config.get("metrics", []):
        query = metric_conf.get("query")
        if query:
            value = await client.query(query)
            if value is not None:
                name = extract_metric_name(query)
                metrics[name] = value
    return metrics


class OperatorHandler:
    """Handles all Kubernetes-specific logic and orchestrates the scaling process."""

    def __init__(self):
        self.apps_api: Optional[kubernetes.client.AppsV1Api] = None

    async def initialize(self):
        """Initializes the Kubernetes API clients."""
        try:
            # Use in-cluster config if running inside Kubernetes, otherwise use local kubeconfig
            try:
                kubernetes.config.load_incluster_config()
                LOG.info("Loaded in-cluster Kubernetes config.")
            except kubernetes.config.ConfigException:
                kubernetes.config.load_kube_config()
                LOG.info("Loaded local Kubernetes config (for development).")

            self.apps_api = kubernetes.client.AppsV1Api()
            health_status["kubernetes"] = True
            LOG.info("Kubernetes client initialized successfully.")
        except Exception as e:
            LOG.critical(f"Kubernetes client initialization failed: {e}", exc_info=True)
            health_status["kubernetes"] = False
            raise

    async def evaluate_scaling_logic(self, body: Dict[str, Any], namespace: str) -> Dict[str, Any]:
        """The main evaluation loop for a given IntelligentScaling resource."""
        spec = body.get("spec", {})
        target_namespace = spec.get("namespace", namespace)
        target_labels = spec.get("target_labels", {})
        metrics_config = spec.get("metrics_config", {})
        min_replicas = spec.get("minReplicas", 1)
        max_replicas = spec.get("maxReplicas", 10)

        # 1. Get current state from Kubernetes
        current_replicas, deployment_name = await self._get_current_replicas(target_labels, target_namespace)
        if current_replicas is None:
            raise kopf.PermanentError(f"No deployment found for labels {target_labels} in {target_namespace}.")

        # 2. Fetch metrics from Prometheus
        metrics = await _fetch_metrics(metrics_config)

        # 3. Make a scaling decision
        decision = make_decision(
            metrics=metrics,
            current_replicas=current_replicas,
            metric_configs=metrics_config.get("metrics", []),
            min_replicas=min_replicas,
            max_replicas=max_replicas
        )

        # 4. Execute the decision if necessary
        if decision["action"] != "none":
            await self._execute_scaling(decision, deployment_name, target_namespace)

        return {
            "current_replicas": current_replicas,
            **decision
        }

    async def _get_current_replicas(self, labels: Dict[str, str], ns: str) -> tuple[Optional[int], Optional[str]]:
        """Finds a deployment by labels and returns its replica count and name."""
        if not labels:
            LOG.error("No target_labels specified in the spec.")
            return None, None
        try:
            selector = ",".join(f"{k}={v}" for k, v in labels.items())
            deploys = self.apps_api.list_namespaced_deployment(namespace=ns, label_selector=selector)
            if not deploys.items:
                return None, None

            deployment = deploys.items[0]
            replicas = deployment.status.replicas if deployment.status.replicas is not None else 0
            return replicas, deployment.metadata.name
        except Exception as e:
            LOG.error(f"Failed to get replicas for labels '{labels}': {e}", exc_info=True)
            health_status["kubernetes"] = False
            return None, None

    async def _execute_scaling(self, decision: Dict, name: str, ns: str):
        """Patches the scale subresource of a deployment to change replicas."""
        target_replicas = decision["target_replicas"]
        action = decision["action"]
        LOG.info(f"Executing scaling action '{action}' on deployment '{name}' to {target_replicas} replicas.")
        try:
            body = {"spec": {"replicas": target_replicas}}
            self.apps_api.patch_namespaced_deployment_scale(name=name, namespace=ns, body=body)
        except Exception as e:
            LOG.error(f"Failed to scale deployment '{name}': {e}", exc_info=True)
            health_status["kubernetes"] = False
            raise
