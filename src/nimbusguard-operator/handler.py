# engine/handler.py - Optimized for the new 20-feature DQN model
# ============================================================================
# Kubernetes Interaction and Orchestration
# ============================================================================

import logging
from typing import Dict, Any, Optional

import kopf
import kubernetes

from config import health_status
from observability import ObservabilityCollector
from rules import make_decision

LOG = logging.getLogger(__name__)


async def _fetch_unified_observability_data(
    metrics_config: Dict[str, Any],
    target_labels: Dict[str, str],
    service_name: str
) -> Dict[str, Any]:
    """
    Fetches the focused 7-feature state vector from the observability collector.
    """
    # --- SIMPLIFIED --- Only requires the prometheus_url now.
    prometheus_url = metrics_config.get("prometheus_url", "http://prometheus.monitoring.svc.cluster.local:9090")

    # Initialize the new, lean collector.
    collector = ObservabilityCollector(prometheus_url)

    # Collect the unified state. The new collector is much more efficient.
    unified_state = await collector.collect_unified_state(metrics_config, target_labels, service_name)

    return unified_state


class OperatorHandler:
    """Handles all Kubernetes-specific logic and orchestrates the scaling process."""

    def __init__(self):
        """Initializes the handler and Kubernetes API client placeholder."""
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

        resource_name = body.get("metadata", {}).get("name", "unknown")
        LOG.info(f"Evaluating scaling for '{resource_name}' in namespace '{target_namespace}'.")

        # 1. Get current state from the Kubernetes cluster.
        current_replicas, deployment_name = await self._get_current_replicas(target_labels, target_namespace)
        if current_replicas is None:
            msg = f"No deployment found for labels {target_labels} in namespace '{target_namespace}'."
            LOG.error(msg)
            raise kopf.PermanentError(msg)
        LOG.info(f"Found deployment '{deployment_name}' with {current_replicas} current replicas.")

        # 2. Fetch the unified 7-feature state from our optimized collector.
        service_name = target_labels.get("app", "unknown-service")
        unified_state = await _fetch_unified_observability_data(metrics_config, target_labels, service_name)

        # 3. Make a scaling decision using the new, streamlined interface.
        # --- COMPLETELY REFACTORED ---
        # No more legacy metrics or complex context objects.
        decision = make_decision(
            current_replicas=current_replicas,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            spec=spec,  # Pass the whole spec for access to ml_config
            unified_state=unified_state
        )

        # 4. Execute the scaling decision if an action is required.
        if decision.get("action") != "none":
            await self._execute_scaling(decision, deployment_name, target_namespace)

        # 5. Return the result to be patched into the CRD status.
        return {
            "current_replicas": current_replicas,
            **decision
        }

    async def _get_current_replicas(self, labels: Dict[str, str], ns: str) -> tuple[Optional[int], Optional[str]]:
        """Finds a deployment by its labels and returns its current replica count and name."""
        if not self.apps_api:
            await self.initialize()

        if not labels:
            LOG.error("No target_labels specified in the spec.")
            return None, None
        try:
            selector = ",".join(f"{k}={v}" for k, v in labels.items())
            LOG.debug(f"Querying for deployment with label selector: '{selector}' in namespace '{ns}'")

            deploys = self.apps_api.list_namespaced_deployment(namespace=ns, label_selector=selector)
            if not deploys.items:
                return None, None

            deployment = deploys.items[0]
            # status.replicas can be None if the deployment is new, default to 0.
            replicas = deployment.status.replicas if deployment.status.replicas is not None else 0
            return replicas, deployment.metadata.name
        except Exception as e:
            LOG.error(f"Failed to get replicas for labels '{labels}' in '{ns}': {e}", exc_info=True)
            health_status["kubernetes"] = False
            return None, None

    async def _execute_scaling(self, decision: Dict, name: str, ns: str):
        """Patches the scale subresource of a deployment to change the number of replicas."""
        target_replicas = decision["target_replicas"]
        action = decision["action"]
        LOG.info(f"Executing scaling action: {action.upper()} on '{name}' to {target_replicas} replicas.")
        try:
            body = {"spec": {"replicas": target_replicas}}
            self.apps_api.patch_namespaced_deployment_scale(name=name, namespace=ns, body=body)
        except Exception as e:
            LOG.error(f"Failed to scale deployment '{name}': {e}", exc_info=True)
            health_status["kubernetes"] = False
            raise
