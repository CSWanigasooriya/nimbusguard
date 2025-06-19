# engine/handler.py - Fixed with proper scaling logic and loop prevention
# ============================================================================
# Kubernetes Interaction and Orchestration
# ============================================================================

import logging
from typing import Dict, Any, Optional
from collections import deque
from datetime import datetime, timedelta

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
    """Fetches the focused 7-feature state vector from the observability collector."""
    prometheus_url = metrics_config.get("prometheus_url", "http://prometheus.monitoring.svc.cluster.local:9090")
    collector = ObservabilityCollector(prometheus_url)
    unified_state = await collector.collect_unified_state(metrics_config, target_labels, service_name)
    return unified_state


class OperatorHandler:
    """
    Handles all Kubernetes-specific logic and orchestrates the scaling process.
    This version includes proper scaling logic and loop prevention.
    """

    def __init__(self):
        """Initializes the handler, Kubernetes API client, and action history store."""
        self.apps_api: Optional[kubernetes.client.AppsV1Api] = None
        self.action_histories: Dict[str, deque] = {}
        # Cooldown logic removed for proactive scaling

    async def initialize(self):
        """Initializes the Kubernetes API clients."""
        try:
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
        meta = body.get("metadata", {})
        resource_uid = meta.get("uid")
        resource_name = meta.get("name")

        if not resource_uid:
            raise kopf.PermanentError("Cannot operate on a resource without a UID. Check resource metadata.")

        target_namespace = spec.get("namespace", namespace)
        target_labels = spec.get("target_labels", {})
        metrics_config = spec.get("metrics_config", {})
        min_replicas = spec.get("minReplicas", 1)
        max_replicas = spec.get("maxReplicas", 10)

        LOG.info(f"Evaluating scaling for '{resource_name}' (uid: {resource_uid}).")

        # 1. Get current state from Kubernetes.
        current_replicas, deployment_name = await self._get_current_replicas(target_labels, target_namespace)
        if current_replicas is None:
            msg = f"No deployment found for labels {target_labels} in namespace '{target_namespace}'."
            LOG.error(msg)
            raise kopf.PermanentError(msg)
        LOG.info(f"Found deployment '{deployment_name}' with {current_replicas} current replicas.")

        # 3. Fetch observability data.
        service_name = target_labels.get("app", "unknown-service")
        unified_state = await _fetch_unified_observability_data(metrics_config, target_labels, service_name)

        # 4. Check confidence threshold to prevent actions with poor data quality
        confidence = unified_state.get("confidence_score", 0.0)
        confidence_threshold = spec.get("ml_config", {}).get("confidence_threshold", 0.3)
        
        if confidence < confidence_threshold:
            LOG.warning(f"Confidence score ({confidence:.2f}) below threshold ({confidence_threshold}). Skipping scaling decision.")
            return {
                "current_replicas": current_replicas,
                "action": "none", 
                "target_replicas": current_replicas,
                "reason": f"Low confidence score: {confidence:.2f} < {confidence_threshold}"
            }

        # 5. Retrieve this resource's action history from our stateful memory.
        if resource_uid not in self.action_histories:
            self.action_histories[resource_uid] = deque([2] * 5, maxlen=5)  # Initialize with NO_ACTION
        recent_actions = list(self.action_histories[resource_uid])

        # 6. Make a scaling decision, now passing the required 'recent_actions' argument.
        decision = make_decision(
            current_replicas=current_replicas,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            spec=spec,
            unified_state=unified_state,
            recent_actions=recent_actions  # Pass the history to the decision logic
        )

        # 7. Execute the decision if needed.
        if decision.get("action") != "none":
            success = await self._execute_scaling(decision, deployment_name, target_namespace)
            if success:
                LOG.info(f"Successfully executed scaling action for '{resource_name}'")
            else:
                LOG.error(f"Failed to execute scaling action for '{resource_name}'")
                decision["reason"] += " (execution failed)"

        # 8. Update the action history with the new decision.
        newly_chosen_action_value = decision.get("ml_decision", {}).get("action_value", 2)  # Default to NO_ACTION
        self.action_histories[resource_uid].append(newly_chosen_action_value)

        return {"current_replicas": current_replicas, **decision}

    async def _get_current_replicas(self, labels: Dict[str, str], ns: str) -> tuple[Optional[int], Optional[str]]:
        """Finds a deployment by its labels and returns its current replica count and name."""
        if not self.apps_api: 
            await self.initialize()

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
            LOG.error(f"Failed to get replicas for labels '{labels}' in '{ns}': {e}", exc_info=True)
            health_status["kubernetes"] = False
            return None, None

    async def _execute_scaling(self, decision: Dict, name: str, ns: str) -> bool:
        """
        Properly scales a deployment using both scale subresource and direct deployment patch.
        Returns True if successful, False otherwise.
        """
        target_replicas = decision["target_replicas"]
        action = decision["action"]
        LOG.info(f"Executing scaling action: {action.upper()} on '{name}' to {target_replicas} replicas.")
        
        try:
            # Method 1: Try using the scale subresource (recommended approach)
            try:
                scale_body = kubernetes.client.V1Scale(
                    spec=kubernetes.client.V1ScaleSpec(replicas=target_replicas)
                )
                self.apps_api.patch_namespaced_deployment_scale(
                    name=name, 
                    namespace=ns, 
                    body=scale_body
                )
                LOG.info(f"Successfully scaled '{name}' using scale subresource")
                return True
            except Exception as scale_error:
                LOG.warning(f"Scale subresource failed: {scale_error}. Trying direct deployment patch.")
                
                # Method 2: Fallback to direct deployment patch
                deployment = self.apps_api.read_namespaced_deployment(name=name, namespace=ns)
                deployment.spec.replicas = target_replicas
                
                self.apps_api.patch_namespaced_deployment(
                    name=name,
                    namespace=ns, 
                    body=deployment
                )
                LOG.info(f"Successfully scaled '{name}' using direct deployment patch")
                return True
                
        except Exception as e:
            LOG.error(f"Failed to scale deployment '{name}': {e}", exc_info=True)
            health_status["kubernetes"] = False
            return False

    async def verify_scaling_effect(self, name: str, ns: str, expected_replicas: int, max_wait_seconds: int = 30) -> bool:
        """
        Verify that a scaling action actually took effect by checking the deployment.
        """
        import asyncio
        
        for i in range(max_wait_seconds):
            try:
                current_replicas, _ = await self._get_current_replicas({"app": name}, ns)
                if current_replicas == expected_replicas:
                    LOG.info(f"Scaling verified: '{name}' now has {current_replicas} replicas")
                    return True
                await asyncio.sleep(1)
            except Exception as e:
                LOG.warning(f"Error during scaling verification: {e}")
                
        LOG.warning(f"Scaling verification failed: '{name}' did not reach {expected_replicas} replicas in {max_wait_seconds}s")
        return False
