# engine/handler.py - DQN Implementation
# ============================================================================
# Kubernetes Interaction and Orchestration with KServe ML Pipeline
# ============================================================================

import logging
import asyncio
from collections import deque
from typing import Dict, Any, Optional, List

import kopf
import kubernetes

from config import health_status

# DQN imports
from ml.dqn_agent import DQNAgent, create_dqn_agent
from ml.state_representation import EnvironmentState

LOG = logging.getLogger(__name__)


class OperatorHandler:
    """
    Handler for ML pipeline management with KServe model serving capabilities.
    """

    def __init__(self):
        """Initializes the handler with Kubernetes and KServe integration."""
        self.apps_api: Optional[kubernetes.client.AppsV1Api] = None
        self.action_histories: Dict[str, deque] = {}
        
        # DQN configuration
        self.dqn_agent: Optional[DQNAgent] = None
        self.kserve_enabled = False
        self.kserve_endpoint: Optional[str] = None
        
        # Performance tracking
        self.total_decisions = 0
        self.successful_decisions = 0
        self.kserve_failures = 0

    async def initialize(self):
        """Initializes Kubernetes API clients and KServe integration."""
        try:
            # Initialize Kubernetes client
            try:
                kubernetes.config.load_incluster_config()
                LOG.info("Loaded in-cluster Kubernetes config.")
            except kubernetes.config.ConfigException:
                kubernetes.config.load_kube_config()
                LOG.info("Loaded local Kubernetes config (for development).")
            
            self.apps_api = kubernetes.client.AppsV1Api()
            health_status["kubernetes"] = True
            LOG.info("Kubernetes client initialized successfully.")
            
            # Initialize DQN integration
            await self._initialize_dqn()
            
        except Exception as e:
            LOG.critical(f"Handler initialization failed: {e}", exc_info=True)
            health_status["kubernetes"] = False
            raise

    async def _initialize_dqn(self):
        """Initialize DQN components and model serving."""
        try:
            # Get KServe endpoint from environment
            import os
            kserve_endpoint = os.getenv('KSERVE_ENDPOINT')
            
            if not kserve_endpoint:
                LOG.error("KSERVE_ENDPOINT environment variable is required")
                health_status["kserve"] = False
                return
                
            # DQN configuration
            dqn_config = {
                'kserve_endpoint': kserve_endpoint,
                'model_name': os.getenv('KSERVE_MODEL_NAME', 'nimbusguard-dqn'),
                'state_dim': 11,
                'action_dim': 5,
                'confidence_threshold': float(os.getenv('DQN_CONFIDENCE_THRESHOLD', '0.7')),
                'health_check_interval': int(os.getenv('KSERVE_HEALTH_CHECK_INTERVAL', '300'))
            }
            
            self.dqn_agent = await create_dqn_agent(dqn_config)
            self.kserve_enabled = True
            self.kserve_endpoint = kserve_endpoint
            
            health_status["kserve"] = True
            health_status["ml_decision_engine"] = True
            
            LOG.info("DQN integration initialized successfully")
            LOG.info(f"Model serving endpoint: {self.kserve_endpoint}")
            LOG.info(f"Confidence threshold: {dqn_config['confidence_threshold']}")
            
        except Exception as e:
            LOG.error(f"DQN initialization failed: {e}")
            health_status["kserve"] = False
            health_status["ml_decision_engine"] = False
            self.kserve_enabled = False
            # This is a critical failure
            raise RuntimeError(f"DQN initialization failed: {e}")

    async def evaluate_scaling_logic(self, body: Dict[str, Any], namespace: str) -> Dict[str, Any]:
        """DQN evaluation loop for scaling decisions."""
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

        LOG.info(f"Evaluating scaling for '{resource_name}' (uid: {resource_uid}) using DQN.")

        # 1. Ensure DQN agent is available
        if not self.kserve_enabled or not self.dqn_agent:
            error_msg = "DQN agent not available - operator requires KServe endpoint configuration"
            LOG.error(error_msg)
            raise kopf.PermanentError(error_msg)

        # 2. Get current state from Kubernetes
        current_replicas, deployment_name = await self._get_current_replicas(target_labels, target_namespace)
        if current_replicas is None:
            msg = f"No deployment found for labels {target_labels} in namespace '{target_namespace}'."
            LOG.error(msg)
            raise kopf.PermanentError(msg)
        LOG.info(f"Found deployment '{deployment_name}' with {current_replicas} current replicas.")

        # 3. Retrieve action history
        if resource_uid not in self.action_histories:
            self.action_histories[resource_uid] = deque([2] * 5, maxlen=5)
        recent_actions = list(self.action_histories[resource_uid])

        # 4. Make scaling decision with DQN (using trained model, no live metrics needed)
        decision = await self._make_dqn_decision(
            current_replicas=current_replicas,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            recent_actions=recent_actions
        )

        # 5. Execute the decision
        success = True
        if decision.get("action") != "none":
            success = await self._execute_scaling(decision, deployment_name, target_namespace)
            if success:
                LOG.info(f"Successfully executed DQN scaling action for '{resource_name}'")
                self.successful_decisions += 1
            else:
                LOG.error(f"Failed to execute DQN scaling action for '{resource_name}'")
                decision["reason"] += " (execution failed)"

        # 6. Update action history
        newly_chosen_action_value = decision.get("ml_decision", {}).get("action_value", 2)
        self.action_histories[resource_uid].append(newly_chosen_action_value)

        # 7. Update performance metrics
        self.total_decisions += 1
        if not success:
            self.kserve_failures += 1

        # 8. Add KServe status to response
        decision["kserve_status"] = self._get_kserve_status()

        return {"current_replicas": current_replicas, **decision}

    async def _make_dqn_decision(self,
                                   current_replicas: int,
                                   min_replicas: int,
                                   max_replicas: int,
                                   recent_actions: List[int]) -> Dict[str, Any]:
        """
        Make scaling decision using DQN model inference with trained model
        """
        
        try:
            # Create simplified environment state (no live metrics needed)
            # The trained model already learned patterns from TrainData.csv
            simplified_state = {
                "feature_vector": [0.5] * 7,  # Neutral values - model uses learned patterns
                "feature_names": ["cpu_usage", "memory_usage", "request_rate", 
                                "response_time_p95", "error_rate", "pod_restart_rate", "queue_size"]
            }
            
            env_state = EnvironmentState.from_observability_data(
                unified_state=simplified_state,
                current_replicas=current_replicas,
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                recent_actions=recent_actions
            )
            
            # Use DQN for prediction
            action, metadata = await self.dqn_agent.select_action(
                state=env_state,
                force_valid=True
            )
            
            # Convert to decision format
            target_replicas = self._calculate_target_replicas(
                current_replicas, action, min_replicas, max_replicas
            )
            
            decision = {
                "action": "scale_up" if target_replicas > current_replicas else 
                         ("scale_down" if target_replicas < current_replicas else "none"),
                "target_replicas": target_replicas,
                "reason": f"DQN decision: {action.name} (confidence: {metadata.get('confidence', 0.0):.3f})",
                "ml_decision": {
                    "action_name": action.name,
                    "action_value": action.value,
                    "confidence": metadata.get("confidence", 0.0),
                    "prediction_source": metadata.get("prediction_source", "kserve"),
                    "q_values": metadata.get("q_values", []),
                    "kserve_metadata": metadata
                }
            }
            
            LOG.info(f"DQN decision: {action.name} -> {target_replicas} replicas "
                    f"(confidence: {metadata.get('confidence', 0.0):.3f})")
            
            return decision
            
        except Exception as e:
            LOG.error(f"DQN decision failed: {e}")
            return {
                "action": "none",
                "target_replicas": current_replicas,
                "reason": f"DQN decision failed: {str(e)}",
                "ml_decision": {
                    "action_name": "NO_ACTION",
                    "action_value": 2,
                    "confidence": 0.0,
                    "prediction_source": "error",
                    "error": str(e)
                }
            }

    def _calculate_target_replicas(self, current_replicas: int, action, min_replicas: int, max_replicas: int) -> int:
        """Calculate target replicas based on action."""
        from ml.state_representation import ScalingActions
        
        # Use the actual replica change method from ScalingActions
        change = action.get_replica_change()
        target = current_replicas + change
        return max(min_replicas, min(max_replicas, target))

    def _get_kserve_status(self) -> Dict[str, Any]:
        """Get current KServe integration status."""
        status = {
            "kserve_enabled": self.kserve_enabled,
            "kserve_endpoint": self.kserve_endpoint,
            "total_decisions": self.total_decisions,
            "successful_decisions": self.successful_decisions,
            "kserve_failures": self.kserve_failures,
            "success_rate": (self.successful_decisions / self.total_decisions) if self.total_decisions > 0 else 0.0
        }
        
        if self.dqn_agent:
            status.update(self.dqn_agent.get_kserve_metrics())
            
        return status

    async def get_kserve_metrics(self) -> Dict[str, Any]:
        """Get comprehensive KServe metrics for monitoring."""
        metrics = {
            "kserve_integration": {
                "enabled": self.kserve_enabled,
                "agent_available": self.dqn_agent is not None,
                "serving_endpoint": self.kserve_endpoint
            }
        }
        
        if self.dqn_agent:
            metrics["kserve"] = self.dqn_agent.get_kserve_metrics()
            metrics["dqn_performance"] = self.dqn_agent.get_performance_metrics()
            
        return metrics

    # Core Kubernetes operations (unchanged)
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
