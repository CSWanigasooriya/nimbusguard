# engine/handler.py - Fixed with proper scaling logic and loop prevention
# ============================================================================
# Kubernetes Interaction and Orchestration
# ============================================================================

import logging
from collections import deque
from typing import Dict, Any, Optional, List

import kopf
import kubernetes

from config import health_status
from observability import ObservabilityCollector
from rules import make_decision
from ml.state_representation import EnvironmentState

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
        # DQN Experience Collection - stores (state, action) pairs waiting for next state
        self.pending_experiences: Dict[str, Dict[str, Any]] = {}
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

        # 9. DQN Experience Collection (Proper Implementation)
        if spec.get("ml_config", {}).get("training_mode", False):
            await self._handle_dqn_experience_collection(
                resource_uid=resource_uid,
                resource_name=resource_name,
                spec=spec,
                unified_state=unified_state,
                current_replicas=current_replicas,
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                recent_actions=recent_actions,
                decision=decision,
                execution_success=success if 'success' in locals() else True
            )

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

    async def _handle_dqn_experience_collection(
        self, 
        resource_uid: str,
        resource_name: str,
        spec: Dict[str, Any],
        unified_state: Dict[str, Any],
        current_replicas: int,
        min_replicas: int,
        max_replicas: int,
        recent_actions: List[int],
        decision: Dict[str, Any],
        execution_success: bool
    ):
        """
        Proper DQN experience collection that handles state transitions over time.
        This is the REAL implementation that properly stores (s, a, r, s') tuples.
        
        The key insight: We can't calculate proper rewards immediately because we need to:
        1. Execute the action 
        2. Wait for the system to respond (30s evaluation interval)
        3. Observe the new state 
        4. Then calculate reward based on the actual state transition
        """
        try:
            from ml.state_representation import EnvironmentState, ScalingActions
            from rules import _initialize_dqn_agent
            import time
            
            # Create current environment state
            current_env_state = EnvironmentState.from_observability_data(
                unified_state=unified_state,
                current_replicas=current_replicas,
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                recent_actions=recent_actions
            )
            
            newly_chosen_action_value = decision.get("ml_decision", {}).get("action_value", 2)
            action_taken = ScalingActions(newly_chosen_action_value)
            
            # Step 1: Process any pending experience from previous iteration
            if resource_uid in self.pending_experiences:
                await self._complete_pending_experience(resource_uid, resource_name, current_env_state, spec)
            
            # Step 2: Store current state and action for next iteration
            # Only if we actually took an action (not NO_ACTION)
            if action_taken != ScalingActions.NO_ACTION:
                self.pending_experiences[resource_uid] = {
                    "previous_state": current_env_state,
                    "action": action_taken,
                    "timestamp": time.time(),
                    "execution_success": execution_success,
                    "resource_name": resource_name
                }
                LOG.info(f"[{resource_name}] Stored experience for action {action_taken.name}, waiting for next state...")
            else:
                # For NO_ACTION, we can calculate reward immediately since there's no state change expected
                agent = _initialize_dqn_agent(spec)
                reward_components = agent.evaluate_decision(
                    previous_state=current_env_state,
                    action=action_taken,
                    current_state=current_env_state,  # Same state for NO_ACTION
                    execution_success=execution_success
                )
                
                LOG.info(f"[{resource_name}] NO_ACTION Reward: "
                        f"Total: {reward_components.total_reward:.2f} | "
                        f"Performance: {reward_components.performance_reward:.2f} | "
                        f"Efficiency: {reward_components.efficiency_reward:.2f} | "
                        f"Stability: {reward_components.stability_reward:.2f}")
                        
        except Exception as e:
            LOG.warning(f"Failed to handle DQN experience collection: {e}")

    async def _complete_pending_experience(
        self, 
        resource_uid: str, 
        resource_name: str, 
        next_state: EnvironmentState, 
        spec: Dict[str, Any]
    ):
        """
        Complete a pending experience by calculating reward and storing in replay buffer.
        This creates the proper (s, a, r, s') tuple for DQN training.
        
        This is called when we have:
        - Previous state (from last evaluation)  
        - Action taken (from last evaluation)
        - Current state (from this evaluation)
        - Can now calculate the reward based on actual state transition!
        """
        try:
            from rules import _initialize_dqn_agent
            import time
            
            pending = self.pending_experiences[resource_uid]
            previous_state = pending["previous_state"]
            action = pending["action"]
            execution_success = pending["execution_success"]
            time_elapsed = time.time() - pending["timestamp"]
            
            # Calculate reward based on actual state transition
            agent = _initialize_dqn_agent(spec)
            reward_components = agent.evaluate_decision(
                previous_state=previous_state,
                action=action,
                current_state=next_state,
                execution_success=execution_success
            )
            
            # Store the complete experience in the agent's replay buffer
            # This is where the real learning happens!
            agent.store_experience(
                state=previous_state,
                action=action,
                reward_components=reward_components,
                next_state=next_state,
                done=False  # Episodes don't really "end" in continuous control
            )
            
            # Perform training step if enough experiences are collected
            training_result = agent.train_step()
            
            LOG.info(f"[{resource_name}] COMPLETE EXPERIENCE | Action: {action.name} | "
                    f"Time elapsed: {time_elapsed:.1f}s | "
                    f"Reward: {reward_components.total_reward:.2f} "
                    f"(Perf: {reward_components.performance_reward:.2f}, "
                    f"Eff: {reward_components.efficiency_reward:.2f}, "
                    f"Stab: {reward_components.stability_reward:.2f})")
            
            if training_result:
                LOG.info(f"[{resource_name}] DQN Training Step | "
                        f"Loss: {training_result['loss']:.4f} | "
                        f"Epsilon: {training_result['epsilon']:.3f}")
            
            # Remove the completed experience
            del self.pending_experiences[resource_uid]
            
        except Exception as e:
            LOG.error(f"Failed to complete pending experience: {e}")
            # Remove the failed experience to avoid getting stuck
            if resource_uid in self.pending_experiences:
                del self.pending_experiences[resource_uid]
