# engine/langgraph_handler.py
# ============================================================================
# LangGraph-based Handler for NimbusGuard Operator - DQN Primary with LLM Validation
# ============================================================================

import logging
import asyncio
from collections import deque
from typing import Dict, Any, Optional, List
from datetime import datetime

import kopf
import kubernetes

from config import health_status
from langgraph_state import NimbusGuardState, ScalingDecision
from langgraph_workflow import run_scaling_workflow

LOG = logging.getLogger(__name__)


class LangGraphOperatorHandler:
    """
    LangGraph-based handler that preserves DQN as primary decision maker with LLM validation.
    This maintains compatibility with the existing KServe DQN agent while adding intelligent validation.
    """

    def __init__(self):
        """Initializes the handler with Kubernetes integration and workflow state management."""
        self.apps_api: Optional[kubernetes.client.AppsV1Api] = None
        self.workflow_histories: Dict[str, List[Dict[str, Any]]] = {}
        self.action_histories: Dict[str, deque] = {}  # Preserve action history for DQN
        
        # Performance tracking (compatible with original handler)
        self.total_decisions = 0
        self.successful_decisions = 0
        self.workflow_failures = 0
        
        # LangGraph specific tracking
        self.total_workflows = 0
        self.successful_workflows = 0

    async def initialize(self):
        """Initializes Kubernetes API clients and validates LangGraph environment."""
        try:
            # Initialize Kubernetes client (same as original)
            try:
                kubernetes.config.load_incluster_config()
                LOG.info("Loaded in-cluster Kubernetes config.")
            except kubernetes.config.ConfigException:
                kubernetes.config.load_kube_config()
                LOG.info("Loaded local Kubernetes config (for development).")
            
            self.apps_api = kubernetes.client.AppsV1Api()
            health_status["kubernetes"] = True
            LOG.info("Kubernetes client initialized successfully.")
            
            # Validate LangGraph environment
            await self._validate_langgraph_environment()
            
        except Exception as e:
            LOG.critical(f"LangGraph handler initialization failed: {e}", exc_info=True)
            health_status["kubernetes"] = False
            raise

    async def _validate_langgraph_environment(self):
        """Validate LangGraph and DQN environment setup."""
        try:
            import os
            
            # Check for required environment variables
            required_vars = {
                "KSERVE_ENDPOINT": "KServe DQN model endpoint (REQUIRED for DQN decisions)"
            }
            
            optional_vars = {
                "OPENAI_API_KEY": "OpenAI API for LLM validation (optional)",
                "PROMETHEUS_URL": "Prometheus metrics endpoint (optional)"
            }
            
            missing_required = []
            missing_optional = []
            
            for var, description in required_vars.items():
                if not os.getenv(var):
                    missing_required.append(f"{var} ({description})")
            
            for var, description in optional_vars.items():
                if not os.getenv(var):
                    missing_optional.append(f"{var} ({description})")
            
            if missing_required:
                error_msg = f"Missing REQUIRED environment variables: {', '.join(missing_required)}"
                LOG.error(error_msg)
                health_status["langgraph"] = False
                health_status["ml_decision_engine"] = False
                raise RuntimeError(error_msg)
            
            if missing_optional:
                LOG.warning(f"Missing optional environment variables: {', '.join(missing_optional)}")
                LOG.warning("Some LangGraph features may not be available")
                
            health_status["langgraph"] = True
            health_status["ml_decision_engine"] = True
            LOG.info("LangGraph environment validation passed - DQN decision engine ready")
            
        except Exception as e:
            LOG.error(f"LangGraph environment validation failed: {e}")
            health_status["langgraph"] = False
            health_status["ml_decision_engine"] = False

    async def evaluate_scaling_logic(self, body: Dict[str, Any], namespace: str) -> Dict[str, Any]:
        """
        LangGraph-based evaluation that preserves DQN as primary decision maker.
        This replaces the original evaluate_scaling_logic while maintaining compatibility.
        """
        spec = body.get("spec", {})
        meta = body.get("metadata", {})
        resource_uid = meta.get("uid")
        resource_name = meta.get("name")

        if not resource_uid:
            raise kopf.PermanentError("Cannot operate on a resource without a UID. Check resource metadata.")

        target_namespace = spec.get("namespace", namespace)
        target_labels = spec.get("target_labels", {})
        min_replicas = spec.get("minReplicas", 1)
        max_replicas = spec.get("maxReplicas", 10)

        LOG.info(f"Starting LangGraph workflow for '{resource_name}' (uid: {resource_uid}) - DQN primary decision")

        # Initialize action history for DQN (preserve compatibility)
        if resource_uid not in self.action_histories:
            self.action_histories[resource_uid] = deque([2] * 5, maxlen=5)  # NO_ACTION = 2

        # Get current replicas first (needed for initial state)
        current_replicas = await self._get_current_replicas_simple(target_labels, target_namespace)
        if current_replicas is None:
            msg = f"No deployment found for labels {target_labels} in namespace '{target_namespace}'."
            LOG.error(msg)
            raise kopf.PermanentError(msg)

        # Run the LangGraph workflow with DQN primary decision
        try:
            workflow_result = await run_scaling_workflow(
                resource_name=resource_name,
                resource_uid=resource_uid,
                namespace=target_namespace,
                current_replicas=current_replicas,
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                target_labels=target_labels
            )
            
            # Store workflow result for history tracking
            if resource_uid not in self.workflow_histories:
                self.workflow_histories[resource_uid] = deque(maxlen=10)
            self.workflow_histories[resource_uid].append(dict(workflow_result))
            
            # Update action history for DQN continuity
            if workflow_result.get("scaling_decision"):
                action_value = self._scaling_decision_to_action_value(workflow_result["scaling_decision"])
                self.action_histories[resource_uid].append(action_value)
            
            # Update performance metrics
            self.total_workflows += 1
            self.total_decisions += 1  # Maintain compatibility with original metrics
            
            if workflow_result.get("execution_success", False):
                self.successful_workflows += 1
                self.successful_decisions += 1  # Maintain compatibility
                LOG.info(f"LangGraph workflow completed successfully for '{resource_name}' - DQN decision executed")
            else:
                self.workflow_failures += 1
                LOG.error(f"LangGraph workflow failed for '{resource_name}': {workflow_result.get('execution_errors', [])}")

            # Format response for Kopf (compatible with original handler format)
            return self._format_workflow_response(workflow_result, resource_name)
            
        except Exception as e:
            LOG.error(f"LangGraph workflow execution failed for '{resource_name}': {e}", exc_info=True)
            self.workflow_failures += 1
            
            # Return error response (compatible with original handler format)
            return {
                "current_replicas": current_replicas,
                "action": "none",
                "target_replicas": current_replicas,
                "reason": f"LangGraph workflow failed: {str(e)}",
                "ml_decision": {
                    "action_name": "NO_ACTION",
                    "action_value": 2,
                    "confidence": 0.0,
                    "prediction_source": "error",
                    "error": str(e)
                },
                "langgraph_status": {
                    "workflow_failed": True,
                    "error": str(e),
                    "workflow_step": "failed"
                }
            }

    def _scaling_decision_to_action_value(self, scaling_decision) -> int:
        """Convert ScalingDecision enum to action value for DQN history."""
        if hasattr(scaling_decision, 'value'):
            return scaling_decision.value
        elif isinstance(scaling_decision, str):
            mapping = {
                "scale_down_2": 0,
                "scale_down_1": 1,
                "no_action": 2,
                "scale_up_1": 3,
                "scale_up_2": 4
            }
            return mapping.get(scaling_decision, 2)
        return 2  # Default to NO_ACTION

    def _format_workflow_response(self, workflow_state: Dict[str, Any], resource_name: str) -> Dict[str, Any]:
        """Format the LangGraph workflow result for Kopf response (compatible with original format)."""
        
        # Map ScalingDecision back to action string (same as original handler)
        scaling_decision = workflow_state.get("scaling_decision")
        if hasattr(scaling_decision, 'value'):
            decision_value = scaling_decision.value
        else:
            decision_value = str(scaling_decision) if scaling_decision else "no_action"
            
        action_mapping = {
            "scale_down_2": "scale_down",
            "scale_down_1": "scale_down", 
            "no_action": "none",
            "scale_up_1": "scale_up",
            "scale_up_2": "scale_up"
        }
        
        action = action_mapping.get(decision_value, "none")
        
        # Extract DQN metadata (preserve original ML decision format)
        dqn_metadata = workflow_state.get("dqn_metadata", {})
        
        response = {
            "current_replicas": workflow_state.get("current_replicas", 0),
            "action": action,
            "target_replicas": workflow_state.get("target_replicas", workflow_state.get("current_replicas", 0)),
            "reason": workflow_state.get("decision_reason", "LangGraph workflow completed"),
            
            # ML decision format (compatible with original handler)
            "ml_decision": {
                "action_name": dqn_metadata.get("dqn_action", "NO_ACTION"),
                "action_value": dqn_metadata.get("dqn_action_value", 2),
                "confidence": dqn_metadata.get("confidence", workflow_state.get("decision_confidence", 0.0)),
                "prediction_source": dqn_metadata.get("prediction_source", "kserve"),
                "q_values": dqn_metadata.get("q_values", []),
                "dqn_metadata": dqn_metadata
            },
            
            # LangGraph-specific information (additional)
            "langgraph_status": {
                "workflow_completed": workflow_state.get("workflow_step") == "finalized",
                "execution_success": workflow_state.get("execution_success", False),
                "decision_confidence": workflow_state.get("decision_confidence", 0.0),
                "observability_confidence": workflow_state.get("confidence_score", 0.0),
                "observability_health": str(workflow_state.get("observability_health", "unknown")),
                "reasoning_steps": len(workflow_state.get("reasoning_steps", [])),
                "workflow_duration": workflow_state.get("performance_metrics", {}).get("total_workflow_duration", 0.0),
                "llm_validation": bool(workflow_state.get("llm_analysis")),
                "llm_override": "LLM override" in workflow_state.get("decision_reason", "")
            },
            
            # System Health (compatible with original format)
            "kserve_status": {
                "kserve_available": workflow_state.get("kserve_available", False),
                "total_decisions": self.total_decisions,
                "successful_decisions": self.successful_decisions,
                "success_rate": (self.successful_decisions / self.total_decisions) if self.total_decisions > 0 else 0.0
            },
            
            # Execution details
            "execution_details": {
                "execution_plan": workflow_state.get("execution_plan", []),
                "execution_errors": workflow_state.get("execution_errors", []),
                "deployment_name": workflow_state.get("deployment_name", "unknown")
            }
        }
        
        return response

    async def _get_current_replicas_simple(self, labels: Dict[str, str], ns: str) -> Optional[int]:
        """Simple method to get current replicas (used for initial state)."""
        if not self.apps_api:
            await self.initialize()

        if not labels:
            LOG.error("No target_labels specified in the spec.")
            return None
            
        try:
            selector = ",".join(f"{k}={v}" for k, v in labels.items())
            deploys = self.apps_api.list_namespaced_deployment(namespace=ns, label_selector=selector)
            if not deploys.items: 
                return None
            deployment = deploys.items[0]
            replicas = deployment.status.replicas if deployment.status.replicas is not None else 0
            return replicas
        except Exception as e:
            LOG.error(f"Failed to get replicas for labels '{labels}' in '{ns}': {e}")
            return None

    async def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get comprehensive LangGraph workflow metrics for monitoring (compatible with original)."""
        metrics = {
            "langgraph_workflows": {
                "total_workflows": self.total_workflows,
                "successful_workflows": self.successful_workflows,
                "workflow_failures": self.workflow_failures,
                "success_rate": (self.successful_workflows / self.total_workflows) if self.total_workflows > 0 else 0.0
            },
            
            # Compatible with original handler metrics
            "dqn_performance": {
                "total_decisions": self.total_decisions,
                "successful_decisions": self.successful_decisions,
                "success_rate": (self.successful_decisions / self.total_decisions) if self.total_decisions > 0 else 0.0
            },
            
            "recent_decisions": []
        }
        
        # Calculate average performance metrics from recent workflows
        all_workflows = []
        for history in self.workflow_histories.values():
            all_workflows.extend(history)
        
        if all_workflows:
            recent_workflows = all_workflows[-10:]  # Last 10 workflows
            
            avg_duration = sum(w.get("performance_metrics", {}).get("total_workflow_duration", 0.0) for w in recent_workflows) / len(recent_workflows)
            avg_confidence = sum(w.get("decision_confidence", 0.0) for w in recent_workflows) / len(recent_workflows)
            
            metrics["workflow_performance"] = {
                "avg_workflow_duration": avg_duration,
                "avg_decision_confidence": avg_confidence
            }
            
            # Recent decisions summary
            for workflow in recent_workflows[-5:]:
                if workflow.get("scaling_decision"):
                    decision_str = str(workflow.get("scaling_decision", "unknown"))
                    metrics["recent_decisions"].append({
                        "resource": workflow.get("resource_name", "unknown"),
                        "decision": decision_str,
                        "confidence": workflow.get("decision_confidence", 0.0),
                        "timestamp": workflow.get("decision_made_at", 0.0)
                    })
        
        return metrics

    def get_workflow_history(self, resource_uid: str) -> List[Dict[str, Any]]:
        """Get workflow history for a specific resource."""
        return list(self.workflow_histories.get(resource_uid, []))

    # Maintain compatibility with original handler methods
    async def verify_scaling_effect(self, name: str, ns: str, expected_replicas: int, max_wait_seconds: int = 30) -> bool:
        """
        Verify that a scaling action actually took effect by checking the deployment.
        (Preserved from original handler for compatibility)
        """
        for i in range(max_wait_seconds):
            try:
                current_replicas = await self._get_current_replicas_simple({"app": name}, ns)
                if current_replicas == expected_replicas:
                    LOG.info(f"Scaling verified: '{name}' now has {current_replicas} replicas")
                    return True
                await asyncio.sleep(1)
            except Exception as e:
                LOG.warning(f"Error during scaling verification: {e}")
                
        LOG.warning(f"Scaling verification failed: '{name}' did not reach {expected_replicas} replicas in {max_wait_seconds}s")
        return False 