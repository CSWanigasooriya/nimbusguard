#!/usr/bin/env python3
"""
NimbusGuard LangGraph Operator

AI-powered Kubernetes operator for intelligent autoscaling using LangGraph workflows
and Q-learning reinforcement learning. This operator follows Kubernetes operator 
framework best practices using kopf.
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import kopf
import kubernetes
from kubernetes.client.rest import ApiException
import structlog

# Import our AI components
from workflows.scaling_state import (
    ScalingWorkflowState, 
    WorkflowStatus, 
    create_initial_state,
    MetricData,
    ActionResult
)
from agents.supervisor_agent import SupervisorAgent
from agents.state_observer_agent import StateObserverAgent  
from ml_models.q_learning import QLearningAgent

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class NimbusGuardOperator:
    """
    Main NimbusGuard operator class that manages AI-powered scaling workflows.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.k8s_apps_v1 = kubernetes.client.AppsV1Api()
        self.k8s_core_v1 = kubernetes.client.CoreV1Api()
        
        # Initialize AI agents
        self.q_agent = QLearningAgent(config)
        self.supervisor = SupervisorAgent(config)
        self.state_observer = StateObserverAgent(config)
        
        # Track active workflows
        self.active_workflows: Dict[str, ScalingWorkflowState] = {}
        
        logger.info("NimbusGuard operator initialized", 
                   version="0.1.0", 
                   agents=["supervisor", "state_observer", "q_learning"])

    async def trigger_scaling_workflow(self, event_data: Dict[str, Any]) -> str:
        """
        Trigger a new scaling workflow based on an event.
        
        Args:
            event_data: Event that triggered the scaling workflow
            
        Returns:
            Workflow ID
        """
        workflow_id = f"workflow-{uuid.uuid4().hex[:8]}"
        
        # Create initial state
        initial_state = create_initial_state(
            workflow_id=workflow_id,
            trigger_event=event_data,
            config=self.config
        )
        
        # Store workflow
        self.active_workflows[workflow_id] = initial_state
        
        logger.info("Started scaling workflow", 
                   workflow_id=workflow_id,
                   trigger_event=event_data)
        
        # Start the workflow execution
        asyncio.create_task(self._execute_workflow(workflow_id))
        
        return workflow_id

    async def _execute_workflow(self, workflow_id: str):
        """Execute a scaling workflow using the supervisor and specialized agents."""
        try:
            state = self.active_workflows[workflow_id]
            state["status"] = WorkflowStatus.OBSERVING
            
            # Execute workflow steps
            max_iterations = 20
            iteration = 0
            
            while not state["completed"] and iteration < max_iterations:
                iteration += 1
                
                # Route to appropriate agent via supervisor
                command = await self.supervisor.invoke(state)
                
                # Execute command based on supervisor decision
                if command.update:
                    state.update(command.update)
                
                if command.goto == "__end__":
                    state["completed"] = True
                    state["status"] = WorkflowStatus.COMPLETED
                    break
                elif command.goto == "state_observer":
                    await self._handle_state_observation(state)
                elif command.goto == "decision_agent":
                    await self._handle_decision_making(state)
                elif command.goto == "action_executor":
                    await self._handle_action_execution(state)
                elif command.goto == "reward_calculator":
                    await self._handle_reward_calculation(state)
                
                await asyncio.sleep(0.1)
            
            # Clean up completed workflow
            if state["completed"]:
                logger.info("Workflow completed", 
                           workflow_id=workflow_id,
                           iterations=iteration)
                del self.active_workflows[workflow_id]
            
        except Exception as e:
            logger.error("Workflow execution error", 
                        workflow_id=workflow_id, 
                        error=str(e))
            
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id]["status"] = WorkflowStatus.FAILED
                self.active_workflows[workflow_id]["errors"].append(str(e))

    async def _handle_state_observation(self, state: ScalingWorkflowState):
        """Handle state observation phase."""
        try:
            state["status"] = WorkflowStatus.OBSERVING
            state["current_agent"] = "state_observer"
            
            # Get current metrics
            metrics = await self.state_observer.collect_metrics()
            
            if metrics:
                metric_data = MetricData(
                    cpu_utilization=metrics.get("cpu_utilization", 0),
                    memory_utilization=metrics.get("memory_utilization", 0),
                    request_rate=metrics.get("request_rate", 0),
                    error_rate=metrics.get("error_rate", 0),
                    pod_count=metrics.get("pod_count", 1)
                )
                
                state["current_metrics"] = metric_data
                state["metrics_history"].append(metric_data)
                
                logger.info("Collected metrics", 
                           workflow_id=state["workflow_id"],
                           cpu=metric_data.cpu_utilization,
                           memory=metric_data.memory_utilization)
            
        except Exception as e:
            logger.error("State observation error", error=str(e))
            state["errors"].append(f"State observation failed: {str(e)}")

    async def _handle_decision_making(self, state: ScalingWorkflowState):
        """Handle AI decision making phase."""
        try:
            state["status"] = WorkflowStatus.DECIDING
            state["current_agent"] = "decision_agent"
            
            if not state["current_metrics"]:
                logger.warning("No metrics available for decision making")
                return
            
            # Convert metrics for Q-learning agent
            metrics_dict = {
                "cpu_utilization": state["current_metrics"].cpu_utilization,
                "memory_utilization": state["current_metrics"].memory_utilization,
                "request_rate": state["current_metrics"].request_rate,
                "error_rate": state["current_metrics"].error_rate,
                "pod_count": state["current_metrics"].pod_count
            }
            
            # Get Q-learning recommendation
            q_decision = self.q_agent.choose_scaling_action(metrics_dict)
            
            # Create scaling decision
            from workflows.scaling_state import ScalingDecision, ScalingAction
            
            action_map = {
                "scale_up": ScalingAction.SCALE_UP,
                "scale_down": ScalingAction.SCALE_DOWN,
                "no_action": ScalingAction.MAINTAIN
            }
            
            decision = ScalingDecision(
                action=action_map.get(q_decision["action"], ScalingAction.MAINTAIN),
                target_replicas=q_decision["target_replicas"],
                current_replicas=state["current_metrics"].pod_count,
                confidence=q_decision["confidence"],
                reasoning=q_decision["reasoning"],
                metrics_snapshot=state["current_metrics"]
            )
            
            state["scaling_decision"] = decision
            state["decisions_history"].append(decision)
            
            logger.info("AI decision made", 
                       workflow_id=state["workflow_id"],
                       action=decision.action.value,
                       target_replicas=decision.target_replicas)
            
        except Exception as e:
            logger.error("Decision making error", error=str(e))
            state["errors"].append(f"Decision making failed: {str(e)}")

    async def _handle_action_execution(self, state: ScalingWorkflowState):
        """Handle scaling action execution."""
        try:
            state["status"] = WorkflowStatus.EXECUTING
            state["current_agent"] = "action_executor"
            
            decision = state["scaling_decision"]
            if not decision:
                logger.warning("No scaling decision available")
                return
            
            # Execute the scaling action
            result = await self._execute_scaling_action(decision)
            
            state["last_action"] = result
            state["action_history"].append(result)
            
            logger.info("Action executed", 
                       workflow_id=state["workflow_id"],
                       success=result.success)
            
        except Exception as e:
            logger.error("Action execution error", error=str(e))
            state["errors"].append(f"Action execution failed: {str(e)}")

    async def _handle_reward_calculation(self, state: ScalingWorkflowState):
        """Handle reward calculation for Q-learning."""
        try:
            state["current_agent"] = "reward_calculator"
            
            # Get metrics after action
            post_action_metrics = await self.state_observer.collect_metrics()
            
            if state["last_action"] and post_action_metrics:
                # Calculate reward using Q-learning module
                from ml_models.q_learning import calculate_reward
                
                pre_metrics = {
                    "cpu_utilization": state["scaling_decision"].metrics_snapshot.cpu_utilization,
                    "memory_utilization": state["scaling_decision"].metrics_snapshot.memory_utilization,
                    "request_rate": state["scaling_decision"].metrics_snapshot.request_rate,
                    "error_rate": state["scaling_decision"].metrics_snapshot.error_rate,
                    "pod_count": state["scaling_decision"].metrics_snapshot.pod_count
                }
                
                reward, components = calculate_reward(
                    pre_metrics, 
                    post_action_metrics,
                    state["last_action"].action_taken.value,
                    self.config
                )
                
                # Provide feedback to Q-learning agent
                self.q_agent.provide_feedback(reward, post_action_metrics)
                
                # Store reward
                from workflows.scaling_state import RewardSignal
                reward_signal = RewardSignal(
                    reward=reward,
                    components=components,
                    metrics_before=state["scaling_decision"].metrics_snapshot,
                    metrics_after=None  # Will be populated later
                )
                
                state["last_reward"] = reward_signal
                state["reward_history"].append(reward_signal)
                state["cumulative_reward"] += reward
                
                logger.info("Reward calculated", 
                           workflow_id=state["workflow_id"],
                           reward=reward)
            
        except Exception as e:
            logger.error("Reward calculation error", error=str(e))
            state["errors"].append(f"Reward calculation failed: {str(e)}")

    async def _execute_scaling_action(self, decision) -> ActionResult:
        """Execute the actual scaling action on Kubernetes."""
        
        try:
            # Target deployment configuration
            deployment_name = "nimbusguard-consumer"
            namespace = "default"
            
            # Get current deployment
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=deployment_name, 
                namespace=namespace
            )
            
            current_replicas = deployment.spec.replicas
            target_replicas = decision.target_replicas
            
            # Update deployment if needed
            if current_replicas != target_replicas and decision.action.value != "maintain":
                deployment.spec.replicas = target_replicas
                
                self.k8s_apps_v1.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace,
                    body=deployment
                )
                
                return ActionResult(
                    success=True,
                    action_taken=decision.action,
                    old_replicas=current_replicas,
                    new_replicas=target_replicas
                )
            else:
                return ActionResult(
                    success=True,
                    action_taken=decision.action,
                    old_replicas=current_replicas,
                    new_replicas=current_replicas
                )
                
        except ApiException as e:
            logger.error("Kubernetes API error", error=str(e))
            return ActionResult(
                success=False,
                action_taken=decision.action,
                old_replicas=decision.current_replicas,
                new_replicas=decision.current_replicas,
                error_message=str(e)
            )
        except Exception as e:
            logger.error("Scaling action error", error=str(e))
            return ActionResult(
                success=False,
                action_taken=decision.action,
                old_replicas=decision.current_replicas,
                new_replicas=decision.current_replicas,
                error_message=str(e)
            )


# Global operator instance
nimbusguard_operator: Optional[NimbusGuardOperator] = None


@kopf.on.startup()
async def startup_handler(settings: kopf.OperatorSettings, **kwargs):
    """Initialize the operator on startup."""
    global nimbusguard_operator
    
    # Configure kopf settings
    settings.posting.level = logging.INFO
    settings.watching.connect_timeout = 1 * 60
    settings.watching.server_timeout = 10 * 60
    
    # Load configuration
    config = {
        "agents": {
            "supervisor": {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 1000
            }
        },
        "q_learning": {
            "learning_rate": 0.1,
            "discount_factor": 0.95,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995
        }
    }
    
    # Initialize Kubernetes client
    try:
        kubernetes.config.load_incluster_config()
        logger.info("Loaded in-cluster Kubernetes config")
    except kubernetes.config.ConfigException:
        try:
            kubernetes.config.load_kube_config()
            logger.info("Loaded local Kubernetes config")
        except kubernetes.config.ConfigException:
            logger.error("Could not load Kubernetes config")
            raise
    
    # Initialize operator
    nimbusguard_operator = NimbusGuardOperator(config)
    
    logger.info("NimbusGuard operator started successfully")


@kopf.on.cleanup()
async def cleanup_handler(**kwargs):
    """Clean up on operator shutdown."""
    global nimbusguard_operator
    
    if nimbusguard_operator:
        # Save Q-learning model
        try:
            nimbusguard_operator.q_agent.save_model("/tmp/q_learning_model.pkl")
            logger.info("Saved Q-learning model on shutdown")
        except Exception as e:
            logger.error("Failed to save Q-learning model", error=str(e))
    
    logger.info("NimbusGuard operator shut down successfully")


# Custom Resource Definition handlers
@kopf.on.create('nimbusguard.io', 'v1', 'scalingpolicies')
async def create_scaling_policy(spec, name, namespace, logger, **kwargs):
    """Handle creation of NimbusGuard scaling policies."""
    logger.info(f"Creating scaling policy: {name}")
    
    return {"status": "created", "message": f"Scaling policy {name} created successfully"}


@kopf.on.update('nimbusguard.io', 'v1', 'scalingpolicies')
async def update_scaling_policy(spec, name, namespace, logger, **kwargs):
    """Handle updates to NimbusGuard scaling policies."""
    logger.info(f"Updating scaling policy: {name}")
    return {"status": "updated", "message": f"Scaling policy {name} updated successfully"}


@kopf.on.delete('nimbusguard.io', 'v1', 'scalingpolicies')
async def delete_scaling_policy(spec, name, namespace, logger, **kwargs):
    """Handle deletion of NimbusGuard scaling policies."""
    logger.info(f"Deleting scaling policy: {name}")
    return {"status": "deleted", "message": f"Scaling policy {name} deleted successfully"}


# Event handlers for triggering scaling workflows
@kopf.on.event('apps', 'v1', 'deployments')
async def deployment_event_handler(event, name, namespace, **kwargs):
    """Handle deployment events that might trigger scaling."""
    global nimbusguard_operator
    
    if not nimbusguard_operator:
        return
    
    # Only handle events for deployments we're managing
    if name == "nimbusguard-consumer":
        event_data = {
            "event_type": "deployment_change",
            "resource": f"{namespace}/{name}",
            "event": event,
            "timestamp": datetime.now().isoformat()
        }
        
        # Trigger scaling workflow
        workflow_id = await nimbusguard_operator.trigger_scaling_workflow(event_data)
        logger.info("Triggered scaling workflow for deployment event", 
                   workflow_id=workflow_id)


if __name__ == "__main__":
    # Run the operator
    kopf.run(
        module=__name__,
        liveness_endpoint=":8080/healthz",
        ready_endpoint=":8080/readyz"
    ) 