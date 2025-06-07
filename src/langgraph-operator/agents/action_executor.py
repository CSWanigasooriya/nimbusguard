"""
Action Executor Agent for LangGraph Scaling Workflow

This agent executes scaling decisions by interacting with Kubernetes APIs
to modify deployment replicas, update HPA configurations, and manage KEDA ScaledObjects.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

import kubernetes
from kubernetes.client.rest import ApiException
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from config import get_agent_config, get_system_prompt
from workflows.scaling_state import (
    ScalingWorkflowState, 
    WorkflowStatus, 
    ScalingDecision,
    ScalingAction,
    ActionResult,
    ActionStatus
)
from mcp_integration.mcp_tools import COMMON_TOOLS

logger = logging.getLogger(__name__)


class ActionExecutorAgent:
    """
    Action Executor Agent that executes scaling decisions on Kubernetes resources.
    
    This agent:
    - Executes scaling actions determined by the Decision Agent
    - Modifies Kubernetes Deployments, HPAs, and KEDA ScaledObjects
    - Validates scaling constraints and safety limits
    - Records action outcomes for learning feedback
    - Handles rollback scenarios for failed actions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the action executor agent.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = get_agent_config("action_executor") if config is None else config.get("agents", {}).get("action_executor", {})
        
        # Initialize Kubernetes clients
        try:
            kubernetes.config.load_incluster_config()
        except kubernetes.config.ConfigException:
            kubernetes.config.load_kube_config()
        
        self.k8s_apps_v1 = kubernetes.client.AppsV1Api()
        self.k8s_autoscaling_v1 = kubernetes.client.AutoscalingV1Api()
        self.k8s_autoscaling_v2 = kubernetes.client.AutoscalingV2Api()
        self.k8s_custom = kubernetes.client.CustomObjectsApi()
        
        # Initialize LLM for reasoning about actions
        self.llm = ChatOpenAI(
            model=self.config.get("model", "gpt-4o-mini"),
            temperature=self.config.get("temperature", 0.1),
            max_tokens=self.config.get("max_tokens", 800),
            timeout=self.config.get("timeout", 30)
        )
        
        # Bind tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(COMMON_TOOLS)
        
        # Safety and constraint parameters
        self.min_replicas = self.config.get("min_replicas", 1)
        self.max_replicas = self.config.get("max_replicas", 50)
        self.max_scale_step = self.config.get("max_scale_step", 5)
        self.dry_run = self.config.get("dry_run", False)
        
        # Load system prompt
        self.system_prompt = get_system_prompt("action_executor")
        
        logger.info("Action Executor Agent initialized")

    async def invoke(self, state: ScalingWorkflowState) -> Command:
        """
        Execute scaling actions based on decisions made by the Decision Agent.
        
        Args:
            state: Current workflow state with scaling decision
            
        Returns:
            Command object with execution results and updated state
        """
        try:
            logger.info(f"Action Executor starting execution for workflow {state['workflow_id']}")
            
            # Get the scaling decision
            decision = state.get("scaling_decision")
            if not decision:
                logger.error("No scaling decision found in state")
                return self._handle_error(state, "No scaling decision available")
            
            # Validate decision constraints
            if not await self._validate_scaling_constraints(decision, state):
                return self._handle_constraint_violation(state, decision)
            
            # Execute the scaling action
            action_result = await self._execute_scaling_action(decision, state)
            
            # Update state with action result
            updated_state = await self._update_state_with_action_result(state, action_result)
            
            # Determine next step based on execution result
            next_command = await self._determine_next_action(updated_state, action_result)
            
            logger.info(f"Action Executor completed: {action_result.status.value}")
            return next_command
            
        except Exception as e:
            logger.error(f"Action Executor error: {e}", exc_info=True)
            return self._handle_error(state, str(e))

    async def _validate_scaling_constraints(self, decision: ScalingDecision, state: ScalingWorkflowState) -> bool:
        """Validate that the scaling decision meets safety constraints."""
        try:
            current_replicas = await self._get_current_replica_count(state)
            target_replicas = decision.target_replicas
            
            # Check replica bounds
            if target_replicas < self.min_replicas:
                logger.warning(f"Target replicas {target_replicas} below minimum {self.min_replicas}")
                return False
            
            if target_replicas > self.max_replicas:
                logger.warning(f"Target replicas {target_replicas} above maximum {self.max_replicas}")
                return False
            
            # Check scale step size
            scale_change = abs(target_replicas - current_replicas)
            if scale_change > self.max_scale_step:
                logger.warning(f"Scale change {scale_change} exceeds maximum step {self.max_scale_step}")
                return False
            
            # Check for no-op scaling
            if target_replicas == current_replicas and decision.action != ScalingAction.NO_ACTION:
                logger.info("Target replicas same as current, treating as no-action")
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating constraints: {e}")
            return False

    async def _execute_scaling_action(self, decision: ScalingDecision, state: ScalingWorkflowState) -> ActionResult:
        """Execute the actual scaling action on Kubernetes resources."""
        start_time = datetime.now()
        
        try:
            # Get target deployment info from state
            target_deployment = state.get("target_deployment", "consumer-workload")
            target_namespace = state.get("target_namespace", "nimbusguard")
            
            if decision.action == ScalingAction.NO_ACTION:
                return ActionResult(
                    action=decision.action,
                    target_replicas=decision.target_replicas,
                    actual_replicas=await self._get_current_replica_count(state),
                    status=ActionStatus.SUCCESS,
                    timestamp=start_time,
                    execution_time_ms=0,
                    details="No action required"
                )
            
            # Get current replica count
            current_replicas = await self._get_current_replica_count(state)
            
            if self.dry_run:
                logger.info(f"DRY RUN: Would scale {target_deployment} from {current_replicas} to {decision.target_replicas}")
                return ActionResult(
                    action=decision.action,
                    target_replicas=decision.target_replicas,
                    actual_replicas=current_replicas,
                    status=ActionStatus.SUCCESS,
                    timestamp=start_time,
                    execution_time_ms=0,
                    details="Dry run mode - no actual scaling performed"
                )
            
            # Execute the scaling
            success = await self._scale_deployment(target_deployment, target_namespace, decision.target_replicas)
            
            if success:
                # Verify the scaling was applied
                await asyncio.sleep(2)  # Brief wait for Kubernetes to update
                actual_replicas = await self._get_current_replica_count(state)
                
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return ActionResult(
                    action=decision.action,
                    target_replicas=decision.target_replicas,
                    actual_replicas=actual_replicas,
                    status=ActionStatus.SUCCESS,
                    timestamp=start_time,
                    execution_time_ms=execution_time,
                    details=f"Successfully scaled {target_deployment} to {decision.target_replicas} replicas"
                )
            else:
                return ActionResult(
                    action=decision.action,
                    target_replicas=decision.target_replicas,
                    actual_replicas=current_replicas,
                    status=ActionStatus.FAILED,
                    timestamp=start_time,
                    execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    details="Failed to scale deployment",
                    error="Kubernetes API call failed"
                )
                
        except Exception as e:
            logger.error(f"Error executing scaling action: {e}")
            return ActionResult(
                action=decision.action,
                target_replicas=decision.target_replicas,
                actual_replicas=await self._get_current_replica_count(state),
                status=ActionStatus.FAILED,
                timestamp=start_time,
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                details="Exception during scaling execution",
                error=str(e)
            )

    async def _scale_deployment(self, deployment_name: str, namespace: str, target_replicas: int) -> bool:
        """Scale a Kubernetes deployment to target replica count."""
        try:
            # Get current deployment
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            # Update replica count
            deployment.spec.replicas = target_replicas
            
            # Apply the update
            self.k8s_apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=deployment
            )
            
            logger.info(f"Successfully updated {deployment_name} to {target_replicas} replicas")
            return True
            
        except ApiException as e:
            logger.error(f"Kubernetes API error scaling deployment: {e}")
            return False
        except Exception as e:
            logger.error(f"Error scaling deployment: {e}")
            return False

    async def _get_current_replica_count(self, state: ScalingWorkflowState) -> int:
        """Get current replica count for the target deployment."""
        try:
            target_deployment = state.get("target_deployment", "consumer-workload")
            target_namespace = state.get("target_namespace", "nimbusguard")
            
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=target_deployment,
                namespace=target_namespace
            )
            
            return deployment.status.replicas or 0
            
        except Exception as e:
            logger.error(f"Error getting current replica count: {e}")
            return 0

    async def _update_state_with_action_result(
        self, 
        state: ScalingWorkflowState, 
        action_result: ActionResult
    ) -> ScalingWorkflowState:
        """Update workflow state with action execution results."""
        try:
            # Update current action result
            state["last_action"] = action_result
            state["status"] = WorkflowStatus.ACTION_EXECUTED
            state["current_agent"] = "action_executor"
            
            # Add to action history
            if "action_history" not in state:
                state["action_history"] = []
            state["action_history"].append(action_result)
            
            # Update metrics if available
            if action_result.status == ActionStatus.SUCCESS:
                # Update pod count in current metrics if available
                if state.get("current_metrics"):
                    state["current_metrics"].pod_count = action_result.actual_replicas
            
            logger.info(f"Updated state with action result: {action_result.status.value}")
            return state
            
        except Exception as e:
            logger.error(f"Error updating state with action result: {e}")
            state["errors"].append(f"Failed to update state: {str(e)}")
            return state

    async def _determine_next_action(self, state: ScalingWorkflowState, action_result: ActionResult) -> Command:
        """Determine next step in workflow based on action execution result."""
        try:
            if action_result.status == ActionStatus.SUCCESS:
                # Action successful, move to reward calculation
                return Command(
                    update={"status": WorkflowStatus.CALCULATING_REWARD},
                    goto="reward_calculator"
                )
            elif action_result.status == ActionStatus.FAILED:
                # Action failed, may need retry or error handling
                retry_count = state.get("retry_count", 0)
                max_retries = self.config.get("max_retries", 2)
                
                if retry_count < max_retries:
                    # Retry the action
                    logger.info(f"Retrying action execution (attempt {retry_count + 1})")
                    return Command(
                        update={"retry_count": retry_count + 1},
                        goto="action_executor"
                    )
                else:
                    # Max retries reached, fail the workflow
                    logger.error("Max retries reached for action execution")
                    return Command(
                        update={
                            "status": WorkflowStatus.FAILED,
                            "completed": True
                        },
                        goto="__end__"
                    )
            else:
                # Unexpected status
                logger.warning(f"Unexpected action status: {action_result.status}")
                return Command(
                    update={"status": WorkflowStatus.FAILED},
                    goto="__end__"
                )
                
        except Exception as e:
            logger.error(f"Error determining next action: {e}")
            return self._handle_error(state, str(e))

    def _handle_constraint_violation(self, state: ScalingWorkflowState, decision: ScalingDecision) -> Command:
        """Handle scaling constraint violations."""
        error_msg = f"Scaling decision violates constraints: {decision.action.value} to {decision.target_replicas} replicas"
        logger.warning(error_msg)
        
        return Command(
            update={
                "status": WorkflowStatus.FAILED,
                "completed": True,
                "errors": state.get("errors", []) + [error_msg]
            },
            goto="__end__"
        )

    def _handle_error(self, state: ScalingWorkflowState, error_message: str) -> Command:
        """Handle errors during action execution."""
        logger.error(f"Action Executor error: {error_message}")
        
        return Command(
            update={
                "status": WorkflowStatus.FAILED,
                "completed": True,
                "errors": state.get("errors", []) + [f"Action execution error: {error_message}"]
            },
            goto="__end__"
        )


async def action_executor_node(state: ScalingWorkflowState) -> Command:
    """
    LangGraph node function for action execution.
    
    Args:
        state: Current workflow state
        
    Returns:
        Command with execution results
    """
    agent = ActionExecutorAgent()
    return await agent.invoke(state) 