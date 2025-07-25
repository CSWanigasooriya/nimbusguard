"""
LangGraph workflow definition for the NimbusGuard scaling system.
"""

import logging
from datetime import datetime
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from metrics.collector import metrics
from .state import WorkflowState
from .nodes import WorkflowNodes

logger = logging.getLogger(__name__)


class ScalingWorkflow:
    """LangGraph workflow for intelligent scaling decisions."""
    
    def __init__(self, config, prometheus_client, predictor):
        """
        Initialize the scaling workflow.
        
        Args:
            config: Configuration object
            prometheus_client: Prometheus client instance
            predictor: Model predictor instance
        """
        self.config = config
        self.nodes = WorkflowNodes(config, prometheus_client, predictor)
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow with the specified flow:
        collect_metrics -> generate_forecast -> dqn_decision -> 
        validate_decision -> execute_scaling -> calculate_reward -> 
        store_experience -> train_dqn -> END
        """
        # Create the workflow graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes to the workflow
        workflow.add_node("collect_metrics", self.nodes.collect_metrics)
        workflow.add_node("generate_forecast", self.nodes.generate_forecast)
        workflow.add_node("dqn_decision", self.nodes.dqn_decision)
        workflow.add_node("validate_decision", self.nodes.validate_decision)
        workflow.add_node("execute_scaling", self.nodes.execute_scaling)
        workflow.add_node("calculate_reward", self.nodes.calculate_reward)
        workflow.add_node("store_experience", self.nodes.store_experience)
        workflow.add_node("train_dqn", self.nodes.train_dqn)
        
        # Define the workflow edges in the specified order
        workflow.add_edge("collect_metrics", "generate_forecast")
        workflow.add_edge("generate_forecast", "dqn_decision")
        workflow.add_edge("dqn_decision", "validate_decision")
        workflow.add_edge("validate_decision", "execute_scaling")
        workflow.add_edge("execute_scaling", "calculate_reward")
        workflow.add_edge("calculate_reward", "store_experience")
        workflow.add_edge("store_experience", "train_dqn")
        workflow.add_edge("train_dqn", END)
        
        # Set the entry point
        workflow.set_entry_point("collect_metrics")
        
        return workflow
    
    async def execute_scaling_workflow(
        self, 
        deployment_name: str, 
        namespace: str, 
        current_replicas: int, 
        ready_replicas: int
    ) -> WorkflowState:
        """
        Execute the scaling workflow for a deployment.
        
        Args:
            deployment_name: Name of the deployment
            namespace: Kubernetes namespace
            current_replicas: Current number of replicas
            ready_replicas: Number of ready replicas
            
        Returns:
            Final workflow state with all results
        """
        logger.info(f"Starting scaling workflow for {deployment_name} in {namespace}")
        
        # Initialize the workflow state
        initial_state = WorkflowState(
            deployment_name=deployment_name,
            namespace=namespace,
            current_replicas=current_replicas,
            ready_replicas=ready_replicas,
            workflow_start_time=datetime.now().isoformat()
        )
        
        try:
            # Compile and run the workflow
            app = self.workflow.compile()
            
            # Execute the workflow
            result = await app.ainvoke(initial_state)
            
            # Convert result back to WorkflowState if it's a dict
            if isinstance(result, dict):
                final_state = WorkflowState(**result)
            else:
                final_state = result
            
            # Calculate workflow duration
            if final_state.workflow_start_time:
                start_time = datetime.fromisoformat(final_state.workflow_start_time)
                duration = (datetime.now() - start_time).total_seconds()
                final_state.workflow_duration = duration
            
            logger.info(f"Scaling workflow completed in {final_state.workflow_duration:.2f}s")
            
            # Record workflow execution metrics
            success = not final_state.error_occurred
            error_type = "workflow_error" if final_state.error_occurred else None
            metrics.record_workflow_execution(
                duration=final_state.workflow_duration or 0,
                success=success,
                error_type=error_type
            )
            
            # Log final results
            self._log_workflow_results(final_state)
            
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            
            # Return error state
            error_state = WorkflowState(
                deployment_name=deployment_name,
                namespace=namespace,
                current_replicas=current_replicas,
                ready_replicas=ready_replicas,
                workflow_start_time=initial_state.workflow_start_time,
                error_occurred=True,
                error_message=f"Workflow execution failed: {e}"
            )
            
            if error_state.workflow_start_time:
                start_time = datetime.fromisoformat(error_state.workflow_start_time)
                error_state.workflow_duration = (datetime.now() - start_time).total_seconds()
            
            # Record failed workflow execution
            metrics.record_workflow_execution(
                duration=error_state.workflow_duration or 0,
                success=False,
                error_type="workflow_exception"
            )
            
            return error_state
    
    def _log_workflow_results(self, state: WorkflowState):
        """Log the final workflow results."""
        logger.info("=== Scaling Workflow Results ===")
        logger.info(f"Deployment: {state.deployment_name}")
        logger.info(f"Current Replicas: {state.current_replicas}")
        logger.info(f"Recommended Action: {state.recommended_action}")
        logger.info(f"Recommended Replicas: {state.recommended_replicas}")
        logger.info(f"Action Confidence: {state.action_confidence}")
        logger.info(f"Validation Passed: {state.validation_passed}")
        logger.info(f"Scaling Executed: {state.scaling_executed}")
        logger.info(f"Reward Value: {state.reward_value}")
        logger.info(f"Error Occurred: {state.error_occurred}")
        if state.error_message:
            logger.info(f"Error Message: {state.error_message}")
        logger.info(f"Duration: {state.workflow_duration:.2f}s")
        logger.info("=== End Results ===")


def create_scaling_workflow(config, prometheus_client, predictor) -> ScalingWorkflow:
    """
    Factory function to create a scaling workflow instance.
    
    Args:
        config: Configuration object
        prometheus_client: Prometheus client instance
        predictor: Model predictor instance
        
    Returns:
        Configured scaling workflow
    """
    return ScalingWorkflow(config, prometheus_client, predictor) 