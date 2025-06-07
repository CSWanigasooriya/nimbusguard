"""
Main Scaling Workflow for NimbusGuard LangGraph Operator

This module defines the complete LangGraph workflow that orchestrates AI agents
to make intelligent scaling decisions using Q-learning and LLM reasoning.
"""

import logging
from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from .scaling_state import ScalingWorkflowState, WorkflowStatus, create_initial_state
from agents.supervisor_agent import SupervisorAgent, supervisor_node
from agents.state_observer_agent import StateObserverAgent, state_observer_node
from agents.decision_agent import DecisionAgent, decision_node
from agents.action_executor import ActionExecutorAgent, action_executor_node
from agents.reward_calculator import RewardCalculatorAgent, reward_calculator_node

logger = logging.getLogger(__name__)


def create_scaling_workflow(config: Optional[Dict[str, Any]] = None) -> StateGraph:
    """
    Create the complete scaling workflow graph.
    
    Args:
        config: Optional configuration for the workflow
        
    Returns:
        StateGraph: Compiled LangGraph workflow
    """
    
    # Create the state graph
    workflow = StateGraph(ScalingWorkflowState)
    
    # Add all agent nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("state_observer", state_observer_node)
    workflow.add_node("decision_agent", decision_node)
    workflow.add_node("action_executor", action_executor_node)
    workflow.add_node("reward_calculator", reward_calculator_node)
    
    # Define the workflow entry point
    workflow.add_edge(START, "supervisor")
    
    # Add conditional edges from supervisor to route workflow
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor_decision,
        {
            "state_observer": "state_observer",
            "decision_agent": "decision_agent", 
            "action_executor": "action_executor",
            "reward_calculator": "reward_calculator",
            "__end__": END
        }
    )
    
    # Add edges from each agent back to supervisor for routing
    workflow.add_edge("state_observer", "supervisor")
    workflow.add_edge("decision_agent", "supervisor")
    workflow.add_edge("action_executor", "supervisor")
    workflow.add_edge("reward_calculator", "supervisor")
    
    # Compile the workflow
    compiled_workflow = workflow.compile()
    
    logger.info("Scaling workflow created and compiled successfully")
    return compiled_workflow


def route_supervisor_decision(state: ScalingWorkflowState) -> str:
    """
    Route workflow based on supervisor decisions.
    
    Args:
        state: Current workflow state
        
    Returns:
        str: Next node to execute
    """
    try:
        # Get the last command from supervisor
        last_command = state.get("last_command")
        if last_command and hasattr(last_command, 'goto'):
            return last_command.goto
        
        # Fallback routing based on workflow status
        status = state.get("status", WorkflowStatus.STARTING)
        
        if status == WorkflowStatus.STARTING:
            return "state_observer"
        elif status == WorkflowStatus.OBSERVING:
            return "decision_agent"
        elif status == WorkflowStatus.DECIDING:
            return "action_executor"
        elif status == WorkflowStatus.ACTION_EXECUTED:
            return "reward_calculator"
        elif status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
            return "__end__"
        else:
            # Default to supervisor for decision
            return "state_observer"
            
    except Exception as e:
        logger.error(f"Error routing supervisor decision: {e}")
        return "__end__"


async def execute_scaling_workflow(
    trigger_event: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> ScalingWorkflowState:
    """
    Execute a complete scaling workflow.
    
    Args:
        trigger_event: Event that triggered the scaling workflow
        config: Optional configuration for the workflow
        
    Returns:
        ScalingWorkflowState: Final workflow state
    """
    try:
        # Create the workflow
        workflow = create_scaling_workflow(config)
        
        # Create initial state
        initial_state = create_initial_state(
            workflow_id=f"workflow-{trigger_event.get('timestamp', 'unknown')}",
            trigger_event=trigger_event,
            config=config or {}
        )
        
        logger.info(f"Starting scaling workflow for event: {trigger_event}")
        
        # Execute the workflow
        final_state = await workflow.ainvoke(initial_state)
        
        logger.info(f"Scaling workflow completed with status: {final_state.get('status')}")
        return final_state
        
    except Exception as e:
        logger.error(f"Error executing scaling workflow: {e}", exc_info=True)
        
        # Return error state
        error_state = create_initial_state(
            workflow_id="error-workflow",
            trigger_event=trigger_event,
            config=config or {}
        )
        error_state["status"] = WorkflowStatus.FAILED
        error_state["completed"] = True
        error_state["errors"] = [f"Workflow execution error: {str(e)}"]
        
        return error_state


# Simplified workflow for testing and development
def create_simple_scaling_workflow() -> StateGraph:
    """
    Create a simplified scaling workflow for testing purposes.
    
    Returns:
        StateGraph: Compiled simple workflow
    """
    
    # Create simplified state graph
    workflow = StateGraph(ScalingWorkflowState)
    
    # Add simplified nodes
    workflow.add_node("observe", simple_observe_node)
    workflow.add_node("decide", simple_decide_node)
    workflow.add_node("execute", simple_execute_node)
    workflow.add_node("reward", simple_reward_node)
    
    # Define linear workflow
    workflow.add_edge(START, "observe")
    workflow.add_edge("observe", "decide")
    workflow.add_edge("decide", "execute")
    workflow.add_edge("execute", "reward")
    workflow.add_edge("reward", END)
    
    # Compile the workflow
    compiled_workflow = workflow.compile()
    
    logger.info("Simple scaling workflow created")
    return compiled_workflow


async def simple_observe_node(state: ScalingWorkflowState) -> ScalingWorkflowState:
    """Simplified observation node for testing."""
    state["status"] = WorkflowStatus.OBSERVING
    state["current_agent"] = "state_observer"
    
    # Mock metrics collection
    state["current_metrics"] = {
        "cpu_utilization": 75.0,
        "memory_utilization": 60.0,
        "request_rate": 100.0,
        "error_rate": 0.5,
        "pod_count": 3
    }
    
    logger.info("Simple observation completed")
    return state


async def simple_decide_node(state: ScalingWorkflowState) -> ScalingWorkflowState:
    """Simplified decision node for testing."""
    state["status"] = WorkflowStatus.DECIDING
    state["current_agent"] = "decision_agent"
    
    # Mock scaling decision
    current_metrics = state.get("current_metrics", {})
    cpu_utilization = current_metrics.get("cpu_utilization", 0)
    
    if cpu_utilization > 80:
        action = "SCALE_UP"
        target_replicas = current_metrics.get("pod_count", 3) + 2
    elif cpu_utilization < 30:
        action = "SCALE_DOWN"
        target_replicas = max(1, current_metrics.get("pod_count", 3) - 1)
    else:
        action = "NO_ACTION"
        target_replicas = current_metrics.get("pod_count", 3)
    
    state["scaling_decision"] = {
        "action": action,
        "target_replicas": target_replicas,
        "confidence": 0.8,
        "reasoning": f"CPU utilization at {cpu_utilization}%"
    }
    
    logger.info(f"Simple decision: {action} to {target_replicas} replicas")
    return state


async def simple_execute_node(state: ScalingWorkflowState) -> ScalingWorkflowState:
    """Simplified execution node for testing."""
    state["status"] = WorkflowStatus.ACTION_EXECUTED
    state["current_agent"] = "action_executor"
    
    # Mock action execution
    decision = state.get("scaling_decision", {})
    
    state["last_action"] = {
        "action": decision.get("action", "NO_ACTION"),
        "target_replicas": decision.get("target_replicas", 3),
        "actual_replicas": decision.get("target_replicas", 3),
        "status": "SUCCESS",
        "execution_time_ms": 1500,
        "details": "Mock scaling execution successful"
    }
    
    logger.info("Simple execution completed")
    return state


async def simple_reward_node(state: ScalingWorkflowState) -> ScalingWorkflowState:
    """Simplified reward calculation node for testing."""
    state["status"] = WorkflowStatus.COMPLETED
    state["current_agent"] = "reward_calculator"
    state["completed"] = True
    
    # Mock reward calculation
    action_result = state.get("last_action", {})
    
    if action_result.get("status") == "SUCCESS":
        total_reward = 5.0
    else:
        total_reward = -2.0
    
    state["reward_result"] = {
        "total_reward": total_reward,
        "resource_efficiency_reward": 3.0,
        "sla_compliance_reward": 2.0,
        "stability_reward": 0.0,
        "reasoning": "Mock reward calculation"
    }
    
    logger.info(f"Simple reward calculation: {total_reward}")
    return state 