"""
Supervisor Agent for LangGraph Scaling Workflow

This agent orchestrates the entire scaling workflow, deciding which specialized 
agents to invoke and managing the overall flow control using LangGraph's 
Command pattern for dynamic routing.
"""

import logging
import asyncio
from typing import Dict, Any, Literal, Optional
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from ..config import get_agent_config, get_system_prompt
from ..workflows.scaling_state import (
    ScalingWorkflowState, 
    WorkflowStatus, 
    validate_state,
    serialize_state_for_logging
)
from ..mcp_integration.mcp_tools import COMMON_TOOLS, get_cluster_metrics

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """
    Supervisor Agent that orchestrates the scaling workflow.
    
    The supervisor acts as the central coordinator, analyzing the current state
    and deciding which specialized agent should handle the next step in the
    scaling process.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the supervisor agent.
        
        Args:
            config: Optional configuration dictionary. If None, loads from config system.
        """
        # Load configuration from config system
        self.config = get_agent_config("supervisor") if config is None else config.get("agents", {}).get("supervisor", {})
        
        # Initialize LLM with configuration
        self.llm = ChatOpenAI(
            model=self.config.get("model", "gpt-4o-mini"),
            temperature=self.config.get("temperature", 0.1),
            max_tokens=self.config.get("max_tokens", 1000),
            timeout=self.config.get("timeout", 30)
        )
        self.retry_attempts = self.config.get("retry_attempts", 3)
        
        # Load system prompt from configuration
        self.system_prompt = get_system_prompt("supervisor")

    async def invoke(self, state: ScalingWorkflowState) -> Command:
        """
        Main supervisor logic - analyzes state and routes to appropriate agent.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command object with routing decision and state updates
        """
        try:
            # Validate current state
            validation_errors = validate_state(state)
            if validation_errors:
                logger.error(f"State validation failed: {validation_errors}")
                return self._handle_error(state, f"State validation failed: {validation_errors}")
            
            # Log current state for debugging
            logger.info(f"Supervisor processing workflow {state['workflow_id']} in status {state['status']}")
            
            # Check for workflow timeout
            if self._is_workflow_timed_out(state):
                return self._handle_timeout(state)
            
            # Check for too many retries
            if state.get("retry_count", 0) >= state.get("max_retries", 3):
                return self._handle_max_retries_exceeded(state)
            
            # Route based on current status
            routing_decision = await self._make_routing_decision(state)
            
            # Log routing decision
            logger.info(f"Supervisor routing decision: {routing_decision}")
            
            return routing_decision
            
        except Exception as e:
            logger.error(f"Supervisor agent error: {e}", exc_info=True)
            return self._handle_error(state, f"Supervisor agent error: {str(e)}")

    async def _make_routing_decision(self, state: ScalingWorkflowState) -> Command:
        """
        Make routing decision based on current state using LLM reasoning.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command with routing decision
        """
        # Prepare context for LLM
        context = self._prepare_llm_context(state)
        
        # Create prompt for routing decision
        prompt = f"""
        Analyze the current scaling workflow state and decide the next action:

        Current Status: {state['status'].value}
        Current Agent: {state['current_agent']}
        Workflow ID: {state['workflow_id']}
        
        Context:
        {context}
        
        Available next agents:
        - state_observer: Monitor cluster metrics
        - decision_agent: Make scaling decisions
        - action_executor: Execute scaling actions
        - reward_calculator: Calculate learning rewards
        - __end__: Complete workflow
        
        Decide the next agent and provide reasoning.
        Format your response as:
        NEXT_AGENT: [agent_name]
        REASONING: [your reasoning]
        UPDATE_STATUS: [new_status if needed]
        """
        
        # Get LLM decision
        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            response = await self.llm.ainvoke(messages)
            decision = self._parse_llm_response(response.content)
            
            # Log LLM decision for debugging
            logger.info(f"Supervisor LLM decision: {decision}")
            
            # Create command based on decision
            return self._create_routing_command(state, decision)
            
        except Exception as e:
            logger.error(f"LLM routing decision failed: {e}")
            return self._fallback_routing_decision(state)

    def _prepare_llm_context(self, state: ScalingWorkflowState) -> str:
        """Prepare context string for LLM decision making."""
        context_parts = []
        
        # Current metrics
        if state.get("current_metrics"):
            metrics = state["current_metrics"]
            context_parts.append(f"Current Metrics: CPU={metrics.cpu_utilization}%, Memory={metrics.memory_utilization}%, Pods={metrics.pod_count}")
        
        # Recent decisions
        if state.get("scaling_decision"):
            decision = state["scaling_decision"]
            context_parts.append(f"Last Decision: {decision.action.value} to {decision.target_replicas} replicas (confidence: {decision.confidence})")
        
        # Recent actions
        if state.get("last_action"):
            action = state["last_action"]
            context_parts.append(f"Last Action: {action.action_taken.value} {'succeeded' if action.success else 'failed'}")
        
        # Errors
        if state.get("errors"):
            context_parts.append(f"Recent Errors: {state['errors'][-3:]}")  # Last 3 errors
        
        # Human approval
        if state.get("human_approval_required"):
            context_parts.append("Human approval required for next action")
        
        return "\n".join(context_parts) if context_parts else "No additional context available"

    def _parse_llm_response(self, response: str) -> Dict[str, str]:
        """Parse structured LLM response."""
        decision = {
            "next_agent": "__end__",
            "reasoning": "Default fallback",
            "update_status": None
        }
        
        for line in response.strip().split('\n'):
            if line.startswith("NEXT_AGENT:"):
                decision["next_agent"] = line.split(":", 1)[1].strip()
            elif line.startswith("REASONING:"):
                decision["reasoning"] = line.split(":", 1)[1].strip()
            elif line.startswith("UPDATE_STATUS:"):
                decision["update_status"] = line.split(":", 1)[1].strip()
        
        return decision

    def _create_routing_command(self, state: ScalingWorkflowState, decision: Dict[str, str]) -> Command:
        """Create a Command object from the routing decision."""
        next_agent = decision["next_agent"]
        reasoning = decision["reasoning"]
        
        # Prepare state updates
        updates = {
            "current_agent": "supervisor",
            "next_agent": next_agent,
            "messages": [AIMessage(content=f"Supervisor routing to {next_agent}: {reasoning}")]
        }
        
        # Update status if specified
        if decision.get("update_status"):
            try:
                new_status = WorkflowStatus(decision["update_status"])
                updates["status"] = new_status
            except ValueError:
                logger.warning(f"Invalid status update: {decision['update_status']}")
        
        # Handle special routing cases
        if next_agent == "__end__":
            updates.update({
                "completed": True,
                "status": WorkflowStatus.COMPLETED,
                "final_result": {
                    "workflow_id": state["workflow_id"],
                    "completed_at": datetime.now(),
                    "total_actions": len(state.get("action_history", [])),
                    "cumulative_reward": state.get("cumulative_reward", 0.0),
                    "final_status": "completed"
                }
            })
        
        return Command(
            goto=next_agent,
            update=updates
        )

    def _fallback_routing_decision(self, state: ScalingWorkflowState) -> Command:
        """Fallback routing logic when LLM fails."""
        current_status = state.get("status", WorkflowStatus.INITIALIZING)
        
        # Simple state machine fallback
        if current_status == WorkflowStatus.INITIALIZING:
            next_agent = "state_observer"
            new_status = WorkflowStatus.OBSERVING
        elif current_status == WorkflowStatus.OBSERVING:
            next_agent = "decision_agent" if state.get("current_metrics") else "state_observer"
            new_status = WorkflowStatus.ANALYZING if state.get("current_metrics") else WorkflowStatus.OBSERVING
        elif current_status == WorkflowStatus.ANALYZING:
            next_agent = "decision_agent"
            new_status = WorkflowStatus.DECIDING
        elif current_status == WorkflowStatus.DECIDING:
            next_agent = "action_executor" if state.get("scaling_decision") else "decision_agent"
            new_status = WorkflowStatus.EXECUTING if state.get("scaling_decision") else WorkflowStatus.DECIDING
        elif current_status == WorkflowStatus.EXECUTING:
            next_agent = "reward_calculator" if state.get("last_action") else "action_executor"
            new_status = WorkflowStatus.MONITORING if state.get("last_action") else WorkflowStatus.EXECUTING
        else:
            next_agent = "__end__"
            new_status = WorkflowStatus.COMPLETED
        
        return Command(
            goto=next_agent,
            update={
                "current_agent": "supervisor",
                "next_agent": next_agent,
                "status": new_status,
                "messages": [AIMessage(content=f"Fallback routing to {next_agent}")]
            }
        )

    def _handle_error(self, state: ScalingWorkflowState, error_message: str) -> Command:
        """Handle workflow errors."""
        logger.error(f"Workflow error: {error_message}")
        
        return Command(
            goto="__end__",
            update={
                "status": WorkflowStatus.FAILED,
                "completed": True,
                "errors": state.get("errors", []) + [error_message],
                "final_result": {
                    "workflow_id": state["workflow_id"],
                    "error": error_message,
                    "failed_at": datetime.now(),
                    "final_status": "failed"
                },
                "messages": [AIMessage(content=f"Workflow failed: {error_message}")]
            }
        )

    def _handle_timeout(self, state: ScalingWorkflowState) -> Command:
        """Handle workflow timeout."""
        return self._handle_error(state, "Workflow exceeded maximum execution time")

    def _handle_max_retries_exceeded(self, state: ScalingWorkflowState) -> Command:
        """Handle maximum retries exceeded."""
        return self._handle_error(state, f"Maximum retries ({state['max_retries']}) exceeded")

    def _is_workflow_timed_out(self, state: ScalingWorkflowState, max_duration_minutes: int = 30) -> bool:
        """Check if workflow has exceeded maximum execution time."""
        if not state.get("started_at"):
            return False
        
        elapsed = datetime.now() - state["started_at"]
        return elapsed > timedelta(minutes=max_duration_minutes)


# Node function for LangGraph integration
async def supervisor_node(state: ScalingWorkflowState) -> Command:
    """
    LangGraph node function for the supervisor agent.
    
    Args:
        state: Current workflow state
        
    Returns:
        Command with routing decision
    """
    # Initialize supervisor agent using configuration system
    supervisor = SupervisorAgent()
    
    # Process the state and return routing decision
    return await supervisor.invoke(state) 