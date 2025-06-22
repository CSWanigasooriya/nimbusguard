# engine/langgraph_workflow.py
# ============================================================================
# LangGraph Agent for NimbusGuard Operator - Using create_react_agent
# ============================================================================

import logging
import os
from typing import Dict, Any, List, Optional, Annotated
from typing_extensions import TypedDict

from langgraph.prebuilt import create_react_agent, ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph_tools import get_nimbusguard_tools

LOG = logging.getLogger(__name__)


# ============================================================================
# Agent State Schema (Compatible with LangGraph AgentState)
# ============================================================================

class NimbusGuardAgentState(TypedDict):
    """Extended agent state for NimbusGuard with DQN decision context."""
    # Standard LangGraph agent state
    messages: Annotated[List[Dict[str, Any]], "The conversation messages"]
    
    # NimbusGuard specific context
    resource_name: str
    resource_uid: str
    namespace: str
    current_replicas: int
    min_replicas: int
    max_replicas: int
    target_labels: Dict[str, str]
    
    # DQN decision results
    dqn_decision: Optional[Dict[str, Any]]
    final_decision: Optional[Dict[str, Any]]
    execution_result: Optional[Dict[str, Any]]


# ============================================================================
# System Prompt for DQN Validation Agent
# ============================================================================

NIMBUSGUARD_SYSTEM_PROMPT = """You are NimbusGuard, an intelligent Kubernetes autoscaling validation agent.

Your PRIMARY role is to validate DQN (Deep Q-Network) scaling decisions, NOT to make scaling decisions yourself.

WORKFLOW:
1. A DQN ML model makes the PRIMARY scaling decision
2. You validate this decision for safety and reasonableness
3. You can override ONLY if there are serious safety concerns

CONTEXT:
- You have access to observability data from Prometheus
- You can check current Kubernetes deployment status
- The DQN model is trained on historical scaling patterns
- Your job is VALIDATION, not decision-making

VALIDATION CRITERIA:
- Safety: Will this scaling action cause service disruption?
- Resource constraints: Are min/max replica limits respected?
- Timing: Is it too soon after the last scaling action?
- System health: Are there any red flags in the metrics?

RESPONSES:
- If DQN decision is SAFE: Approve with brief reasoning
- If DQN decision is RISKY: Suggest alternative or deny with clear reasoning
- Always explain your validation logic briefly

Remember: The DQN model is the primary decision maker. You are the safety validator."""


# ============================================================================
# Agent Creation and Execution
# ============================================================================

def create_nimbusguard_agent():
    """
    Create a NimbusGuard agent using LangGraph's create_react_agent.
    This follows the proper LangGraph agent pattern from the documentation.
    """
    # Get OpenAI model (optional for validation)
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        LOG.warning("OPENAI_API_KEY not set - LLM validation will be skipped")
        return None
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=openai_api_key
    )
    
    # Get NimbusGuard tools
    tools = get_nimbusguard_tools()
    
    # Create the agent using LangGraph's prebuilt pattern
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_schema=NimbusGuardAgentState,
        interrupt_before=None,  # No human interrupts for now
        interrupt_after=None,
        debug=False
    )
    
    LOG.info("NimbusGuard LangGraph agent created with create_react_agent")
    return agent


async def run_scaling_workflow(
    resource_name: str,
    resource_uid: str,
    namespace: str,
    current_replicas: int,
    min_replicas: int,
    max_replicas: int,
    target_labels: Dict[str, str]
) -> Dict[str, Any]:
    """
    Run the NimbusGuard scaling workflow using LangGraph agent.
    This preserves DQN as primary decision maker with LLM validation.
    """
    
    # Create the agent
    agent = create_nimbusguard_agent()
    
    # If no LLM available, fall back to DQN-only decision
    if agent is None:
        LOG.info("No LLM available - using DQN-only decision")
        return await _run_dqn_only_workflow(
            resource_name, resource_uid, namespace, 
            current_replicas, min_replicas, max_replicas, target_labels
        )
    
    # Prepare the initial state
    initial_state = {
        "messages": [
            SystemMessage(content=NIMBUSGUARD_SYSTEM_PROMPT),
            HumanMessage(content=f"""
Please help me validate a DQN scaling decision for Kubernetes deployment:

Resource: {resource_name}
Namespace: {namespace}
Current Replicas: {current_replicas}
Min Replicas: {min_replicas}
Max Replicas: {max_replicas}
Target Labels: {target_labels}

WORKFLOW STEPS:
1. First, collect current observability data
2. Get the DQN scaling decision (PRIMARY decision)
3. Validate the DQN decision for safety
4. If safe, execute the scaling
5. Report the results

Start by collecting observability data for this deployment.
""")
        ],
        "resource_name": resource_name,
        "resource_uid": resource_uid,
        "namespace": namespace,
        "current_replicas": current_replicas,
        "min_replicas": min_replicas,
        "max_replicas": max_replicas,
        "target_labels": target_labels,
        "dqn_decision": None,
        "final_decision": None,
        "execution_result": None
    }
    
    try:
        # Run the agent workflow
        LOG.info(f"Starting LangGraph agent workflow for {resource_name}")
        
        result = await agent.ainvoke(initial_state)
        
        # Extract results from the agent execution
        messages = result.get("messages", [])
        final_message = messages[-1] if messages else None
        
        # Parse the final decision from agent messages
        final_decision = _extract_decision_from_messages(messages)
        
        return {
            "resource_name": resource_name,
            "resource_uid": resource_uid,
            "workflow_step": "completed",
            "execution_success": final_decision.get("success", False),
            "scaling_decision": final_decision.get("action", "no_action"),
            "target_replicas": final_decision.get("target_replicas", current_replicas),
            "current_replicas": current_replicas,
            "decision_reason": final_decision.get("reason", "LangGraph agent completed"),
            "decision_confidence": final_decision.get("confidence", 0.0),
            "dqn_metadata": final_decision.get("dqn_metadata", {}),
            "llm_analysis": final_message.content if final_message else None,
            "agent_messages": len(messages),
            "execution_errors": final_decision.get("errors", []),
            "kserve_available": final_decision.get("kserve_available", False),
            "performance_metrics": {
                "total_workflow_duration": 0.0,  # Will be calculated by handler
                "agent_steps": len(messages)
            }
        }
        
    except Exception as e:
        LOG.error(f"LangGraph agent workflow failed: {e}")
        return {
            "resource_name": resource_name,
            "resource_uid": resource_uid,
            "workflow_step": "failed",
            "execution_success": False,
            "scaling_decision": "no_action",
            "target_replicas": current_replicas,
            "current_replicas": current_replicas,
            "decision_reason": f"Agent workflow failed: {str(e)}",
            "decision_confidence": 0.0,
            "execution_errors": [str(e)],
            "kserve_available": False,
            "performance_metrics": {
                "total_workflow_duration": 0.0,
                "agent_steps": 0
            }
        }


async def _run_dqn_only_workflow(
    resource_name: str,
    resource_uid: str,
    namespace: str,
    current_replicas: int,
    min_replicas: int,
    max_replicas: int,
    target_labels: Dict[str, str]
) -> Dict[str, Any]:
    """
    Fallback to DQN-only decision when LLM is not available.
    This preserves the core DQN functionality.
    """
    
    try:
        from langgraph_tools import (
            collect_observability_data,
            make_dqn_decision,
            execute_kubernetes_scaling,
            get_current_replicas
        )
        
        # Step 1: Collect observability data
        LOG.info(f"Collecting observability data for {resource_name}")
        prometheus_url = os.getenv('PROMETHEUS_URL', 'http://prometheus:9090')
        
        observability_result = await collect_observability_data.ainvoke({
            "prometheus_url": prometheus_url,
            "target_labels": target_labels,
            "service_name": resource_name
        })
        
        # Step 2: Make DQN decision
        LOG.info(f"Making DQN decision for {resource_name}")
        dqn_result = await make_dqn_decision.ainvoke({
            "current_replicas": current_replicas,
            "min_replicas": min_replicas,
            "max_replicas": max_replicas,
            "unified_state": observability_result,
            "recent_actions": [2, 2, 2, 2, 2]  # Default NO_ACTION history
        })
        
        if not dqn_result.get("success", False):
            return {
                "resource_name": resource_name,
                "resource_uid": resource_uid,
                "workflow_step": "failed",
                "execution_success": False,
                "scaling_decision": "no_action",
                "target_replicas": current_replicas,
                "current_replicas": current_replicas,
                "decision_reason": f"DQN decision failed: {dqn_result.get('error', 'Unknown error')}",
                "decision_confidence": 0.0,
                "execution_errors": [dqn_result.get('error', 'DQN decision failed')],
                "kserve_available": False,
                "dqn_metadata": dqn_result
            }
        
        # Step 3: Execute scaling if needed
        target_replicas = dqn_result.get("target_replicas", current_replicas)
        execution_result = {"success": True, "message": "No scaling needed"}
        
        if target_replicas != current_replicas:
            # Get deployment name first
            replica_info = await get_current_replicas.ainvoke({
                "target_labels": target_labels,
                "namespace": namespace
            })
            
            if replica_info.get("success", False):
                deployment_name = replica_info.get("deployment_name")
                
                LOG.info(f"Executing scaling for {deployment_name}: {current_replicas} -> {target_replicas}")
                execution_result = await execute_kubernetes_scaling.ainvoke({
                    "deployment_name": deployment_name,
                    "namespace": namespace,
                    "target_replicas": target_replicas,
                    "dry_run": False
                })
            else:
                execution_result = {"success": False, "error": "Could not find deployment"}
        
        return {
            "resource_name": resource_name,
            "resource_uid": resource_uid,
            "workflow_step": "completed",
            "execution_success": execution_result.get("success", False),
            "scaling_decision": dqn_result.get("action", "no_action"),
            "target_replicas": target_replicas,
            "current_replicas": current_replicas,
            "decision_reason": f"DQN-only decision: {dqn_result.get('reasoning', 'No LLM validation')}",
            "decision_confidence": dqn_result.get("confidence", 0.0),
            "dqn_metadata": {
                "dqn_action": dqn_result.get("dqn_action"),
                "dqn_action_value": dqn_result.get("dqn_action_value"),
                "confidence": dqn_result.get("confidence"),
                "q_values": dqn_result.get("q_values", []),
                "prediction_source": dqn_result.get("prediction_source")
            },
            "execution_errors": [] if execution_result.get("success", False) else [execution_result.get("error", "Execution failed")],
            "kserve_available": dqn_result.get("kserve_available", False),
            "llm_analysis": None,
            "performance_metrics": {
                "total_workflow_duration": 0.0,
                "agent_steps": 0
            }
        }
        
    except Exception as e:
        LOG.error(f"DQN-only workflow failed: {e}")
        return {
            "resource_name": resource_name,
            "resource_uid": resource_uid,
            "workflow_step": "failed",
            "execution_success": False,
            "scaling_decision": "no_action",
            "target_replicas": current_replicas,
            "current_replicas": current_replicas,
            "decision_reason": f"DQN-only workflow failed: {str(e)}",
            "decision_confidence": 0.0,
            "execution_errors": [str(e)],
            "kserve_available": False,
            "performance_metrics": {
                "total_workflow_duration": 0.0,
                "agent_steps": 0
            }
        }


def _extract_decision_from_messages(messages: List[Any]) -> Dict[str, Any]:
    """
    Extract the final scaling decision from agent messages.
    This parses the agent's tool calls and responses.
    """
    
    # Look for tool calls and responses in the messages
    dqn_decision = None
    execution_result = None
    
    for message in messages:
        # Check for tool calls in AI messages
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.get('name') == 'make_dqn_decision':
                    # This is the DQN decision
                    pass
                elif tool_call.get('name') == 'execute_kubernetes_scaling':
                    # This is the execution result
                    pass
        
        # Check for tool responses
        if hasattr(message, 'content') and isinstance(message.content, str):
            try:
                # Try to parse JSON responses from tools
                import json
                if message.content.startswith('{'):
                    content = json.loads(message.content)
                    if 'dqn_action' in content:
                        dqn_decision = content
                    elif 'target_replicas' in content and 'success' in content:
                        execution_result = content
            except:
                pass
    
    # Build the final decision
    if dqn_decision:
        return {
            "success": execution_result.get("success", True) if execution_result else True,
            "action": dqn_decision.get("action", "no_action"),
            "target_replicas": dqn_decision.get("target_replicas"),
            "confidence": dqn_decision.get("confidence", 0.0),
            "reason": dqn_decision.get("reasoning", "DQN decision with LLM validation"),
            "dqn_metadata": dqn_decision,
            "kserve_available": dqn_decision.get("kserve_available", False),
            "errors": [] if execution_result.get("success", True) else [execution_result.get("error", "Unknown error")]
        }
    
    # Fallback if no clear decision found
    return {
        "success": False,
        "action": "no_action",
        "target_replicas": None,
        "confidence": 0.0,
        "reason": "Could not extract decision from agent messages",
        "dqn_metadata": {},
        "kserve_available": False,
        "errors": ["Failed to parse agent decision"]
    } 