"""LangGraph state management for operator workflow."""
from typing import Dict, List, Optional, Any, TypedDict
import numpy as np
from datetime import datetime


class OperatorState(TypedDict):
    """State passed between LangGraph nodes."""
    
    # Timing
    timestamp: str
    execution_id: str
    
    # Current metrics
    current_metrics: Dict[str, float]
    current_replicas: int
    
    # Forecast data
    forecast_result: Optional[Dict[str, Any]]
    forecast_confidence: float
    
    # DQN decision
    dqn_state: Optional[List[float]]  # Converted from numpy array for serialization
    dqn_action: Optional[str]
    dqn_q_values: Optional[Dict[str, float]]
    dqn_confidence: float
    
    # Scaling decision
    desired_replicas: int
    scaling_decision: str  # "scale_up", "scale_down", "keep_same"
    decision_reason: str
    
    # Validation
    validation_passed: bool
    validation_errors: List[str]
    
    # Execution
    scaling_applied: bool
    scaling_error: Optional[str]
    
    # Rewards and learning
    reward_calculated: bool
    reward_value: Optional[float]
    
    # Metadata
    node_outputs: Dict[str, Any]
    execution_time_ms: Dict[str, float]
    errors: List[str]


def create_initial_state(execution_id: str, current_replicas: int) -> OperatorState:
    """Create initial state for workflow execution."""
    return OperatorState(
        # Timing
        timestamp=datetime.utcnow().isoformat(),
        execution_id=execution_id,
        
        # Current metrics
        current_metrics={},
        current_replicas=current_replicas,
        
        # Forecast data
        forecast_result=None,
        forecast_confidence=0.0,
        
        # DQN decision
        dqn_state=None,
        dqn_action=None,
        dqn_q_values=None,
        dqn_confidence=0.0,
        
        # Scaling decision
        desired_replicas=current_replicas,
        scaling_decision="keep_same",
        decision_reason="initial",
        
        # Validation
        validation_passed=False,
        validation_errors=[],
        
        # Execution
        scaling_applied=False,
        scaling_error=None,
        
        # Rewards and learning
        reward_calculated=False,
        reward_value=None,
        
        # Metadata
        node_outputs={},
        execution_time_ms={},
        errors=[]
    )


def update_state_with_metrics(state: OperatorState, metrics: Dict[str, float]) -> OperatorState:
    """Update state with current metrics."""
    state["current_metrics"] = metrics
    return state


def update_state_with_forecast(state: OperatorState, forecast_result: Dict[str, Any]) -> OperatorState:
    """Update state with forecast results."""
    state["forecast_result"] = forecast_result
    state["forecast_confidence"] = forecast_result.get("confidence", 0.0)
    return state


def update_state_with_dqn(state: OperatorState, action: str, q_values: Dict[str, float], 
                         dqn_state: List[float], confidence: float) -> OperatorState:
    """Update state with DQN decision results."""
    state["dqn_action"] = action
    state["dqn_q_values"] = q_values
    state["dqn_state"] = dqn_state
    state["dqn_confidence"] = confidence
    return state


def add_node_output(state: OperatorState, node_name: str, output: Dict[str, Any], 
                   execution_time_ms: float) -> OperatorState:
    """Add output from a workflow node."""
    
    def make_serializable(obj):
        """Convert numpy types to Python native types."""
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        else:
            return obj
    
    # Ensure all values are serializable
    serializable_output = make_serializable(output)
    serializable_time = float(execution_time_ms)
    
    state["node_outputs"][node_name] = serializable_output
    state["execution_time_ms"][node_name] = serializable_time
    return state


def add_error(state: OperatorState, error_message: str) -> OperatorState:
    """Add an error to the state."""
    state["errors"].append(error_message)
    return state


def get_state_summary(state: OperatorState) -> str:
    """Get a summary of the current state."""
    return (f"execution_id={state['execution_id']}, "
            f"action={state['scaling_decision']}, "
            f"replicas={state['current_replicas']}→{state['desired_replicas']}, "
            f"confidence={state['dqn_confidence']:.3f}, "
            f"total_errors={len(state['errors'])}") 