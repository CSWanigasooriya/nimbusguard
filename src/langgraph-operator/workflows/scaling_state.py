"""
State Management for LangGraph Scaling Workflow

This module defines the state structure and reducers for the multi-agent scaling system,
following LangGraph best practices for complex state management.
"""

from typing import Dict, List, Any, Optional, Literal, TypedDict, Annotated, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class ScalingAction(Enum):
    """Possible scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"  # Changed from NO_ACTION to MAINTAIN for consistency
    ROLLBACK = "rollback"


class WorkflowStatus(Enum):
    """Workflow execution status"""
    INITIALIZING = "initializing"
    OBSERVING = "observing"
    ANALYZING = "analyzing"
    DECIDING = "deciding"
    EXECUTING = "executing"
    MONITORING = "monitoring"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MetricData:
    """Container for cluster metrics"""
    cpu_utilization: float
    memory_utilization: float
    request_rate: float
    error_rate: float
    pod_count: int
    health_score: float = 0.5
    response_time: float = 100.0  # Default response time in ms
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ScalingDecision:
    """Scaling decision with rationale"""
    action: ScalingAction
    target_replicas: int
    current_replicas: int
    confidence: float
    reasoning: str
    metrics_snapshot: MetricData
    q_learning_recommendation: Optional[ScalingAction] = None
    llm_recommendation: Optional[ScalingAction] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ActionResult:
    """Result of executing a scaling action"""
    success: bool
    action_taken: ScalingAction
    old_replicas: int
    new_replicas: int
    error_message: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def actual_replicas(self) -> int:
        """Compatibility property for old field name"""
        return self.new_replicas


@dataclass
class RewardSignal:
    """Reward calculation for Q-learning"""
    reward: float
    components: Dict[str, float]  # Breakdown of reward calculation
    metrics_before: MetricData
    metrics_after: Optional[MetricData]
    timestamp: datetime = field(default_factory=datetime.now)


def merge_messages(existing: List[BaseMessage], new: List[BaseMessage]) -> List[BaseMessage]:
    """Custom reducer for conversation messages"""
    return existing + new


def merge_metrics_history(existing: List[MetricData], new: List[MetricData]) -> List[MetricData]:
    """Custom reducer for metrics history with sliding window"""
    combined = existing + new
    # Keep only last 100 metrics (about 16 minutes at 10s intervals)
    return combined[-100:] if len(combined) > 100 else combined


def merge_decisions_history(existing: List[ScalingDecision], new: List[ScalingDecision]) -> List[ScalingDecision]:
    """Custom reducer for decision history"""
    combined = existing + new
    # Keep only last 50 decisions
    return combined[-50:] if len(combined) > 50 else combined


def merge_action_history(existing: List[ActionResult], new: List[ActionResult]) -> List[ActionResult]:
    """Custom reducer for action history"""
    combined = existing + new
    # Keep only last 50 actions
    return combined[-50:] if len(combined) > 50 else combined


def merge_reward_history(existing: List[RewardSignal], new: List[RewardSignal]) -> List[RewardSignal]:
    """Custom reducer for reward history"""
    combined = existing + new
    # Keep only last 100 rewards
    return combined[-100:] if len(combined) > 100 else combined


def update_context(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Custom reducer for context updates"""
    return {**existing, **new}


class ScalingWorkflowState(TypedDict):
    """
    Complete state for the LangGraph scaling workflow.
    
    This state is shared across all agents and maintains the context
    of the entire scaling decision process.
    """
    
    # Core Workflow State
    status: WorkflowStatus
    current_agent: str
    next_agent: Optional[str]
    workflow_id: str
    started_at: datetime
    
    # Agent Communication
    messages: Annotated[List[BaseMessage], add_messages]
    agent_outputs: Dict[str, Any]
    
    # Metrics and Observations
    current_metrics: Optional[MetricData]
    metrics_history: Annotated[List[MetricData], merge_metrics_history]
    baseline_metrics: Optional[MetricData]
    
    # Decision Making
    scaling_decision: Optional[ScalingDecision]
    decisions_history: Annotated[List[ScalingDecision], merge_decisions_history]
    q_learning_state: Dict[str, Any]
    
    # Action Execution
    last_action: Optional[ActionResult]
    action_history: Annotated[List[ActionResult], merge_action_history]
    rollback_available: bool
    
    # Learning and Rewards
    last_reward: Optional[RewardSignal]
    reward_history: Annotated[List[RewardSignal], merge_reward_history]
    cumulative_reward: float
    
    # Context and Configuration
    context: Annotated[Dict[str, Any], update_context]
    config: Dict[str, Any]
    
    # Error Handling
    errors: List[str]
    retry_count: int
    max_retries: int
    
    # Human-in-the-Loop
    human_approval_required: bool
    human_feedback: Optional[str]
    
    # Completion and Results
    completed: bool
    final_result: Optional[Dict[str, Any]]


def create_initial_state(
    workflow_id: str,
    trigger_event: Dict[str, Any],
    config: Dict[str, Any]
) -> ScalingWorkflowState:
    """
    Create the initial state for a new scaling workflow.
    
    Args:
        workflow_id: Unique identifier for this workflow execution
        trigger_event: Event that triggered this scaling workflow
        config: Configuration for the workflow
        
    Returns:
        Initialized ScalingWorkflowState
    """
    return ScalingWorkflowState(
        # Core Workflow State
        status=WorkflowStatus.INITIALIZING,
        current_agent="supervisor",
        next_agent="state_observer",
        workflow_id=workflow_id,
        started_at=datetime.now(),
        
        # Agent Communication
        messages=[],
        agent_outputs={},
        
        # Metrics and Observations
        current_metrics=None,
        metrics_history=[],
        baseline_metrics=None,
        
        # Decision Making
        scaling_decision=None,
        decisions_history=[],
        q_learning_state={},
        
        # Action Execution
        last_action=None,
        action_history=[],
        rollback_available=False,
        
        # Learning and Rewards
        last_reward=None,
        reward_history=[],
        cumulative_reward=0.0,
        
        # Context and Configuration
        context={
            "trigger_event": trigger_event,
            "environment": config.get("environment", "production"),
            "namespace": config.get("kubernetes", {}).get("namespace", "default"),
        },
        config=config,
        
        # Error Handling
        errors=[],
        retry_count=0,
        max_retries=config.get("max_retries", 3),
        
        # Human-in-the-Loop
        human_approval_required=config.get("human_approval_required", False),
        human_feedback=None,
        
        # Completion and Results
        completed=False,
        final_result=None,
    )


def serialize_state_for_logging(state: ScalingWorkflowState) -> Dict[str, Any]:
    """
    Serialize state for logging, handling non-JSON serializable objects.
    
    Args:
        state: Current workflow state
        
    Returns:
        JSON-serializable representation of the state
    """
    def serialize_obj(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    serialized = {}
    for key, value in state.items():
        try:
            if isinstance(value, list):
                serialized[key] = [serialize_obj(item) for item in value]
            elif isinstance(value, dict):
                serialized[key] = {k: serialize_obj(v) for k, v in value.items()}
            else:
                serialized[key] = serialize_obj(value)
        except Exception as e:
            serialized[key] = f"<serialization_error: {str(e)}>"
    
    return serialized


def validate_state(state: ScalingWorkflowState) -> List[str]:
    """
    Validate the state for consistency and completeness.
    
    Args:
        state: State to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required fields
    if not state.get("workflow_id"):
        errors.append("workflow_id is required")
    
    if not state.get("status"):
        errors.append("status is required")
    
    # Check state consistency
    if state.get("completed") and not state.get("final_result"):
        errors.append("completed workflows must have final_result")
    
    if state.get("retry_count", 0) > state.get("max_retries", 3):
        errors.append("retry_count exceeds max_retries")
    
    # Check metrics consistency
    if state.get("current_metrics") and state.get("baseline_metrics"):
        current = state["current_metrics"]
        baseline = state["baseline_metrics"]
        if current.timestamp < baseline.timestamp:
            errors.append("current_metrics timestamp is before baseline_metrics")
    
    return errors 