# engine/langgraph_state.py
# ============================================================================
# LangGraph State Schema for NimbusGuard Operator
# ============================================================================

import time
from typing import Dict, Any, List, Optional, Annotated
from typing_extensions import TypedDict
from enum import Enum

import operator
from langgraph.graph import add_messages


class ScalingDecision(Enum):
    """Available scaling decisions."""
    SCALE_DOWN_2 = "scale_down_2"
    SCALE_DOWN_1 = "scale_down_1"
    NO_ACTION = "no_action"
    SCALE_UP_1 = "scale_up_1"
    SCALE_UP_2 = "scale_up_2"


class ObservabilityHealth(Enum):
    """Health status of observability systems."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class NimbusGuardState(TypedDict):
    """
    Central state for the NimbusGuard LangGraph workflow.
    This replaces the scattered state management in the current operator.
    """
    
    # === Resource Information ===
    resource_name: str
    resource_uid: str
    namespace: str
    current_replicas: int
    min_replicas: int
    max_replicas: int
    target_labels: Dict[str, str]
    
    # === Observability Data ===
    metrics: Dict[str, Any]
    feature_vector: List[float]
    feature_names: List[str]
    confidence_score: float
    health_score: float
    
    # === System Health ===
    prometheus_available: bool
    loki_available: bool
    tempo_available: bool
    kserve_available: bool
    observability_health: ObservabilityHealth
    
    # === Decision Context ===
    recent_actions: List[str]
    time_since_last_scale: float
    scaling_history: List[Dict[str, Any]]
    decision_confidence: float
    
    # === LLM Analysis ===
    llm_analysis: Optional[str]
    reasoning_steps: List[str]
    risk_assessment: Dict[str, float]
    
    # === Decision Making ===
    scaling_decision: Optional[ScalingDecision]
    target_replicas: int
    decision_reason: str
    execution_plan: List[str]
    
    # === Results ===
    execution_success: bool
    execution_errors: List[str]
    performance_metrics: Dict[str, Any]
    
    # === Messages (for LangGraph communication) ===
    messages: Annotated[List[Dict[str, Any]], add_messages]
    
    # === Workflow Control ===
    workflow_step: str
    retry_count: int
    max_retries: int
    should_continue: bool
    
    # === Timestamps ===
    workflow_started_at: float
    last_updated_at: float
    decision_made_at: Optional[float]
    execution_completed_at: Optional[float]
    
    # === DQN Metadata ===
    dqn_metadata: Optional[Dict[str, Any]]
    deployment_name: Optional[str]


def create_initial_state(
    resource_name: str,
    resource_uid: str,
    namespace: str,
    current_replicas: int,
    min_replicas: int,
    max_replicas: int,
    target_labels: Dict[str, str]
) -> NimbusGuardState:
    """Create initial state for a scaling workflow."""
    current_time = time.time()
    
    return NimbusGuardState(
        # Resource Information
        resource_name=resource_name,
        resource_uid=resource_uid,
        namespace=namespace,
        current_replicas=current_replicas,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        target_labels=target_labels,
        
        # Observability Data
        metrics={},
        feature_vector=[],
        feature_names=[],
        confidence_score=0.0,
        health_score=0.0,
        
        # System Health
        prometheus_available=False,
        loki_available=False,
        tempo_available=False,
        kserve_available=False,
        observability_health=ObservabilityHealth.UNAVAILABLE,
        
        # Decision Context
        recent_actions=[],
        time_since_last_scale=0.0,
        scaling_history=[],
        decision_confidence=0.0,
        
        # LLM Analysis
        llm_analysis=None,
        reasoning_steps=[],
        risk_assessment={},
        
        # Decision Making
        scaling_decision=None,
        target_replicas=current_replicas,
        decision_reason="",
        execution_plan=[],
        
        # Results
        execution_success=False,
        execution_errors=[],
        performance_metrics={},
        
        # Messages
        messages=[],
        
        # Workflow Control
        workflow_step="start",
        retry_count=0,
        max_retries=3,
        should_continue=True,
        
        # Timestamps
        workflow_started_at=current_time,
        last_updated_at=current_time,
        decision_made_at=None,
        execution_completed_at=None,
        
        # DQN Metadata
        dqn_metadata=None,
        deployment_name=None
    )


def update_state_timestamp(state: NimbusGuardState) -> NimbusGuardState:
    """Update the last_updated_at timestamp."""
    state["last_updated_at"] = time.time()
    return state


def add_reasoning_step(state: NimbusGuardState, step: str) -> NimbusGuardState:
    """Add a reasoning step to the workflow."""
    state["reasoning_steps"].append(f"[{time.time():.2f}] {step}")
    return update_state_timestamp(state)


def add_execution_error(state: NimbusGuardState, error: str) -> NimbusGuardState:
    """Add an execution error to the state."""
    state["execution_errors"].append(f"[{time.time():.2f}] {error}")
    return update_state_timestamp(state)


def is_scaling_action_valid(state: NimbusGuardState, decision: ScalingDecision) -> bool:
    """Check if a scaling action is valid given current constraints."""
    replica_changes = {
        ScalingDecision.SCALE_DOWN_2: -2,
        ScalingDecision.SCALE_DOWN_1: -1,
        ScalingDecision.NO_ACTION: 0,
        ScalingDecision.SCALE_UP_1: 1,
        ScalingDecision.SCALE_UP_2: 2
    }
    
    change = replica_changes.get(decision, 0)
    new_replicas = state["current_replicas"] + change
    
    return state["min_replicas"] <= new_replicas <= state["max_replicas"]


def calculate_target_replicas(state: NimbusGuardState, decision: ScalingDecision) -> int:
    """Calculate target replicas for a scaling decision."""
    replica_changes = {
        ScalingDecision.SCALE_DOWN_2: -2,
        ScalingDecision.SCALE_DOWN_1: -1,
        ScalingDecision.NO_ACTION: 0,
        ScalingDecision.SCALE_UP_1: 1,
        ScalingDecision.SCALE_UP_2: 2
    }
    
    change = replica_changes.get(decision, 0)
    target = state["current_replicas"] + change
    
    # Clamp to valid range
    return max(state["min_replicas"], min(target, state["max_replicas"])) 