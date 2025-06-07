"""
Unit tests for scaling state management.

Tests the core data structures, reducers, and state management functions
used in the LangGraph scaling workflow.
"""

import pytest
from datetime import datetime, timedelta
from typing import List

from workflows.scaling_state import (
    MetricData, ScalingDecision, ActionResult, RewardSignal,
    ScalingAction, WorkflowStatus, ActionStatus,
    create_initial_state, serialize_state_for_logging, validate_state,
    merge_messages, merge_metrics_history, merge_decisions_history,
    merge_action_history, merge_reward_history, update_context
)
from langchain_core.messages import HumanMessage, AIMessage


class TestMetricData:
    """Test cases for MetricData dataclass."""
    
    def test_metric_data_creation(self):
        """Test basic MetricData creation."""
        metrics = MetricData(
            cpu_utilization=75.0,
            memory_utilization=60.0,
            request_rate=1500.0,
            error_rate=2.5,
            pod_count=5,
            health_score=0.8,
            response_time=150.0
        )
        
        assert metrics.cpu_utilization == 75.0
        assert metrics.memory_utilization == 60.0
        assert metrics.request_rate == 1500.0
        assert metrics.error_rate == 2.5
        assert metrics.pod_count == 5
        assert metrics.health_score == 0.8
        assert metrics.response_time == 150.0
        assert isinstance(metrics.timestamp, datetime)
    
    def test_metric_data_defaults(self):
        """Test MetricData with default values."""
        metrics = MetricData(
            cpu_utilization=50.0,
            memory_utilization=40.0,
            request_rate=1000.0,
            error_rate=1.0,
            pod_count=3
        )
        
        assert metrics.health_score == 0.5  # Default value
        assert metrics.response_time == 100.0  # Default value
        assert isinstance(metrics.timestamp, datetime)


class TestScalingDecision:
    """Test cases for ScalingDecision dataclass."""
    
    def test_scaling_decision_creation(self, sample_metrics):
        """Test basic ScalingDecision creation."""
        decision = ScalingDecision(
            action=ScalingAction.SCALE_UP,
            target_replicas=7,
            current_replicas=5,
            confidence=0.85,
            reasoning="High CPU utilization",
            metrics_snapshot=sample_metrics,
            q_learning_recommendation=ScalingAction.SCALE_UP,
            llm_recommendation=ScalingAction.SCALE_UP
        )
        
        assert decision.action == ScalingAction.SCALE_UP
        assert decision.target_replicas == 7
        assert decision.current_replicas == 5
        assert decision.confidence == 0.85
        assert decision.reasoning == "High CPU utilization"
        assert decision.metrics_snapshot == sample_metrics
        assert decision.q_learning_recommendation == ScalingAction.SCALE_UP
        assert decision.llm_recommendation == ScalingAction.SCALE_UP
        assert isinstance(decision.timestamp, datetime)
    
    def test_scaling_decision_optional_fields(self, sample_metrics):
        """Test ScalingDecision with optional fields."""
        decision = ScalingDecision(
            action=ScalingAction.MAINTAIN,
            target_replicas=5,
            current_replicas=5,
            confidence=0.6,
            reasoning="Current load is acceptable",
            metrics_snapshot=sample_metrics
        )
        
        assert decision.q_learning_recommendation is None
        assert decision.llm_recommendation is None


class TestActionResult:
    """Test cases for ActionResult dataclass."""
    
    def test_action_result_creation(self):
        """Test basic ActionResult creation."""
        result = ActionResult(
            action=ScalingAction.SCALE_UP,
            target_replicas=7,
            actual_replicas=7,
            status=ActionStatus.SUCCESS,
            execution_time_ms=1500.0,
            details="Successfully scaled up"
        )
        
        assert result.action == ScalingAction.SCALE_UP
        assert result.target_replicas == 7
        assert result.actual_replicas == 7
        assert result.status == ActionStatus.SUCCESS
        assert result.execution_time_ms == 1500.0
        assert result.details == "Successfully scaled up"
        assert isinstance(result.timestamp, datetime)
    
    def test_action_result_compatibility_properties(self):
        """Test backward compatibility properties."""
        result = ActionResult(
            action=ScalingAction.SCALE_DOWN,
            target_replicas=3,
            actual_replicas=3,
            status=ActionStatus.SUCCESS,
            execution_time_ms=2000.0
        )
        
        # Test compatibility properties
        assert result.success is True
        assert result.action_taken == ScalingAction.SCALE_DOWN
        assert result.new_replicas == 3
        assert result.execution_time == 2.0  # Convert ms to seconds
    
    def test_action_result_failed_status(self):
        """Test ActionResult with failed status."""
        result = ActionResult(
            action=ScalingAction.SCALE_UP,
            target_replicas=8,
            actual_replicas=5,
            status=ActionStatus.FAILED,
            error_message="Insufficient resources"
        )
        
        assert result.success is False
        assert result.error_message == "Insufficient resources"


class TestRewardSignal:
    """Test cases for RewardSignal dataclass."""
    
    def test_reward_signal_creation(self, sample_metrics):
        """Test basic RewardSignal creation."""
        reward = RewardSignal(
            reward=0.75,
            components={
                "resource_efficiency": 0.3,
                "performance": 0.25,
                "stability": 0.2
            },
            metrics_before=sample_metrics,
            metrics_after=sample_metrics
        )
        
        assert reward.reward == 0.75
        assert reward.components["resource_efficiency"] == 0.3
        assert reward.components["performance"] == 0.25
        assert reward.components["stability"] == 0.2
        assert reward.metrics_before == sample_metrics
        assert reward.metrics_after == sample_metrics
        assert isinstance(reward.timestamp, datetime)


class TestStateMergers:
    """Test cases for state reducer functions."""
    
    def test_merge_messages(self):
        """Test message merging."""
        existing = [HumanMessage(content="Hello")]
        new = [AIMessage(content="Hi there"), HumanMessage(content="How are you?")]
        
        result = merge_messages(existing, new)
        
        assert len(result) == 3
        assert result[0].content == "Hello"
        assert result[1].content == "Hi there"
        assert result[2].content == "How are you?"
    
    def test_merge_metrics_history(self, sample_metrics):
        """Test metrics history merging with sliding window."""
        # Create metrics with different timestamps
        metrics1 = [
            MetricData(cpu_utilization=50.0, memory_utilization=40.0, 
                      request_rate=1000.0, error_rate=1.0, pod_count=3)
        ]
        metrics2 = [sample_metrics]
        
        result = merge_metrics_history(metrics1, metrics2)
        
        assert len(result) == 2
        assert result[0].cpu_utilization == 50.0
        assert result[1] == sample_metrics
    
    def test_merge_metrics_history_sliding_window(self):
        """Test metrics history sliding window (max 100 items)."""
        # Create more than 100 metrics
        existing = [
            MetricData(cpu_utilization=float(i), memory_utilization=40.0,
                      request_rate=1000.0, error_rate=1.0, pod_count=3)
            for i in range(95)
        ]
        new = [
            MetricData(cpu_utilization=float(i + 95), memory_utilization=40.0,
                      request_rate=1000.0, error_rate=1.0, pod_count=3)
            for i in range(10)
        ]
        
        result = merge_metrics_history(existing, new)
        
        assert len(result) == 100
        assert result[0].cpu_utilization == 5.0  # Should start from index 5
        assert result[-1].cpu_utilization == 104.0  # Should end at 95 + 9
    
    def test_merge_decisions_history(self, sample_metrics):
        """Test decision history merging."""
        decision1 = ScalingDecision(
            action=ScalingAction.SCALE_UP,
            target_replicas=6,
            current_replicas=5,
            confidence=0.8,
            reasoning="First decision",
            metrics_snapshot=sample_metrics
        )
        decision2 = ScalingDecision(
            action=ScalingAction.MAINTAIN,
            target_replicas=6,
            current_replicas=6,
            confidence=0.7,
            reasoning="Second decision",
            metrics_snapshot=sample_metrics
        )
        
        existing = [decision1]
        new = [decision2]
        
        result = merge_decisions_history(existing, new)
        
        assert len(result) == 2
        assert result[0].reasoning == "First decision"
        assert result[1].reasoning == "Second decision"
    
    def test_merge_action_history(self):
        """Test action history merging."""
        action1 = ActionResult(
            action=ScalingAction.SCALE_UP,
            target_replicas=6,
            actual_replicas=6,
            status=ActionStatus.SUCCESS
        )
        action2 = ActionResult(
            action=ScalingAction.MAINTAIN,
            target_replicas=6,
            actual_replicas=6,
            status=ActionStatus.SUCCESS
        )
        
        existing = [action1]
        new = [action2]
        
        result = merge_action_history(existing, new)
        
        assert len(result) == 2
        assert result[0].action == ScalingAction.SCALE_UP
        assert result[1].action == ScalingAction.MAINTAIN
    
    def test_merge_reward_history(self, sample_metrics):
        """Test reward history merging."""
        reward1 = RewardSignal(
            reward=0.5,
            components={"test": 0.5},
            metrics_before=sample_metrics
        )
        reward2 = RewardSignal(
            reward=0.8,
            components={"test": 0.8},
            metrics_before=sample_metrics
        )
        
        existing = [reward1]
        new = [reward2]
        
        result = merge_reward_history(existing, new)
        
        assert len(result) == 2
        assert result[0].reward == 0.5
        assert result[1].reward == 0.8
    
    def test_update_context(self):
        """Test context updating."""
        existing = {"key1": "value1", "key2": "value2"}
        new = {"key2": "updated_value2", "key3": "value3"}
        
        result = update_context(existing, new)
        
        assert result["key1"] == "value1"
        assert result["key2"] == "updated_value2"  # Should be updated
        assert result["key3"] == "value3"  # Should be added


class TestWorkflowState:
    """Test cases for workflow state management."""
    
    def test_create_initial_state(self, sample_config):
        """Test initial state creation."""
        workflow_id = "test-workflow-123"
        trigger_event = {
            "type": "metrics_threshold",
            "deployment": "test-deployment",
            "namespace": "default"
        }
        
        state = create_initial_state(workflow_id, trigger_event, sample_config)
        
        assert state["workflow_id"] == workflow_id
        assert state["status"] == WorkflowStatus.INITIALIZING
        assert state["current_agent"] == "supervisor"
        assert state["next_agent"] is None
        assert isinstance(state["started_at"], datetime)
        assert state["messages"] == []
        assert state["agent_outputs"] == {}
        assert state["current_metrics"] is None
        assert state["metrics_history"] == []
        assert state["scaling_decision"] is None
        assert state["decisions_history"] == []
        assert state["last_action"] is None
        assert state["action_history"] == []
        assert state["rollback_available"] is False
        assert state["last_reward"] is None
        assert state["reward_history"] == []
        assert state["cumulative_reward"] == 0.0
        assert state["errors"] == []
        assert state["retry_count"] == 0
        assert state["max_retries"] == 3
        assert state["human_approval_required"] is False
        assert state["human_feedback"] is None
        assert state["completed"] is False
        assert state["final_result"] is None
        
        # Check context
        assert "trigger_event" in state["context"]
        assert "config" in state["context"]
        assert state["context"]["trigger_event"] == trigger_event
        assert state["context"]["config"] == sample_config
    
    def test_serialize_state_for_logging(self, sample_workflow_state):
        """Test state serialization for logging."""
        serialized = serialize_state_for_logging(sample_workflow_state)
        
        assert isinstance(serialized, dict)
        assert "workflow_id" in serialized
        assert "status" in serialized
        assert "current_agent" in serialized
        
        # Check that enums are converted to strings
        assert serialized["status"] == "initializing"
    
    def test_validate_state_valid(self, sample_workflow_state):
        """Test state validation with valid state."""
        errors = validate_state(sample_workflow_state)
        assert len(errors) == 0
    
    def test_validate_state_invalid(self, sample_workflow_state):
        """Test state validation with invalid state."""
        # Make state invalid by removing required fields
        del sample_workflow_state["workflow_id"]
        sample_workflow_state["retry_count"] = -1
        sample_workflow_state["max_retries"] = 0
        
        errors = validate_state(sample_workflow_state)
        
        assert len(errors) > 0
        assert any("workflow_id" in error for error in errors)
        assert any("retry_count" in error for error in errors)
        assert any("max_retries" in error for error in errors) 