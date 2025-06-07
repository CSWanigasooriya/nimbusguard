"""
Integration tests for NimbusGuard components.

Tests the interaction between different components of the system.
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from workflows.scaling_state import (
    create_initial_state, MetricData, ScalingAction, WorkflowStatus
)
from ml_models.q_learning import QLearningAgent


class TestWorkflowIntegration:
    """Integration tests for workflow components."""
    
    def test_initial_state_creation_with_real_config(self, sample_config):
        """Test creating initial workflow state with real configuration."""
        workflow_id = "integration-test-workflow"
        trigger_event = {
            "type": "high_cpu_utilization",
            "deployment": "test-app",
            "namespace": "production",
            "metrics": {
                "cpu_utilization": 85.0,
                "memory_utilization": 70.0
            }
        }
        
        state = create_initial_state(workflow_id, trigger_event, sample_config)
        
        # Verify state structure
        assert state["workflow_id"] == workflow_id
        assert state["status"] == WorkflowStatus.INITIALIZING
        assert state["context"]["trigger_event"] == trigger_event
        assert state["context"]["config"] == sample_config
        
        # Verify configuration is properly embedded
        q_learning_config = state["context"]["config"]["q_learning"]
        assert q_learning_config["learning_rate"] == 0.1
        assert q_learning_config["discount_factor"] == 0.95
    
    def test_q_learning_agent_with_workflow_metrics(self, sample_config):
        """Test Q-learning agent with metrics from workflow state."""
        # Create Q-learning agent
        agent = QLearningAgent(sample_config)
        
        # Create sample metrics that would come from workflow state
        workflow_metrics = {
            "cpu_utilization": 75.0,
            "memory_utilization": 60.0,
            "request_rate": 1500.0,
            "pod_count": 5,
            "error_rate": 2.5
        }
        
        # Test discretization
        state = agent.discretize_metrics(workflow_metrics)
        assert state is not None
        
        # Test action selection
        decision = agent.choose_scaling_action(workflow_metrics)
        assert "action" in decision
        assert "confidence" in decision
        assert "reasoning" in decision
        
        # Test that the decision is valid
        valid_actions = {"scale_up", "scale_down", "no_action"}
        assert decision["action"] in valid_actions
    
    def test_metrics_flow_through_system(self, sample_config):
        """Test how metrics flow through the entire system."""
        # Start with raw metrics
        raw_metrics = MetricData(
            cpu_utilization=85.0,
            memory_utilization=75.0,
            request_rate=2000.0,
            error_rate=3.5,
            pod_count=4,
            health_score=0.7,
            response_time=180.0
        )
        
        # Create workflow state with these metrics
        workflow_id = "metrics-flow-test"
        trigger_event = {"type": "metrics_threshold"}
        
        state = create_initial_state(workflow_id, trigger_event, sample_config)
        
        # Add metrics to state
        state["current_metrics"] = raw_metrics
        state["metrics_history"] = [raw_metrics]
        
        # Convert to format expected by Q-learning agent
        q_learning_metrics = {
            "cpu_utilization": raw_metrics.cpu_utilization,
            "memory_utilization": raw_metrics.memory_utilization,
            "request_rate": raw_metrics.request_rate,
            "pod_count": raw_metrics.pod_count,
            "error_rate": raw_metrics.error_rate
        }
        
        # Test with Q-learning agent
        agent = QLearningAgent(sample_config)
        discretized_state = agent.discretize_metrics(q_learning_metrics)
        
        # Verify the flow
        assert discretized_state is not None
        assert state["current_metrics"].cpu_utilization == 85.0
        assert q_learning_metrics["cpu_utilization"] == 85.0
    
    @pytest.mark.asyncio
    async def test_scaling_decision_workflow(self, sample_config):
        """Test the complete scaling decision workflow."""
        # Create initial state
        workflow_id = "scaling-decision-test"
        trigger_event = {
            "type": "high_load",
            "deployment": "web-app",
            "namespace": "production"
        }
        
        state = create_initial_state(workflow_id, trigger_event, sample_config)
        
        # Add current metrics indicating high load
        high_load_metrics = MetricData(
            cpu_utilization=90.0,
            memory_utilization=85.0,
            request_rate=5000.0,
            error_rate=7.0,
            pod_count=3,
            health_score=0.4,
            response_time=400.0
        )
        
        state["current_metrics"] = high_load_metrics
        state["metrics_history"] = [high_load_metrics]
        
        # Use Q-learning agent to make a decision
        agent = QLearningAgent(sample_config)
        
        metrics_dict = {
            "cpu_utilization": high_load_metrics.cpu_utilization,
            "memory_utilization": high_load_metrics.memory_utilization,
            "request_rate": high_load_metrics.request_rate,
            "pod_count": high_load_metrics.pod_count,
            "error_rate": high_load_metrics.error_rate
        }
        
        decision = agent.choose_scaling_action(metrics_dict)
        
        # Verify decision makes sense for high load
        # With high CPU/memory and low pod count, should likely scale up
        assert decision["action"] in ["scale_up", "no_action"]  # Should not scale down
        
        # Update state with decision
        state["status"] = WorkflowStatus.DECIDING
        state["agent_outputs"]["q_learning"] = decision
        
        # Verify state is properly updated
        assert state["status"] == WorkflowStatus.DECIDING
        assert "q_learning" in state["agent_outputs"]
        assert state["agent_outputs"]["q_learning"]["action"] == decision["action"]


class TestErrorHandling:
    """Integration tests for error handling scenarios."""
    
    def test_q_learning_with_invalid_metrics(self, sample_config):
        """Test Q-learning agent behavior with invalid metrics."""
        agent = QLearningAgent(sample_config)
        
        # Test with missing metrics
        incomplete_metrics = {
            "cpu_utilization": 75.0,
            # Missing other required metrics
        }
        
        # Should handle gracefully without crashing
        state = agent.discretize_metrics(incomplete_metrics)
        assert state is not None
        
        # Test with extreme values
        extreme_metrics = {
            "cpu_utilization": 1000.0,  # Over 100%
            "memory_utilization": -50.0,  # Negative
            "request_rate": float('inf'),  # Infinity
            "pod_count": 0,  # Zero pods
            "error_rate": 200.0  # Over 100%
        }
        
        # Should handle gracefully
        state = agent.discretize_metrics(extreme_metrics)
        assert state is not None
        
        # Values should be clamped to valid ranges
        assert 0 <= state.cpu_utilization_bin <= 9
        assert 0 <= state.memory_utilization_bin <= 9
    
    def test_workflow_state_validation(self, sample_config):
        """Test workflow state validation with various error conditions."""
        from workflows.scaling_state import validate_state
        
        # Create valid initial state
        state = create_initial_state(
            "test-workflow", 
            {"type": "test"}, 
            sample_config
        )
        
        # Should be valid initially
        errors = validate_state(state)
        assert len(errors) == 0
        
        # Introduce errors
        state["retry_count"] = -1  # Invalid negative retry count
        state["max_retries"] = 0   # Invalid max retries
        del state["workflow_id"]   # Missing required field
        
        # Should detect errors
        errors = validate_state(state)
        assert len(errors) > 0
        
        # Check specific error messages
        error_messages = " ".join(errors)
        assert "workflow_id" in error_messages
        assert "retry_count" in error_messages 