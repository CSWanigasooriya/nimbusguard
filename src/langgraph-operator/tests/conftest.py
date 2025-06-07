"""
Pytest configuration and shared fixtures for NimbusGuard tests.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, MagicMock

# Import the components we're testing
from workflows.scaling_state import (
    MetricData, ScalingDecision, ActionResult, ScalingAction, 
    WorkflowStatus, ActionStatus, create_initial_state
)


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_metrics():
    """Sample metrics data for testing."""
    return MetricData(
        cpu_utilization=75.0,
        memory_utilization=60.0,
        request_rate=1500.0,
        error_rate=2.5,
        pod_count=5,
        health_score=0.8,
        response_time=150.0,
        timestamp=datetime.now()
    )


@pytest.fixture
def low_load_metrics():
    """Low load metrics for testing scale-down scenarios."""
    return MetricData(
        cpu_utilization=25.0,
        memory_utilization=30.0,
        request_rate=200.0,
        error_rate=0.5,
        pod_count=8,
        health_score=0.9,
        response_time=80.0,
        timestamp=datetime.now()
    )


@pytest.fixture
def high_load_metrics():
    """High load metrics for testing scale-up scenarios."""
    return MetricData(
        cpu_utilization=95.0,
        memory_utilization=85.0,
        request_rate=5000.0,
        error_rate=8.0,
        pod_count=3,
        health_score=0.4,
        response_time=500.0,
        timestamp=datetime.now()
    )


@pytest.fixture
def sample_scaling_decision(sample_metrics):
    """Sample scaling decision for testing."""
    return ScalingDecision(
        action=ScalingAction.SCALE_UP,
        target_replicas=7,
        current_replicas=5,
        confidence=0.85,
        reasoning="High CPU utilization detected, scaling up to handle load",
        metrics_snapshot=sample_metrics,
        q_learning_recommendation=ScalingAction.SCALE_UP,
        llm_recommendation=ScalingAction.SCALE_UP,
        timestamp=datetime.now()
    )


@pytest.fixture
def sample_action_result():
    """Sample action result for testing."""
    return ActionResult(
        action=ScalingAction.SCALE_UP,
        target_replicas=7,
        actual_replicas=7,
        status=ActionStatus.SUCCESS,
        timestamp=datetime.now(),
        execution_time_ms=1500.0,
        details="Successfully scaled deployment from 5 to 7 replicas"
    )


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "q_learning": {
            "learning_rate": 0.1,
            "discount_factor": 0.95,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995,
            "exploration_steps": 1000
        },
        "scaling": {
            "min_replicas": 1,
            "max_replicas": 20,
            "cpu_threshold_up": 80.0,
            "cpu_threshold_down": 30.0,
            "memory_threshold_up": 80.0,
            "memory_threshold_down": 30.0,
            "scale_up_cooldown": 300,
            "scale_down_cooldown": 600
        },
        "rewards": {
            "resource_efficiency_weight": 0.3,
            "performance_weight": 0.4,
            "stability_weight": 0.3,
            "target_cpu_utilization": 70.0,
            "target_memory_utilization": 70.0,
            "max_response_time": 200.0,
            "penalty_scale_factor": 2.0
        }
    }


@pytest.fixture
def mock_k8s_client():
    """Mock Kubernetes client for testing."""
    mock_client = MagicMock()
    
    # Mock deployment object
    mock_deployment = MagicMock()
    mock_deployment.spec.replicas = 5
    mock_deployment.metadata.name = "test-deployment"
    mock_deployment.metadata.namespace = "default"
    
    mock_client.read_namespaced_deployment.return_value = mock_deployment
    mock_client.patch_namespaced_deployment_scale.return_value = mock_deployment
    
    return mock_client


@pytest.fixture
def sample_workflow_state(sample_config):
    """Create a sample workflow state for testing."""
    return create_initial_state(
        workflow_id="test-workflow-123",
        trigger_event={
            "type": "metrics_threshold",
            "deployment": "test-deployment",
            "namespace": "default"
        },
        config=sample_config
    )


@pytest.fixture
def q_learning_config():
    """Configuration specific to Q-learning tests."""
    return {
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "epsilon_start": 0.8,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
        "exploration_steps": 500
    }


@pytest.fixture
def sample_q_metrics():
    """Sample metrics in the format expected by Q-learning agent."""
    return {
        "cpu_utilization": 75.0,
        "memory_utilization": 60.0,
        "request_rate": 1500.0,
        "pod_count": 5,
        "error_rate": 2.5
    }


# Test utilities
class AsyncMock(MagicMock):
    """Mock class for async functions."""
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


def assert_metrics_equal(actual: MetricData, expected: MetricData, tolerance: float = 0.01):
    """Assert that two MetricData objects are equal within tolerance."""
    assert abs(actual.cpu_utilization - expected.cpu_utilization) < tolerance
    assert abs(actual.memory_utilization - expected.memory_utilization) < tolerance
    assert abs(actual.request_rate - expected.request_rate) < tolerance
    assert abs(actual.error_rate - expected.error_rate) < tolerance
    assert actual.pod_count == expected.pod_count
    assert abs(actual.health_score - expected.health_score) < tolerance
    assert abs(actual.response_time - expected.response_time) < tolerance 