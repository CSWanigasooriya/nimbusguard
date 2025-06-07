"""
Unit tests for NimbusGuard operator.

Tests the main controller logic, metrics management, and policy reconciliation.
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

from nimbusguard_operator import NimbusGuardController
from workflows.scaling_state import WorkflowStatus, ScalingAction, ActionStatus


class TestNimbusGuardController:
    """Test cases for NimbusGuardController."""
    
    def test_controller_initialization(self, sample_config, mock_k8s_client):
        """Test controller initialization."""
        with patch('nimbusguard_operator.kubernetes.client.AppsV1Api') as mock_apps, \
             patch('nimbusguard_operator.kubernetes.client.CoreV1Api') as mock_core, \
             patch('nimbusguard_operator.QLearningAgent') as mock_q_agent, \
             patch('nimbusguard_operator.SupervisorAgent') as mock_supervisor, \
             patch('nimbusguard_operator.StateObserverAgent') as mock_observer:
            
            mock_apps.return_value = mock_k8s_client
            mock_core.return_value = mock_k8s_client
            
            controller = NimbusGuardController(sample_config)
            
            assert controller.config == sample_config
            assert isinstance(controller.active_policies, dict)
            assert isinstance(controller.active_workflows, dict)
            assert len(controller.active_policies) == 0
            assert len(controller.active_workflows) == 0
            
            # Check that agents were initialized
            mock_q_agent.assert_called_once_with(sample_config)
            mock_supervisor.assert_called_once_with(sample_config)
            mock_observer.assert_called_once_with(sample_config)
    
    def test_update_metrics(self, sample_config, mock_k8s_client):
        """Test metrics updating."""
        with patch('nimbusguard_operator.kubernetes.client.AppsV1Api') as mock_apps, \
             patch('nimbusguard_operator.kubernetes.client.CoreV1Api') as mock_core, \
             patch('nimbusguard_operator.QLearningAgent'), \
             patch('nimbusguard_operator.SupervisorAgent'), \
             patch('nimbusguard_operator.StateObserverAgent'):
            
            mock_apps.return_value = mock_k8s_client
            mock_core.return_value = mock_k8s_client
            
            controller = NimbusGuardController(sample_config)
            
            # Add some test data
            controller.active_policies["test/policy1"] = {
                "spec": {"target": {"name": "test-deployment"}},
                "status": {"phase": "Active"},
                "uid": "test-uid"
            }
            controller.active_policies["test/policy2"] = {
                "spec": {"target": {"name": "test-deployment2"}},
                "status": {"phase": "Learning"},
                "uid": "test-uid2"
            }
            
            # Test metrics update
            controller.update_metrics()
            
            # Check that metrics were set
            assert controller.active_policies_gauge._value._value == 2
            assert controller.active_workflows_gauge._value._value == 0
            assert controller.controller_uptime._value._value > 0
    
    def test_get_metrics(self, sample_config, mock_k8s_client):
        """Test metrics endpoint."""
        with patch('nimbusguard_operator.kubernetes.client.AppsV1Api') as mock_apps, \
             patch('nimbusguard_operator.kubernetes.client.CoreV1Api') as mock_core, \
             patch('nimbusguard_operator.QLearningAgent'), \
             patch('nimbusguard_operator.SupervisorAgent'), \
             patch('nimbusguard_operator.StateObserverAgent'):
            
            mock_apps.return_value = mock_k8s_client
            mock_core.return_value = mock_k8s_client
            
            controller = NimbusGuardController(sample_config)
            
            metrics_output = controller.get_metrics()
            
            assert isinstance(metrics_output, str)
            assert "nimbusguard_" in metrics_output
            assert "controller_uptime_seconds" in metrics_output
    
    @pytest.mark.asyncio
    async def test_reconcile_scaling_policy_new(self, sample_config, mock_k8s_client):
        """Test reconciling a new scaling policy."""
        with patch('nimbusguard_operator.kubernetes.client.AppsV1Api') as mock_apps, \
             patch('nimbusguard_operator.kubernetes.client.CoreV1Api') as mock_core, \
             patch('nimbusguard_operator.QLearningAgent'), \
             patch('nimbusguard_operator.SupervisorAgent'), \
             patch('nimbusguard_operator.StateObserverAgent'):
            
            mock_apps.return_value = mock_k8s_client
            mock_core.return_value = mock_k8s_client
            
            controller = NimbusGuardController(sample_config)
            
            spec = {
                "target": {
                    "name": "test-deployment",
                    "namespace": "default"
                },
                "scaling": {
                    "minReplicas": 1,
                    "maxReplicas": 10
                }
            }
            
            status = {}
            name = "test-policy"
            namespace = "default"
            uid = "test-uid-123"
            
            result = await controller.reconcile_scaling_policy(
                spec, status, name, namespace, uid
            )
            
            assert isinstance(result, dict)
            
            # Check that policy was added to active policies
            policy_key = f"{namespace}/{name}"
            assert policy_key in controller.active_policies
            assert controller.active_policies[policy_key]["spec"] == spec
            assert controller.active_policies[policy_key]["uid"] == uid
    
    @pytest.mark.asyncio
    async def test_reconcile_scaling_policy_missing_deployment(self, sample_config):
        """Test reconciling a policy with missing target deployment."""
        mock_k8s_client = MagicMock()
        mock_k8s_client.read_namespaced_deployment.side_effect = Exception("Not found")
        
        with patch('nimbusguard_operator.kubernetes.client.AppsV1Api') as mock_apps, \
             patch('nimbusguard_operator.kubernetes.client.CoreV1Api') as mock_core, \
             patch('nimbusguard_operator.QLearningAgent'), \
             patch('nimbusguard_operator.SupervisorAgent'), \
             patch('nimbusguard_operator.StateObserverAgent'):
            
            mock_apps.return_value = mock_k8s_client
            mock_core.return_value = mock_k8s_client
            
            controller = NimbusGuardController(sample_config)
            
            spec = {
                "target": {
                    "name": "nonexistent-deployment",
                    "namespace": "default"
                }
            }
            
            with pytest.raises(Exception):
                await controller.reconcile_scaling_policy(
                    spec, {}, "test-policy", "default", "test-uid"
                )
    
    @pytest.mark.asyncio
    async def test_initialize_policy_status(self, sample_config, mock_k8s_client):
        """Test policy status initialization."""
        with patch('nimbusguard_operator.kubernetes.client.AppsV1Api') as mock_apps, \
             patch('nimbusguard_operator.kubernetes.client.CoreV1Api') as mock_core, \
             patch('nimbusguard_operator.QLearningAgent'), \
             patch('nimbusguard_operator.SupervisorAgent'), \
             patch('nimbusguard_operator.StateObserverAgent'):
            
            mock_apps.return_value = mock_k8s_client
            mock_core.return_value = mock_k8s_client
            
            controller = NimbusGuardController(sample_config)
            
            # Mock deployment
            mock_deployment = MagicMock()
            mock_deployment.spec.replicas = 5
            mock_deployment.metadata.name = "test-deployment"
            
            policy_key = "default/test-policy"
            status = await controller._initialize_policy_status(policy_key, mock_deployment)
            
            assert isinstance(status, dict)
            assert status["phase"] == "Initializing"
            assert "observedGeneration" in status
            assert "conditions" in status
    
    @pytest.mark.asyncio
    async def test_update_policy_status(self, sample_config, mock_k8s_client):
        """Test policy status updates."""
        with patch('nimbusguard_operator.kubernetes.client.AppsV1Api') as mock_apps, \
             patch('nimbusguard_operator.kubernetes.client.CoreV1Api') as mock_core, \
             patch('nimbusguard_operator.QLearningAgent'), \
             patch('nimbusguard_operator.SupervisorAgent'), \
             patch('nimbusguard_operator.StateObserverAgent'):
            
            mock_apps.return_value = mock_k8s_client
            mock_core.return_value = mock_k8s_client
            
            controller = NimbusGuardController(sample_config)
            
            # Set up policy
            policy_key = "default/test-policy"
            controller.active_policies[policy_key] = {
                "spec": {"target": {"name": "test-deployment"}},
                "status": {"phase": "Initializing"},
                "uid": "test-uid"
            }
            
            status = await controller._update_policy_status(
                policy_key, "Active", error_message=None
            )
            
            assert status["phase"] == "Active"
            assert "lastTransitionTime" in status
            
            # Test with error
            error_status = await controller._update_policy_status(
                policy_key, "Error", error_message="Test error"
            )
            
            assert error_status["phase"] == "Error"
            assert error_status["message"] == "Test error"
    
    @pytest.mark.asyncio 
    async def test_should_trigger_scaling_decision(self, sample_config, mock_k8s_client):
        """Test scaling decision trigger logic."""
        with patch('nimbusguard_operator.kubernetes.client.AppsV1Api') as mock_apps, \
             patch('nimbusguard_operator.kubernetes.client.CoreV1Api') as mock_core, \
             patch('nimbusguard_operator.QLearningAgent'), \
             patch('nimbusguard_operator.SupervisorAgent'), \
             patch('nimbusguard_operator.StateObserverAgent'):
            
            mock_apps.return_value = mock_k8s_client
            mock_core.return_value = mock_k8s_client
            
            controller = NimbusGuardController(sample_config)
            
            policy_key = "default/test-policy"
            spec = {
                "scaling": {
                    "evaluationInterval": "30s"
                }
            }
            status = {
                "lastScalingDecision": (datetime.now() - timedelta(minutes=2)).isoformat()
            }
            
            # Mock deployment
            mock_deployment = MagicMock()
            mock_deployment.spec.replicas = 5
            
            # Should trigger because enough time has passed
            should_trigger = await controller._should_trigger_scaling_decision(
                policy_key, spec, status, mock_deployment
            )
            
            assert isinstance(should_trigger, bool)
    
    def test_prometheus_metrics_initialization(self, sample_config, mock_k8s_client):
        """Test that Prometheus metrics are properly initialized."""
        with patch('nimbusguard_operator.kubernetes.client.AppsV1Api') as mock_apps, \
             patch('nimbusguard_operator.kubernetes.client.CoreV1Api') as mock_core, \
             patch('nimbusguard_operator.QLearningAgent'), \
             patch('nimbusguard_operator.SupervisorAgent'), \
             patch('nimbusguard_operator.StateObserverAgent'):
            
            mock_apps.return_value = mock_k8s_client
            mock_core.return_value = mock_k8s_client
            
            controller = NimbusGuardController(sample_config)
            
            # Check that all expected metrics exist
            assert hasattr(controller, 'policies_total')
            assert hasattr(controller, 'workflows_total')
            assert hasattr(controller, 'workflows_successful')
            assert hasattr(controller, 'workflows_failed')
            assert hasattr(controller, 'scaling_actions_total')
            assert hasattr(controller, 'active_policies_gauge')
            assert hasattr(controller, 'active_workflows_gauge')
            assert hasattr(controller, 'controller_uptime')
            assert hasattr(controller, 'policies_by_phase')
            assert hasattr(controller, 'q_learning_epsilon')
            assert hasattr(controller, 'q_learning_episodes')


class TestControllerHelperMethods:
    """Test cases for controller helper methods."""
    
    @pytest.mark.asyncio
    async def test_start_scaling_workflow(self, sample_config, mock_k8s_client):
        """Test starting a scaling workflow."""
        with patch('nimbusguard_operator.kubernetes.client.AppsV1Api') as mock_apps, \
             patch('nimbusguard_operator.kubernetes.client.CoreV1Api') as mock_core, \
             patch('nimbusguard_operator.QLearningAgent'), \
             patch('nimbusguard_operator.SupervisorAgent'), \
             patch('nimbusguard_operator.StateObserverAgent'):
            
            mock_apps.return_value = mock_k8s_client
            mock_core.return_value = mock_k8s_client
            
            controller = NimbusGuardController(sample_config)
            
            policy_key = "default/test-policy"
            spec = {"target": {"name": "test-deployment"}}
            status = {"phase": "Active"}
            
            workflow_id = await controller._start_scaling_workflow(
                policy_key, spec, status
            )
            
            assert isinstance(workflow_id, str)
            assert workflow_id in controller.active_workflows
            assert controller.active_workflows[workflow_id]["status"] == WorkflowStatus.INITIALIZING 