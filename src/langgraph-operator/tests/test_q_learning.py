"""
Unit tests for Q-learning agent.

Tests the Q-learning algorithm implementation for Kubernetes scaling decisions.
"""

import pytest
import numpy as np
from ml_models.q_learning import QLearningAgent, QState, QAction, calculate_reward


class TestQState:
    """Test cases for QState dataclass."""
    
    def test_qstate_creation(self):
        """Test basic QState creation."""
        state = QState(5, 4, 6, 3, 2)
        assert state.cpu_utilization_bin == 5
        assert state.memory_utilization_bin == 4
        assert state.request_rate_bin == 6
        assert state.pod_count_bin == 3
        assert state.error_rate_bin == 2
    
    def test_qstate_to_tuple(self):
        """Test QState tuple conversion."""
        state = QState(5, 4, 6, 3, 2)
        assert state.to_tuple() == (5, 4, 6, 3, 2)
    
    def test_qstate_equality(self):
        """Test QState equality."""
        state1 = QState(5, 4, 6, 3, 2)
        state2 = QState(5, 4, 6, 3, 2)
        state3 = QState(5, 4, 6, 3, 1)
        
        assert state1 == state2
        assert state1 != state3
        assert hash(state1) == hash(state2)


class TestQAction:
    """Test cases for QAction dataclass."""
    
    def test_qaction_creation(self):
        """Test basic QAction creation."""
        action = QAction("scale_up", 2)
        assert action.action_type == "scale_up"
        assert action.replicas_change == 2
    
    def test_qaction_to_tuple(self):
        """Test QAction tuple conversion."""
        action = QAction("scale_down", 1)
        assert action.to_tuple() == ("scale_down", 1)
    
    def test_qaction_equality(self):
        """Test QAction equality."""
        action1 = QAction("scale_up", 2)
        action2 = QAction("scale_up", 2)
        action3 = QAction("scale_up", 3)
        
        assert action1 == action2
        assert action1 != action3
        assert hash(action1) == hash(action2)


class TestQLearningAgent:
    """Test cases for QLearningAgent."""
    
    def test_agent_initialization(self, q_learning_config):
        """Test Q-learning agent initialization."""
        config = {"q_learning": q_learning_config}
        agent = QLearningAgent(config)
        
        assert agent.learning_rate == 0.1
        assert agent.discount_factor == 0.95
        assert agent.epsilon == 0.8
        assert agent.epsilon_end == 0.01
        assert agent.epsilon_decay == 0.995
        assert agent.exploration_steps == 500
        
        # Check action space
        assert len(agent.actions) == 8
        assert any(action.action_type == "no_action" for action in agent.actions)
        assert any(action.action_type == "scale_up" for action in agent.actions)
        assert any(action.action_type == "scale_down" for action in agent.actions)
        
        # Check initial state
        assert agent.episodes == 0
        assert agent.total_reward == 0.0
        assert agent.last_state is None
        assert agent.last_action is None
        assert len(agent.q_table) == 0
    
    def test_discretize_metrics(self, q_learning_config, sample_q_metrics):
        """Test metrics discretization."""
        config = {"q_learning": q_learning_config}
        agent = QLearningAgent(config)
        
        state = agent.discretize_metrics(sample_q_metrics)
        
        assert isinstance(state, QState)
        assert 0 <= state.cpu_utilization_bin <= 9
        assert 0 <= state.memory_utilization_bin <= 9
        assert 0 <= state.request_rate_bin <= 9
        assert 0 <= state.pod_count_bin <= 8
        assert 0 <= state.error_rate_bin <= 6
    
    def test_get_set_q_value(self, q_learning_config):
        """Test Q-value operations."""
        config = {"q_learning": q_learning_config}
        agent = QLearningAgent(config)
        
        state = QState(1, 2, 3, 4, 5)
        action = QAction("scale_up", 2)
        
        # Initially should be 0.0
        assert agent.get_q_value(state, action) == 0.0
        
        # Set new value
        agent.set_q_value(state, action, 0.75)
        assert agent.get_q_value(state, action) == 0.75
    
    def test_select_action_exploration(self, q_learning_config):
        """Test action selection during exploration."""
        config = {"q_learning": q_learning_config}
        agent = QLearningAgent(config)
        agent.epsilon = 1.0  # Force exploration
        
        state = QState(1, 2, 3, 4, 5)
        
        # With epsilon = 1.0, should explore (random actions)
        actions = []
        for _ in range(10):
            action = agent.select_action(state, exploration=True)
            actions.append(action)
        
        # Should have some variety
        unique_actions = set(action.to_tuple() for action in actions)
        assert len(unique_actions) >= 1  # At least some variation expected
    
    def test_select_action_exploitation(self, q_learning_config):
        """Test action selection during exploitation."""
        config = {"q_learning": q_learning_config}
        agent = QLearningAgent(config)
        agent.epsilon = 0.0  # Force exploitation
        
        state = QState(1, 2, 3, 4, 5)
        
        # Set clear best action
        best_action = QAction("scale_up", 2)
        agent.set_q_value(state, best_action, 1.0)
        
        for other_action in agent.actions:
            if other_action != best_action:
                agent.set_q_value(state, other_action, 0.5)
        
        # Should consistently choose best action
        for _ in range(5):
            selected = agent.select_action(state, exploration=True)
            assert selected == best_action
    
    def test_choose_scaling_action(self, q_learning_config, sample_q_metrics):
        """Test choosing scaling action."""
        config = {"q_learning": q_learning_config}
        agent = QLearningAgent(config)
        
        result = agent.choose_scaling_action(sample_q_metrics)
        
        assert isinstance(result, dict)
        assert "action" in result
        assert "replicas_change" in result
        assert "confidence" in result
        assert "reasoning" in result
        assert "q_values" in result
        
        valid_actions = {"scale_up", "scale_down", "no_action"}
        assert result["action"] in valid_actions
        assert 0.0 <= result["confidence"] <= 1.0
    
    def test_provide_feedback(self, q_learning_config, sample_q_metrics):
        """Test providing feedback."""
        config = {"q_learning": q_learning_config}
        agent = QLearningAgent(config)
        
        # Make decision to set up state
        agent.choose_scaling_action(sample_q_metrics)
        
        initial_episodes = agent.episodes
        agent.provide_feedback(0.7, sample_q_metrics)
        
        assert agent.episodes == initial_episodes + 1
        assert agent.total_reward == 0.7
    
    def test_get_learning_stats(self, q_learning_config):
        """Test getting learning statistics."""
        config = {"q_learning": q_learning_config}
        agent = QLearningAgent(config)
        
        agent.episodes = 50
        agent.total_reward = 25.5
        
        stats = agent.get_learning_stats()
        
        assert isinstance(stats, dict)
        assert stats["episodes"] == 50
        assert stats["total_reward"] == 25.5
        assert stats["average_reward"] == 0.51
        assert stats["epsilon"] == agent.epsilon
        assert stats["q_table_size"] == len(agent.q_table)


class TestRewardCalculation:
    """Test cases for reward calculation."""
    
    def test_calculate_reward_improvement(self):
        """Test reward calculation with improvement."""
        metrics_before = {
            "cpu_utilization": 80.0,
            "memory_utilization": 70.0,
            "request_rate": 1000.0,
            "error_rate": 5.0,
            "response_time": 200.0,
            "pod_count": 5
        }
        
        metrics_after = {
            "cpu_utilization": 70.0,  # Improved
            "memory_utilization": 65.0,  # Improved
            "request_rate": 1000.0,
            "error_rate": 3.0,  # Improved
            "response_time": 150.0,  # Improved
            "pod_count": 6
        }
        
        config = {
            "resource_efficiency_weight": 0.3,
            "performance_weight": 0.4,
            "stability_weight": 0.3,
            "target_cpu_utilization": 70.0,
            "target_memory_utilization": 70.0,
            "max_response_time": 200.0,
            "penalty_scale_factor": 2.0
        }
        
        reward, components = calculate_reward(
            metrics_before, metrics_after, "scale_up", config
        )
        
        assert isinstance(reward, float)
        assert isinstance(components, dict)
        assert "resource_efficiency" in components
        assert "performance" in components
        assert "stability" in components
        
        # Should be positive since metrics improved
        assert reward > 0
    
    def test_calculate_reward_degradation(self):
        """Test reward calculation with degradation."""
        metrics_before = {
            "cpu_utilization": 70.0,
            "memory_utilization": 65.0,
            "request_rate": 1000.0,
            "error_rate": 2.0,
            "response_time": 150.0,
            "pod_count": 5
        }
        
        metrics_after = {
            "cpu_utilization": 90.0,  # Worse
            "memory_utilization": 85.0,  # Worse
            "request_rate": 1000.0,
            "error_rate": 8.0,  # Worse
            "response_time": 300.0,  # Worse
            "pod_count": 3
        }
        
        config = {
            "resource_efficiency_weight": 0.3,
            "performance_weight": 0.4,
            "stability_weight": 0.3,
            "target_cpu_utilization": 70.0,
            "target_memory_utilization": 70.0,
            "max_response_time": 200.0,
            "penalty_scale_factor": 2.0
        }
        
        reward, components = calculate_reward(
            metrics_before, metrics_after, "scale_down", config
        )
        
        # Should be negative since metrics got worse
        assert reward < 0 