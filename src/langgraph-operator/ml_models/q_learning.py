"""
Q-Learning Implementation for Kubernetes Scaling Decisions

This module implements a Q-learning algorithm for making intelligent scaling decisions
based on cluster metrics and historical performance data.
"""

import numpy as np
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class QState:
    """Represents a state in the Q-learning environment."""
    cpu_utilization_bin: int
    memory_utilization_bin: int  
    request_rate_bin: int
    pod_count_bin: int
    error_rate_bin: int
    
    def to_tuple(self) -> Tuple[int, int, int, int, int]:
        """Convert state to tuple for use as dictionary key."""
        return (
            self.cpu_utilization_bin,
            self.memory_utilization_bin,
            self.request_rate_bin,
            self.pod_count_bin,
            self.error_rate_bin
        )
    
    def __hash__(self):
        return hash(self.to_tuple())
    
    def __eq__(self, other):
        if not isinstance(other, QState):
            return False
        return self.to_tuple() == other.to_tuple()


@dataclass
class QAction:
    """Represents an action in the Q-learning environment."""
    action_type: str  # 'scale_up', 'scale_down', 'no_action'
    replicas_change: int  # How many replicas to add/remove
    
    def to_tuple(self) -> Tuple[str, int]:
        """Convert action to tuple for use as dictionary key."""
        return (self.action_type, self.replicas_change)
    
    def __hash__(self):
        return hash(self.to_tuple())
    
    def __eq__(self, other):
        if not isinstance(other, QAction):
            return False
        return self.to_tuple() == other.to_tuple()


class QLearningAgent:
    """
    Q-Learning agent for Kubernetes scaling decisions.
    
    The agent learns to make optimal scaling decisions based on:
    - Current cluster metrics (CPU, memory, request rate, error rate)
    - Number of pods currently running
    - Historical performance and rewards
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Q-learning agent.
        
        Args:
            config: Configuration containing hyperparameters
        """
        self.config = config.get("q_learning", {})
        
        # Q-learning hyperparameters
        self.learning_rate = self.config.get("learning_rate", 0.1)
        self.discount_factor = self.config.get("discount_factor", 0.95)
        self.epsilon = self.config.get("epsilon_start", 1.0)
        self.epsilon_end = self.config.get("epsilon_end", 0.01)
        self.epsilon_decay = self.config.get("epsilon_decay", 0.995)
        self.exploration_steps = self.config.get("exploration_steps", 1000)
        
        # State space discretization
        self.cpu_bins = np.linspace(0, 100, 11)  # 0-10%, 10-20%, ..., 90-100%
        self.memory_bins = np.linspace(0, 100, 11)
        self.request_rate_bins = np.logspace(0, 4, 11)  # 1, 10, 100, 1000, 10000 req/s
        self.pod_count_bins = np.array([1, 2, 3, 5, 8, 12, 20, 30, 50, 100])
        self.error_rate_bins = np.array([0, 0.1, 0.5, 1, 2, 5, 10, 20])  # Error percentages
        
        # Action space
        self.actions = [
            QAction("no_action", 0),
            QAction("scale_up", 1),
            QAction("scale_up", 2), 
            QAction("scale_up", 3),
            QAction("scale_up", 5),
            QAction("scale_down", 1),
            QAction("scale_down", 2),
            QAction("scale_down", 3),
        ]
        
        # Q-table: state -> action -> Q-value
        self.q_table: Dict[Tuple, Dict[Tuple, float]] = {}
        
        # Learning statistics
        self.episodes = 0
        self.total_reward = 0.0
        self.last_state: Optional[QState] = None
        self.last_action: Optional[QAction] = None
        
        logger.info(f"Initialized Q-learning agent with {len(self.actions)} actions")

    def discretize_metrics(self, metrics: Dict[str, float]) -> QState:
        """
        Convert continuous metrics to discrete state representation.
        
        Args:
            metrics: Dictionary containing cluster metrics
            
        Returns:
            Discretized state representation
        """
        cpu_bin = np.digitize(metrics.get("cpu_utilization", 0), self.cpu_bins) - 1
        memory_bin = np.digitize(metrics.get("memory_utilization", 0), self.memory_bins) - 1
        request_bin = np.digitize(metrics.get("request_rate", 1), self.request_rate_bins) - 1
        pod_bin = np.digitize(metrics.get("pod_count", 1), self.pod_count_bins) - 1
        error_bin = np.digitize(metrics.get("error_rate", 0), self.error_rate_bins) - 1
        
        # Clamp bins to valid ranges
        cpu_bin = max(0, min(cpu_bin, len(self.cpu_bins) - 2))
        memory_bin = max(0, min(memory_bin, len(self.memory_bins) - 2))
        request_bin = max(0, min(request_bin, len(self.request_rate_bins) - 2))
        pod_bin = max(0, min(pod_bin, len(self.pod_count_bins) - 2))
        error_bin = max(0, min(error_bin, len(self.error_rate_bins) - 2))
        
        return QState(
            cpu_utilization_bin=cpu_bin,
            memory_utilization_bin=memory_bin,
            request_rate_bin=request_bin,
            pod_count_bin=pod_bin,
            error_rate_bin=error_bin
        )

    def get_q_value(self, state: QState, action: QAction) -> float:
        """Get Q-value for state-action pair."""
        state_key = state.to_tuple()
        action_key = action.to_tuple()
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0.0
        
        return self.q_table[state_key][action_key]

    def set_q_value(self, state: QState, action: QAction, value: float):
        """Set Q-value for state-action pair."""
        state_key = state.to_tuple()
        action_key = action.to_tuple()
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        self.q_table[state_key][action_key] = value

    def select_action(self, state: QState, exploration: bool = True) -> QAction:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            exploration: Whether to use exploration (epsilon-greedy)
            
        Returns:
            Selected action
        """
        if exploration and np.random.random() < self.epsilon:
            # Explore: choose random action
            action = np.random.choice(self.actions)
            logger.debug(f"Exploration: selected random action {action.action_type}")
            return action
        else:
            # Exploit: choose best action based on Q-values
            best_action = None
            best_q_value = float('-inf')
            
            for action in self.actions:
                q_value = self.get_q_value(state, action)
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            
            if best_action is None:
                best_action = self.actions[0]  # Fallback to no-action
            
            logger.debug(f"Exploitation: selected best action {best_action.action_type} (Q={best_q_value:.3f})")
            return best_action

    def update_q_value(self, reward: float, next_state: Optional[QState] = None):
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            reward: Reward received from the environment
            next_state: Next state (None if terminal)
        """
        if self.last_state is None or self.last_action is None:
            logger.warning("Cannot update Q-value: no previous state/action")
            return
        
        current_q = self.get_q_value(self.last_state, self.last_action)
        
        if next_state is not None:
            # Find maximum Q-value for next state
            max_next_q = max(
                self.get_q_value(next_state, action) 
                for action in self.actions
            )
            target_q = reward + self.discount_factor * max_next_q
        else:
            # Terminal state
            target_q = reward
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.set_q_value(self.last_state, self.last_action, new_q)
        
        # Update statistics
        self.total_reward += reward
        self.episodes += 1
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        logger.debug(
            f"Q-update: reward={reward:.3f}, old_Q={current_q:.3f}, "
            f"new_Q={new_q:.3f}, epsilon={self.epsilon:.3f}"
        )

    def choose_scaling_action(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Choose scaling action based on current metrics.
        
        Args:
            metrics: Current cluster metrics
            
        Returns:
            Dictionary containing action details and confidence
        """
        # Discretize current metrics
        current_state = self.discretize_metrics(metrics)
        
        # Select action
        action = self.select_action(current_state)
        
        # Calculate confidence based on Q-values
        q_values = [self.get_q_value(current_state, a) for a in self.actions]
        if max(q_values) > min(q_values):
            confidence = (max(q_values) - min(q_values)) / (abs(max(q_values)) + 1e-6)
        else:
            confidence = 0.1  # Low confidence when all Q-values are similar
        
        confidence = min(1.0, max(0.0, confidence))  # Clamp to [0, 1]
        
        # Calculate target replicas
        current_pods = int(metrics.get("pod_count", 1))
        if action.action_type == "scale_up":
            target_replicas = current_pods + action.replicas_change
        elif action.action_type == "scale_down":
            target_replicas = max(1, current_pods - action.replicas_change)
        else:
            target_replicas = current_pods
        
        # Store current state and action for learning
        self.last_state = current_state
        self.last_action = action
        
        return {
            "action_type": action.action_type,
            "target_replicas": target_replicas,
            "replicas_change": action.replicas_change,
            "confidence": confidence,
            "q_values": dict(zip([a.action_type for a in self.actions], q_values)),
            "state_representation": current_state.to_tuple(),
            "exploration_rate": self.epsilon
        }

    def provide_feedback(self, reward: float, new_metrics: Optional[Dict[str, float]] = None):
        """
        Provide feedback to the agent for learning.
        
        Args:
            reward: Reward signal for the last action
            new_metrics: New metrics after action (None if terminal)
        """
        next_state = None
        if new_metrics is not None:
            next_state = self.discretize_metrics(new_metrics)
        
        self.update_q_value(reward, next_state)

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        avg_reward = self.total_reward / max(1, self.episodes)
        
        return {
            "episodes": self.episodes,
            "total_reward": self.total_reward,
            "average_reward": avg_reward,
            "exploration_rate": self.epsilon,
            "q_table_size": len(self.q_table),
            "total_state_action_pairs": sum(len(actions) for actions in self.q_table.values())
        }

    def save_model(self, filepath: str):
        """Save Q-table and agent state to file."""
        state = {
            "q_table": self.q_table,
            "episodes": self.episodes,
            "total_reward": self.total_reward,
            "epsilon": self.epsilon,
            "config": self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved Q-learning model to {filepath}")

    def load_model(self, filepath: str):
        """Load Q-table and agent state from file."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.q_table = state.get("q_table", {})
            self.episodes = state.get("episodes", 0)
            self.total_reward = state.get("total_reward", 0.0)
            self.epsilon = state.get("epsilon", self.epsilon)
            
            logger.info(f"Loaded Q-learning model from {filepath}")
            logger.info(f"Model stats: {self.get_learning_stats()}")
            
        except Exception as e:
            logger.error(f"Failed to load Q-learning model: {e}")
            logger.info("Starting with fresh Q-table")


def calculate_reward(
    metrics_before: Dict[str, float],
    metrics_after: Dict[str, float],
    action_taken: str,
    config: Dict[str, Any]
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate reward for a scaling action.
    
    Args:
        metrics_before: Metrics before the action
        metrics_after: Metrics after the action  
        action_taken: Type of action taken
        config: Reward configuration
        
    Returns:
        Tuple of (total_reward, reward_components)
    """
    reward_config = config.get("rewards", {})
    
    # Reward components with weights
    sla_weight = reward_config.get("sla_compliance_weight", 0.4)
    efficiency_weight = reward_config.get("resource_efficiency_weight", 0.4)
    stability_weight = reward_config.get("stability_weight", 0.2)
    
    # Penalties and rewards
    violation_penalty = reward_config.get("violation_penalty", -5)
    optimal_reward = reward_config.get("optimal_reward", 10)
    waste_penalty = reward_config.get("waste_penalty", -1)
    
    reward_components = {}
    
    # SLA Compliance (CPU and Memory thresholds)
    cpu_before = metrics_before.get("cpu_utilization", 0)
    cpu_after = metrics_after.get("cpu_utilization", 0)
    memory_before = metrics_before.get("memory_utilization", 0)
    memory_after = metrics_after.get("memory_utilization", 0)
    error_rate_after = metrics_after.get("error_rate", 0)
    
    # SLA violations (high resource usage or errors)
    sla_reward = 0
    if cpu_after > 90 or memory_after > 90 or error_rate_after > 5:
        sla_reward = violation_penalty
    elif 60 <= cpu_after <= 80 and 60 <= memory_after <= 80 and error_rate_after < 1:
        sla_reward = optimal_reward
    
    reward_components["sla_compliance"] = sla_reward * sla_weight
    
    # Resource Efficiency
    efficiency_reward = 0
    if action_taken == "scale_down" and cpu_after > 30 and memory_after > 30:
        efficiency_reward = optimal_reward  # Good downscaling
    elif action_taken == "scale_up" and (cpu_before > 85 or memory_before > 85):
        efficiency_reward = optimal_reward  # Good upscaling
    elif action_taken == "no_action" and 40 <= cpu_after <= 75 and 40 <= memory_after <= 75:
        efficiency_reward = optimal_reward * 0.5  # Good stability
    elif cpu_after < 20 or memory_after < 20:
        efficiency_reward = waste_penalty  # Resource waste
    
    reward_components["resource_efficiency"] = efficiency_reward * efficiency_weight
    
    # Stability (minimize frequent changes)
    pods_before = metrics_before.get("pod_count", 1)
    pods_after = metrics_after.get("pod_count", 1)
    pod_change = abs(pods_after - pods_before)
    
    if pod_change == 0:
        stability_reward = optimal_reward * 0.3  # Reward stability
    elif pod_change <= 2:
        stability_reward = 0  # Neutral for small changes
    else:
        stability_reward = waste_penalty * 0.5  # Penalize large changes
    
    reward_components["stability"] = stability_reward * stability_weight
    
    # Calculate total reward
    total_reward = sum(reward_components.values())
    
    return total_reward, reward_components 