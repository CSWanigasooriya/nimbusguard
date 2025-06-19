# engine/ml/reward_system.py - Optimized and Encapsulated
# ============================================================================
# Reward System for DQN Autoscaling
# ============================================================================

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .state_representation import EnvironmentState, ScalingActions

LOG = logging.getLogger(__name__)


@dataclass
class RewardComponents:
    """Individual components of the reward calculation for transparency."""
    performance_reward: float = 0.0
    efficiency_reward: float = 0.0
    stability_reward: float = 0.0
    sla_reward: float = 0.0
    cost_reward: float = 0.0
    proactivity_reward: float = 0.0
    total_reward: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "performance_reward": self.performance_reward,
            "efficiency_reward": self.efficiency_reward,
            "stability_reward": self.stability_reward,
            "sla_reward": self.sla_reward,
            "cost_reward": self.cost_reward,
            "proactivity_reward": self.proactivity_reward,
            "total_reward": self.total_reward
        }


def _detect_oscillation(recent_actions: list) -> float:
    """Detects oscillation patterns. Returns value from 0 (stable) to 1 (oscillating)."""
    if len(recent_actions) < 3: return 0.0

    # Count changes in scaling direction (e.g., up -> down)
    direction_changes = 0
    for i in range(1, len(recent_actions)):
        if recent_actions[i] != recent_actions[i - 1]:
            direction_changes += 1

    return direction_changes / (len(recent_actions) - 1)


def _utilization_score(usage: float, optimal_min: float, optimal_max: float) -> float:
    """Calculate score where 1.0 is optimal, tapering off outside the range."""
    if optimal_min <= usage <= optimal_max:
        return 1.0
    elif usage < optimal_min:
        return usage / optimal_min
    else:
        return max(0.0, 2.0 - (usage / optimal_max))


def _calculate_cost_reward(action: ScalingActions, state: EnvironmentState) -> float:
    """Reward for optimizing cost by using fewer replicas. Range: -3 to +3."""
    reward = 0.0
    replica_efficiency = state.current_replicas / max(state.max_replicas, 1)

    if replica_efficiency < 0.5:
        reward += 2.0
    elif replica_efficiency > 0.9:
        reward -= 1.0

    if action in (ScalingActions.SCALE_DOWN_1, ScalingActions.SCALE_DOWN_2):
        reward += 1.0

    return np.clip(reward, -3.0, 3.0)


def _calculate_stability_reward(state: EnvironmentState) -> float:
    """Reward for avoiding rapid, oscillating scaling actions. Range: -5 to +5."""
    reward = 0.0

    # Penalize for oscillating (up/down/up/down)
    oscillation_penalty = _detect_oscillation(state.recent_scaling_actions)
    reward -= oscillation_penalty * 3.0

    # Reward for long periods of stability
    if state.time_since_last_scale > 300:
        reward += 2.0  # 5 mins
    elif state.time_since_last_scale < 60:
        reward -= 1.0  # 1 min

    return np.clip(reward, -5.0, 5.0)


def _get_feature(state: EnvironmentState, feature_name: str, default: float = 0.0) -> float:
    """Safely get a normalized feature value by its name."""
    try:
        idx = state.feature_names.index(feature_name)
        return state.feature_vector[idx]
    except (ValueError, IndexError):
        LOG.debug(f"Feature '{feature_name}' not found in state, using default.")
        return default


def _calculate_performance_reward(prev: EnvironmentState, curr: EnvironmentState) -> float:
    """Reward based on performance improvement or degradation. Range: -15 to +15."""
    reward = 0.0

    # Reward for improving response time
    prev_p95 = _get_feature(prev, 'response_time_p95')
    curr_p95 = _get_feature(curr, 'response_time_p95')
    if prev_p95 > 0:
        reward += ((prev_p95 - curr_p95) / prev_p95) * 10.0

    # Reward for reducing error rate
    prev_err = _get_feature(prev, 'error_rate')
    curr_err = _get_feature(curr, 'error_rate')
    reward += (prev_err - curr_err) * 50.0

    # Reward for moving CPU utilization towards the optimal range (60-80%)
    prev_cpu = _get_feature(prev, 'cpu_usage')
    curr_cpu = _get_feature(curr, 'cpu_usage')
    prev_cpu_score = _utilization_score(prev_cpu, 0.6, 0.8)
    curr_cpu_score = _utilization_score(curr_cpu, 0.6, 0.8)
    reward += (curr_cpu_score - prev_cpu_score) * 5.0

    return np.clip(reward, -15.0, 15.0)


def _calculate_efficiency_reward(action: ScalingActions, state: EnvironmentState) -> float:
    """Reward for efficient resource usage. Range: -8 to +8."""
    reward = 0.0
    cpu = _get_feature(state, 'cpu_usage')
    mem = _get_feature(state, 'memory_usage')

    # Reward for being in the optimal utilization range
    reward += (_utilization_score(cpu, 0.6, 0.8) + _utilization_score(mem, 0.6, 0.85)) * 3.0

    # Reward/penalize based on action magnitude
    action_change = abs(action.get_replica_change())
    if action_change == 0:
        reward += 2.0
    elif action_change == 1:
        reward += 1.0
    else:
        reward -= 1.0

    return np.clip(reward, -8.0, 8.0)


def _calculate_proactivity_reward(action: ScalingActions, state: EnvironmentState) -> float:
    """Reward for scaling before problems become critical. Range: -2 to +5."""
    reward = 0.0
    cpu = _get_feature(state, 'cpu_usage')
    mem = _get_feature(state, 'memory_usage')

    # Proactive scale-up before hitting critical thresholds
    if action in (ScalingActions.SCALE_UP_1, ScalingActions.SCALE_UP_2):
        if 0.7 < cpu < 0.85 or 0.75 < mem < 0.9:
            reward += 3.0
        elif cpu > 0.9 or mem > 0.95:
            reward += 1.0  # Reactive but necessary

    # Proactive scale-down during low utilization
    elif action in (ScalingActions.SCALE_DOWN_1, ScalingActions.SCALE_DOWN_2):
        if cpu < 0.3 and mem < 0.4: reward += 2.0

    return np.clip(reward, -2.0, 5.0)


def get_reward_explanation(components: RewardComponents) -> str:
    """Generate a human-readable explanation of the reward calculation."""
    return (
        f"Total Reward: {components.total_reward:.2f}\n"
        f"  - Performance: {components.performance_reward:.2f}\n"
        f"  - Efficiency: {components.efficiency_reward:.2f}\n"
        f"  - Stability: {components.stability_reward:.2f}\n"
        f"  - SLA Compliance: {components.sla_reward:.2f}\n"
        f"  - Cost Optimization: {components.cost_reward:.2f}\n"
        f"  - Proactivity: {components.proactivity_reward:.2f}"
    )


class RewardSystem:
    """
    An encapsulated and configurable reward system that evaluates scaling decisions
    based on multiple objectives: performance, efficiency, stability, and cost.
    """

    def __init__(self):
        """Initialize the reward system with configurable thresholds and weights."""
        # Reward component weights (should sum to 1.0)
        self.reward_weights = {
            "performance": 0.3,
            "efficiency": 0.2,
            "stability": 0.2,
            "sla": 0.2,
            "cost": 0.05,
            "proactivity": 0.05
        }

        # SLA thresholds for denormalized values
        self.sla_thresholds = {
            "response_time_sla_ms": 1000.0,  # 1s SLA
            "error_rate_sla_percent": 5.0,  # 5% error rate SLA
        }

        # --- NEW: Config-driven feature scaling for denormalization ---
        # These should match the 'scale' values in your observability collector
        self.feature_scales = {
            "cpu_usage": 100.0,
            "memory_usage": 100.0,
            "request_rate": 1000.0,
            "response_time_p95": 1.0,  # Represents 1s or 1000ms
            "error_rate": 0.05,  # Represents 5%
            "pod_restart_rate": 1.0,
            "queue_size": 100.0,
        }

    # --- PRIMARY PUBLIC METHOD ---
    def calculate_reward(self,
                         previous_state: EnvironmentState,
                         action: ScalingActions,
                         current_state: EnvironmentState,
                         execution_success: bool = True) -> RewardComponents:
        """
        Calculate a comprehensive reward for a scaling decision.
        This is the main entry point for the class.
        """
        if not execution_success:
            return RewardComponents(total_reward=-50.0)  #

        # Calculate individual reward components using internal methods
        performance = _calculate_performance_reward(previous_state, current_state)  #
        efficiency = _calculate_efficiency_reward(action, current_state)  #
        stability = _calculate_stability_reward(current_state)  #
        sla = self._calculate_sla_reward(current_state)  #
        cost = _calculate_cost_reward(action, current_state)  #
        proactivity = _calculate_proactivity_reward(action, current_state)  #

        # Calculate weighted total reward
        total_reward = (
                self.reward_weights["performance"] * performance +
                self.reward_weights["efficiency"] * efficiency +
                self.reward_weights["stability"] * stability +
                self.reward_weights["sla"] * sla +
                self.reward_weights["cost"] * cost +
                self.reward_weights["proactivity"] * proactivity
        )  #

        components = RewardComponents(
            performance_reward=performance,
            efficiency_reward=efficiency,
            stability_reward=stability,
            sla_reward=sla,
            cost_reward=cost,
            proactivity_reward=proactivity,
            total_reward=np.clip(total_reward, -50.0, 50.0)
        )

        # Log the reward explanation for better visibility and debugging
        explanation = get_reward_explanation(components)
        LOG.info(f"Reward calculation for action {action.name}:\n{explanation}")

        return components

    def _calculate_sla_reward(self, state: EnvironmentState) -> float:
        """Reward for meeting Service Level Agreements. Range: -10 to +10."""
        reward = 0.0

        # Denormalize p95 response time to milliseconds for comparison
        p95_norm = _get_feature(state, 'response_time_p95')
        p95_ms = p95_norm * self.feature_scales['response_time_p95'] * 1000

        # Denormalize error rate to percent for comparison
        err_norm = _get_feature(state, 'error_rate')
        err_percent = err_norm * self.feature_scales['error_rate'] * 100

        reward += 5.0 if p95_ms < self.sla_thresholds["response_time_sla_ms"] else -5.0
        reward += 5.0 if err_percent < self.sla_thresholds["error_rate_sla_percent"] else -5.0

        return np.clip(reward, -10.0, 10.0)

