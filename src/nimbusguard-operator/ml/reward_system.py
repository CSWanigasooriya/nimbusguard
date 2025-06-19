# engine/ml/reward_system.py
# ============================================================================
# Reward System for DQN Autoscaling
# ============================================================================

import numpy as np
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .state_representation import EnvironmentState, ScalingActions

LOG = logging.getLogger(__name__)

@dataclass
class RewardComponents:
    """Individual components of the reward calculation for transparency"""
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

class RewardSystem:
    """
    Comprehensive reward system that evaluates scaling decisions based on 
    multiple objectives: performance, efficiency, stability, and cost.
    """
    
    def __init__(self):
        """Initialize reward system with configurable thresholds"""
        
        # Performance thresholds (based on your observability features)
        self.performance_thresholds = {
            "response_time_p95_ms": 500.0,  # 500ms target
            "response_time_p99_ms": 1000.0,  # 1s target
            "error_rate": 0.01,  # 1% error rate
            "cpu_usage": 0.8,   # 80% CPU usage
            "memory_usage": 0.85,  # 85% memory usage
        }
        
        # SLA thresholds
        self.sla_thresholds = {
            "availability": 0.999,  # 99.9% availability
            "response_time_sla": 1000.0,  # 1s SLA
            "error_rate_sla": 0.05,  # 5% error rate SLA
        }
        
        # Reward weights (sum to 1.0)
        self.reward_weights = {
            "performance": 0.3,
            "efficiency": 0.2,
            "stability": 0.2,
            "sla": 0.2,
            "cost": 0.05,
            "proactivity": 0.05
        }
    
    def calculate_reward(self, 
                        previous_state: EnvironmentState,
                        action: ScalingActions,
                        current_state: EnvironmentState,
                        execution_success: bool = True) -> RewardComponents:
        """
        Calculate comprehensive reward for a scaling decision.
        
        Args:
            previous_state: State before scaling action
            action: Action that was taken
            current_state: State after scaling action
            execution_success: Whether the scaling action succeeded
            
        Returns:
            RewardComponents with detailed breakdown
        """
        
        if not execution_success:
            # Severe penalty for failed actions
            return RewardComponents(total_reward=-50.0)
        
        # Calculate individual reward components
        performance_reward = self._calculate_performance_reward(previous_state, current_state)
        efficiency_reward = self._calculate_efficiency_reward(action, current_state)
        stability_reward = self._calculate_stability_reward(action, current_state)
        sla_reward = self._calculate_sla_reward(current_state)
        cost_reward = self._calculate_cost_reward(action, current_state)
        proactivity_reward = self._calculate_proactivity_reward(previous_state, action, current_state)
        
        # Weighted total reward
        total_reward = (
            self.reward_weights["performance"] * performance_reward +
            self.reward_weights["efficiency"] * efficiency_reward +
            self.reward_weights["stability"] * stability_reward +
            self.reward_weights["sla"] * sla_reward +
            self.reward_weights["cost"] * cost_reward +
            self.reward_weights["proactivity"] * proactivity_reward
        )
        
        # Clip total reward to reasonable range
        total_reward = np.clip(total_reward, -50.0, 50.0)
        
        return RewardComponents(
            performance_reward=performance_reward,
            efficiency_reward=efficiency_reward,
            stability_reward=stability_reward,
            sla_reward=sla_reward,
            cost_reward=cost_reward,
            proactivity_reward=proactivity_reward,
            total_reward=total_reward
        )
    
    def _calculate_performance_reward(self, 
                                    previous_state: EnvironmentState, 
                                    current_state: EnvironmentState) -> float:
        """
        Reward based on performance improvement/degradation.
        Range: -15 to +15 points
        """
        reward = 0.0
        
        # Extract performance metrics from feature vectors
        # Note: These indices correspond to your ObservabilityCollector feature order
        prev_features = previous_state.feature_vector
        curr_features = current_state.feature_vector
        
        # Response time improvement (indices might need adjustment based on your feature order)
        if len(curr_features) > 10:  # Ensure we have enough features
            # Assuming response time features are in positions 10-13 (application metrics)
            prev_p95 = prev_features[11] if len(prev_features) > 11 else 0.0
            curr_p95 = curr_features[11] if len(curr_features) > 11 else 0.0
            
            if prev_p95 > 0:  # Avoid division by zero
                response_time_improvement = (prev_p95 - curr_p95) / prev_p95
                reward += response_time_improvement * 10.0  # Up to ±10 points
        
        # Error rate improvement
        if len(curr_features) > 14:
            prev_error_rate = prev_features[14] if len(prev_features) > 14 else 0.0
            curr_error_rate = curr_features[14] if len(curr_features) > 14 else 0.0
            
            error_rate_improvement = prev_error_rate - curr_error_rate
            reward += error_rate_improvement * 50.0  # Up to ±5 points
        
        # CPU and memory utilization improvement
        if len(curr_features) > 1:
            prev_cpu = prev_features[0] if len(prev_features) > 0 else 0.0
            curr_cpu = curr_features[0] if len(curr_features) > 0 else 0.0
            
            # Reward for keeping CPU in optimal range (0.6-0.8)
            prev_cpu_score = self._calculate_utilization_score(prev_cpu, 0.6, 0.8)
            curr_cpu_score = self._calculate_utilization_score(curr_cpu, 0.6, 0.8)
            reward += (curr_cpu_score - prev_cpu_score) * 5.0
        
        return np.clip(reward, -15.0, 15.0)
    
    def _calculate_efficiency_reward(self, 
                                   action: ScalingActions, 
                                   current_state: EnvironmentState) -> float:
        """
        Reward based on resource efficiency.
        Range: -8 to +8 points
        """
        reward = 0.0
        
        # Resource utilization efficiency
        if len(current_state.feature_vector) > 1:
            cpu_usage = current_state.feature_vector[0]
            memory_usage = current_state.feature_vector[1]
            
            # Reward for optimal utilization ranges
            cpu_efficiency = self._calculate_utilization_score(cpu_usage, 0.6, 0.8)
            memory_efficiency = self._calculate_utilization_score(memory_usage, 0.6, 0.85)
            
            reward += (cpu_efficiency + memory_efficiency) * 3.0  # Up to 6 points
        
        # Action efficiency (prefer smaller changes when possible)
        action_change = abs(action.get_replica_change())
        if action_change == 0:
            reward += 2.0  # Bonus for maintaining when appropriate
        elif action_change == 1:
            reward += 1.0  # Small bonus for conservative scaling
        else:
            reward -= 1.0  # Small penalty for aggressive scaling
        
        return np.clip(reward, -8.0, 8.0)
    
    def _calculate_stability_reward(self, 
                                  action: ScalingActions, 
                                  current_state: EnvironmentState) -> float:
        """
        Reward based on system stability (avoid oscillations).
        Range: -5 to +5 points
        """
        reward = 0.0
        
        # Analyze recent scaling actions for oscillations
        recent_actions = current_state.recent_scaling_actions
        if len(recent_actions) >= 3:
            # Check for oscillation patterns
            oscillation_penalty = self._detect_oscillation(recent_actions)
            reward -= oscillation_penalty * 3.0  # Up to -3 points
        
        # Time since last scale (reward stability)
        if current_state.time_since_last_scale > 300:  # 5 minutes
            reward += 2.0  # Bonus for stability
        elif current_state.time_since_last_scale < 60:  # 1 minute
            reward -= 1.0  # Penalty for frequent changes
        
        return np.clip(reward, -5.0, 5.0)
    
    def _calculate_sla_reward(self, current_state: EnvironmentState) -> float:
        """
        Reward based on SLA compliance.
        Range: -10 to +10 points
        """
        reward = 0.0
        
        features = current_state.feature_vector
        
        # SLA compliance based on response time
        if len(features) > 11:
            response_time_p95 = features[11] * 10000  # Denormalize (assuming ms)
            if response_time_p95 < self.sla_thresholds["response_time_sla"]:
                reward += 5.0
            else:
                sla_violation = response_time_p95 / self.sla_thresholds["response_time_sla"]
                reward -= min(sla_violation * 5.0, 8.0)
        
        # SLA compliance based on error rate
        if len(features) > 14:
            error_rate = features[14]
            if error_rate < self.sla_thresholds["error_rate_sla"]:
                reward += 5.0
            else:
                reward -= (error_rate / self.sla_thresholds["error_rate_sla"]) * 5.0
        
        return np.clip(reward, -10.0, 10.0)
    
    def _calculate_cost_reward(self, 
                             action: ScalingActions, 
                             current_state: EnvironmentState) -> float:
        """
        Reward based on cost optimization.
        Range: -3 to +3 points
        """
        reward = 0.0
        
        # Cost is roughly proportional to replica count
        replica_efficiency = current_state.current_replicas / max(current_state.max_replicas, 1)
        
        # Reward for running fewer replicas when possible
        if replica_efficiency < 0.5:
            reward += 2.0  # Running efficiently with fewer replicas
        elif replica_efficiency > 0.9:
            reward -= 1.0  # Running close to max (potentially wasteful)
        
        # Additional cost consideration based on action
        if action == ScalingActions.SCALE_DOWN_1 or action == ScalingActions.SCALE_DOWN_2:
            reward += 1.0  # Bonus for cost reduction attempts
        
        return np.clip(reward, -3.0, 3.0)
    
    def _calculate_proactivity_reward(self, 
                                    previous_state: EnvironmentState,
                                    action: ScalingActions, 
                                    current_state: EnvironmentState) -> float:
        """
        Reward for proactive scaling (anticipating problems).
        Range: -2 to +5 points
        """
        reward = 0.0
        
        # Check if action was proactive based on trends
        if len(current_state.feature_vector) > 15:
            # Look for early indicators of stress
            cpu_usage = current_state.feature_vector[0]
            memory_usage = current_state.feature_vector[1]
            
            # Proactive scale-up before reaching critical thresholds
            if action in [ScalingActions.SCALE_UP_1, ScalingActions.SCALE_UP_2]:
                if 0.7 < cpu_usage < 0.85 or 0.75 < memory_usage < 0.9:
                    reward += 3.0  # Good proactive scaling
                elif cpu_usage > 0.9 or memory_usage > 0.95:
                    reward += 1.0  # Reactive but necessary
            
            # Proactive scale-down during low utilization
            elif action in [ScalingActions.SCALE_DOWN_1, ScalingActions.SCALE_DOWN_2]:
                if cpu_usage < 0.3 and memory_usage < 0.4:
                    reward += 2.0  # Good proactive downscaling
        
        # Penalty for unnecessary actions
        if action != ScalingActions.NO_ACTION and current_state.health_score < 0.5:
            reward -= 2.0  # Don't scale when data quality is poor
        
        return np.clip(reward, -2.0, 5.0)
    
    def _calculate_utilization_score(self, usage: float, optimal_min: float, optimal_max: float) -> float:
        """Calculate efficiency score for resource utilization"""
        if optimal_min <= usage <= optimal_max:
            return 1.0  # Optimal usage
        elif usage < optimal_min:
            return usage / optimal_min  # Underutilization penalty
        else:
            return max(0.0, 2.0 - (usage / optimal_max))  # Overutilization penalty
    
    def _detect_oscillation(self, recent_actions: list) -> float:
        """Detect oscillation patterns in recent actions"""
        if len(recent_actions) < 3:
            return 0.0
        
        # Look for alternating patterns
        direction_changes = 0
        for i in range(1, len(recent_actions)):
            if recent_actions[i] != recent_actions[i-1]:
                direction_changes += 1
        
        # High direction changes indicate oscillation
        oscillation_ratio = direction_changes / (len(recent_actions) - 1)
        return oscillation_ratio  # 0.0 = no oscillation, 1.0 = maximum oscillation
    
    def get_reward_explanation(self, reward_components: RewardComponents) -> str:
        """Generate human-readable explanation of reward calculation"""
        explanation = f"Total Reward: {reward_components.total_reward:.2f}\n"
        explanation += f"  Performance: {reward_components.performance_reward:.2f}\n"
        explanation += f"  Efficiency: {reward_components.efficiency_reward:.2f}\n"
        explanation += f"  Stability: {reward_components.stability_reward:.2f}\n"
        explanation += f"  SLA Compliance: {reward_components.sla_reward:.2f}\n"
        explanation += f"  Cost Optimization: {reward_components.cost_reward:.2f}\n"
        explanation += f"  Proactivity: {reward_components.proactivity_reward:.2f}"
        return explanation
