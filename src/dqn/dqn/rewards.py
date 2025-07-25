"""
Context-aware reward system for DQN scaling decisions.
"""

import logging
import math
from typing import Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ContextAwareRewardCalculator:
    """
    Calculates rewards based on deployment context, resource utilization,
    and forecasted future states.
    """
    
    def __init__(self):
        # Target utilization levels (aligned with HPA/KEDA thresholds)
        self.target_cpu_util = 0.70  # 70% CPU target (matches HPA)
        self.target_memory_util = 0.80  # 80% Memory target (matches HPA)
        
        # Reward shape parameters
        self.cpu_reward_width = 0.3
        self.memory_reward_width = 0.3
        
        # Weights for different reward components
        self.cpu_weight = 0.4
        self.memory_weight = 0.4
        self.stability_weight = 0.2
        
        # Cost parameters
        self.cost_per_replica = 0.05
        self.underutilization_penalty = 2.0
        self.overutilization_penalty = 3.0
        
        # Forecasting weights
        self.current_weight = 0.3
        self.forecast_weight = 0.7
        
        # Action-specific bonuses/penalties
        self.action_bonuses = {
            'appropriate_scale_up': 0.2,
            'appropriate_scale_down': 0.15,
            'proactive_action': 0.1,
            'stability_bonus': 0.1
        }
        
        self.action_penalties = {
            'unnecessary_scaling': -0.3,
            'thrashing': -0.5,
            'resource_waste': -0.2,
            'performance_risk': -0.4
        }
        
        logger.info("Initialized ContextAwareRewardCalculator")
    
    def _normalize_metrics(self, metrics: Dict[str, float], deployment_context: Dict[str, Any]) -> Dict[str, float]:
        """Normalize metrics based on deployment context."""
        normalized = {}
        
        # Get resource requests from deployment context
        resource_requests = deployment_context.get('resource_requests', {})
        cpu_request = resource_requests.get('cpu', 0.5)  # Default 0.5 cores
        memory_request = resource_requests.get('memory', 512 * 1024 * 1024)  # Default 512MB
        
        replicas = deployment_context.get('current_replicas', 1)
        
        # Calculate total resource capacity
        total_cpu_capacity = cpu_request * replicas
        total_memory_capacity = memory_request * replicas
        
        # Normalize CPU utilization
        cpu_usage = metrics.get('process_cpu_seconds_total_rate', 0)
        normalized['cpu_utilization'] = cpu_usage / max(total_cpu_capacity, 0.001)
        
        # Normalize Memory utilization
        memory_usage = metrics.get('process_resident_memory_bytes', 0)
        normalized['memory_utilization'] = memory_usage / max(total_memory_capacity, 1)
        
        # Add per-pod utilizations
        normalized['cpu_per_pod'] = cpu_usage / max(replicas, 1)
        normalized['memory_per_pod'] = memory_usage / max(replicas, 1)
        
        logger.debug(f"Normalized metrics: CPU={normalized['cpu_utilization']:.3f}, "
                    f"Memory={normalized['memory_utilization']:.3f}")
        
        return normalized
    
    def _calculate_utilization_reward(self, normalized_metrics: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Calculate reward based on resource utilization efficiency."""
        cpu_util = normalized_metrics['cpu_utilization']
        memory_util = normalized_metrics['memory_utilization']
        
        # Gaussian reward curves centered on target utilization
        def gaussian_reward(util: float, target: float, width: float) -> float:
            return math.exp(-((util - target) ** 2) / (2 * width ** 2))
        
        cpu_reward = gaussian_reward(cpu_util, self.target_cpu_util, self.cpu_reward_width)
        memory_reward = gaussian_reward(memory_util, self.target_memory_util, self.memory_reward_width)
        
        # Apply penalties for extreme utilizations
        if cpu_util < 0.1:  # Very low CPU usage
            cpu_reward *= 0.3  # Heavy penalty for waste
        elif cpu_util > 0.9:  # Very high CPU usage
            cpu_reward *= 0.5  # Penalty for performance risk
            
        if memory_util < 0.1:  # Very low memory usage
            memory_reward *= 0.3
        elif memory_util > 0.9:  # Very high memory usage
            memory_reward *= 0.4  # Higher penalty for memory pressure
        
        total_util_reward = (self.cpu_weight * cpu_reward + 
                           self.memory_weight * memory_reward)
        
        components = {
            'cpu_reward': cpu_reward,
            'memory_reward': memory_reward,
            'utilization_reward': total_util_reward
        }
        
        return total_util_reward, components
    
    def _calculate_cost_penalty(self, deployment_context: Dict[str, Any]) -> float:
        """Calculate cost penalty based on resource allocation."""
        replicas = deployment_context.get('current_replicas', 1)
        target_replicas = deployment_context.get('target_replicas', replicas)
        
        # Base cost penalty
        cost_penalty = self.cost_per_replica * target_replicas
        
        # Additional penalty for excessive replicas
        if target_replicas > 10:
            cost_penalty += (target_replicas - 10) * 0.1
        
        return cost_penalty
    
    def _calculate_stability_reward(self, deployment_context: Dict[str, Any]) -> float:
        """Calculate reward for system stability."""
        ready_replicas = deployment_context.get('ready_replicas', 0)
        current_replicas = deployment_context.get('current_replicas', 1)
        
        # Reward for having all replicas ready
        stability_ratio = ready_replicas / max(current_replicas, 1)
        stability_reward = stability_ratio * self.stability_weight
        
        # Penalty for instability
        if stability_ratio < 0.8:
            stability_reward *= 0.5
        
        return stability_reward
    
    def _calculate_action_specific_reward(self, 
                                        action: str,
                                        current_metrics: Dict[str, float],
                                        forecast_metrics: Optional[Dict[str, float]],
                                        deployment_context: Dict[str, Any]) -> float:
        """Calculate action-specific bonuses and penalties."""
        bonus = 0.0
        
        current_norm = self._normalize_metrics(current_metrics, deployment_context)
        current_replicas = deployment_context.get('current_replicas', 1)
        target_replicas = deployment_context.get('target_replicas', current_replicas)
        
        # Get recent scaling history (if available)
        scaling_history = deployment_context.get('scaling_history', [])
        recent_actions = scaling_history[-3:] if scaling_history else []
        
        # Detect thrashing (alternating scale up/down)
        if len(recent_actions) >= 2:
            if (recent_actions[-1] == 'scale_up' and action == 'scale_down') or \
               (recent_actions[-1] == 'scale_down' and action == 'scale_up'):
                bonus += self.action_penalties['thrashing']
                logger.debug("Thrashing detected - applying penalty")
        
        # Action-specific logic
        if action == 'scale_up':
            cpu_util = current_norm['cpu_utilization']
            memory_util = current_norm['memory_utilization']
            
            # Reward appropriate scale up
            if cpu_util > 0.8 or memory_util > 0.85:
                bonus += self.action_bonuses['appropriate_scale_up']
                logger.debug("Appropriate scale up detected")
            
            # Penalty for unnecessary scale up
            elif cpu_util < 0.5 and memory_util < 0.6:
                bonus += self.action_penalties['unnecessary_scaling']
                logger.debug("Unnecessary scale up detected")
            
            # Proactive bonus if forecast shows increased load
            if forecast_metrics:
                forecast_norm = self._normalize_metrics(forecast_metrics, 
                                                      {**deployment_context, 'current_replicas': target_replicas})
                if (forecast_norm['cpu_utilization'] > current_norm['cpu_utilization'] + 0.1 or
                    forecast_norm['memory_utilization'] > current_norm['memory_utilization'] + 0.1):
                    bonus += self.action_bonuses['proactive_action']
                    logger.debug("Proactive scale up bonus")
        
        elif action == 'scale_down':
            cpu_util = current_norm['cpu_utilization']
            memory_util = current_norm['memory_utilization']
            
            # Reward appropriate scale down
            if cpu_util < 0.4 and memory_util < 0.5 and current_replicas > 1:
                bonus += self.action_bonuses['appropriate_scale_down']
                logger.debug("Appropriate scale down detected")
            
            # Penalty for risky scale down
            elif cpu_util > 0.7 or memory_util > 0.8:
                bonus += self.action_penalties['performance_risk']
                logger.debug("Risky scale down detected")
            
            # Proactive bonus if forecast shows decreased load
            if forecast_metrics:
                forecast_norm = self._normalize_metrics(forecast_metrics,
                                                      {**deployment_context, 'current_replicas': target_replicas})
                if (forecast_norm['cpu_utilization'] < current_norm['cpu_utilization'] - 0.1 and
                    forecast_norm['memory_utilization'] < current_norm['memory_utilization'] - 0.1):
                    bonus += self.action_bonuses['proactive_action']
                    logger.debug("Proactive scale down bonus")
        
        elif action == 'keep_same':  # Fixed: changed from 'no_action' to 'keep_same'
            # Reward for maintaining current state when appropriate
            cpu_util = current_norm['cpu_utilization']
            memory_util = current_norm['memory_utilization']
            
            # Stability bonus for no action when metrics are in good range
            if 0.4 <= cpu_util <= 0.8 and 0.4 <= memory_util <= 0.8:
                bonus += self.action_bonuses['stability_bonus']
                logger.debug("Stability bonus for appropriate no-action")
        
        return bonus
    
    def calculate_reward(self,
                        action: str,
                        current_metrics: Dict[str, float],
                        deployment_context: Dict[str, Any],
                        forecast_metrics: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, float]]:
        """
        Calculate comprehensive reward for the given action and state.
        
        Args:
            action: Action taken ('scale_up', 'scale_down', 'no_action')
            current_metrics: Current system metrics
            deployment_context: Deployment context information
            forecast_metrics: Forecasted metrics (optional)
            
        Returns:
            Tuple of (total_reward, reward_components)
        """
        try:
            components = {}
            
            # Calculate current state reward
            current_norm = self._normalize_metrics(current_metrics, deployment_context)
            current_util_reward, util_components = self._calculate_utilization_reward(current_norm)
            components.update(util_components)
            
            # Calculate forecast state reward if available
            forecast_reward = 0.0
            if forecast_metrics:
                # Update deployment context with target replicas for forecast calculation
                forecast_context = deployment_context.copy()
                forecast_context['current_replicas'] = deployment_context.get('target_replicas', 
                                                                            deployment_context.get('current_replicas', 1))
                
                forecast_norm = self._normalize_metrics(forecast_metrics, forecast_context)
                forecast_util_reward, forecast_components = self._calculate_utilization_reward(forecast_norm)
                
                forecast_reward = forecast_util_reward
                components['forecast_utilization_reward'] = forecast_util_reward
            
            # Combine current and forecast rewards
            if forecast_metrics:
                combined_util_reward = (self.current_weight * current_util_reward + 
                                      self.forecast_weight * forecast_reward)
            else:
                combined_util_reward = current_util_reward
            
            components['combined_utilization_reward'] = combined_util_reward
            
            # Calculate cost penalty
            cost_penalty = self._calculate_cost_penalty(deployment_context)
            components['cost_penalty'] = -cost_penalty
            
            # Calculate stability reward
            stability_reward = self._calculate_stability_reward(deployment_context)
            components['stability_reward'] = stability_reward
            
            # Calculate action-specific bonuses/penalties
            action_bonus = self._calculate_action_specific_reward(
                action, current_metrics, forecast_metrics, deployment_context
            )
            components['action_bonus'] = action_bonus
            
            # Calculate total reward
            total_reward = (combined_util_reward + 
                          stability_reward + 
                          action_bonus - 
                          cost_penalty)
            
            components['total_reward'] = total_reward
            
            logger.debug(f"Reward calculation: action={action}, "
                        f"util={combined_util_reward:.3f}, "
                        f"stability={stability_reward:.3f}, "
                        f"action_bonus={action_bonus:.3f}, "
                        f"cost={cost_penalty:.3f}, "
                        f"total={total_reward:.3f}")
            
            return total_reward, components
            
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0, {'error': str(e)}
    
    def get_deployment_context(self, 
                             current_replicas: int,
                             ready_replicas: int,
                             target_replicas: int,
                             resource_requests: Optional[Dict[str, float]] = None,
                             scaling_history: Optional[list] = None) -> Dict[str, Any]:
        """
        Create deployment context for reward calculation.
        
        Args:
            current_replicas: Current number of replicas
            ready_replicas: Number of ready replicas
            target_replicas: Target number of replicas after action
            resource_requests: Resource requests per pod
            scaling_history: Recent scaling actions
            
        Returns:
            Deployment context dictionary
        """
        context = {
            'current_replicas': current_replicas,
            'ready_replicas': ready_replicas,
            'target_replicas': target_replicas,
            'resource_requests': resource_requests or {
                'cpu': 0.5,  # 0.5 CPU cores
                'memory': 512 * 1024 * 1024  # 512MB
            },
            'scaling_history': scaling_history or []
        }
        
        return context 