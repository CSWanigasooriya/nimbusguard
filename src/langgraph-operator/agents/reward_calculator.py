"""
Reward Calculator Agent for LangGraph Scaling Workflow

This agent calculates rewards for the Q-learning system based on scaling action outcomes,
resource efficiency, SLA compliance, and system stability metrics.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command

logger = logging.getLogger(__name__)


class RewardCalculatorAgent:
    """
    Reward Calculator Agent that evaluates scaling action outcomes and provides learning feedback.
    
    This agent:
    - Calculates rewards based on resource efficiency, SLA compliance, and stability
    - Tracks performance metrics before and after scaling actions
    - Provides detailed feedback for Q-learning model updates
    - Identifies successful vs. unsuccessful scaling patterns
    - Logs reward signals for performance analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the reward calculator agent.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize LLM for reward reasoning
        self.llm = ChatOpenAI(
            model=self.config.get("model", "gpt-4o-mini"),
            temperature=self.config.get("temperature", 0.1),
            max_tokens=self.config.get("max_tokens", 800),
            timeout=self.config.get("timeout", 30)
        )
        
        # Reward calculation parameters
        self.reward_weights = self.config.get("reward_weights", {
            "resource_efficiency": 0.4,
            "sla_compliance": 0.4,
            "stability": 0.2
        })
        
        # Performance thresholds
        self.optimal_cpu_range = self.config.get("optimal_cpu_range", [50, 80])
        self.optimal_memory_range = self.config.get("optimal_memory_range", [50, 80])
        self.max_error_rate = self.config.get("max_error_rate", 1.0)  # 1%
        
        logger.info("Reward Calculator Agent initialized")

    async def invoke(self, state):
        """
        Calculate reward for the scaling action taken.
        
        Args:
            state: Current workflow state with action results
            
        Returns:
            Command object with reward calculation and updated state
        """
        try:
            logger.info(f"Reward Calculator starting analysis for workflow {state['workflow_id']}")
            
            # Calculate a reward based on action success and metrics
            action_result = state.get("last_action")
            current_metrics = state.get("current_metrics")
            
            if not action_result:
                total_reward = 0.0
                reasoning = "No action result available for reward calculation"
            else:
                # Calculate reward components
                resource_efficiency = self._calculate_resource_efficiency(current_metrics)
                sla_compliance = self._calculate_sla_compliance(current_metrics)
                stability = self._calculate_stability(action_result)
                
                # Calculate weighted total reward
                total_reward = (
                    resource_efficiency * self.reward_weights["resource_efficiency"] +
                    sla_compliance * self.reward_weights["sla_compliance"] +
                    stability * self.reward_weights["stability"]
                )
                
                reasoning = f"Resource efficiency: {resource_efficiency:.2f}, SLA compliance: {sla_compliance:.2f}, Stability: {stability:.2f}"
            
            # Update state with reward
            reward_result = {
                "total_reward": total_reward,
                "resource_efficiency_reward": resource_efficiency if action_result else 0.0,
                "sla_compliance_reward": sla_compliance if action_result else 0.0,
                "stability_reward": stability if action_result else 0.0,
                "timestamp": datetime.now(),
                "reasoning": reasoning
            }
            
            state["reward_result"] = reward_result
            state["completed"] = True
            state["status"] = "COMPLETED"
            
            logger.info(f"Reward Calculator completed: reward={total_reward:.3f}")
            
            return Command(
                update={
                    "status": "COMPLETED",
                    "completed": True
                },
                goto="__end__"
            )
            
        except Exception as e:
            logger.error(f"Reward Calculator error: {e}", exc_info=True)
            return Command(
                update={
                    "status": "FAILED",
                    "completed": True,
                    "errors": state.get("errors", []) + [f"Reward calculation error: {str(e)}"]
                },
                goto="__end__"
            )

    def _calculate_resource_efficiency(self, metrics):
        """Calculate reward based on resource utilization efficiency."""
        if not metrics:
            return 0.0
        
        cpu_utilization = metrics.get("cpu_utilization", 0)
        memory_utilization = metrics.get("memory_utilization", 0)
        
        # Calculate efficiency scores
        cpu_efficiency = self._calculate_utilization_efficiency(cpu_utilization, self.optimal_cpu_range)
        memory_efficiency = self._calculate_utilization_efficiency(memory_utilization, self.optimal_memory_range)
        
        # Average efficiency scaled to reward range
        avg_efficiency = (cpu_efficiency + memory_efficiency) / 2
        return (avg_efficiency - 0.5) * 10  # Scale to -5 to +5 range

    def _calculate_sla_compliance(self, metrics):
        """Calculate reward based on SLA compliance."""
        if not metrics:
            return 0.0
        
        error_rate = metrics.get("error_rate", 0)
        
        if error_rate <= self.max_error_rate:
            return 5.0  # Good SLA compliance
        elif error_rate <= self.max_error_rate * 2:
            return 0.0  # Acceptable
        else:
            return -5.0  # Poor SLA compliance

    def _calculate_stability(self, action_result):
        """Calculate reward based on system stability."""
        if not action_result:
            return 0.0
        
        # Basic stability assessment based on action success
        if action_result.get("status") == "SUCCESS":
            return 3.0
        else:
            return -2.0

    def _calculate_utilization_efficiency(self, utilization: float, optimal_range: List[float]) -> float:
        """Calculate efficiency score for resource utilization."""
        if optimal_range[0] <= utilization <= optimal_range[1]:
            # Within optimal range - calculate how close to center
            center = (optimal_range[0] + optimal_range[1]) / 2
            distance_from_center = abs(utilization - center)
            max_distance = (optimal_range[1] - optimal_range[0]) / 2
            efficiency = 1.0 - (distance_from_center / max_distance)
            return efficiency
        elif utilization < optimal_range[0]:
            # Under-utilized
            if utilization <= 0:
                return 0.0
            return 0.3 + (utilization / optimal_range[0]) * 0.3
        else:  # utilization > optimal_range[1]
            # Over-utilized
            if utilization >= 100:
                return 0.0
            remaining = 100 - optimal_range[1]
            excess = utilization - optimal_range[1]
            return 0.3 - (excess / remaining) * 0.3


async def reward_calculator_node(state):
    """
    LangGraph node function for reward calculation.
    
    Args:
        state: Current workflow state
        
    Returns:
        Command with reward calculation results
    """
    agent = RewardCalculatorAgent()
    return await agent.invoke(state)
