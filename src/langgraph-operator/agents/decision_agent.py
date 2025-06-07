"""
Decision Agent for LangGraph Scaling Workflow

This agent uses Q-learning and ML models to make intelligent scaling decisions
based on cluster metrics and historical data. It analyzes the current state
and determines the optimal scaling action.
"""

import logging
import asyncio
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from config import get_agent_config, get_system_prompt
from workflows.scaling_state import (
    ScalingWorkflowState, 
    WorkflowStatus, 
    ScalingDecision,
    ScalingAction,
    MetricData,
    validate_state
)
from ml_models.q_learning import QLearningAgent, QState, QAction
from mcp_integration.mcp_tools import COMMON_TOOLS, get_cluster_metrics

logger = logging.getLogger(__name__)


class DecisionAgent:
    """
    Decision Agent that makes intelligent scaling decisions using Q-learning.
    
    This agent:
    - Analyzes current and historical metrics
    - Uses Q-learning for decision making
    - Considers confidence thresholds and constraints
    - Provides reasoning for decisions
    - Updates ML models based on feedback
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the decision agent.
        
        Args:
            config: Optional configuration dictionary. If None, loads from config system.
        """
        # Load configuration from config system
        self.config = get_agent_config("decision_agent") if config is None else config.get("agents", {}).get("decision", {})
        
        # Initialize LLM with configuration
        self.llm = ChatOpenAI(
            model=self.config.get("model", "gpt-4o-mini"),
            temperature=self.config.get("temperature", 0.1),
            max_tokens=self.config.get("max_tokens", 1200),
            timeout=self.config.get("timeout", 30)
        )
        
        # Bind tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(COMMON_TOOLS)
        
        # Initialize Q-learning agent
        q_config = config.get("q_learning", {})
        self.q_agent = QLearningAgent(
            learning_rate=q_config.get("learning_rate", 0.1),
            epsilon=q_config.get("epsilon", 0.1),
            epsilon_decay=q_config.get("epsilon_decay", 0.995),
            epsilon_min=q_config.get("epsilon_min", 0.01),
            discount_factor=q_config.get("discount_factor", 0.95)
        )
        
        # Decision parameters
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.min_observations = self.config.get("min_observations", 2)
        self.max_scale_step = self.config.get("max_scale_step", 3)
        self.scale_cooldown = self.config.get("scale_cooldown", 300)  # seconds
        
        # Load system prompt from configuration
        self.system_prompt = get_system_prompt("decision_agent")

    async def invoke(self, state: ScalingWorkflowState) -> Command:
        """
        Main decision logic - analyzes metrics and makes scaling decisions.
        
        Args:
            state: Current workflow state with metrics and history
            
        Returns:
            Command object with scaling decision and updated state
        """
        try:
            logger.info(f"Decision Agent starting analysis for workflow {state['workflow_id']}")
            
            # Validate we have sufficient data for decision making
            if not self._validate_decision_inputs(state):
                return self._request_more_data(state)
            
            # Extract current metrics and history
            current_metrics = state.get("current_metrics")
            metrics_history = state.get("metrics_history", [])
            
            # Get Q-learning recommendation
            q_recommendation = await self._get_q_learning_decision(current_metrics, metrics_history)
            
            # Get LLM analysis and reasoning
            llm_analysis = await self._get_llm_analysis(current_metrics, metrics_history, q_recommendation, state)
            
            # Make final decision combining Q-learning and LLM reasoning
            final_decision = await self._make_final_decision(
                current_metrics, q_recommendation, llm_analysis, state
            )
            
            # Create updated state with decision
            updated_state = await self._update_state_with_decision(state, final_decision)
            
            # Determine next action
            next_command = await self._determine_next_action(updated_state, final_decision)
            
            logger.info(f"Decision Agent completed analysis: {final_decision.action.value} (confidence: {final_decision.confidence:.2f})")
            return next_command
            
        except Exception as e:
            logger.error(f"Decision Agent error: {e}", exc_info=True)
            return self._handle_error(state, str(e))

    def _validate_decision_inputs(self, state: ScalingWorkflowState) -> bool:
        """Validate we have sufficient data for decision making."""
        try:
            # Check current metrics
            if not state.get("current_metrics"):
                logger.warning("No current metrics available for decision")
                return False
            
            # Check minimum observations
            metrics_history = state.get("metrics_history", [])
            if len(metrics_history) < self.min_observations:
                logger.warning(f"Insufficient metrics history: {len(metrics_history)} < {self.min_observations}")
                return False
            
            # Check if we're in cooldown period
            if self._in_scaling_cooldown(state):
                logger.info("In scaling cooldown period")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating decision inputs: {e}")
            return False

    def _in_scaling_cooldown(self, state: ScalingWorkflowState) -> bool:
        """Check if we're in scaling cooldown period."""
        try:
            last_action = state.get("last_action")
            if not last_action or not last_action.timestamp:
                return False
            
            # Check if last action was recent
            time_since_action = datetime.now() - last_action.timestamp
            return time_since_action.total_seconds() < self.scale_cooldown
            
        except Exception:
            return False

    async def _get_q_learning_decision(
        self, 
        current_metrics: MetricData, 
        metrics_history: List[MetricData]
    ) -> Dict[str, Any]:
        """Get scaling recommendation from Q-learning model."""
        try:
            # Convert current metrics to Q-learning state
            q_state = QState(
                cpu_utilization=current_metrics.cpu_utilization,
                memory_utilization=current_metrics.memory_utilization,
                request_rate=current_metrics.request_rate,
                pod_count=current_metrics.pod_count,
                error_rate=current_metrics.error_rate
            )
            
            # Get action recommendation from Q-learning
            recommended_action = self.q_agent.choose_action(q_state)
            q_value = self.q_agent.get_q_value(q_state, recommended_action)
            
            # Convert Q-action to scaling action
            scaling_action = self._convert_q_action_to_scaling(recommended_action)
            
            # Calculate confidence based on Q-value and exploration
            confidence = self._calculate_q_confidence(q_value, q_state)
            
            return {
                "action": scaling_action,
                "confidence": confidence,
                "q_value": q_value,
                "q_state": q_state,
                "q_action": recommended_action,
                "model_episodes": len(self.q_agent.q_table),
                "exploration_rate": self.q_agent.epsilon
            }
            
        except Exception as e:
            logger.error(f"Error getting Q-learning decision: {e}")
            return {
                "action": ScalingAction.MAINTAIN,
                "confidence": 0.5,
                "error": str(e)
            }

    def _convert_q_action_to_scaling(self, q_action: QAction) -> ScalingAction:
        """Convert Q-learning action to scaling action."""
        action_mapping = {
            "scale_up_1": ScalingAction.SCALE_UP,
            "scale_up_2": ScalingAction.SCALE_UP,
            "scale_up_3": ScalingAction.SCALE_UP,
            "scale_down_1": ScalingAction.SCALE_DOWN,
            "scale_down_2": ScalingAction.SCALE_DOWN,
            "scale_down_3": ScalingAction.SCALE_DOWN,
            "maintain": ScalingAction.MAINTAIN
        }
        return action_mapping.get(q_action.action_type, ScalingAction.MAINTAIN)

    def _calculate_q_confidence(self, q_value: float, q_state: QState) -> float:
        """Calculate confidence in Q-learning recommendation."""
        try:
            # Base confidence on Q-value magnitude and state visits
            visits = self.q_agent.state_visits.get(q_state, 0)
            visit_confidence = min(visits / 10.0, 1.0)  # More visits = higher confidence
            
            # Normalize Q-value to 0-1 range
            q_confidence = max(0, min(1, (q_value + 10) / 20))  # Assume Q-values range -10 to +10
            
            # Combined confidence
            return (visit_confidence * 0.4 + q_confidence * 0.6)
            
        except Exception:
            return 0.5

    async def _get_llm_analysis(
        self,
        current_metrics: MetricData,
        metrics_history: List[MetricData],
        q_recommendation: Dict[str, Any],
        state: ScalingWorkflowState
    ) -> Dict[str, Any]:
        """Get LLM analysis and reasoning for the decision."""
        try:
            # Prepare context for LLM
            context = self._prepare_decision_context(current_metrics, metrics_history, q_recommendation, state)
            
            # Create analysis prompt
            prompt = f"""
            Analyze the following cluster state and provide scaling decision reasoning:

            {context}

            Q-Learning Recommendation: {q_recommendation.get('action', 'MAINTAIN').value} (confidence: {q_recommendation.get('confidence', 0):.2f})

            Please provide:
            1. Analysis of current cluster health and trends
            2. Risk assessment (SLA, stability, cost)
            3. Scaling recommendation with reasoning
            4. Confidence level (0.0 to 1.0)
            5. Key factors influencing the decision

            Consider:
            - Resource efficiency vs SLA compliance
            - Cost implications
            - System stability
            - Trend analysis from historical data

            Format your response as structured analysis with clear recommendations.
            """
            
            # Get LLM response with system and human messages
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            response = await self.llm.ainvoke(messages)
            
            # Parse LLM response
            analysis = {
                "llm_reasoning": response.content,
                "llm_recommendation": self._extract_llm_recommendation(response.content),
                "llm_confidence": self._extract_llm_confidence(response.content),
                "risk_assessment": self._extract_risk_assessment(response.content),
                "key_factors": self._extract_key_factors(response.content)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting LLM analysis: {e}")
            return {
                "error": str(e),
                "llm_recommendation": ScalingAction.MAINTAIN,
                "llm_confidence": 0.5,
                "llm_reasoning": "Error occurred during analysis"
            }

    def _prepare_decision_context(
        self,
        current_metrics: MetricData,
        metrics_history: List[MetricData],
        q_recommendation: Dict[str, Any],
        state: ScalingWorkflowState
    ) -> str:
        """Prepare context string for LLM decision analysis."""
        context_parts = [
            f"Current Metrics:",
            f"  CPU: {current_metrics.cpu_utilization:.1f}%",
            f"  Memory: {current_metrics.memory_utilization:.1f}%",
            f"  Request Rate: {current_metrics.request_rate:.1f} req/s",
            f"  Error Rate: {current_metrics.error_rate:.2f}%",
            f"  Pod Count: {current_metrics.pod_count}",
            f"  Health Score: {current_metrics.health_score:.2f}",
            f"  Timestamp: {current_metrics.timestamp}",
            ""
        ]
        
        # Add trend analysis
        if len(metrics_history) >= 2:
            prev_metrics = metrics_history[-2]
            cpu_trend = current_metrics.cpu_utilization - prev_metrics.cpu_utilization
            memory_trend = current_metrics.memory_utilization - prev_metrics.memory_utilization
            context_parts.extend([
                f"Recent Trends:",
                f"  CPU trend: {cpu_trend:+.1f}% (last observation)",
                f"  Memory trend: {memory_trend:+.1f}% (last observation)",
                ""
            ])
        
        # Add historical summary
        if len(metrics_history) >= 3:
            avg_cpu = sum(m.cpu_utilization for m in metrics_history[-5:]) / min(len(metrics_history), 5)
            avg_memory = sum(m.memory_utilization for m in metrics_history[-5:]) / min(len(metrics_history), 5)
            context_parts.extend([
                f"Historical Average (last 5 observations):",
                f"  CPU: {avg_cpu:.1f}%",
                f"  Memory: {avg_memory:.1f}%",
                ""
            ])
        
        # Add Q-learning context
        context_parts.extend([
            f"Q-Learning Context:",
            f"  Model Episodes: {q_recommendation.get('model_episodes', 0)}",
            f"  Exploration Rate: {q_recommendation.get('exploration_rate', 0):.3f}",
            f"  Q-Value: {q_recommendation.get('q_value', 0):.2f}",
            ""
        ])
        
        # Add recent actions
        if state.get("last_action"):
            last_action = state["last_action"]
            time_since = datetime.now() - last_action.timestamp
            context_parts.extend([
                f"Recent Actions:",
                f"  Last action: {last_action.action_taken.value}",
                f"  Time since: {time_since.total_seconds():.0f} seconds ago",
                f"  Success: {last_action.success}",
                ""
            ])
        
        return "\n".join(context_parts)

    def _extract_llm_recommendation(self, llm_response: str) -> ScalingAction:
        """Extract scaling recommendation from LLM response."""
        response_lower = llm_response.lower()
        if "scale up" in response_lower or "scale_up" in response_lower:
            return ScalingAction.SCALE_UP
        elif "scale down" in response_lower or "scale_down" in response_lower:
            return ScalingAction.SCALE_DOWN
        else:
            return ScalingAction.MAINTAIN

    def _extract_llm_confidence(self, llm_response: str) -> float:
        """Extract confidence score from LLM response."""
        import re
        # Look for confidence patterns like "confidence: 0.8" or "85% confident"
        confidence_patterns = [
            r"confidence[:\s]+([0-9]*\.?[0-9]+)",
            r"([0-9]+)%\s*confident",
            r"confidence\s*level[:\s]+([0-9]*\.?[0-9]+)"
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, llm_response.lower())
            if match:
                value = float(match.group(1))
                return value if value <= 1.0 else value / 100.0
        
        return 0.7  # Default confidence

    def _extract_risk_assessment(self, llm_response: str) -> str:
        """Extract risk assessment from LLM response."""
        response_lower = llm_response.lower()
        if "high risk" in response_lower or "critical" in response_lower:
            return "HIGH"
        elif "medium risk" in response_lower or "moderate" in response_lower:
            return "MEDIUM"
        elif "low risk" in response_lower:
            return "LOW"
        else:
            return "MEDIUM"

    def _extract_key_factors(self, llm_response: str) -> List[str]:
        """Extract key factors from LLM response."""
        # Simple extraction - look for bullet points or numbered lists
        factors = []
        lines = llm_response.split('\n')
        for line in lines:
            line = line.strip()
            if (line.startswith('- ') or line.startswith('• ') or 
                line.startswith('* ') or re.match(r'^\d+\.', line)):
                factors.append(line)
        
        return factors[:5]  # Limit to top 5 factors

    async def _make_final_decision(
        self,
        current_metrics: MetricData,
        q_recommendation: Dict[str, Any],
        llm_analysis: Dict[str, Any],
        state: ScalingWorkflowState
    ) -> ScalingDecision:
        """Make final scaling decision combining Q-learning and LLM reasoning."""
        try:
            # Get recommendations
            q_action = q_recommendation.get("action", ScalingAction.MAINTAIN)
            q_confidence = q_recommendation.get("confidence", 0.5)
            llm_action = llm_analysis.get("llm_recommendation", ScalingAction.MAINTAIN)
            llm_confidence = llm_analysis.get("llm_confidence", 0.5)
            
            # Decision fusion logic
            if q_action == llm_action:
                # Both agree - high confidence
                final_action = q_action
                final_confidence = min(0.95, (q_confidence + llm_confidence) / 2 + 0.2)
            elif abs(q_confidence - llm_confidence) > 0.3:
                # Significant confidence difference - go with higher confidence
                if q_confidence > llm_confidence:
                    final_action = q_action
                    final_confidence = q_confidence * 0.8  # Reduce due to disagreement
                else:
                    final_action = llm_action
                    final_confidence = llm_confidence * 0.8
            else:
                # Similar confidence but different actions - be conservative
                final_action = ScalingAction.MAINTAIN
                final_confidence = 0.5
            
            # Apply confidence threshold
            if final_confidence < self.confidence_threshold:
                final_action = ScalingAction.MAINTAIN
                final_confidence = 0.5
            
            # Calculate target replicas
            current_pods = current_metrics.pod_count
            target_replicas = self._calculate_target_replicas(final_action, current_pods)
            
            # Create decision object
            decision = ScalingDecision(
                timestamp=datetime.now(),
                action=final_action,
                target_replicas=target_replicas,
                current_replicas=current_pods,
                confidence=final_confidence,
                reasoning=self._create_decision_reasoning(q_recommendation, llm_analysis, final_action),
                metrics_snapshot=current_metrics,
                q_learning_recommendation=q_action,
                llm_recommendation=llm_action
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Error making final decision: {e}")
            return ScalingDecision(
                timestamp=datetime.now(),
                action=ScalingAction.MAINTAIN,
                target_replicas=current_metrics.pod_count,
                current_replicas=current_metrics.pod_count,
                confidence=0.5,
                reasoning=f"Error in decision making: {str(e)}",
                metrics_snapshot=current_metrics
            )

    def _calculate_target_replicas(self, action: ScalingAction, current_pods: int) -> int:
        """Calculate target replica count based on scaling action."""
        if action == ScalingAction.SCALE_UP:
            # Scale up by 1-3 pods based on urgency
            scale_factor = min(self.max_scale_step, max(1, current_pods // 3))
            return current_pods + scale_factor
        elif action == ScalingAction.SCALE_DOWN:
            # Scale down by 1-2 pods, but never below 1
            scale_factor = min(2, max(1, current_pods // 4))
            return max(1, current_pods - scale_factor)
        else:
            return current_pods

    def _create_decision_reasoning(
        self,
        q_recommendation: Dict[str, Any],
        llm_analysis: Dict[str, Any],
        final_action: ScalingAction
    ) -> str:
        """Create comprehensive reasoning for the decision."""
        reasoning_parts = [
            f"Final Decision: {final_action.value}",
            "",
            f"Q-Learning Recommendation: {q_recommendation.get('action', 'MAINTAIN').value} "
            f"(confidence: {q_recommendation.get('confidence', 0):.2f})",
            f"LLM Recommendation: {llm_analysis.get('llm_recommendation', 'MAINTAIN').value} "
            f"(confidence: {llm_analysis.get('llm_confidence', 0):.2f})",
            "",
            "Key Factors:"
        ]
        
        # Add key factors from LLM
        key_factors = llm_analysis.get("key_factors", [])
        for factor in key_factors[:3]:  # Top 3 factors
            reasoning_parts.append(f"  • {factor}")
        
        # Add Q-learning context
        reasoning_parts.extend([
            "",
            f"Model Context: {q_recommendation.get('model_episodes', 0)} episodes trained, "
            f"exploration rate: {q_recommendation.get('exploration_rate', 0):.1%}"
        ])
        
        return "\n".join(reasoning_parts)

    async def _update_state_with_decision(
        self, 
        state: ScalingWorkflowState, 
        decision: ScalingDecision
    ) -> ScalingWorkflowState:
        """Update workflow state with the scaling decision."""
        try:
            updated_state = state.copy()
            updated_state["scaling_decision"] = decision
            updated_state["current_agent"] = "decision_agent"
            updated_state["status"] = WorkflowStatus.DECIDING
            updated_state["last_decision_time"] = datetime.now()
            
            # Add to decisions history
            if "decisions_history" not in updated_state:
                updated_state["decisions_history"] = []
            updated_state["decisions_history"].append(decision)
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Error updating state with decision: {e}")
            return state

    async def _determine_next_action(
        self, 
        state: ScalingWorkflowState, 
        decision: ScalingDecision
    ) -> Command:
        """Determine next action based on the scaling decision."""
        try:
            if decision.action == ScalingAction.MAINTAIN:
                # No scaling needed - go to monitoring
                next_agent = "__end__"
                new_status = WorkflowStatus.COMPLETED
            else:
                # Scaling action needed - go to action executor
                next_agent = "action_executor"
                new_status = WorkflowStatus.EXECUTING
            
            updated_state = state.copy()
            updated_state["status"] = new_status
            updated_state["current_agent"] = next_agent
            
            return Command(
                goto=next_agent,
                update=updated_state
            )
            
        except Exception as e:
            logger.error(f"Error determining next action: {e}")
            return self._handle_error(state, str(e))

    def _request_more_data(self, state: ScalingWorkflowState) -> Command:
        """Request more data from state observer."""
        updated_state = state.copy()
        updated_state["current_agent"] = "state_observer"
        updated_state["status"] = WorkflowStatus.OBSERVING
        
        return Command(
            goto="state_observer",
            update=updated_state
        )

    def _handle_error(self, state: ScalingWorkflowState, error_message: str) -> Command:
        """Handle errors in decision making."""
        logger.error(f"Decision Agent error: {error_message}")
        
        updated_state = state.copy()
        updated_state["status"] = WorkflowStatus.ERROR
        updated_state["current_agent"] = "supervisor"
        
        if "errors" not in updated_state:
            updated_state["errors"] = []
        updated_state["errors"].append(f"DecisionAgent: {error_message}")
        
        return Command(
            goto="supervisor",
            update=updated_state
        )


async def decision_node(state: ScalingWorkflowState) -> Command:
    """
    LangGraph node function for Decision Agent.
    
    Args:
        state: Current workflow state
        
    Returns:
        Command with scaling decision and routing
    """
    # Initialize agent using configuration system
    agent = DecisionAgent()
    return await agent.invoke(state) 