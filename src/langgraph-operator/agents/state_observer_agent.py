"""
State Observer Agent for LangGraph Scaling Workflow

This agent monitors cluster metrics and system state using MCP integration
with Prometheus and Kubernetes. It gathers comprehensive metrics and updates
the workflow state with current observations.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from ..workflows.scaling_state import (
    ScalingWorkflowState, 
    WorkflowStatus, 
    MetricData,
    validate_state
)
from ..mcp_integration.mcp_tools import (
    STATE_OBSERVER_TOOLS,
    get_cluster_metrics,
    get_cpu_utilization,
    get_memory_utilization,
    get_request_rate,
    get_error_rate,
    list_kubernetes_pods
)

logger = logging.getLogger(__name__)


class StateObserverAgent:
    """
    State Observer Agent that monitors cluster metrics and system state.
    
    This agent uses MCP integration to:
    - Query Prometheus for system metrics
    - Monitor Kubernetes pod status
    - Analyze trends and anomalies
    - Update workflow state with observations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the state observer agent.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        self.config = config.get("agents", {}).get("state_observer", {})
        self.llm = ChatOpenAI(
            model=self.config.get("model", "gpt-4o-mini"),
            temperature=self.config.get("temperature", 0.0),
            max_tokens=self.config.get("max_tokens", 800),
            timeout=self.config.get("timeout", 30)
        )
        
        # Bind tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(STATE_OBSERVER_TOOLS)
        
        # Observation parameters
        self.observation_interval = self.config.get("observation_interval", 30)  # seconds
        self.metrics_history_size = self.config.get("metrics_history_size", 10)
        self.anomaly_threshold = self.config.get("anomaly_threshold", 2.0)  # standard deviations
        
        # System prompt for analysis
        self.system_prompt = """You are the State Observer Agent in NimbusGuard's AI-powered Kubernetes scaling system.

Your role is to monitor cluster metrics and system state to inform scaling decisions. You have access to tools that can:
- Query Prometheus for CPU, memory, request rate, and error rate metrics
- List Kubernetes pods and their status
- Get comprehensive cluster metrics

Your tasks:
1. Gather current metrics from all available sources
2. Analyze trends and detect anomalies
3. Assess system health and performance
4. Provide insights for scaling decisions
5. Update the workflow state with observations

Always:
- Use tools to get real-time data
- Look for patterns and trends
- Consider both current state and historical context
- Identify potential issues or optimization opportunities
- Provide clear, actionable insights

When you complete your observations, summarize your findings and recommend next steps."""

    async def invoke(self, state: ScalingWorkflowState) -> Command:
        """
        Main state observer logic - gathers metrics and analyzes system state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command object with updated state and routing decision
        """
        try:
            logger.info(f"State Observer starting observation for workflow {state['workflow_id']}")
            
            # Gather all metrics in parallel
            metrics_data = await self._gather_comprehensive_metrics()
            
            # Analyze metrics and detect anomalies
            analysis = await self._analyze_metrics(metrics_data, state)
            
            # Create updated state with new observations
            updated_state = await self._update_state_with_observations(state, metrics_data, analysis)
            
            # Determine next action based on observations
            next_command = await self._determine_next_action(updated_state, analysis)
            
            logger.info(f"State Observer completed observation for workflow {state['workflow_id']}")
            return next_command
            
        except Exception as e:
            logger.error(f"State Observer error: {e}", exc_info=True)
            return self._handle_error(state, str(e))

    async def _gather_comprehensive_metrics(self) -> Dict[str, Any]:
        """
        Gather comprehensive metrics from all available sources.
        
        Returns:
            Dictionary containing all gathered metrics
        """
        try:
            # Execute all metric gathering operations in parallel
            tasks = [
                get_cpu_utilization(),
                get_memory_utilization(),
                get_request_rate(),
                get_error_rate(),
                list_kubernetes_pods("nimbusguard"),
                get_cluster_metrics()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            metrics_data = {
                "cpu_utilization": self._extract_metric_value(results[0]),
                "memory_utilization": self._extract_metric_value(results[1]),
                "request_rate": self._extract_metric_value(results[2]),
                "error_rate": self._extract_metric_value(results[3]),
                "pod_status": results[4] if not isinstance(results[4], Exception) else "Error getting pod status",
                "cluster_metrics": results[5] if not isinstance(results[5], Exception) else "{}",
                "timestamp": datetime.now().isoformat(),
                "collection_success": sum(1 for r in results if not isinstance(r, Exception)),
                "collection_total": len(results)
            }
            
            logger.info(f"Gathered metrics: {metrics_data['collection_success']}/{metrics_data['collection_total']} successful")
            return metrics_data
            
        except Exception as e:
            logger.error(f"Error gathering metrics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "collection_success": 0,
                "collection_total": 0
            }

    def _extract_metric_value(self, tool_result: str) -> float:
        """Extract numeric value from tool result string."""
        try:
            if isinstance(tool_result, Exception):
                return 0.0
            
            # Extract number from strings like "Current CPU utilization: 65.5%"
            import re
            numbers = re.findall(r'(\d+\.?\d*)', tool_result)
            if numbers:
                return float(numbers[0])
            return 0.0
        except:
            return 0.0

    async def _analyze_metrics(self, metrics_data: Dict[str, Any], state: ScalingWorkflowState) -> Dict[str, Any]:
        """
        Analyze metrics using LLM to detect patterns and anomalies.
        
        Args:
            metrics_data: Current metrics data
            state: Current workflow state
            
        Returns:
            Analysis results and insights
        """
        try:
            # Prepare context for analysis
            context = self._prepare_analysis_context(metrics_data, state)
            
            # Create analysis prompt
            prompt = f"""
            Analyze the following cluster metrics and provide insights:

            {context}

            Please analyze:
            1. Current system health and performance
            2. Resource utilization patterns
            3. Any anomalies or concerning trends
            4. Scaling recommendations based on current state
            5. Urgency level (LOW, MEDIUM, HIGH, CRITICAL)

            Provide a concise analysis with specific recommendations.
            """
            
            # Get LLM analysis
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            
            # Parse analysis response
            analysis = {
                "llm_analysis": response.content,
                "health_score": self._calculate_health_score(metrics_data),
                "anomaly_detected": self._detect_anomalies(metrics_data, state),
                "scaling_signal": self._determine_scaling_signal(metrics_data),
                "urgency": self._assess_urgency(metrics_data),
                "timestamp": datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing metrics: {e}")
            return {
                "error": str(e),
                "health_score": 0.5,
                "anomaly_detected": False,
                "scaling_signal": "none",
                "urgency": "MEDIUM"
            }

    def _prepare_analysis_context(self, metrics_data: Dict[str, Any], state: ScalingWorkflowState) -> str:
        """Prepare context string for LLM analysis."""
        context_parts = [
            f"Timestamp: {metrics_data.get('timestamp', 'Unknown')}",
            f"CPU Utilization: {metrics_data.get('cpu_utilization', 0):.1f}%",
            f"Memory Utilization: {metrics_data.get('memory_utilization', 0):.1f}%",
            f"Request Rate: {metrics_data.get('request_rate', 0):.1f} req/s",
            f"Error Rate: {metrics_data.get('error_rate', 0):.2f}%",
            f"Pod Status: {metrics_data.get('pod_status', 'Unknown')}",
            f"Metrics Collection: {metrics_data.get('collection_success', 0)}/{metrics_data.get('collection_total', 0)} successful"
        ]
        
        # Add historical context if available
        if state.get("metrics_history"):
            history = state["metrics_history"]
            if len(history) > 1:
                context_parts.append(f"Previous metrics available: {len(history)} data points")
                latest = history[-1]
                context_parts.append(f"Previous CPU: {latest.cpu_utilization:.1f}%, Memory: {latest.memory_utilization:.1f}%")
        
        return "\n".join(context_parts)

    def _calculate_health_score(self, metrics_data: Dict[str, Any]) -> float:
        """Calculate overall system health score (0.0 to 1.0)."""
        try:
            cpu = metrics_data.get("cpu_utilization", 0)
            memory = metrics_data.get("memory_utilization", 0)
            error_rate = metrics_data.get("error_rate", 0)
            success_rate = metrics_data.get("collection_success", 0) / max(metrics_data.get("collection_total", 1), 1)
            
            # Normalize scores (lower is better for CPU/memory, higher is better for success rate)
            cpu_score = max(0, min(1, (100 - cpu) / 100))
            memory_score = max(0, min(1, (100 - memory) / 100))
            error_score = max(0, min(1, (10 - error_rate) / 10))
            
            # Weighted average
            health_score = (cpu_score * 0.3 + memory_score * 0.3 + error_score * 0.2 + success_rate * 0.2)
            return round(health_score, 3)
            
        except Exception:
            return 0.5

    def _detect_anomalies(self, metrics_data: Dict[str, Any], state: ScalingWorkflowState) -> bool:
        """Detect anomalies in current metrics compared to historical data."""
        try:
            # Simple anomaly detection based on threshold
            cpu = metrics_data.get("cpu_utilization", 0)
            memory = metrics_data.get("memory_utilization", 0)
            error_rate = metrics_data.get("error_rate", 0)
            
            # Threshold-based anomaly detection
            if cpu > 90 or memory > 90 or error_rate > 5:
                return True
            
            # TODO: Implement statistical anomaly detection using historical data
            return False
            
        except Exception:
            return False

    def _determine_scaling_signal(self, metrics_data: Dict[str, Any]) -> str:
        """Determine scaling signal based on metrics."""
        try:
            cpu = metrics_data.get("cpu_utilization", 0)
            memory = metrics_data.get("memory_utilization", 0)
            error_rate = metrics_data.get("error_rate", 0)
            
            # Scale up conditions
            if cpu > 80 or memory > 80 or error_rate > 2:
                return "scale_up"
            
            # Scale down conditions  
            elif cpu < 20 and memory < 20 and error_rate < 0.1:
                return "scale_down"
            
            # No scaling needed
            else:
                return "stable"
                
        except Exception:
            return "unknown"

    def _assess_urgency(self, metrics_data: Dict[str, Any]) -> str:
        """Assess urgency level based on metrics."""
        try:
            cpu = metrics_data.get("cpu_utilization", 0)
            memory = metrics_data.get("memory_utilization", 0)
            error_rate = metrics_data.get("error_rate", 0)
            
            if cpu > 95 or memory > 95 or error_rate > 10:
                return "CRITICAL"
            elif cpu > 85 or memory > 85 or error_rate > 5:
                return "HIGH"
            elif cpu > 70 or memory > 70 or error_rate > 2:
                return "MEDIUM"
            else:
                return "LOW"
                
        except Exception:
            return "MEDIUM"

    async def _update_state_with_observations(
        self, 
        state: ScalingWorkflowState, 
        metrics_data: Dict[str, Any], 
        analysis: Dict[str, Any]
    ) -> ScalingWorkflowState:
        """Update workflow state with new observations."""
        try:
            # Create new metric data point
            new_metric = MetricData(
                timestamp=datetime.now(),
                cpu_utilization=metrics_data.get("cpu_utilization", 0),
                memory_utilization=metrics_data.get("memory_utilization", 0),
                request_rate=metrics_data.get("request_rate", 0),
                error_rate=metrics_data.get("error_rate", 0),
                pod_count=self._extract_pod_count(metrics_data.get("pod_status", "")),
                health_score=analysis.get("health_score", 0.5)
            )
            
            # Update state
            updated_state = state.copy()
            updated_state["current_metrics"] = new_metric
            updated_state["last_observation"] = datetime.now()
            updated_state["current_agent"] = "state_observer"
            
            # Add to metrics history
            if "metrics_history" not in updated_state:
                updated_state["metrics_history"] = []
            updated_state["metrics_history"].append(new_metric)
            
            # Update status if needed
            if state["status"] == WorkflowStatus.INITIALIZING:
                updated_state["status"] = WorkflowStatus.OBSERVING
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Error updating state: {e}")
            return state

    def _extract_pod_count(self, pod_status_string: str) -> int:
        """Extract pod count from pod status string."""
        try:
            import re
            # Look for patterns like "3 pods" or count ✅/❌ markers
            pod_lines = pod_status_string.split('\n')
            return len([line for line in pod_lines if '✅' in line or '❌' in line])
        except:
            return 0

    async def _determine_next_action(self, state: ScalingWorkflowState, analysis: Dict[str, Any]) -> Command:
        """Determine next action based on observations."""
        try:
            urgency = analysis.get("urgency", "MEDIUM")
            scaling_signal = analysis.get("scaling_signal", "stable")
            anomaly_detected = analysis.get("anomaly_detected", False)
            
            # Determine next agent based on analysis
            if anomaly_detected or urgency in ["HIGH", "CRITICAL"]:
                # Critical situations go directly to decision making
                next_agent = "decision_agent"
                new_status = WorkflowStatus.ANALYZING
            elif scaling_signal in ["scale_up", "scale_down"]:
                # Scaling signals trigger decision making
                next_agent = "decision_agent"
                new_status = WorkflowStatus.ANALYZING
            else:
                # Stable system - continue monitoring or end workflow
                observation_count = len(state.get("metrics_history", []))
                if observation_count >= 3:  # Enough observations
                    next_agent = "__end__"
                    new_status = WorkflowStatus.COMPLETED
                else:
                    # Continue observing
                    next_agent = "state_observer"
                    new_status = WorkflowStatus.OBSERVING
            
            # Create command with updated state
            updated_state = state.copy()
            updated_state["status"] = new_status
            updated_state["current_agent"] = next_agent
            updated_state["last_analysis"] = analysis
            
            return Command(
                goto=next_agent,
                update=updated_state
            )
            
        except Exception as e:
            logger.error(f"Error determining next action: {e}")
            return self._handle_error(state, str(e))

    def _handle_error(self, state: ScalingWorkflowState, error_message: str) -> Command:
        """Handle errors in state observation."""
        logger.error(f"State Observer error: {error_message}")
        
        updated_state = state.copy()
        updated_state["status"] = WorkflowStatus.ERROR
        updated_state["current_agent"] = "supervisor"
        
        if "errors" not in updated_state:
            updated_state["errors"] = []
        updated_state["errors"].append(f"StateObserver: {error_message}")
        
        return Command(
            goto="supervisor",
            update=updated_state
        )


async def state_observer_node(state: ScalingWorkflowState) -> Command:
    """
    LangGraph node function for State Observer Agent.
    
    Args:
        state: Current workflow state
        
    Returns:
        Command with updated state and routing decision
    """
    # This would be loaded from config in production
    config = {
        "agents": {
            "state_observer": {
                "model": "gpt-4o-mini",
                "temperature": 0.0,
                "max_tokens": 800,
                "timeout": 30,
                "observation_interval": 30,
                "metrics_history_size": 10,
                "anomaly_threshold": 2.0
            }
        }
    }
    
    agent = StateObserverAgent(config)
    return await agent.invoke(state) 