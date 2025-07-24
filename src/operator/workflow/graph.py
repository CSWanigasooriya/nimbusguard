"""
LangGraph workflow for proactive scaling decisions.
"""

import logging
import uuid
from typing import Dict, Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from workflow.nodes import (
    collect_metrics_node,
    generate_forecast_node,
    dqn_decision_node,
    validate_decision_node,
    execute_scaling_node,
    calculate_reward_node
)
from workflow.state import OperatorState, create_initial_state, get_state_summary, update_state_with_forecast, \
    update_state_with_dqn, add_error

logger = logging.getLogger(__name__)


class ScalingWorkflow:
    def __init__(self, graph, services, config):
        self.graph = graph
        self.services = services
        self.config = config
        self.logger = logger

    async def ainvoke(self, initial_input: Dict[str, Any], config_dict: Dict[str, Any] = None):
        """Execute the scaling workflow."""
        try:
            # Generate unique execution ID
            execution_id = str(uuid.uuid4())[:8]

            # Get current replica count from initial input or services
            current_replicas = initial_input.get('current_replicas', 1)
            if 'k8s_client' in self.services and 'scaler' in self.services:
                try:
                    deployment_info = await self.services['k8s_client'].get_deployment(
                        self.config.scaling.target_deployment,
                        self.config.scaling.target_namespace
                    )
                    if deployment_info:
                        current_replicas = deployment_info['replicas']
                except Exception as e:
                    self.logger.warning(f"Failed to get current replicas: {e}")

            # Create initial state
            initial_state = create_initial_state(execution_id, current_replicas)

            self.logger.info(f"🚀 Starting scaling workflow execution {execution_id}")

            # Execute workflow
            final_state = await self.graph.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": execution_id}}
            )

            # Log execution summary
            summary = get_state_summary(final_state)
            self.logger.info(f"✅ Workflow {execution_id} completed: {summary}")

            # Create result summary
            result = {
                'execution_id': execution_id,
                'success': len(final_state['errors']) == 0,
                'final_decision': {
                    'replicas': final_state['desired_replicas'],
                    'action': final_state['scaling_decision'],
                    'confidence': final_state['dqn_confidence'],
                    'reason': final_state['decision_reason']
                },
                'execution_summary': {
                    'metrics_collected': 'collect_metrics' in final_state['node_outputs'],
                    'forecast_generated': 'generate_forecast' in final_state['node_outputs'],
                    'dqn_decision_made': 'dqn_decision' in final_state['node_outputs'],
                    'validation_passed': final_state['validation_passed'],
                    'scaling_applied': final_state['scaling_applied'],
                    'reward_calculated': final_state['reward_calculated']
                },
                'performance': {
                    'total_time_ms': sum(final_state['execution_time_ms'].values()),
                    'node_times': final_state['execution_time_ms']
                },
                'errors': final_state['errors']
            }

            # Update metrics if available
            if 'metrics' in self.services:
                await self.services['metrics'].update_decision_metrics(result)

            return result

        except Exception as e:
            self.logger.error(f"❌ Workflow execution failed: {e}")
            return {
                'execution_id': execution_id if 'execution_id' in locals() else 'unknown',
                'success': False,
                'error': str(e),
                'final_decision': {
                    'replicas': current_replicas if 'current_replicas' in locals() else 1,
                    'action': 'keep_same',
                    'confidence': 0.0,
                    'reason': 'workflow_error'
                }
            }

    async def get_workflow_status(self):
        """Get current workflow status and statistics."""
        return {
            'workflow_type': 'proactive_scaling',
            'nodes': [
                'collect_metrics',
                'generate_forecast',
                'dqn_decision',
                'validate_decision',
                'execute_scaling',
                'calculate_reward'
            ],
            'services_available': {
                'prometheus': 'prometheus' in self.services,
                'forecaster': 'forecaster' in self.services,
                'dqn_agent': 'dqn_agent' in self.services,
                'k8s_client': 'k8s_client' in self.services,
                'scaler': 'scaler' in self.services,
                'metrics': 'metrics' in self.services
            },
            'config': {
                'forecasting_enabled': self.config.forecasting.enabled,
                'target_deployment': self.config.scaling.target_deployment,
                'target_namespace': self.config.scaling.target_namespace
            }
        }

    def visualize_workflow(self):
        """Get a text representation of the workflow."""
        return """
🔄 NimbusGuard Proactive Scaling Workflow
==========================================

1. 📊 collect_metrics
   ├─ Query Prometheus for 9 selected features
   ├─ Get current deployment replica count
   └─ Prepare metrics for analysis

2. 🔮 generate_forecast  
   ├─ Generate LSTM forecast (if enabled)
   ├─ Calculate forecast confidence
   └─ Fallback to current metrics if needed

3. 🧠 dqn_decision
   ├─ Create 22-dimensional state vector
   ├─ Get Q-values for all actions
   ├─ Select action with forecast-guided exploration
   └─ Determine desired replica count

4. ✅ validate_decision
   ├─ Check scaling constraints
   ├─ Validate deployment health
   └─ Approve or reject scaling

5. 🚀 execute_scaling
   ├─ Apply scaling via Kubernetes API
   ├─ Wait for rollout completion
   └─ Record scaling results

6. 🎯 calculate_reward
   ├─ Calculate DQN reward
   ├─ Store experience in replay buffer
   └─ Enable continuous learning

🔄 Result: Intelligent, proactive scaling decision
        """


def create_workflow(services: Dict[str, Any], config: Any):
    """Create the LangGraph workflow for scaling decisions."""

    # Create workflow with checkpointing for resilience
    workflow = StateGraph(OperatorState)
    checkpointer = MemorySaver()

    # Define workflow nodes with service injection
    async def collect_metrics_wrapper(state: OperatorState) -> OperatorState:
        return await collect_metrics_node(state, services, config)

    async def generate_forecast_wrapper(state: OperatorState) -> OperatorState:
        return await generate_forecast_node(state, services, config)

    async def dqn_decision_wrapper(state: OperatorState) -> OperatorState:
        return await dqn_decision_node(state, services, config)

    async def validate_decision_wrapper(state: OperatorState) -> OperatorState:
        return await validate_decision_node(state, services, config)

    async def execute_scaling_wrapper(state: OperatorState) -> OperatorState:
        return await execute_scaling_node(state, services, config)

    async def calculate_reward_wrapper(state: OperatorState) -> OperatorState:
        return await calculate_reward_node(state, services, config)

    # Add nodes to workflow
    workflow.add_node("collect_metrics", collect_metrics_wrapper)
    workflow.add_node("generate_forecast", generate_forecast_wrapper)
    workflow.add_node("dqn_decision", dqn_decision_wrapper)
    workflow.add_node("validate_decision", validate_decision_wrapper)
    workflow.add_node("execute_scaling", execute_scaling_wrapper)
    workflow.add_node("calculate_reward", calculate_reward_wrapper)

    # Define workflow flow
    workflow.set_entry_point("collect_metrics")

    # Linear workflow: metrics → forecast → DQN → validate → execute → reward
    workflow.add_edge("collect_metrics", "generate_forecast")
    workflow.add_edge("generate_forecast", "dqn_decision")
    workflow.add_edge("dqn_decision", "validate_decision")
    workflow.add_edge("validate_decision", "execute_scaling")
    workflow.add_edge("execute_scaling", "calculate_reward")
    workflow.add_edge("calculate_reward", END)

    # Compile workflow with checkpointing
    compiled_workflow = workflow.compile(checkpointer=checkpointer)

    logger.info("✅ LangGraph workflow created with 6 nodes and checkpointing")

    return ScalingWorkflow(compiled_workflow, services, config)


def create_simple_workflow(services: Dict[str, Any], config: Any):
    """Create a simplified workflow for testing or minimal deployments."""

    workflow = StateGraph(OperatorState)

    # Simplified workflow: just metrics → DQN → execute
    async def simple_decision_node(state: OperatorState) -> OperatorState:
        """Simplified decision node that combines multiple steps."""
        try:
            # Collect metrics
            state = await collect_metrics_node(state, services, config)

            # Make DQN decision (with fallback forecast)
            forecast_result = {
                'predicted_metrics': state['current_metrics'].copy(),
                'confidence': 0.5,
                'method': 'simple_fallback'
            }
            state = update_state_with_forecast(state, forecast_result)
            state = await dqn_decision_node(state, services, config)

            # Validate and execute
            state = await validate_decision_node(state, services, config)
            if state['validation_passed']:
                state = await execute_scaling_node(state, services, config)

            # Calculate reward
            state = await calculate_reward_node(state, services, config)

            return state

        except Exception as e:
            logger.error(f"Simple workflow failed: {e}")
            state = add_error(state, f"simple_workflow: {str(e)}")
            return state

    workflow.add_node("simple_decision", simple_decision_node)
    workflow.set_entry_point("simple_decision")
    workflow.add_edge("simple_decision", END)

    compiled_workflow = workflow.compile()

    logger.info("✅ Simple LangGraph workflow created")
    return ScalingWorkflow(compiled_workflow, services, config)
