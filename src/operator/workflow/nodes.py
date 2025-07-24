"""LangGraph workflow nodes for the NimbusGuard operator."""
import logging
import time
from typing import Dict, Any

import numpy as np
from workflow.state import (
    OperatorState,
    update_state_with_metrics,
    update_state_with_forecast,
    update_state_with_dqn,
    add_node_output,
    add_error
)
import json
import re


# Helper functions for state updates
def update_state_with_scaling_decision(state: OperatorState, desired_replicas: int,
                                       action: str, reason: str) -> OperatorState:
    """Update state with scaling decision."""
    state["desired_replicas"] = desired_replicas
    state["scaling_decision"] = action
    state["decision_reason"] = reason
    return state


def update_state_with_validation(state: OperatorState, passed: bool,
                                 errors: list = None) -> OperatorState:
    """Update state with validation results."""
    state["validation_passed"] = passed
    state["validation_errors"] = errors or []
    return state


def update_state_with_execution(state: OperatorState, applied: bool,
                                error: str = None) -> OperatorState:
    """Update state with execution results."""
    state["scaling_applied"] = applied
    state["scaling_error"] = error
    return state


def update_state_with_reward(state: OperatorState, reward: float) -> OperatorState:
    """Update state with reward calculation."""
    state["reward_calculated"] = True
    state["reward_value"] = reward
    return state


logger = logging.getLogger(__name__)


async def collect_metrics_node(state: OperatorState, services: Dict[str, Any], config: Any) -> OperatorState:
    """Collect current metrics from Prometheus."""
    start_time = time.time()

    try:
        logger.info(f"📊 Collecting metrics for execution {state['execution_id']}")

        # Get Prometheus client
        prometheus = services.get('prometheus')
        if not prometheus:
            raise Exception("Prometheus client not available")

        # Collect current metrics using PrometheusClient
        raw_metrics = prometheus.get_current_metrics()

        # Log the fetched metrics for debugging and monitoring
        logger.info(f"📊 Current metrics fetched from Prometheus:")
        for metric_name, value in raw_metrics.items():
            if isinstance(value, (int, float)):
                if 'memory' in metric_name.lower():
                    # Convert memory to MB for readability
                    logger.info(f"  🧠 {metric_name}: {value/1024/1024:.2f} MB ({value:,.0f} bytes)")
                elif 'cpu' in metric_name.lower():
                    logger.info(f"  ⚙️  {metric_name}: {value:.4f} cores")
                else:
                    logger.info(f"  📈 {metric_name}: {value}")
            else:
                logger.info(f"  📊 {metric_name}: {value}")

        # Ensure all metrics are serializable (convert numpy types to Python types)
        def make_serializable(obj):
            """Convert numpy types to Python native types."""
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            else:
                return obj

        metrics = make_serializable(raw_metrics)

        # Get current replica count
        k8s_client = services.get('k8s_client')
        if k8s_client:
            deployment_info = await k8s_client.get_deployment(
                config.scaling.target_deployment,
                config.scaling.target_namespace
            )
            current_replicas = deployment_info['replicas'] if deployment_info else 1
        else:
            current_replicas = state['current_replicas']

        # Update state
        state = update_state_with_metrics(state, metrics)
        state['current_replicas'] = current_replicas
        state['deployment_info'] = deployment_info  # Store deployment info for other nodes

        # CRITICAL FIX: Always update current replicas metric during collection
        if 'metrics' in services:
            services['metrics'].current_replicas.set(current_replicas)

        execution_time = (time.time() - start_time) * 1000
        state = add_node_output(state, 'collect_metrics', {
            'metrics_count': len(metrics),
            'current_replicas': current_replicas
        }, execution_time)

        logger.info(f"✅ Metrics collected: {len(metrics)} features, {current_replicas} replicas")
        return state

    except Exception as e:
        logger.error(f"❌ Metrics collection failed: {e}")
        execution_time = (time.time() - start_time) * 1000
        state = add_error(state, f"metrics_collection: {str(e)}")
        state = add_node_output(state, 'collect_metrics', {'error': str(e)}, execution_time)
        return state


async def generate_forecast_node(state: OperatorState, services: Dict[str, Any], config: Any) -> OperatorState:
    """Generate LSTM forecast for proactive scaling."""
    start_time = time.time()

    try:
        logger.info(f"🔮 Generating forecast for execution {state['execution_id']}")

        if not config.forecasting.enabled:
            raise Exception("Forecasting is disabled but required for proactive scaling")

        forecaster = services.get('forecaster')
        prometheus = services.get('prometheus')

        if not forecaster or not prometheus:
            raise Exception("Forecaster or Prometheus client not available")

        if not forecaster.is_loaded:
            raise Exception("LSTM model not loaded - check forecaster.keras and forecaster_scaler.pkl")

        # Fetch historical data for the LSTM model
        historical_data = await prometheus.get_historical_metrics(
            duration_minutes=config.forecasting.lookback_minutes
        )

        # Generate forecast using the pre-trained LSTMPredictor
        predicted_metrics = await forecaster.predict(historical_data)

        if not predicted_metrics:
            raise Exception("LSTM prediction returned None - insufficient historical data")

        forecast_result = {
            'predicted_metrics': predicted_metrics,
            'method': 'lstm',
            'horizon_seconds': config.forecasting.forecast_horizon_seconds
        }

        # Ensure all values are serializable (convert numpy types to Python types)
        def make_serializable(obj):
            """Convert numpy types to Python native types."""
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            else:
                return obj

        forecast_result = make_serializable(forecast_result)

        # Update state with forecast
        state = update_state_with_forecast(state, forecast_result)

        # Update forecast metrics
        if 'metrics' in services:
            await services['metrics'].update_forecast_metrics(forecast_result)

        execution_time = (time.time() - start_time) * 1000
        state = add_node_output(state, 'generate_forecast', {
            'method': 'lstm',
            'horizon_seconds': config.forecasting.forecast_horizon_seconds
        }, execution_time)

        logger.info(f"✅ LSTM forecast generated: horizon={config.forecasting.forecast_horizon_seconds}s")
        return state

    except Exception as e:
        logger.error(f"❌ Forecast generation failed: {e}")
        execution_time = (time.time() - start_time) * 1000
        state = add_error(state, f"forecast_generation: {str(e)}")
        state = add_node_output(state, 'generate_forecast', {'error': str(e)}, execution_time)
        raise Exception(f"Forecasting failed: {e}")


async def dqn_decision_node(state: OperatorState, services: Dict[str, Any], config: Any) -> OperatorState:
    """Make scaling decision using DQN agent."""
    start_time = time.time()

    try:
        logger.info(f"🧠 Making DQN decision for execution {state['execution_id']}")

        # Get DQN agent
        dqn_agent = services.get('dqn_agent')
        if not dqn_agent:
            raise Exception("DQN agent not available")

        # Create state vector (now 3 dimensions)
        dqn_state_vector = dqn_agent.create_state_vector(
            state['current_metrics'],
            state['forecast_result']['predicted_metrics'] if state['forecast_result'] else None,
            state['current_replicas']
        )

        # Get Q-values and make decision
        q_values = dqn_agent.get_q_values(dqn_state_vector)
        action, confidence, is_exploration = dqn_agent.select_action(
            dqn_state_vector,
            use_forecast_guidance=True
        )

        # Get deployment-specific constraints for action bounds
        deployment_info = state.get('deployment_info')
        min_replicas = deployment_info.get('min_replicas', 1) if deployment_info else 1
        max_replicas = deployment_info.get('max_replicas', 50) if deployment_info else 50

        # Convert action to replica count
        if action == 0:  # scale_down
            desired_replicas = max(min_replicas, state['current_replicas'] - 1)
            action_name = "scale_down"
        elif action == 1:  # keep_same
            desired_replicas = state['current_replicas']
            action_name = "keep_same"
        else:  # scale_up (action == 2)
            desired_replicas = min(max_replicas, state['current_replicas'] + 1)
            action_name = "scale_up"

        # Create Q-values dict for metrics
        # Ensure all Q-values are properly converted to Python scalars
        def make_serializable(obj):
            """Convert numpy types to Python native types."""
            if hasattr(obj, 'item') and hasattr(obj, 'size'):
                # numpy array or scalar
                if obj.size == 1:
                    return obj.item()  # Single element array/scalar
                else:
                    return obj.tolist()  # Multi-element array
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            else:
                return obj

        # Convert Q-values to proper scalars
        q_values_serializable = make_serializable(q_values)
        if isinstance(q_values_serializable, list) and len(q_values_serializable) >= 3:
            q_values_dict = {
                'scale_down': float(q_values_serializable[0]),
                'keep_same': float(q_values_serializable[1]),
                'scale_up': float(q_values_serializable[2])
            }
        else:
            # Fallback if Q-values format is unexpected
            q_values_dict = {
                'scale_down': 0.0,
                'keep_same': 0.0,
                'scale_up': 0.0
            }

        # Update state with DQN decision

        # Convert state vector and confidence to serializable types
        dqn_state_list = make_serializable(dqn_state_vector)
        confidence_float = float(confidence)

        # Update state
        state = update_state_with_dqn(state, action_name, q_values_dict, dqn_state_list, confidence_float)
        state['desired_replicas'] = desired_replicas
        state['scaling_decision'] = action_name
        state['decision_reason'] = f"DQN_confidence_{confidence:.3f}"

        # Update DQN metrics
        if 'metrics' in services:
            training_stats = {
                'buffer_size': len(dqn_agent.replay_buffer) if hasattr(dqn_agent, 'replay_buffer') else 0,
                'training_steps': getattr(dqn_agent, 'training_steps', 0),
                'epsilon': getattr(dqn_agent, 'epsilon', 0.0)  # Include current epsilon value
            }

            # Add loss if available
            if hasattr(dqn_agent, 'last_loss') and dqn_agent.last_loss is not None:
                training_stats['loss'] = float(dqn_agent.last_loss)

            await services['metrics'].update_dqn_metrics(
                training_stats=training_stats,
                q_values=q_values_dict,
                decision_confidence=float(confidence),
                is_exploration=is_exploration
            )

        execution_time = (time.time() - start_time) * 1000
        state = add_node_output(state, 'dqn_decision', {
            'action': action_name,
            'desired_replicas': desired_replicas,
            'confidence': float(confidence),
            'q_values': q_values_dict,
            'is_exploration': is_exploration
        }, execution_time)

        logger.info(f"✅ DQN decision: {action_name} → {desired_replicas} replicas "
                    f"(confidence: {confidence:.3f}, exploration: {is_exploration})")
        return state

    except Exception as e:
        logger.error(f"❌ DQN decision failed: {e}", exc_info=True)  # Add stack trace
        state = add_error(state, f"dqn_decision: {str(e)}")

        # Fallback decision
        state = update_state_with_scaling_decision(
            state,
            state['current_replicas'],
            "keep_same",
            "dqn_error_fallback"
        )
        state['dqn_confidence'] = 0.0
        state['dqn_state'] = []
        state['dqn_q_values'] = {'scale_down': 0.0, 'keep_same': 0.0, 'scale_up': 0.0}

        return state


# Robust LLM JSON response parser with safety fallback
def parse_llm_json_response(response_text: str, action_name: str) -> dict:
    """Parse LLM JSON response with robust error handling and safety-first fallbacks."""
    try:
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)

            # Validate required fields for safety monitoring
            required_fields = ['approved', 'confidence', 'reasoning']
            if all(field in parsed for field in required_fields):

                # Ensure all expected safety fields exist with defaults
                safety_defaults = {
                    'safety_risk': parsed.get('safety_risk', 'none'),
                    'extreme_factors': parsed.get('extreme_factors', []),
                    'alternative_suggestion': parsed.get('alternative_suggestion', ''),
                    'cluster_check_performed': parsed.get('cluster_check_performed', False),
                    'tool_findings': parsed.get('tool_findings', []),
                    'validation_score': 0.8 if parsed.get('approved', True) else 0.3
                }

                # Merge parsed response with safety defaults
                result = {**parsed, **safety_defaults}

                # Ensure confidence is valid
                if result['confidence'] not in ['low', 'medium', 'high']:
                    result['confidence'] = 'medium'

                return result

        # If JSON parsing failed, log and use safety fallback
        logger.warning(f"LLM_PARSING: json_extraction_failed response_preview={response_text[:100]}")

    except json.JSONDecodeError as e:
        logger.warning(f"LLM_PARSING: json_decode_error position={e.pos}")

    except Exception as e:
        logger.warning(f"LLM_PARSING: general_parsing_error error={e}")

    # SAFETY FALLBACK: Default to APPROVE for any parsing failures
    # This preserves DQN learning and only blocks when LLM explicitly identifies extreme risk
    return {
        'approved': True,  # Safety-first: approve unless explicitly dangerous
        'confidence': 'low',
        'reasoning': f'Parsing failure, unable to verify cluster state. Defaulting to APPROVE for safety. Response: {response_text[:200]}...',
        'safety_risk': 'unknown',
        'extreme_factors': ['parsing_failure', 'tool_verification_failed'],
        'alternative_suggestion': 'Monitor decision closely due to validation parsing failure',
        'cluster_check_performed': False,
        'validation_score': 0.5,
        'fallback_mode': True,
        'tool_findings': ['Parsing failed - no tool verification performed']
    }


async def validate_decision_node(state: OperatorState, services: Dict[str, Any], config: Any) -> OperatorState:
    start_time = time.time()

    try:
        logger.info(f"✅ Validating decision for execution {state['execution_id']}")

        scaler = services.get('scaler')
        if not scaler:
            raise Exception("Scaler not available")

        # LLM validation logic
        llm_validation_enabled = hasattr(config, "ai") and getattr(config.ai, "enable_llm_validation", False)
        agent = services.get("langgraph_agent")
        llm_blocked = False
        llm_reason = None
        if llm_validation_enabled:
            if agent:
                try:
                    # Build detailed safety monitor prompt
                    dqn_prediction = state.get('dqn_prediction', {'action_name': state.get('scaling_decision', 'keep_same')})
                    action_name = dqn_prediction.get('action_name', state.get('scaling_decision', 'keep_same'))
                    dqn_confidence = dqn_prediction.get('confidence', state.get('dqn_confidence', 'unknown'))
                    dqn_risk = dqn_prediction.get('risk_assessment', 'unknown')
                    dqn_explanation = dqn_prediction.get('explanation', {})
                    metrics = state.get('current_metrics', {})
                    # Build the prompt string
                    prompt = f"""You are a SAFETY MONITOR for Kubernetes autoscaling. Your ONLY role is to detect and prevent EXTREME or DANGEROUS scaling decisions that could harm the cluster.\n\n🚨 CRITICAL: Only intervene for EXTREME decisions. Allow normal DQN learning to proceed uninterrupted.\n\nDQN SCALING DECISION TO EVALUATE:\n- Recommended Action: {action_name}\n- Current Replicas: {state.get('current_replicas', 1)}\n- DQN Confidence: {dqn_confidence}\n- DQN Risk Assessment: {dqn_risk}\n\nDQN REASONING FACTORS:\n{chr(10).join(f'- {factor}' for factor in dqn_explanation.get('reasoning_factors', ['No DQN reasoning available']))}\n\nCURRENT SYSTEM METRICS:\n- Pod Readiness: {metrics.get('kube_pod_container_status_ready', 1.0):.1%}\n- Unavailable Replicas: {metrics.get('kube_deployment_status_replicas_unavailable', 0)}\n- CPU Limits: {metrics.get('kube_pod_container_resource_limits_cpu', 0):.2f} cores\n- Memory Limits: {metrics.get('kube_pod_container_resource_limits_memory', 0)/1000000:.0f} MB\n- Running Containers: {metrics.get('kube_pod_container_status_running', 0)}\n\n🔍 EXTREME DECISION CRITERIA (Block these ONLY):\n\n1. **RUNAWAY SCALING**: Scaling to >15 replicas without clear justification\n2. **RESOURCE EXHAUSTION**: Scaling up when cluster resources are constrained  \n3. **MASSIVE OVER-SCALING**: Requesting 3x+ more replicas than needed (consider system capacity)\n4. **DANGEROUS DOWN-SCALING**: Scaling to 0 or very low when system shows stress\n5. **RAPID OSCILLATION**: Frequent large scaling changes without stabilization\n6. **IGNORES HIGH RISK**: DQN marked decision as \"high\" risk but proceeding anyway\n\n⚠️  **DEFAULT: APPROVE** - Only block if decision meets extreme criteria above.\n\n🛡️  **MANDATORY KUBERNETES VERIFICATION**: You MUST use the available Kubernetes MCP tools to verify cluster state before making any safety decision.\n\nTARGET DEPLOYMENT DETAILS:\n- Namespace: {getattr(config.scaling, 'target_namespace', 'nimbusguard')}\n- Deployment Name: {getattr(config.scaling, 'target_deployment', 'consumer')}\n- Current Replicas: {state.get('current_replicas', 1)}\n\nREQUIRED ASSESSMENT PROTOCOL:\n1. **USE RELEVANT TOOLS ONLY** - Choose the tools that are most relevant for this specific safety decision:\n   \n   **Available Tools:**\n   - `mcp_kubernetes_pods_list` - Check current pod status and health in \"{getattr(config.scaling, 'target_namespace', 'nimbusguard')}\" namespace\n   - `mcp_kubernetes_pods_top` - Verify actual resource consumption for \"{getattr(config.scaling, 'target_deployment', 'consumer')}\" pods  \n   - `mcp_kubernetes_resources_get` - Check deployment status (apiVersion: apps/v1, kind: Deployment, name: {getattr(config.scaling, 'target_deployment', 'consumer')}, namespace: {getattr(config.scaling, 'target_namespace', 'nimbusguard')})\n   - `mcp_kubernetes_events_list` - Look for recent cluster issues or warnings in \"{getattr(config.scaling, 'target_namespace', 'nimbusguard')}\" namespace\n\n   **Tool Selection Guide:**\n   - For scaling decisions: Use `mcp_kubernetes_resources_get` to verify current deployment state\n   - For resource concerns: Use `mcp_kubernetes_pods_top` to check actual resource consumption\n   - For stability issues: Use `mcp_kubernetes_events_list` to check for recent problems\n   - For health verification: Use `mcp_kubernetes_pods_list` to verify pod status\n\n2. **EFFICIENT VERIFICATION**: Use only 1-2 tools that provide the most relevant data for your safety assessment\n\n3. **MAKE INFORMED DECISION** based on:\n   - Real-time cluster data from the tools you chose to use\n   - Provided DQN metrics and reasoning\n\nCRITICAL REQUIREMENTS:\n1. You MUST use at least 1 relevant MCP Kubernetes tool before making a decision\n2. Choose tools based on what's most important for the specific scaling decision being evaluated\n3. Set \"cluster_check_performed\": true in your response (required)\n4. Include specific findings from the tools you used in your reasoning\n5. If tools show different data than provided metrics, prioritize tool data\n\nIMPORTANT: All tools are READ-ONLY. You cannot modify the cluster - only observe and assess.\n\nCRITICAL: Respond with ONLY a valid JSON object. No markdown, no explanations.\n\nExample for NORMAL decision (approve):\n{{\n    \"approved\": true,\n    \"confidence\": \"high\",\n    \"reasoning\": \"Verified with mcp_kubernetes_resources_get: deployment shows 3 healthy replicas. DQN scale-down decision is safe.\",\n    \"safety_risk\": \"none\",\n    \"extreme_factors\": [],\n    \"cluster_check_performed\": true,\n    \"tool_findings\": [\"Deployment has 3 healthy replicas\", \"No resource pressure indicated\"]\n}}\n\nExample for EXTREME decision (block):\n{{\n    \"approved\": false,\n    \"confidence\": \"high\",\n    \"reasoning\": \"EXTREME: mcp_kubernetes_events_list shows OutOfMemory errors. Scaling up to 25 replicas would exhaust cluster resources.\",\n    \"safety_risk\": \"high\",\n    \"extreme_factors\": [\"runaway_scaling\", \"resource_exhaustion_risk\"],\n    \"alternative_suggestion\": \"Investigate memory issues before scaling. Consider 8-10 replicas maximum.\",\n    \"cluster_check_performed\": true,\n    \"tool_findings\": [\"OutOfMemory events detected\", \"Cluster showing resource pressure\"]\n}}\n\nYour JSON response:"""
                    logger.info("LLM validation: invoking agent for scaling decision safety check with detailed prompt...")
                    response = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
                    # Parse the last message as JSON
                    last_message = None
                    if isinstance(response, dict) and 'messages' in response and response['messages']:
                        last_message = response['messages'][-1].get('content', '')
                    elif isinstance(response, str):
                        last_message = response
                    else:
                        last_message = str(response)
                    logger.info(f"LLM validation response chars={len(last_message) if last_message else 0}")
                    if last_message:
                        logger.debug(f"LLM RAW RESPONSE: {last_message[:200]}...")
                    # Use robust parser
                    validation_result = parse_llm_json_response(last_message, action_name)
                    # If LLM blocks, override validation
                    if not validation_result.get("approved", True):
                        llm_blocked = True
                        llm_reason = validation_result.get("reasoning", "Blocked by LLM safety monitor.")
                        state["llm_validation_response"] = validation_result
                except Exception as e:
                    logger.error(f"LLM validation error: {e}")
                    # Fallback: approve if LLM fails
            else:
                logger.warning("LLM validation enabled but agent not initialized; skipping LLM validation.")
                # Fallback: approve if agent missing
        # If LLM blocked, override validation
        if llm_blocked:
            state = update_state_with_validation(state, False, [llm_reason or "Blocked by LLM safety monitor."])
            logger.warning(f"SCALING BLOCKED BY LLM: {llm_reason}")
            execution_time = (time.time() - start_time) * 1000
            state = add_node_output(state, 'validate_decision', {
                'valid': False,
                'reason': llm_reason,
                'current_replicas': state['current_replicas'],
                'desired_replicas': state['desired_replicas'],
                'llm_blocked': True
            }, execution_time)
            return state

        # Normal validation
        current_replicas = state['current_replicas']
        desired_replicas = state['desired_replicas']

        is_valid, reason = await scaler.validate_scaling_decision(
            current_replicas=current_replicas,
            desired_replicas=desired_replicas,
            metrics=state['current_metrics']
        )

        errors = [] if is_valid else [reason]

        # Update state
        state = update_state_with_validation(state, is_valid, errors)

        execution_time = (time.time() - start_time) * 1000
        state = add_node_output(state, 'validate_decision', {
            'valid': is_valid,
            'reason': reason,
            'current_replicas': current_replicas,
            'desired_replicas': desired_replicas
        }, execution_time)

        logger.info(f"✅ Decision validation: {'PASSED' if is_valid else 'FAILED'} - {reason}")
        return state

    except Exception as e:
        logger.error(f"❌ Decision validation failed: {e}")
        execution_time = (time.time() - start_time) * 1000
        state = add_error(state, f"decision_validation: {str(e)}")
        state = update_state_with_validation(state, False, [str(e)])
        state = add_node_output(state, 'validate_decision', {'error': str(e)}, execution_time)
        return state


async def execute_scaling_node(state: OperatorState, services: Dict[str, Any], config: Any) -> OperatorState:
    """Execute the scaling decision."""
    start_time = time.time()

    try:
        logger.info(f"🚀 Executing scaling for execution {state['execution_id']}")

        # Check for LLM validation block
        llm_validation = state.get("llm_validation_response")
        if llm_validation and not llm_validation.get("approved", True):
            logger.warning(f"Skipping scaling execution: Blocked by LLM. Reason: {llm_validation.get('reasoning')}")
            state['scaling_applied'] = False
            state['scaling_error'] = llm_validation.get('reasoning', 'Blocked by LLM')
            execution_time = (time.time() - start_time) * 1000
            state = add_node_output(state, 'execute_scaling', {
                'skipped': True,
                'reason': llm_validation.get('reasoning', 'Blocked by LLM'),
                'llm_blocked': True
            }, execution_time)
            return state

        # Skip if validation failed
        if not state['validation_passed']:
            logger.info("⏭️ Skipping scaling execution due to validation failure")
            state['scaling_applied'] = False
            state['scaling_error'] = "validation_failed"
            execution_time = (time.time() - start_time) * 1000
            state = add_node_output(state, 'execute_scaling', {
                'skipped': True,
                'reason': 'validation_failed'
            }, execution_time)
            return state

        # Skip if no scaling needed
        if state['desired_replicas'] == state['current_replicas']:
            logger.info(f"⏭️ No scaling needed: {state['current_replicas']} replicas")
            state['scaling_applied'] = True
            state['scaling_error'] = None

            # CRITICAL FIX: Update metrics even when no scaling occurs
            if 'metrics' in services:
                await services['metrics'].update_scaling_metrics(
                    action=state['scaling_decision'],
                    current_replicas=state['current_replicas'],
                    desired_replicas=state['desired_replicas'],
                    reason="DQN_no_change"
                )

            execution_time = (time.time() - start_time) * 1000
            state = add_node_output(state, 'execute_scaling', {
                'skipped': True,
                'reason': 'no_change_needed'
            }, execution_time)
            return state

        # Get scaler service
        scaler = services.get('scaler')
        if not scaler:
            raise Exception("Scaler service not available")

        # Execute scaling
        old_replicas = state['current_replicas']
        new_replicas = state['desired_replicas']

        scaling_success, scaling_result = await scaler.scale_to(
            new_replicas,
            reason=f"DQN_{state['scaling_decision']}"
        )

        if scaling_success:
            state['scaling_applied'] = True
            state['scaling_error'] = None

            # Update scaling metrics
            if 'metrics' in services:
                await services['metrics'].update_scaling_metrics(
                    action=state['scaling_decision'],
                    current_replicas=new_replicas,
                    desired_replicas=new_replicas,
                    reason="DQN"
                )

            logger.info(f"✅ Scaling successful: {old_replicas} → {new_replicas} replicas")
        else:
            state['scaling_applied'] = False
            state['scaling_error'] = scaling_result.get('error', 'unknown_error')
            logger.error(f"❌ Scaling failed: {scaling_result.get('error')}")

        execution_time = (time.time() - start_time) * 1000
        state = add_node_output(state, 'execute_scaling', {
            'success': scaling_success,
            'old_replicas': old_replicas,
            'new_replicas': new_replicas,
            'error': scaling_result.get('error') if not scaling_success else None
        }, execution_time)

        return state

    except Exception as e:
        logger.error(f"❌ Scaling execution failed: {e}")
        execution_time = (time.time() - start_time) * 1000
        state = add_error(state, f"scaling_execution: {str(e)}")
        state = add_node_output(state, 'execute_scaling', {'error': str(e)}, execution_time)

        state['scaling_applied'] = False
        state['scaling_error'] = str(e)
        return state


async def calculate_reward_node(state: OperatorState, services: Dict[str, Any], config: Any) -> OperatorState:
    """Calculate reward for DQN learning."""
    start_time = time.time()

    try:
        logger.info(f"🎯 Calculating reward for execution {state['execution_id']}")

        # Get DQN agent
        dqn_agent = services.get('dqn_agent')
        if not dqn_agent:
            raise Exception("DQN agent not available")

        # Get deployment resource limits for better reward calculation
        deployment_info = None
        k8s_client = services.get('k8s_client')
        if k8s_client:
            try:
                deployment_info = await k8s_client.get_deployment(
                    config.scaling.target_deployment,
                    config.scaling.target_namespace
                )
                logger.debug(
                    f"Deployment resource info: {deployment_info.get('resource_info', {}) if deployment_info else 'Not available'}")
            except Exception as e:
                logger.warning(f"Failed to get deployment resource info: {e}")

        # Use the sophisticated IoT-inspired reward system with deployment resource limits
        reward = dqn_agent.reward_calculator.calculate(
            action=state['scaling_decision'],
            current_metrics=state['current_metrics'],
            current_replicas=state['current_replicas'],
            desired_replicas=state['desired_replicas'],
            forecast_result=state.get('forecast_result'),
            deployment_info=deployment_info  # NEW: Pass deployment resource limits
        )

        # Additional reward adjustments based on execution success
        if not state['scaling_applied']:
            reward -= 1.0  # Penalty for failed execution
        if not state['validation_passed']:
            reward -= 0.5  # Penalty for validation failure

        # Store experience in replay buffer
        if (state['dqn_state'] is not None and
                hasattr(dqn_agent, 'replay_buffer') and
                state['dqn_action'] is not None):

            # Convert action name to index
            action_map = {"scale_down": 0, "keep_same": 1, "scale_up": 2}
            action_index = action_map.get(state['dqn_action'], 1)

            # Convert state to numpy array if it's a list
            import numpy as np
            current_state = np.array(state['dqn_state'], dtype=np.float32)

            # Create next state (simplified - could use next metrics in production)
            next_state = current_state.copy()  # Placeholder

            # Add experience to replay buffer
            dqn_agent.store_experience(
                state=current_state,
                action=action_index,
                reward=reward,
                next_state=next_state,
                done=True  # Each scaling decision is terminal
            )

            buffer_size = len(dqn_agent.replay_buffer)
            logger.info(f"📝 Experience stored: buffer_size={buffer_size}/{dqn_agent.replay_buffer.capacity}")

            # Update experience metrics
            if 'metrics' in services:
                await services['metrics'].update_experience_metrics(experiences_added=1)

        # Update reward metrics
        if 'metrics' in services:
            await services['metrics'].update_reward_metrics(reward_value=reward)

        # Trigger training if buffer has enough experiences
        buffer_size = len(dqn_agent.replay_buffer) if hasattr(dqn_agent, 'replay_buffer') else 0
        batch_size = config.scaling.dqn_batch_size

        logger.info(f"🧠 Training check: buffer_size={buffer_size}, batch_size={batch_size}")

        if buffer_size >= batch_size:
            try:
                logger.info(f"🚀 Starting DQN async training (buffer: {buffer_size}/{dqn_agent.replay_buffer.capacity})")
                training_loss = await dqn_agent.train_async()

                if training_loss is not None:
                    # Ensure training loss is a Python float
                    if hasattr(training_loss, 'item'):  # torch tensor or numpy scalar
                        training_loss = training_loss.item()
                    else:
                        training_loss = float(training_loss)

                    logger.info(
                        f"✅ DQN async training completed: loss={training_loss:.6f}, steps={dqn_agent.training_steps}")

                    # Update training metrics including the decayed epsilon
                    if 'metrics' in services:
                        training_stats = {
                            'loss': training_loss,
                            'training_steps': dqn_agent.training_steps,
                            'epsilon': dqn_agent.epsilon,
                            'buffer_size': buffer_size
                        }
                        await services['metrics'].update_dqn_metrics(training_stats)
                        logger.info(
                            f"📊 Epsilon updated in metrics: {dqn_agent.epsilon:.6f}")
                else:
                    logger.info("⏳ Training skipped - frequency limit or already in progress")

            except Exception as train_error:
                logger.error(f"❌ DQN async training failed: {train_error}", exc_info=True)
        else:
            logger.info(f"⏳ Waiting for more experiences: {buffer_size}/{batch_size} needed")

        # Update state with reward
        state = update_state_with_reward(state, float(reward))  # Ensure Python float

        # Ensure all state values are serializable before returning
        def make_serializable(obj):
            """Convert numpy types to Python native types."""
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            else:
                return obj

        # Apply serialization to critical state fields that might contain numpy types
        if 'dqn_state' in state and state['dqn_state'] is not None:
            state['dqn_state'] = make_serializable(state['dqn_state'])
        if 'dqn_q_values' in state and state['dqn_q_values'] is not None:
            state['dqn_q_values'] = make_serializable(state['dqn_q_values'])
        if 'current_metrics' in state and state['current_metrics'] is not None:
            state['current_metrics'] = make_serializable(state['current_metrics'])
        if 'forecast_result' in state and state['forecast_result'] is not None:
            state['forecast_result'] = make_serializable(state['forecast_result'])

        execution_time = (time.time() - start_time) * 1000
        state = add_node_output(state, 'calculate_reward', {
            'reward_value': float(reward),  # Ensure Python float
            'scaling_success': state['scaling_applied']
        }, execution_time)

        logger.info(f"✅ Reward calculated: {reward:.3f}")
        return state

    except Exception as e:
        logger.error(f"❌ Reward calculation failed: {e}")
        execution_time = (time.time() - start_time) * 1000
        state = add_error(state, f"reward_calculation: {str(e)}")
        state = add_node_output(state, 'calculate_reward', {'error': str(e)}, execution_time)

        state['reward_calculated'] = False
        state['reward_value'] = 0.0
        return state
