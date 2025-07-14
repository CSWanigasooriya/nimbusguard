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
            logger.info("⏭️ Forecasting disabled, using current metrics as forecast")
            forecast_result = {
                'predicted_metrics': state['current_metrics'].copy(),
                'confidence': 0.5,
                'method': 'current_metrics_fallback',
                'horizon_minutes': config.forecasting.forecast_horizon_minutes
            }
        else:
            # Get forecaster service
            forecaster = services.get('forecaster')
            if not forecaster:
                raise Exception("Forecaster service not available")

            # Generate forecast - LoadForecaster.generate_forecast() takes no arguments
            raw_forecast_result = forecaster.generate_forecast()

            # Ensure the forecast result has the expected structure for DQN
            forecast_result = {
                'predicted_metrics': state['current_metrics'].copy(),  # Default fallback
                'confidence': raw_forecast_result.get('confidence', 0.0),
                'method': raw_forecast_result.get('method', 'lstm'),
                'horizon_minutes': config.forecasting.forecast_horizon_minutes
            }

            # Extract predicted metrics from forecast if available
            if 'current_metrics' in raw_forecast_result:
                forecast_result['predicted_metrics'] = raw_forecast_result['current_metrics']
            elif 'predicted_peak' in raw_forecast_result:
                forecast_result['predicted_metrics'] = raw_forecast_result['predicted_peak']
            elif 'forecast_summary' in raw_forecast_result:
                # Use current metrics with adjustments based on forecast summary
                forecast_summary = raw_forecast_result['forecast_summary']
                predicted_metrics = state['current_metrics'].copy()

                # Apply forecast adjustments if available
                for feature, current_value in predicted_metrics.items():
                    if 'cpu' in feature and 'cpu_peak' in forecast_summary:
                        predicted_metrics[feature] = forecast_summary['cpu_peak']
                    elif 'memory' in feature and 'memory_peak' in forecast_summary:
                        predicted_metrics[feature] = forecast_summary['memory_peak']
                    elif 'request' in feature and 'request_peak' in forecast_summary:
                        predicted_metrics[feature] = forecast_summary['request_peak']

                forecast_result['predicted_metrics'] = predicted_metrics

            # Store additional forecast information
            forecast_result.update({
                'analysis': raw_forecast_result.get('analysis', {}),
                'forecast': raw_forecast_result.get('forecast', []),
                'raw_forecast': raw_forecast_result
            })

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

            # Extract and update LSTM features for dashboard
            lstm_features = {}

            # Calculate pressure forecasts (normalized to 0-1)
            current_cpu = state['current_metrics'].get('process_cpu_seconds_total_rate', 0)
            current_memory = state['current_metrics'].get('process_resident_memory_bytes', 0)
            current_requests = state['current_metrics'].get('http_requests_total_rate', 0)

            # Simulate LSTM pressure forecasts based on current metrics and trends
            base_pressure = min(1.0, (current_cpu * 0.4 + current_memory / 1e9 * 0.3 + current_requests / 100 * 0.3))

            lstm_features['next_30sec_pressure'] = min(1.0, base_pressure * (
                        1 + forecast_result.get('confidence', 0.5) * 0.1))
            lstm_features['next_60sec_pressure'] = min(1.0, base_pressure * (
                        1 + forecast_result.get('confidence', 0.5) * 0.2))

            await services['metrics'].update_lstm_features(lstm_features)

        execution_time = (time.time() - start_time) * 1000
        state = add_node_output(state, 'generate_forecast', {
            'method': forecast_result.get('method', 'lstm'),
            'confidence': float(forecast_result.get('confidence', 0.0)),  # Ensure float
            'horizon_minutes': int(forecast_result.get('horizon_minutes', 0))  # Ensure int
        }, execution_time)

        logger.info(f"✅ Forecast generated: method={forecast_result.get('method')}, "
                    f"confidence={forecast_result.get('confidence', 0):.3f}")
        return state

    except Exception as e:
        logger.error(f"❌ Forecast generation failed: {e}")
        execution_time = (time.time() - start_time) * 1000
        state = add_error(state, f"forecast_generation: {str(e)}")
        state = add_node_output(state, 'generate_forecast', {'error': str(e)}, execution_time)

        # Fallback: use current metrics as forecast with proper structure
        fallback_result = {
            'predicted_metrics': state['current_metrics'].copy(),
            'confidence': 0.0,
            'method': 'error_fallback',
            'horizon_minutes': config.forecasting.forecast_horizon_minutes
        }
        state = update_state_with_forecast(state, fallback_result)
        return state


async def dqn_decision_node(state: OperatorState, services: Dict[str, Any], config: Any) -> OperatorState:
    """Make scaling decision using DQN agent."""
    start_time = time.time()

    try:
        logger.info(f"🧠 Making DQN decision for execution {state['execution_id']}")

        # Get DQN agent
        dqn_agent = services.get('dqn_agent')
        if not dqn_agent:
            raise Exception("DQN agent not available")

        # Create state vector (22 dimensions)
        dqn_state_vector = dqn_agent.create_state_vector(
            state['current_metrics'],
            state['forecast_result']['predicted_metrics'],
            state['current_replicas']
        )

        # Get Q-values and make decision
        q_values = dqn_agent.get_q_values(dqn_state_vector)
        action, confidence, is_exploration = dqn_agent.select_action(
            dqn_state_vector,
            use_forecast_guidance=True
        )

        # Convert action to replica count
        if action == 0:  # scale_down
            desired_replicas = max(config.scaling.min_replicas, state['current_replicas'] - 1)
            action_name = "scale_down"
        elif action == 1:  # keep_same
            desired_replicas = state['current_replicas']
            action_name = "keep_same"
        else:  # scale_up (action == 2)
            desired_replicas = min(config.scaling.max_replicas, state['current_replicas'] + 1)
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


async def validate_decision_node(state: OperatorState, services: Dict[str, Any], config: Any) -> OperatorState:
    """Validate the scaling decision before execution."""
    start_time = time.time()

    try:
        logger.info(f"✅ Validating decision for execution {state['execution_id']}")

        scaler = services.get('scaler')
        if not scaler:
            raise Exception("Scaler not available")

        # Validate scaling decision
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
        reward = dqn_agent.reward_calculator.calculate_reward(
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
                            f"📊 Epsilon updated in metrics: {dqn_agent.epsilon:.6f} (decay: {config.scaling.dqn_epsilon_decay})")
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
