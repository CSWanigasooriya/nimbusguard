from datetime import datetime
import json
import random
from typing import Dict, Any
import logging
from functools import partial

import numpy as np
import asyncio
import torch
from langgraph.graph import StateGraph, END
from models.models import ScalingState
from monitoring.metrics import *
from .reasoning import DecisionReasoning
from core.services import ServiceContainer

logger = logging.getLogger("LangGraph")

decision_reasoning = DecisionReasoning()

# Note: Epsilon and decision count moved to ServiceContainer to eliminate global state

def create_graph(services: ServiceContainer):
    """
    Create the LangGraph workflow with dependency injection.
    🚀 PROACTIVE WORKFLOW: Immediate reward calculation and training (no stabilization wait).
    
    Args:
        services: ServiceContainer with all initialized dependencies
    
    Returns:
        Compiled LangGraph workflow
    
    Raises:
        ValueError: If required services are not initialized
    """
    if not services.is_ready():
        missing = services.get_missing_services()
        raise ValueError(f"Services not ready. Missing: {missing}")
    
    workflow = StateGraph(ScalingState)
    
    # Create node functions with injected dependencies using partial
    workflow.add_node("get_live_metrics", partial(get_live_metrics, services=services))
    workflow.add_node("get_dqn_recommendation", partial(get_dqn_recommendation, services=services))
    workflow.add_node("validate_with_llm", partial(validate_with_llm, services=services))
    workflow.add_node("plan_final_action", partial(plan_final_action, services=services))
    workflow.add_node("immediate_reward_and_training", partial(immediate_reward_and_training, services=services))
    workflow.add_node("log_experience", partial(log_experience, services=services))

    # 🚀 PROACTIVE WORKFLOW: Direct path from action to immediate training
    workflow.set_entry_point("get_live_metrics")
    workflow.add_edge("get_live_metrics", "get_dqn_recommendation")
    workflow.add_edge("get_dqn_recommendation", "validate_with_llm")
    workflow.add_edge("validate_with_llm", "plan_final_action")
    workflow.add_edge("plan_final_action", "immediate_reward_and_training") 
    workflow.add_edge("immediate_reward_and_training", "log_experience")
    workflow.add_edge("log_experience", END)

    return workflow.compile()


# --- LangGraph Nodes ---
async def get_live_metrics(state: ScalingState, services: ServiceContainer, is_next_state: bool = False) -> Dict[str, Any]:
    """
    Get live metrics from Prometheus - UPDATED to use scientifically selected consumer performance features.
    
    FEATURES: 9 scientifically-selected consumer pod performance indicators that directly impact scaling decisions:
    - CPU utilization rate (current usage)
    - Memory pressure rates (garbage collection activity) 
    - HTTP performance rates (request processing)
    - I/O load (file descriptors)
    - Network payload rates (response sizes)
    
    RESOURCE-AWARE: Also retrieves actual resource limits/requests for contextual reward calculation.
    """
    node_name = "observe_next_state" if is_next_state else "get_live_metrics"
    logger.info("=" * 60)
    logger.info(f"NODE_START: {node_name}")
    logger.info("=" * 60)

    from datetime import datetime
    current_time = datetime.now()

    # SCIENTIFIC FEATURES: Consumer pod performance metrics (port 8000)
    # These are the 9 features that were scientifically validated for DQN scaling decisions
    # FIXED: Use instance labels to target consumer pods on port 8000
    queries = {
        # 1. CPU utilization rate - Current CPU usage rate from consumer pods
        "process_cpu_seconds_total_rate": f'rate(process_cpu_seconds_total{{instance=~".*:8000"}}[2m]) or vector(0)',

        # 2. Memory pressure - Garbage collection activity indicating memory pressure
        "python_gc_collections_total_rate": f'rate(python_gc_collections_total{{instance=~".*:8000"}}[2m]) or vector(0)',
        "python_gc_objects_collected_total_rate": f'rate(python_gc_objects_collected_total{{instance=~".*:8000"}}[2m]) or vector(0)',

        # 3. HTTP performance - Request processing rates and latency accumulation
        "http_request_duration_seconds_sum_rate": f'rate(http_request_duration_seconds_sum{{instance=~".*:8000"}}[2m]) or vector(0)',
        "http_requests_total_rate": f'rate(http_requests_total{{instance=~".*:8000"}}[2m]) or vector(0)',
        "http_request_duration_seconds_count_rate": f'rate(http_request_duration_seconds_count{{instance=~".*:8000"}}[2m]) or vector(0)',

        # 4. I/O load - Current number of open file descriptors (gauge metric)
        "process_open_fds": f'process_open_fds{{instance=~".*:8000"}} or vector(10)',

        # 5. Network payload - Response and request payload rates
        "http_response_size_bytes_sum_rate": f'rate(http_response_size_bytes_sum{{instance=~".*:8000"}}[2m]) or vector(0)',
        "http_request_size_bytes_count_rate": f'rate(http_request_size_bytes_count{{instance=~".*:8000"}}[2m]) or vector(0)',

        # RESOURCE AWARENESS: Get actual resource limits/requests for contextual scaling decisions
        "kube_pod_container_resource_limits_cpu": f'kube_pod_container_resource_limits{{deployment="{services.config.kubernetes.target_deployment}",namespace="{services.config.kubernetes.target_namespace}",resource="cpu"}} or vector(0.5)',
        "kube_pod_container_resource_limits_memory": f'kube_pod_container_resource_limits{{deployment="{services.config.kubernetes.target_deployment}",namespace="{services.config.kubernetes.target_namespace}",resource="memory"}} or vector(1073741824)',
        "kube_pod_container_resource_requests_cpu": f'kube_pod_container_resource_requests{{deployment="{services.config.kubernetes.target_deployment}",namespace="{services.config.kubernetes.target_namespace}",resource="cpu"}} or vector(0.2)',
        "kube_pod_container_resource_requests_memory": f'kube_pod_container_resource_requests{{deployment="{services.config.kubernetes.target_deployment}",namespace="{services.config.kubernetes.target_namespace}",resource="memory"}} or vector(536870912)',
    }

    # Debug logging for consumer metric targeting
    logger.info(f"CONSUMER_METRICS: targeting_instance_pattern=.*:8000 (consumer_pods)")
    logger.info(f"FEATURE_ARCHITECTURE: 9_scientifically_selected_features cpu=1 memory=2 http=5 io=1")
    
    # Log key queries for debugging consumer metric collection
    logger.debug(f"CPU_RATE_QUERY: {queries['process_cpu_seconds_total_rate']}")
    logger.debug(f"HTTP_RATE_QUERY: {queries['http_requests_total_rate']}")
    logger.debug(f"IO_GAUGE_QUERY: {queries['process_open_fds']}")

    tasks = {name: services.prometheus_client.query(query) for name, query in queries.items()}

    # Also get current replicas from Kubernetes (still needed for scaling decisions)
    current_replicas_query = f'kube_deployment_status_replicas{{deployment="{services.config.kubernetes.target_deployment}",namespace="{services.config.kubernetes.target_namespace}"}}'
    tasks['current_replicas'] = services.prometheus_client.query(current_replicas_query)
    
    # Debug query to see what consumer pods/jobs are actually being targeted
    debug_consumer_query = f'up{{instance=~".*:8000"}}'
    tasks['debug_consumer_targets'] = services.prometheus_client.query_raw(debug_consumer_query)

    results = await asyncio.gather(*tasks.values())

    metrics = dict(zip(tasks.keys(), results))
    current_replicas = int(metrics.pop('current_replicas', 1))
    debug_consumer_result = metrics.pop('debug_consumer_targets', [])
    
    # Log debug information about targeted consumer pods/jobs
    if debug_consumer_result and isinstance(debug_consumer_result, list) and len(debug_consumer_result) > 0:
        consumer_instances = [result.get('metric', {}).get('instance', 'unknown') for result in debug_consumer_result if 'metric' in result]
        consumer_jobs = [result.get('metric', {}).get('job', 'unknown') for result in debug_consumer_result if 'metric' in result]
        unique_instances = list(set(consumer_instances))
        unique_jobs = list(set(consumer_jobs))
        logger.info(f"CONSUMER_TARGETS: instances={unique_instances} jobs={unique_jobs} "
                    f"consumer_count={len(unique_instances)} pattern=.*:8000")
    else:
        logger.warning(f"CONSUMER_TARGETS: no_consumer_instances_found query_result={debug_consumer_result}")
        logger.warning(f"CONSUMER_TARGETS: check_consumer_pods_expose_metrics_on_port_8000")

    # Ensure consumer performance features have defaults
    metrics = _ensure_consumer_features(metrics)
    
    # Ensure resource configuration metrics have defaults  
    metrics = _ensure_resource_config_features(metrics)

    CURRENT_REPLICAS_GAUGE.set(current_replicas)

    # Enhanced logging with consumer performance metrics
    cpu_rate = metrics.get('process_cpu_seconds_total_rate', 0)
    gc_collections_rate = metrics.get('python_gc_collections_total_rate', 0)
    gc_objects_rate = metrics.get('python_gc_objects_collected_total_rate', 0)
    http_requests_rate = metrics.get('http_requests_total_rate', 0)
    http_duration_rate = metrics.get('http_request_duration_seconds_sum_rate', 0)
    http_count_rate = metrics.get('http_request_duration_seconds_count_rate', 0)
    open_fds = metrics.get('process_open_fds', 10)
    response_size_rate = metrics.get('http_response_size_bytes_sum_rate', 0)
    request_count_rate = metrics.get('http_request_size_bytes_count_rate', 0)

    logger.info(f"CONSUMER_PERFORMANCE: cpu_rate={cpu_rate:.4f} gc_collections_rate={gc_collections_rate:.4f} "
                f"gc_objects_rate={gc_objects_rate:.4f} open_fds={open_fds:.0f}")

    logger.info(f"HTTP_PERFORMANCE: requests_rate={http_requests_rate:.4f} duration_rate={http_duration_rate:.4f} "
                f"count_rate={http_count_rate:.4f} response_size_rate={response_size_rate:.4f} "
                f"request_count_rate={request_count_rate:.4f}")

    logger.info(f"SCALING_CONTEXT: current_replicas={current_replicas} targeting_consumer_pods=True "
                f"feature_source=consumer_pod_metrics")

    # Consumer performance analysis
    logger.info(f"PERFORMANCE_ANALYSIS: cpu_utilization={cpu_rate:.4f}/sec "
                f"memory_pressure_gc_rate={gc_collections_rate + gc_objects_rate:.4f}/sec "
                f"http_throughput={http_requests_rate:.4f}req/sec "
                f"io_load={open_fds:.0f}fds")

    # Log feature availability for debugging
    consumer_features_available = sum(1 for feat in services.config.base_features if feat in metrics)
    total_features_available = consumer_features_available

    logger.info(f"FEATURE_AVAILABILITY: total={total_features_available}/{len(services.config.feature_order)} "
                f"consumer_performance={consumer_features_available}/{len(services.config.base_features)}")

    # Debug: Log missing features if any
    if total_features_available < len(services.config.feature_order):
        missing_consumer = [feat for feat in services.config.base_features if feat not in metrics]
        if missing_consumer:
            logger.warning(f"MISSING_CONSUMER_FEATURES: {missing_consumer}")
            logger.warning(f"CHECK_CONSUMER_PODS: ensure consumer pods expose metrics on /metrics endpoint")

    logger.info("=" * 60)
    logger.info(f"NODE_END: {node_name}")
    logger.info("=" * 60)
    return {"current_metrics": metrics, "current_replicas": current_replicas}


# --- Feature Engineering Helper ---
def _ensure_consumer_features(metrics: Dict[str, float]) -> Dict[str, float]:
    """Ensure consumer performance features have fallback values for the 9 scientifically selected features."""

    # Consumer performance feature defaults (scientifically selected)
    feature_defaults = {
        # CPU utilization rate (default: minimal CPU usage)
        'process_cpu_seconds_total_rate': 0.01,  # 1% CPU usage rate per second
        
        # Memory pressure indicators (default: low GC activity)
        'python_gc_collections_total_rate': 0.1,   # Low garbage collection rate
        'python_gc_objects_collected_total_rate': 1.0,  # Minimal object collection rate
        
        # HTTP performance rates (default: low but healthy activity)
        'http_request_duration_seconds_sum_rate': 0.05,    # Low latency accumulation
        'http_requests_total_rate': 1.0,                   # 1 request per second baseline
        'http_request_duration_seconds_count_rate': 1.0,   # 1 request count per second
        
        # I/O load (default: reasonable file descriptor count for a Python service)
        'process_open_fds': 10.0,  # 10 open file descriptors (typical baseline)
        
        # Network payload rates (default: minimal but present)
        'http_response_size_bytes_sum_rate': 1024.0,  # 1KB per second response rate
        'http_request_size_bytes_count_rate': 1.0,    # 1 request count per second
    }

    # Apply defaults for missing consumer performance features only
    for feature, default_value in feature_defaults.items():
        if feature not in metrics or metrics[feature] == 0:
            metrics[feature] = default_value
            logger.debug(f"CONSUMER_FEATURE_DEFAULT: {feature}={default_value} (metric_missing_or_zero)")

    return metrics


def _ensure_resource_config_features(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Ensure resource configuration features have reasonable defaults based on typical consumer deployment.
    Uses defaults matching the consumer deployment configuration: CPU limit=500m, memory limit=1Gi.
    """
    defaults = {
        'kube_pod_container_resource_limits_cpu': 0.5,        # 500m CPU limit
        'kube_pod_container_resource_limits_memory': 1073741824,  # 1Gi memory limit
        'kube_pod_container_resource_requests_cpu': 0.2,      # 200m CPU request  
        'kube_pod_container_resource_requests_memory': 536870912,  # 512Mi memory request
    }
    
    for feature, default_value in defaults.items():
        if feature not in metrics or metrics[feature] is None or np.isnan(metrics[feature]):
            metrics[feature] = default_value
            logger.debug(f"RESOURCE_CONFIG: {feature}=default({default_value})")
    
    return metrics


async def get_dqn_recommendation(state: ScalingState, services: ServiceContainer) -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("NODE_START: get_dqn_recommendation")
    logger.info("=" * 60)
    if not services.scaler:
        logger.error("DQN_PREDICTION: scaler_not_loaded")
        return {"error": "SCALER_NOT_LOADED"}

    metrics = state['current_metrics'].copy()
    current_replicas = state['current_replicas']

    # Scale the 9 scientifically-selected base features from Prometheus with proper feature names
    raw_features = [metrics.get(feat, 0.0) for feat in services.config.base_features]

    # Transform with feature names to match how scaler was fitted
    import pandas as pd
    raw_features_df = pd.DataFrame([raw_features], columns=services.config.base_features)
    scaled_raw_features = services.scaler.transform(raw_features_df)

    # Use only the scaled raw features for DQN (9 scientifically-selected features)
    final_feature_vector = scaled_raw_features[0].tolist()

    logger.info(f"FEATURE_COMPOSITION: scaled_raw={len(scaled_raw_features[0])} "
                f"total={len(final_feature_vector)}")

    # Count available feature types (check existence, not value > 0)
    base_available = sum(1 for feat in services.config.base_features if feat in metrics)

    # Log consumer performance state
    cpu_rate = metrics.get('process_cpu_seconds_total_rate', 0.01)
    gc_pressure = metrics.get('python_gc_collections_total_rate', 0.1) + metrics.get('python_gc_objects_collected_total_rate', 1.0)
    http_requests_rate = metrics.get('http_requests_total_rate', 1.0)
    open_fds = metrics.get('process_open_fds', 10.0)

    logger.info(f"CONSUMER_STATE: replicas={current_replicas} "
                f"cpu_rate={cpu_rate:.4f}/sec gc_pressure={gc_pressure:.2f}/sec "
                f"http_rate={http_requests_rate:.2f}/sec open_fds={open_fds:.0f}")

    # Calculate per-replica metrics for load assessment
    cpu_per_replica = cpu_rate / max(current_replicas, 1)
    gc_per_replica = gc_pressure / max(current_replicas, 1)
    http_per_replica = http_requests_rate / max(current_replicas, 1)
    fds_per_replica = open_fds / max(current_replicas, 1)

    logger.info(f"PER_REPLICA_LOAD: "
                f"cpu={cpu_per_replica:.4f}/sec gc={gc_per_replica:.2f}/sec "
                f"http={http_per_replica:.2f}/sec fds={fds_per_replica:.1f}")



    try:
        if services.dqn_model is not None:
            # Use local PyTorch model with epsilon-greedy exploration
            current_epsilon = services.get_epsilon()

            device = next(services.dqn_model.parameters()).device
            input_tensor = torch.FloatTensor([final_feature_vector]).to(device)

            # Set model to evaluation mode to disable BatchNorm training behavior
            services.dqn_model.eval()

            with torch.no_grad():
                # Ensure proper batch dimension for BatchNorm layers
                if input_tensor.dim() == 1:
                    input_tensor = input_tensor.unsqueeze(0)
                elif input_tensor.shape[0] == 1 and len(input_tensor.shape) == 2:
                    # Already has batch dimension, keep as is
                    pass

                q_values = services.dqn_model(input_tensor).cpu().numpy().flatten()

            # ENHANCED: Boost exploration when over-provisioned to encourage scale-down learning
            effective_epsilon = current_epsilon
            if current_replicas >= 8:  # Over-provisioned scenario
                effective_epsilon = max(current_epsilon, 0.15)  # Minimum 15% exploration
                logger.info(
                    f"DQN_EXPLORATION_BOOST: over_provisioned={current_replicas} epsilon_boosted={current_epsilon:.3f}=>{effective_epsilon:.3f}")

            # Epsilon-greedy exploration with detailed reasoning
            if random.random() < effective_epsilon:
                action_index = random.randint(0, 2)  # Random action
                exploration_type = "exploration"
                logger.info(f"DQN_EXPLORATION: random_action epsilon={effective_epsilon:.3f}")
            else:
                action_index = np.argmax(q_values)  # Greedy action
                exploration_type = "exploitation"
                logger.info(f"DQN_EXPLOITATION: best_action epsilon={effective_epsilon:.3f}")

            # Update epsilon and decision count through ServiceContainer
            updated_epsilon = services.update_epsilon(services.config.dqn.epsilon_decay, services.config.dqn.epsilon_end)
            decision_count = services.increment_decision_count()

            action_map = {0: "Scale Down", 1: "Keep Same", 2: "Scale Up"}
            dqn_action_name = action_map.get(action_index, "Unknown")

            # Use DQN decision directly
            action_name = dqn_action_name

            # Generate comprehensive explanation using explainable AI system
            explanation = decision_reasoning.explain_dqn_decision(
                metrics=metrics,
                q_values=q_values.tolist(),
                action_name=action_name,
                exploration_type=exploration_type,
                epsilon=updated_epsilon,
                current_replicas=current_replicas
            )

            # Log detailed reasoning
            decision_reasoning.log_decision_reasoning(explanation, metrics, q_values.tolist())

            # Enhanced logging with confidence and reasoning
            confidence = explanation['confidence_metrics']['decision_confidence']
            risk_level = explanation['risk_assessment']
            confidence_gap = explanation['confidence_metrics']['confidence_gap']

            # DIAGNOSTIC: Enhanced Q-value logging for scale-down debugging
            scale_down_q = q_values[0]
            keep_same_q = q_values[1]
            scale_up_q = q_values[2]

            logger.info(
                f"DQN_Q_VALUES_DETAILED: ScaleDown={scale_down_q:.4f} KeepSame={keep_same_q:.4f} ScaleUp={scale_up_q:.4f}")
            logger.info(
                f"DQN_Q_ANALYSIS: winner={'ScaleDown' if scale_down_q == max(q_values) else 'KeepSame' if keep_same_q == max(q_values) else 'ScaleUp'} "
                f"margin={max(q_values) - sorted(q_values, reverse=True)[1]:.4f}")
            logger.info(f"DQN_REPLICA_PRESSURE: current={current_replicas} "
                        f"scale_down_advantage={scale_down_q - keep_same_q:.4f} "
                        f"should_scale_down={current_replicas > 5 and scale_down_q > keep_same_q}")

            logger.info(f"DQN_DECISION: action={action_name} confidence={confidence} risk={risk_level}")
            logger.info(f"DQN_Q_VALUES: values=[{','.join(f'{q:.3f}' for q in q_values)}] gap={confidence_gap:.3f}")
            logger.info(f"DQN_REASONING: factors_count={len(explanation['reasoning_factors'])}")

            # Update Prometheus metrics
            DQN_DECISIONS_COUNTER.inc()
            DQN_EPSILON_GAUGE.set(updated_epsilon)

            # Convert confidence string to numeric value for Prometheus
            confidence_numeric = 0.5  # Default medium confidence
            if isinstance(confidence, str):
                if confidence.lower() == 'high':
                    confidence_numeric = 1.0
                elif confidence.lower() == 'medium':
                    confidence_numeric = 0.5
                elif confidence.lower() == 'low':
                    confidence_numeric = 0.0
            else:
                confidence_numeric = float(confidence)

            DQN_DECISION_CONFIDENCE_GAUGE.set(confidence_numeric)
            DQN_Q_VALUE_SCALE_UP_GAUGE.set(q_values[2])  # Scale up is action 2
            DQN_Q_VALUE_SCALE_DOWN_GAUGE.set(q_values[0])  # Scale down is action 0
            DQN_Q_VALUE_KEEP_SAME_GAUGE.set(q_values[1])  # Keep same is action 1

            # Update action counters
            if action_name == "Scale Up":
                DQN_ACTION_SCALE_UP_COUNTER.inc()
            elif action_name == "Scale Down":
                DQN_ACTION_SCALE_DOWN_COUNTER.inc()
            else:  # Keep Same
                DQN_ACTION_KEEP_SAME_COUNTER.inc()

            # Update exploration/exploitation counters
            if exploration_type == "exploration":
                DQN_EXPLORATION_COUNTER.inc()
            else:
                DQN_EXPLOITATION_COUNTER.inc()



            # Log specific reasoning for this decision
            if services.config.ai.enable_detailed_reasoning:
                logger.info("DQN_FACTORS: summary_top3")
                for i, factor in enumerate(explanation['reasoning_factors'][:3]):  # Top 3 factors
                    logger.info(f"DQN_FACTOR_{i + 1}: {factor}")

            experience_update = {"state": metrics, "action": action_name}

            logger.info("=" * 60)
            logger.info("NODE_END: get_dqn_recommendation")
            logger.info("=" * 60)
            return {
                "dqn_prediction": {
                    "action_name": action_name,
                    "q_values": q_values.tolist(),
                    "epsilon": updated_epsilon,
                    "exploration_type": exploration_type,
                    "explanation": explanation,
                    "confidence": confidence,
                    "risk_assessment": risk_level
                },
                "experience": experience_update
            }
        else:
            # Enhanced fallback: Rule-based logic with consumer performance features
            logger.info("DQN_FALLBACK: using_rule_based_logic model_not_available")

            # Get current consumer performance metrics for fallback decision
            cpu_rate = metrics.get('process_cpu_seconds_total_rate', 0.01)
            gc_collections_rate = metrics.get('python_gc_collections_total_rate', 0.1)
            gc_objects_rate = metrics.get('python_gc_objects_collected_total_rate', 1.0)
            http_requests_rate = metrics.get('http_requests_total_rate', 1.0)
            http_duration_rate = metrics.get('http_request_duration_seconds_sum_rate', 0.05)
            http_count_rate = metrics.get('http_request_duration_seconds_count_rate', 1.0)
            open_fds = metrics.get('process_open_fds', 10.0)
            response_size_rate = metrics.get('http_response_size_bytes_sum_rate', 1024.0)
            request_count_rate = metrics.get('http_request_size_bytes_count_rate', 1.0)

            # Generate analysis for fallback decision
            analysis = decision_reasoning.analyze_metrics(metrics, current_replicas)

            # Enhanced rule-based decision with consumer performance metrics
            reasoning_factors = []

            # Decision logic based on consumer pod performance
            total_gc_pressure = gc_collections_rate + gc_objects_rate
            latency_per_request = http_duration_rate / max(http_requests_rate, 0.1)
            fds_per_replica = open_fds / max(current_replicas, 1)

            if cpu_rate > 0.15 or total_gc_pressure > 3.0 or latency_per_request > 0.08:  # Performance pressure
                action_name = "Scale Up"
                reasoning_factors.append(
                    f"Performance pressure detected: {cpu_rate:.4f} CPU rate, {total_gc_pressure:.2f} GC pressure, {latency_per_request:.4f}s latency per request")
                reasoning_factors.append("Scaling up to improve consumer pod performance")
                risk_level = "high"
            elif http_requests_rate > 15.0 or open_fds > 40:  # High load indicators
                action_name = "Scale Up"
                reasoning_factors.append(
                    f"High load detected: {http_requests_rate:.2f} requests/sec, {open_fds:.0f} file descriptors")
                reasoning_factors.append("Scaling up to handle increased traffic and I/O load")
                risk_level = "medium"
            elif cpu_rate < 0.02 and total_gc_pressure < 0.5 and http_requests_rate < 1.0 and current_replicas > 1:  # Low utilization
                action_name = "Scale Down"
                reasoning_factors.append(
                    f"Low utilization: {cpu_rate:.4f} CPU rate, {total_gc_pressure:.2f} GC pressure, {http_requests_rate:.2f} requests/sec")
                reasoning_factors.append(
                    f"Consumer pods are underutilized with {current_replicas} replicas - cost optimization opportunity")
                risk_level = "low"
            # ENHANCED: Add explicit scale-down for high replica counts with low consumer activity
            elif current_replicas >= 8 and cpu_rate < 0.05 and http_requests_rate < 2.0 and fds_per_replica < 12:  # Over-provisioned
                action_name = "Scale Down"
                reasoning_factors.append(
                    f"Over-provisioned: {current_replicas} replicas with low consumer activity - {cpu_rate:.4f} CPU, {http_requests_rate:.2f} req/s")
                reasoning_factors.append(
                    f"Consumer performance indicators show excess capacity: {fds_per_replica:.1f} FDs per replica")
                risk_level = "low"
            else:
                action_name = "Keep Same"
                reasoning_factors.append(
                    f"Balanced consumer state: {cpu_rate:.4f} CPU rate, {http_requests_rate:.2f} req/s, {current_replicas} replicas")
                reasoning_factors.append("Consumer pods operating within acceptable performance parameters")
                risk_level = "low"

            # Create fallback explanation
            fallback_explanation = {
                'timestamp': datetime.now().isoformat(),
                'decision_type': 'FALLBACK_RULE_BASED',
                'recommended_action': action_name,
                'exploration_strategy': 'rule_based',
                'confidence_metrics': {
                    'decision_confidence': 'medium',
                    'rule_based': True
                },
                'reasoning_factors': reasoning_factors,
                'risk_assessment': risk_level,
                'system_analysis': analysis
            }

            logger.info(f"FALLBACK_DECISION: action={action_name} risk={risk_level}")
            logger.info(f"FALLBACK_REASONING: {reasoning_factors[0]}")

            if services.config.ai.enable_detailed_reasoning:
                logger.info("FALLBACK_ANALYSIS: detailed_factors")
                for i, factor in enumerate(reasoning_factors):
                    logger.info(f"FALLBACK_FACTOR_{i + 1}: {factor}")

            experience_update = {"state": metrics, "action": action_name}

            logger.info("=" * 60)
            logger.info("NODE_END: get_dqn_recommendation")
            logger.info("=" * 60)
            return {
                "dqn_prediction": {
                    "action_name": action_name,
                    "q_values": [0.0, 1.0, 0.0],
                    "explanation": fallback_explanation,
                    "confidence": "medium",
                    "risk_assessment": risk_level,
                    "fallback_mode": True
                },
                "experience": experience_update
            }

    except Exception as e:
        logger.error(f"DQN inference failed: {e}", exc_info=True)
        logger.info("=" * 60)
        logger.info("NODE_END: get_dqn_recommendation")
        logger.info("=" * 60)
        return {"error": f"DQN_INFERENCE_FAILED: {e}"}


async def validate_with_llm(state: ScalingState, services: ServiceContainer) -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("NODE_START: validate_with_llm")
    logger.info("=" * 60)

    # Check if LLM validation is enabled
    if not services.config.ai.enable_llm_validation:
        logger.info("SAFETY_MONITOR: disabled_by_config skipping")
        return {"llm_validation_response": {"approved": True, "reason": "Safety monitor disabled by configuration.",
                                            "confidence": "medium"}}

    if not services.validator_agent:
        if services.config.ai.enable_llm_validation:
            logger.warning("SAFETY_MONITOR: agent_not_initialized skipping")
            return {"llm_validation_response": {"approved": True,
                                                "reason": "Safety monitor enabled but agent not available.",
                                                "confidence": "low"}}
        else:
            # This case should not happen since we check ENABLE_LLM_VALIDATION earlier, but adding for safety
            logger.debug("SAFETY_MONITOR: agent_not_initialized validation_disabled")
            return {"llm_validation_response": {"approved": True, "reason": "Safety monitor disabled by configuration.",
                                                "confidence": "medium"}}

    # Handle missing dqn_prediction gracefully
    dqn_prediction = state.get('dqn_prediction', {'action_name': 'Keep Same'})
    action_name = dqn_prediction.get('action_name', 'Keep Same')
    dqn_confidence = dqn_prediction.get('confidence', 'unknown')
    dqn_risk = dqn_prediction.get('risk_assessment', 'unknown')
    dqn_explanation = dqn_prediction.get('explanation', {})

    # Get metrics for LLM context
    metrics = state['current_metrics']

    # Create comprehensive validation prompt
    prompt = f"""You are a SAFETY MONITOR for Kubernetes autoscaling. Your ONLY role is to detect and prevent EXTREME or DANGEROUS scaling decisions that could harm the cluster.

🚨 CRITICAL: Only intervene for EXTREME decisions. Allow normal DQN learning to proceed uninterrupted.

DQN SCALING DECISION TO EVALUATE:
- Recommended Action: {action_name}
- Current Replicas: {state['current_replicas']}
- DQN Confidence: {dqn_confidence}
- DQN Risk Assessment: {dqn_risk}

DQN REASONING FACTORS:
{chr(10).join(f"- {factor}" for factor in dqn_explanation.get('reasoning_factors', ['No DQN reasoning available']))}

CURRENT SYSTEM METRICS:
- Pod Readiness: {metrics.get('kube_pod_container_status_ready', 1.0):.1%}
- Unavailable Replicas: {metrics.get('kube_deployment_status_replicas_unavailable', 0)}
- CPU Limits: {metrics.get('kube_pod_container_resource_limits_cpu', 0):.2f} cores
- Memory Limits: {metrics.get('kube_pod_container_resource_limits_memory', 0)/1000000:.0f} MB
- Running Containers: {metrics.get('kube_pod_container_status_running', 0)}

🔍 EXTREME DECISION CRITERIA (Block these ONLY):

1. **RUNAWAY SCALING**: Scaling to >15 replicas without clear justification
2. **RESOURCE EXHAUSTION**: Scaling up when cluster resources are constrained  
3. **MASSIVE OVER-SCALING**: Requesting 3x+ more replicas than needed (consider system capacity)
4. **DANGEROUS DOWN-SCALING**: Scaling to 0 or very low when system shows stress
5. **RAPID OSCILLATION**: Frequent large scaling changes without stabilization
6. **IGNORES HIGH RISK**: DQN marked decision as "high" risk but proceeding anyway

⚠️  **DEFAULT: APPROVE** - Only block if decision meets extreme criteria above.

🛡️  **MANDATORY KUBERNETES VERIFICATION**: You MUST use the available Kubernetes MCP tools to verify cluster state before making any safety decision.

TARGET DEPLOYMENT DETAILS:
- Namespace: {services.config.kubernetes.target_namespace}
- Deployment Name: {services.config.kubernetes.target_deployment}
- Current Replicas: {state['current_replicas']}

REQUIRED ASSESSMENT PROTOCOL:
1. **USE RELEVANT TOOLS ONLY** - Choose the tools that are most relevant for this specific safety decision:
   
   **Available Tools:**
   - `mcp_kubernetes_pods_list` - Check current pod status and health in "{services.config.kubernetes.target_namespace}" namespace
   - `mcp_kubernetes_pods_top` - Verify actual resource consumption for "{services.config.kubernetes.target_deployment}" pods  
   - `mcp_kubernetes_resources_get` - Check deployment status (apiVersion: apps/v1, kind: Deployment, name: {services.config.kubernetes.target_deployment}, namespace: {services.config.kubernetes.target_namespace})
   - `mcp_kubernetes_events_list` - Look for recent cluster issues or warnings in "{services.config.kubernetes.target_namespace}" namespace

   **Tool Selection Guide:**
   - For scaling decisions: Use `mcp_kubernetes_resources_get` to verify current deployment state
   - For resource concerns: Use `mcp_kubernetes_pods_top` to check actual resource consumption
   - For stability issues: Use `mcp_kubernetes_events_list` to check for recent problems
   - For health verification: Use `mcp_kubernetes_pods_list` to verify pod status

2. **EFFICIENT VERIFICATION**: Use only 1-2 tools that provide the most relevant data for your safety assessment

3. **MAKE INFORMED DECISION** based on:
   - Real-time cluster data from the tools you chose to use
   - Provided DQN metrics and reasoning

CRITICAL REQUIREMENTS:
1. You MUST use at least 1 relevant MCP Kubernetes tool before making a decision
2. Choose tools based on what's most important for the specific scaling decision being evaluated
3. Set "cluster_check_performed": true in your response (required)
4. Include specific findings from the tools you used in your reasoning
5. If tools show different data than provided metrics, prioritize tool data

IMPORTANT: All tools are READ-ONLY. You cannot modify the cluster - only observe and assess.

CRITICAL: Respond with ONLY a valid JSON object. No markdown, no explanations.

Example for NORMAL decision (approve):
{{
    "approved": true,
    "confidence": "high",
    "reasoning": "Verified with mcp_kubernetes_resources_get: deployment shows 3 healthy replicas. DQN scale-down decision is safe.",
    "safety_risk": "none",
    "extreme_factors": [],
    "cluster_check_performed": true,
    "tool_findings": ["Deployment has 3 healthy replicas", "No resource pressure indicated"]
}}

Example for EXTREME decision (block):
{{
    "approved": false,
    "confidence": "high",
    "reasoning": "EXTREME: mcp_kubernetes_events_list shows OutOfMemory errors. Scaling up to 25 replicas would exhaust cluster resources.",
    "safety_risk": "high",
    "extreme_factors": ["runaway_scaling", "resource_exhaustion_risk"],
    "alternative_suggestion": "Investigate memory issues before scaling. Consider 8-10 replicas maximum.",
    "cluster_check_performed": true,
    "tool_findings": ["OutOfMemory events detected", "Cluster showing resource pressure"]
}}

Your JSON response:"""

    try:
        logger.info(f"SAFETY_MONITOR: evaluating action={action_name} dqn_confidence={dqn_confidence}")

        # Invoke the validator agent
        response = await services.validator_agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
        last_message = response['messages'][-1].content

        logger.info(f"SAFETY_MONITOR: response_received chars={len(last_message)}")

        # Debug: Log first 200 chars of response to see what we're getting
        if len(last_message) > 0:
            logger.debug(f"SAFETY_RAW_RESPONSE: {last_message[:200]}...")

        # Enhanced JSON parsing with fallback
        validation_result = parse_llm_json_response(last_message, action_name)

        # Enhanced logging with safety assessment outcome
        safety_status = "BLOCKED" if not validation_result['approved'] else "APPROVED"
        safety_risk = validation_result.get('safety_risk', 'none')
        cluster_verified = validation_result.get('cluster_check_performed', False)
        tool_findings = validation_result.get('tool_findings', [])
        
        logger.info(f"SAFETY_MONITOR: {safety_status} risk_level={safety_risk} cluster_verified={cluster_verified}")
        
        # Log tool usage compliance
        if cluster_verified:
            logger.info(f"TOOL_VERIFICATION: completed findings_count={len(tool_findings)}")
            if tool_findings:
                for i, finding in enumerate(tool_findings[:3]):  # Log first 3 findings
                    logger.info(f"TOOL_FINDING_{i+1}: {finding}")
        else:
            logger.warning("TOOL_VERIFICATION: not_performed - LLM did not use required Kubernetes tools")
            logger.warning("TOOL_COMPLIANCE: failed - safety assessment may be incomplete")

        # Log safety details only if decision was blocked or high risk detected
        if not validation_result['approved'] or safety_risk in ['high', 'medium']:
            logger.warning("SAFETY_ALERT: detailed_analysis")
            logger.warning(f"SAFETY_DECISION: approved={validation_result['approved']} "
                           f"confidence={validation_result['confidence']} "
                           f"risk={safety_risk}")

            logger.warning(f"SAFETY_REASONING: {validation_result['reasoning']}")

            if validation_result.get('extreme_factors'):
                logger.warning(f"EXTREME_FACTORS: count={len(validation_result['extreme_factors'])}")
                for i, factor in enumerate(validation_result['extreme_factors']):
                    logger.warning(f"EXTREME_FACTOR_{i + 1}: {factor}")

            if validation_result.get('alternative_suggestion'):
                logger.info(f"SAFETY_ALTERNATIVE: {validation_result['alternative_suggestion']}")
        else:
            # Normal approval - minimal logging to avoid noise
            logger.info(f"SAFETY_APPROVED: confidence={validation_result['confidence']}")

        if not validation_result['approved']:
            logger.error(f"SCALING_BLOCKED: reason={validation_result['reasoning']}")
            if validation_result.get('alternative_suggestion'):
                logger.info(f"SUGGESTED_ALTERNATIVE: {validation_result['alternative_suggestion']}")

        logger.info("=" * 60)
        logger.info("NODE_END: validate_with_llm")
        logger.info("=" * 60)
        return {"llm_validation_response": validation_result}

    except Exception as e:
        logger.error(f"SAFETY_MONITOR: failed error={e}")

        # Check if this is a permission-related error or JSON parsing issue
        error_str = str(e).lower()
        is_permission_error = any(keyword in error_str for keyword in
                                  ['permission', 'forbidden', 'unauthorized', 'rbac', 'access denied'])
        is_parsing_error = 'json' in error_str or 'parse' in error_str

        if is_permission_error:
            logger.warning("SAFETY_MONITOR: permission_error_detected suggesting_disable")
            logger.warning("SAFETY_MONITOR: consider_setting ENABLE_LLM_VALIDATION=false")
        elif is_parsing_error:
            logger.warning("SAFETY_MONITOR: json_parsing_issues_detected")
            logger.warning("SAFETY_MONITOR: llm_not_following_json_format")

        # Enhanced fallback - APPROVE by default for safety (preserve DQN learning)
        fallback_result = {
            "approved": True,  # Safety-first: only block explicitly dangerous decisions
            "confidence": "low",
            "reasoning": f"Safety monitor error: {str(e)}. Unable to verify cluster state with tools. Defaulting to APPROVAL to preserve DQN learning.",
            "safety_risk": "unknown",
            "extreme_factors": ["safety_monitor_unavailable", "tool_verification_failed"],
            "alternative_suggestion": "Monitor system closely - cluster verification tools unavailable",
            "cluster_check_performed": False,
            "validation_score": 0.3,
            "fallback_mode": True,
            "permission_error": is_permission_error,
            "tool_findings": ["Tool verification failed due to error"]
        }

        logger.warning("SAFETY_MONITOR: fallback_approved preserving_dqn_learning")
        logger.info("=" * 60)
        logger.info("NODE_END: validate_with_llm")
        logger.info("=" * 60)
        return {"llm_validation_response": fallback_result}


def parse_llm_json_response(response_text: str, action_name: str) -> Dict[str, Any]:
    """Parse LLM JSON response with robust error handling and safety-first fallbacks."""
    try:
        import re

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


async def direct_scale_deployment(desired_replicas: int, services: ServiceContainer) -> bool:
    """
    🎯 SIMPLE SCALING: Directly scale the deployment without KEDA/HPA complexity
    """
    try:
        # Import kubernetes client
        import kubernetes.client
        import kubernetes.config
        from kubernetes.client.rest import ApiException

    except ImportError as e:
        logger.error(f"DIRECT_SCALING: IMPORT_ERROR kubernetes_client_not_available error={e}")
        logger.error("DIRECT_SCALING: install_kubernetes_client: pip install kubernetes")
        return False

    try:
        # Create k8s API client
        kubernetes.config.load_incluster_config()  # We're running inside the cluster
        apps_v1 = kubernetes.client.AppsV1Api()

        # Get target deployment info from config
        target_deployment = services.config.kubernetes.target_deployment
        target_namespace = services.config.kubernetes.target_namespace

        # Get current deployment
        deployment = apps_v1.read_namespaced_deployment(
            name=target_deployment,
            namespace=target_namespace
        )

        current_replicas = deployment.spec.replicas

        # Only scale if there's a difference
        if current_replicas != desired_replicas:
            # Update replica count
            deployment.spec.replicas = desired_replicas

            # Patch the deployment
            apps_v1.patch_namespaced_deployment(
                name=target_deployment,
                namespace=target_namespace,
                body=deployment
            )

            logger.info(f"DIRECT_SCALING: SUCCESS {current_replicas}=>{desired_replicas} replicas")
            return True
        else:
            logger.info(f"DIRECT_SCALING: NO_CHANGE replicas={current_replicas}")
            return True

    except ApiException as e:
        logger.error(f"DIRECT_SCALING: API_ERROR {e.status}:{e.reason}")
        return False
    except Exception as e:
        logger.error(f"DIRECT_SCALING: FAILED error={e}")
        return False


async def plan_final_action(state: ScalingState, services: ServiceContainer) -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("NODE_START: plan_final_action")
    logger.info("=" * 60)
    current_replicas = state['current_replicas']

    # Handle missing dqn_prediction gracefully
    dqn_prediction = state.get('dqn_prediction', {'action_name': 'Keep Same'})
    action_name = dqn_prediction.get('action_name', 'Keep Same')
    dqn_confidence = dqn_prediction.get('confidence', 'unknown')
    dqn_risk = dqn_prediction.get('risk_assessment', 'unknown')

    # Get LLM validation results
    llm_validation = state.get('llm_validation_response', {})
    llm_approved = llm_validation.get('approved', True)
    llm_confidence = llm_validation.get('confidence', 'unknown')
    llm_reasoning = llm_validation.get('reasoning', 'No validation reasoning available')

    # Calculate new replica count
    new_replicas = current_replicas
    if action_name == 'Scale Up':
        new_replicas += 1
    elif action_name == 'Scale Down':
        new_replicas -= 1

    # SAFETY OVERRIDE: If safety monitor blocks the decision, keep current replicas
    if not llm_approved:
        logger.warning(f"SAFETY_OVERRIDE: blocking_{action_name} maintaining_current_replicas={current_replicas}")
        final_decision = current_replicas  # KEEP CURRENT - NO SCALING
    else:
        final_decision = max(1, min(20, new_replicas))  # Normal decision with safety constraints

    # Ensure we never go below 1 replica regardless of safety decisions
    final_decision = max(1, final_decision)

    # Create comprehensive decision explanation AFTER final_decision is known
    decision_explanation = {
        'timestamp': datetime.now().isoformat(),
        'decision_pipeline': {
            'dqn_recommendation': {
                'action': action_name,
                'confidence': dqn_confidence,
                'risk_assessment': dqn_risk
            },
            'llm_validation': {
                'approved': llm_approved,
                'confidence': llm_confidence,
                'reasoning_summary': llm_reasoning
            },
            'final_decision': {
                'from_replicas': current_replicas,
                'to_replicas': final_decision,
                'action_executed': 'Scale Up' if final_decision > current_replicas else 'Scale Down' if final_decision < current_replicas else 'Keep Same'
            }
        },
        'decision_factors': [],
        'risk_mitigation': [],
        'expected_outcomes': []
    }

    # Add decision factors based on safety override result
    if not llm_approved:
        decision_explanation['decision_factors'].append(
            "SAFETY OVERRIDE: Extreme scaling decision blocked - maintaining current replica count")
        decision_explanation['risk_mitigation'].append("CRITICAL: Dangerous scaling prevented by safety monitor")
    else:
        decision_explanation['decision_factors'].append("Safety monitor found no extreme risks - DQN decision approved")

    # Add confidence assessment
    overall_confidence = 'high'
    if dqn_confidence == 'low' or llm_confidence == 'low':
        overall_confidence = 'low'
    elif dqn_confidence == 'medium' or llm_confidence == 'medium':
        overall_confidence = 'medium'

    decision_explanation['overall_confidence'] = overall_confidence

    # Add expected outcomes based on action
    if final_decision > current_replicas:
        decision_explanation['expected_outcomes'].extend([
            "Improved response times and reduced latency",
            "Better handling of increased load",
            "Higher resource costs"
        ])
        decision_explanation['risk_mitigation'].append("Monitor for over-provisioning")
    elif final_decision < current_replicas:
        decision_explanation['expected_outcomes'].extend([
            "Reduced resource costs",
            "Optimized resource utilization",
            "Potential slight increase in response times"
        ])
        decision_explanation['risk_mitigation'].append("Monitor for performance degradation")
    else:
        decision_explanation['expected_outcomes'].append("Maintained current performance and cost balance")

    # 🎯 EXECUTE DIRECT SCALING - Skip KEDA/HPA entirely
    logger.info(f"DIRECT_SCALING: DQN_action={action_name} new_replicas={new_replicas} final_decision={final_decision}")

    # Check if there's actually a change to make
    if final_decision == current_replicas:
        logger.info(f"DIRECT_SCALING: NO_CHANGE replicas={current_replicas}")
        scaling_success = True  # No change needed, so it's successful
    else:
        logger.info(f"DIRECT_SCALING: CHANGE_REQUIRED from={current_replicas} to={final_decision}")
        scaling_success = await direct_scale_deployment(final_decision, services)

    if scaling_success:
        logger.info(f"[SUCCESS] DIRECT_SCALING_SUCCESS: DQN_decided={final_decision}")
        # Update Prometheus gauge to reflect successful scaling
        DESIRED_REPLICAS_GAUGE.set(final_decision)
    else:
        logger.error(f"[FAILED] DIRECT_SCALING_FAILED: keeping_current={current_replicas}")
        # Keep current replicas in metrics if scaling failed
        DESIRED_REPLICAS_GAUGE.set(current_replicas)
        final_decision = current_replicas

    # Create comprehensive audit trail
    if 'dqn_prediction' in state and 'explanation' in dqn_prediction:
        # Simple audit trail creation (inline)
        audit_trail = {
            'decision_id': f"dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'current_replicas': current_replicas,
            'final_decision': final_decision,
            'dqn_action': action_name,
            'dqn_confidence': dqn_confidence,
            'safety_approved': llm_approved,
            'safety_reasoning': llm_reasoning
        }
        decision_explanation['audit_trail_id'] = audit_trail['decision_id']

    # Enhanced logging with complete decision reasoning
    logger.info(f"FINAL_DECISION: scale_from={current_replicas} scale_to={final_decision}")
    logger.info(f"FINAL_CONFIDENCE: overall={overall_confidence} dqn={dqn_confidence} safety={llm_confidence}")
    logger.info(f"SAFETY_STATUS: approved={llm_approved}")
    logger.info(f"PROMETHEUS: gauge_updated value={final_decision}")

    if services.config.ai.enable_detailed_reasoning:
        logger.info("FINAL_DECISION_ANALYSIS: detailed_breakdown")
        logger.info(
            f"FINAL_DECISION_ANALYSIS: action_path={action_name}_to_{decision_explanation['decision_pipeline']['final_decision']['action_executed']}")
        logger.info(
            f"FINAL_DECISION_ANALYSIS: expected_outcomes_count={len(decision_explanation['expected_outcomes'])}")
        for i, outcome in enumerate(decision_explanation['expected_outcomes']):
            logger.info(f"FINAL_DECISION_ANALYSIS: outcome_{i + 1}={outcome}")

        if decision_explanation['risk_mitigation']:
            logger.info(
                f"FINAL_DECISION_ANALYSIS: risk_mitigation_count={len(decision_explanation['risk_mitigation'])}")
            for i, mitigation in enumerate(decision_explanation['risk_mitigation']):
                logger.info(f"FINAL_DECISION_ANALYSIS: mitigation_{i + 1}={mitigation}")

    # Warning for low confidence decisions
    if overall_confidence == 'low':
        logger.warning("LOW_CONFIDENCE: enhanced_monitoring_recommended")
        logger.warning("LOW_CONFIDENCE: consider_manual_review")

    # CRITICAL warning for safety monitor blocking (extreme decisions only)
    if not llm_approved:
        logger.error("EXTREME_DECISION_BLOCKED: safety_monitor_intervention")
        logger.error(f"SAFETY_BLOCK_REASON: {llm_reasoning}")
        logger.error("EXTREME_DECISION_BLOCKED: this_indicates_potentially_dangerous_scaling")

    logger.info("=" * 60)
    logger.info("NODE_END: plan_final_action")
    logger.info("=" * 60)
    return {
        "final_decision": final_decision,
        "decision_explanation": decision_explanation,
        "overall_confidence": overall_confidence,
        "llm_approved": llm_approved
    }


async def immediate_reward_and_training(state: ScalingState, services: ServiceContainer) -> Dict[str, Any]:
    """
    🚀 PROACTIVE TRAINING: Calculate reward immediately after scaling action and train the DQN.
    This eliminates the 15-second delay and enables immediate learning from scaling decisions.
    """
    logger.info("=" * 60)
    logger.info("NODE_START: immediate_reward_and_training")
    logger.info("=" * 60)

    # Get current state for reward calculation
    current_state_metrics = state.get("current_metrics", {})
    current_replicas = state.get("final_decision", state.get("current_replicas", 1))

    # Extract action for context
    experience = state.get('experience')
    if not experience:
        logger.error("IMMEDIATE_TRAINING: experience_missing from_state, creating_minimal_experience")
        # Create minimal experience for training
        dqn_prediction = state.get('dqn_prediction', {'action_name': 'Keep Same'})
        action = dqn_prediction.get('action_name', 'Keep Same')
        experience = {"state": current_state_metrics, "action": action}
    
    action = experience.get('action', 'Keep Same')

    # 🎯 IMMEDIATE REWARD: Calculate reward based on the scaling decision's appropriateness
    # This proactive reward doesn't wait for system effects - it evaluates decision quality immediately
    reward = calculate_proactive_reward(
        current_state_metrics, 
        action, 
        current_replicas, 
        services.config
    )

    # For next state, use current metrics (since we're training immediately)
    # The DQN learns from the decision appropriateness, not delayed system response
    experience['reward'] = reward
    experience['next_state'] = {**current_state_metrics, 'current_replicas': current_replicas}

    # Update Prometheus metrics immediately
    DQN_REWARD_TOTAL_GAUGE.set(reward)
    
    # Estimate reward components for monitoring
    performance_component = reward * services.config.reward.performance_weight
    resource_component = reward * services.config.reward.resource_weight  
    health_component = reward * services.config.reward.health_weight
    cost_component = reward * services.config.reward.cost_weight

    DQN_REWARD_PERFORMANCE_GAUGE.set(performance_component)
    DQN_REWARD_RESOURCE_GAUGE.set(resource_component)
    DQN_REWARD_HEALTH_GAUGE.set(health_component)
    DQN_REWARD_COST_GAUGE.set(cost_component)

    logger.info(f"PROACTIVE_REWARD: action={action} reward={reward:.3f} replicas={current_replicas}")
    logger.info(f"REWARD_COMPONENTS: perf={performance_component:.3f} resource={resource_component:.3f} health={health_component:.3f} cost={cost_component:.3f}")

    logger.info("=" * 60)
    logger.info("NODE_END: immediate_reward_and_training")
    logger.info("=" * 60)
    return {"experience": experience}


def calculate_proactive_reward(current_state, action, current_replicas, config=None):
    """
    🚀 RESOURCE-AWARE PROACTIVE REWARD FUNCTION: Evaluates scaling decisions based on actual deployment resource configuration.
    
    KEY IMPROVEMENTS:
    1. Uses actual CPU/memory limits from deployment configuration  
    2. Calculates utilization percentages instead of absolute values
    3. Makes context-appropriate scaling decisions for any resource configuration
    4. Rewards predictive scaling based on utilization trends
    """
    
    # Extract consumer performance metrics (same 9 scientifically-selected features)
    cpu_rate = current_state.get('process_cpu_seconds_total_rate', 0.01)
    gc_collections_rate = current_state.get('python_gc_collections_total_rate', 0.1)  
    gc_objects_rate = current_state.get('python_gc_objects_collected_total_rate', 1.0)
    http_duration_rate = current_state.get('http_request_duration_seconds_sum_rate', 0.05)
    http_requests_rate = current_state.get('http_requests_total_rate', 1.0)
    http_count_rate = current_state.get('http_request_duration_seconds_count_rate', 1.0)
    open_fds = current_state.get('process_open_fds', 10.0)
    response_size_rate = current_state.get('http_response_size_bytes_sum_rate', 1024.0)
    request_count_rate = current_state.get('http_request_size_bytes_count_rate', 1.0)

    # === RESOURCE-AWARE CONFIGURATION ===
    # Get actual resource limits/requests from the target deployment
    cpu_limit = current_state.get('kube_pod_container_resource_limits_cpu', 0.5)  # Default: 500m
    memory_limit = current_state.get('kube_pod_container_resource_limits_memory', 1073741824)  # Default: 1Gi  
    cpu_request = current_state.get('kube_pod_container_resource_requests_cpu', 0.2)  # Default: 200m
    memory_request = current_state.get('kube_pod_container_resource_requests_memory', 536870912)  # Default: 512Mi

    logger.info(f"RESOURCE_CONFIG: cpu_limit={cpu_limit:.3f}cores memory_limit={memory_limit/1024/1024/1024:.1f}Gi "
                f"cpu_request={cpu_request:.3f}cores memory_request={memory_request/1024/1024:.0f}Mi replicas={current_replicas}")

    # === PERFORMANCE PRESSURE ANALYSIS (Resource-Aware) ===
    # Calculate utilization percentages relative to actual resource configuration
    
    # CPU utilization per replica (as percentage of limit)
    cpu_per_replica = cpu_rate / max(current_replicas, 1)
    cpu_utilization_pct = (cpu_per_replica / cpu_limit) * 100 if cpu_limit > 0 else 0
    
    cpu_pressure_score = 0.0
    if cpu_utilization_pct > 70:  # >70% CPU utilization = high pressure
        cpu_pressure_score = min(2.0, cpu_utilization_pct / 50)  # Scale to 2.0 at 100% util
        logger.debug(f"CPU_PRESSURE: high_utilization {cpu_utilization_pct:.1f}% score={cpu_pressure_score:.2f}")
    elif cpu_utilization_pct < 10:  # <10% CPU utilization = potential over-provisioning
        cpu_pressure_score = -1.0  
        logger.debug(f"CPU_PRESSURE: low_utilization {cpu_utilization_pct:.1f}% score={cpu_pressure_score:.2f}")
    else:
        logger.debug(f"CPU_PRESSURE: moderate_utilization {cpu_utilization_pct:.1f}% score={cpu_pressure_score:.2f}")
    
    # Memory pressure indicator (GC activity)
    total_gc_rate = gc_collections_rate + gc_objects_rate
    gc_pressure_per_replica = total_gc_rate / max(current_replicas, 1)
    gc_pressure_score = 0.0
    if gc_pressure_per_replica > 1.5:  # High GC activity
        gc_pressure_score = min(1.5, gc_pressure_per_replica)
    elif gc_pressure_per_replica < 0.1:  # Very low GC activity
        gc_pressure_score = -0.5
    
    # HTTP performance pressure
    http_latency_per_request = http_duration_rate / max(http_requests_rate, 0.1)
    http_load_per_replica = http_requests_rate / max(current_replicas, 1)
    
    http_pressure_score = 0.0
    if http_latency_per_request > 0.05:  # >50ms average latency
        http_pressure_score += min(2.0, http_latency_per_request * 40)
    if http_load_per_replica > 3.0:  # >3 requests/sec per replica
        http_pressure_score += min(1.0, http_load_per_replica / 3.0)
    if http_load_per_replica < 0.2:  # <0.2 requests/sec per replica
        http_pressure_score -= 0.8
    
    # === PROACTIVE DECISION EVALUATION ===
    
    # System state classification
    total_pressure = cpu_pressure_score + gc_pressure_score + http_pressure_score
    is_under_pressure = total_pressure > 2.0
    is_moderate_load = 0.5 <= total_pressure <= 2.0
    is_low_load = total_pressure < 0.5
    is_over_provisioned = total_pressure < -1.0
    
    # Action appropriateness scoring
    action_score = 0.0
    
    # === BALANCED ACTION SCORING ===
    # FIXED: Balanced rewards that don't bias toward any particular action type
    
    if action == "Scale Up":
        if is_under_pressure:
            # REWARD: Proactive scaling before severe degradation (balanced max: 3.0)
            action_score = 2.5 + min(0.5, total_pressure / 4.0)  # Max 3.0
            logger.info(f"PROACTIVE_REWARD: Scale Up under pressure - EXCELLENT decision! pressure={total_pressure:.2f}")
        elif is_moderate_load and current_replicas <= 3:
            # REWARD: Early scaling to handle growing load
            action_score = 2.0
            logger.info(f"PROACTIVE_REWARD: Scale Up anticipating load growth - GOOD decision!")
        elif is_low_load or is_over_provisioned:
            # PENALTY: Scaling up when not needed
            action_score = -2.0
            logger.info(f"PROACTIVE_REWARD: Scale Up with low load - POOR decision! pressure={total_pressure:.2f}")
        elif current_replicas >= 12:
            # PENALTY: Scaling up to excessive levels
            action_score = -3.0
            logger.info(f"PROACTIVE_REWARD: Scale Up to excessive replicas - VERY POOR decision!")
        else:
            # NEUTRAL: Unclear benefit
            action_score = 0.0
            
    elif action == "Scale Down":
        if is_over_provisioned:
            # REWARD: Proactive cost optimization (balanced max: 3.0)
            action_score = 2.5 + min(0.5, abs(total_pressure) / 4.0)  # Max 3.0
            logger.info(f"PROACTIVE_REWARD: Scale Down over-provisioned system - EXCELLENT decision! pressure={total_pressure:.2f}")
        elif is_low_load and current_replicas > 2:
            # REWARD: Appropriate cost optimization
            action_score = 2.0
            logger.info(f"PROACTIVE_REWARD: Scale Down low load system - GOOD decision!")
        elif is_moderate_load:
            # CAUTION: Might be risky
            action_score = -0.5
            logger.info(f"PROACTIVE_REWARD: Scale Down moderate load - RISKY decision! pressure={total_pressure:.2f}")
        elif is_under_pressure:
            # PENALTY: Scaling down under pressure
            action_score = -3.0
            logger.info(f"PROACTIVE_REWARD: Scale Down under pressure - VERY POOR decision! pressure={total_pressure:.2f}")
        elif current_replicas <= 1:
            # PENALTY: Can't scale below 1
            action_score = -2.0
            logger.info(f"PROACTIVE_REWARD: Scale Down below minimum - INVALID decision!")
        else:
            action_score = 0.0
            
    else:  # Keep Same
        if is_moderate_load:
            # REWARD: Stability when appropriate (enhanced to match scaling actions)
            action_score = 2.5  # Increased from 1.0 to balance with scaling actions
            logger.info(f"PROACTIVE_REWARD: Keep Same with moderate load - EXCELLENT STABLE decision!")
        elif is_low_load and not is_over_provisioned:
            # REWARD: Good stability during low but reasonable load
            action_score = 2.0  # New reward case
            logger.info(f"PROACTIVE_REWARD: Keep Same with low load - GOOD STABLE decision!")
        elif is_under_pressure:
            # PENALTY: Inaction when scaling is needed  
            action_score = -2.0
            logger.info(f"PROACTIVE_REWARD: Keep Same under pressure - MISSED OPPORTUNITY! pressure={total_pressure:.2f}")
        elif is_over_provisioned:
            # PENALTY: Missing cost optimization opportunity (reduced)
            action_score = -1.0  # Reduced from -1.5 to be less harsh
            logger.info(f"PROACTIVE_REWARD: Keep Same when over-provisioned - MISSED SAVINGS! pressure={total_pressure:.2f}")
        else:
            action_score = 1.0  # Increased baseline reward for stability (was 0.5)
    
    # === RESOURCE-AWARE EFFICIENCY BONUS ===
    # Reward efficient resource usage patterns based on actual deployment configuration
    
    fds_per_replica = open_fds / max(current_replicas, 1)
    efficiency_bonus = 0.0
    
    # Calculate efficiency using utilization percentages (more realistic for microservices)
    if 15 <= cpu_utilization_pct <= 65 and 2 <= fds_per_replica <= 25:
        efficiency_bonus = 1.0  # Optimal efficiency: moderate utilization with reasonable I/O
        logger.info(f"PROACTIVE_REWARD: Efficiency bonus - optimal resource usage! cpu_util={cpu_utilization_pct:.1f}% fds_per_replica={fds_per_replica:.1f}")
    elif cpu_utilization_pct < 5 and current_replicas > 2:
        efficiency_bonus = -0.5  # Penalty for significant under-utilization (waste)
        logger.info(f"PROACTIVE_REWARD: Efficiency penalty - significant under-utilization! cpu_util={cpu_utilization_pct:.1f}%")
    elif 8 <= cpu_utilization_pct <= 85:
        # CPU-focused efficiency (good utilization regardless of I/O patterns)
        if 8 <= cpu_utilization_pct <= 30:
            efficiency_bonus = 0.6  # Good efficiency for light workloads
            logger.info(f"PROACTIVE_REWARD: CPU efficiency bonus - good light utilization! cpu_util={cpu_utilization_pct:.1f}%")
        elif 30 < cpu_utilization_pct <= 70:
            efficiency_bonus = 0.8  # Great efficiency for moderate workloads  
            logger.info(f"PROACTIVE_REWARD: CPU efficiency bonus - excellent moderate utilization! cpu_util={cpu_utilization_pct:.1f}%")
        elif 70 < cpu_utilization_pct <= 85:
            efficiency_bonus = 0.4  # Decent efficiency for high workloads
            logger.info(f"PROACTIVE_REWARD: CPU efficiency bonus - handling high load well! cpu_util={cpu_utilization_pct:.1f}%")
    elif 85 < cpu_utilization_pct <= 95:
        efficiency_bonus = 0.1  # Minimal bonus for very high but manageable load
        logger.info(f"PROACTIVE_REWARD: Efficiency minimal bonus - very high but manageable load! cpu_util={cpu_utilization_pct:.1f}%")
    elif cpu_utilization_pct > 95:
        efficiency_bonus = -0.3  # Penalty for dangerous over-utilization
        logger.info(f"PROACTIVE_REWARD: Efficiency penalty - dangerous over-utilization! cpu_util={cpu_utilization_pct:.1f}%")
    else:
        # This should rarely happen now
        logger.info(f"PROACTIVE_REWARD: No efficiency bonus - cpu_util={cpu_utilization_pct:.1f}% fds_per_replica={fds_per_replica:.1f} (outside ranges)")
    
    # === TIMING BONUS ===
    # FIXED: More inclusive timing rewards that consider all actions and realistic pressure ranges
    timing_bonus = 0.0
    
    if action in ["Scale Up", "Scale Down"]:
        # Scaling actions - evaluate timing based on pressure
        if total_pressure > 2.0:  # High pressure = late reaction (lowered from 3.0)
            timing_bonus = -0.5  # Reduced penalty (was -1.0)
            logger.info(f"PROACTIVE_REWARD: Late reaction penalty - should have acted earlier! pressure={total_pressure:.2f}")
        elif 0.5 <= total_pressure <= 1.5:  # Moderate pressure = good timing (lowered thresholds)
            timing_bonus = 1.0
            logger.info(f"PROACTIVE_REWARD: Good timing bonus - proactive scaling! pressure={total_pressure:.2f}")
        elif -0.5 <= total_pressure <= 0.5:  # Low pressure scaling = reasonable timing
            timing_bonus = 0.5
            logger.info(f"PROACTIVE_REWARD: Reasonable timing - moderate pressure scaling! pressure={total_pressure:.2f}")
        else:
            logger.debug(f"PROACTIVE_REWARD: No timing bonus for scaling - pressure={total_pressure:.2f}")
    elif action == "Keep Same":
        # FIXED: "Keep Same" actions should also get timing bonuses for stability
        if is_moderate_load or (-1.0 <= total_pressure <= 1.0):  # Stable conditions
            timing_bonus = 0.5  # Reward stability when appropriate
            logger.info(f"PROACTIVE_REWARD: Stability timing bonus - good decision to maintain! pressure={total_pressure:.2f}")
        elif total_pressure < -2.0:  # Very over-provisioned - should have scaled down
            timing_bonus = -0.3  # Light penalty for missing cost savings
            logger.info(f"PROACTIVE_REWARD: Missed opportunity - should consider scaling down! pressure={total_pressure:.2f}")
        elif total_pressure > 2.0:  # High pressure - should have scaled up
            timing_bonus = -0.5  # Penalty for missing performance opportunity
            logger.info(f"PROACTIVE_REWARD: Missed opportunity - should consider scaling up! pressure={total_pressure:.2f}")
        else:
            logger.debug(f"PROACTIVE_REWARD: No timing bonus for keep same - pressure={total_pressure:.2f}")
    
    # === FINAL REWARD CALCULATION ===
    if config:
        total_reward = (
            action_score * config.reward.performance_weight +    # Primary component
            efficiency_bonus * config.reward.resource_weight +  # Efficiency bonus
            timing_bonus * config.reward.health_weight +        # Timing bonus
            (-0.1 * max(0, current_replicas - 8)) * config.reward.cost_weight  # Cost factor
        )
    else:
        total_reward = (
            action_score * 0.40 +      # 40% - Action appropriateness
            efficiency_bonus * 0.30 +  # 30% - Resource efficiency
            timing_bonus * 0.20 +      # 20% - Timing (proactive vs reactive)
            (-0.1 * max(0, current_replicas - 8)) * 0.10  # 10% - Cost factor
        )
    
    # Clip to reasonable range
    total_reward = np.clip(total_reward, -5.0, 5.0)
    
    # Enhanced logging for proactive decision analysis
    logger.info(f"PROACTIVE_ANALYSIS: cpu_utilization={cpu_utilization_pct:.1f}% gc_pressure={gc_pressure_per_replica:.4f} http_pressure={http_pressure_score:.2f}")
    logger.info(f"PROACTIVE_ANALYSIS: total_pressure={total_pressure:.2f} state={'UNDER_PRESSURE' if is_under_pressure else 'OVER_PROVISIONED' if is_over_provisioned else 'MODERATE' if is_moderate_load else 'LOW_LOAD'}")
    logger.info(f"PROACTIVE_ANALYSIS: action_score={action_score:.2f} efficiency_bonus={efficiency_bonus:.2f} timing_bonus={timing_bonus:.2f}")
    logger.info(f"PROACTIVE_REWARD: final_reward={total_reward:.3f} (resource_aware_immediate_feedback)")
    
    return total_reward


async def wait_for_system_to_stabilize(state: ScalingState, services: ServiceContainer) -> None:
    logger.info("=" * 60)
    logger.info("NODE_START: wait_for_system_to_stabilize")
    logger.info("=" * 60)
    stabilization_seconds = services.config.kubernetes.stabilization_period_seconds
    logger.info(f"STABILIZATION: waiting_seconds={stabilization_seconds}")
    await asyncio.sleep(stabilization_seconds)
    logger.info("=" * 60)
    logger.info("NODE_END: wait_for_system_to_stabilize")
    logger.info("=" * 60)
    return {}


async def observe_next_state_and_calculate_reward(state: ScalingState, services: ServiceContainer) -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("NODE_START: observe_next_state_and_calculate_reward")
    logger.info("=" * 60)

    # Get next state data
    next_state_data = await get_live_metrics(state, services, is_next_state=True)
    next_state_metrics = next_state_data.get("current_metrics", {})
    current_replicas = next_state_data.get("current_replicas", 1)

    # Get current state for comparison
    current_state_metrics = state.get("current_metrics", {})

    # Extract action for context - handle missing experience gracefully
    experience = state.get('experience')
    if not experience:
        logger.error("REWARD_CALCULATION: experience_missing from_state, cannot_calculate_reward")
        return {"error": "Experience data not available for reward calculation"}

    action = experience.get('action', 'Keep Same')

    # Calculate improved reward using the multi-objective function
    reward = calculate_stable_reward(current_state_metrics, next_state_metrics, action,
                                     current_replicas, services.config)

    experience['reward'] = reward
    experience['next_state'] = {**next_state_metrics, 'current_replicas': current_replicas}

    # Update reward metrics (assuming reward components are available)
    DQN_REWARD_TOTAL_GAUGE.set(reward)
    if isinstance(reward, dict):
        # If reward is broken down into components
        DQN_REWARD_PERFORMANCE_GAUGE.set(reward.get('performance', 0))
        DQN_REWARD_RESOURCE_GAUGE.set(reward.get('resource', 0))
        DQN_REWARD_HEALTH_GAUGE.set(reward.get('health', 0))
        DQN_REWARD_COST_GAUGE.set(reward.get('cost', 0))
    else:
        # If reward is a single value, estimate components based on weights
        performance_component = reward * services.config.reward.performance_weight
        resource_component = reward * services.config.reward.resource_weight
        health_component = reward * services.config.reward.health_weight
        cost_component = reward * services.config.reward.cost_weight

        DQN_REWARD_PERFORMANCE_GAUGE.set(performance_component)
        DQN_REWARD_RESOURCE_GAUGE.set(resource_component)
        DQN_REWARD_HEALTH_GAUGE.set(health_component)
        DQN_REWARD_COST_GAUGE.set(cost_component)

    logger.info("=" * 60)
    logger.info("NODE_END: observe_next_state_and_calculate_reward")
    logger.info("=" * 60)
    return {"experience": experience}


def calculate_stable_reward(current_state, next_state, action, current_replicas, config=None):
    """
    RESEARCH-BASED: Reward function using ONLY the consumer performance metrics DQN actually sees.
    FIXED: Now uses the 9 scientifically-selected consumer performance features instead of Kubernetes metrics.
    """

    # === USE ONLY ACTUAL DQN INPUT FEATURES ===
    # These are the 9 scientifically-selected consumer performance features

    # Current state metrics (what DQN saw when making decision)
    curr_cpu_rate = current_state.get('process_cpu_seconds_total_rate', 0.01)
    curr_gc_collections_rate = current_state.get('python_gc_collections_total_rate', 0.1)
    curr_gc_objects_rate = current_state.get('python_gc_objects_collected_total_rate', 1.0)
    curr_http_duration_rate = current_state.get('http_request_duration_seconds_sum_rate', 0.05)
    curr_http_requests_rate = current_state.get('http_requests_total_rate', 1.0)
    curr_http_count_rate = current_state.get('http_request_duration_seconds_count_rate', 1.0)
    curr_open_fds = current_state.get('process_open_fds', 10.0)
    curr_response_size_rate = current_state.get('http_response_size_bytes_sum_rate', 1024.0)
    curr_request_count_rate = current_state.get('http_request_size_bytes_count_rate', 1.0)

    # Next state metrics (outcome after DQN action)
    next_cpu_rate = next_state.get('process_cpu_seconds_total_rate', 0.01)
    next_gc_collections_rate = next_state.get('python_gc_collections_total_rate', 0.1)
    next_gc_objects_rate = next_state.get('python_gc_objects_collected_total_rate', 1.0)
    next_http_duration_rate = next_state.get('http_request_duration_seconds_sum_rate', 0.05)
    next_http_requests_rate = next_state.get('http_requests_total_rate', 1.0)
    next_http_count_rate = next_state.get('http_request_duration_seconds_count_rate', 1.0)
    next_open_fds = next_state.get('process_open_fds', 10.0)
    next_response_size_rate = next_state.get('http_response_size_bytes_sum_rate', 1024.0)
    next_request_count_rate = next_state.get('http_request_size_bytes_count_rate', 1.0)

    # === PERFORMANCE COMPONENT (40%) ===
    # Reward improvements in consumer pod performance

    # CPU efficiency improvement
    cpu_efficiency_improvement = max(0, curr_cpu_rate - next_cpu_rate) if curr_cpu_rate > 0.1 else 0
    
    # Memory pressure reduction (lower GC activity is better)
    memory_pressure_improvement = (curr_gc_collections_rate + curr_gc_objects_rate) - \
                                 (next_gc_collections_rate + next_gc_objects_rate)
    memory_pressure_improvement = max(0, memory_pressure_improvement)  # Only reward improvements
    
    # HTTP performance improvement (lower latency accumulation per request is better)
    http_latency_improvement = 0
    if curr_http_requests_rate > 0 and next_http_requests_rate > 0:
        curr_latency_per_req = curr_http_duration_rate / max(curr_http_requests_rate, 0.1)
        next_latency_per_req = next_http_duration_rate / max(next_http_requests_rate, 0.1)
        http_latency_improvement = max(0, curr_latency_per_req - next_latency_per_req)

    # I/O efficiency (stable file descriptor count is good)
    io_stability = 1.0 - abs(next_open_fds - curr_open_fds) / max(curr_open_fds, 10.0)
    io_stability = max(0, io_stability)

    performance_score = np.clip(
        cpu_efficiency_improvement * 5.0 +      # CPU efficiency is critical
        memory_pressure_improvement * 3.0 +     # Memory pressure reduction
        http_latency_improvement * 100.0 +      # HTTP latency improvement (scaled up)
        io_stability * 1.0,                     # I/O stability
        -5.0, 5.0  # Bound performance component
    )

    # === RESOURCE-AWARE EFFICIENCY COMPONENT (30%) ===
    # Evaluate resource utilization per replica using actual deployment configuration

    # Get actual resource limits for context-aware efficiency calculation
    cpu_limit = next_state.get('kube_pod_container_resource_limits_cpu', 0.5)  # Default: 500m
    memory_limit = next_state.get('kube_pod_container_resource_limits_memory', 1073741824)  # Default: 1Gi

    # Calculate per-replica resource usage and utilization percentages
    cpu_per_replica = next_cpu_rate / max(current_replicas, 1)
    memory_pressure_per_replica = (next_gc_collections_rate + next_gc_objects_rate) / max(current_replicas, 1)
    
    # Resource-aware efficiency calculation
    def resource_efficiency_score(cpu_per_replica, memory_pressure_per_replica, replicas, cpu_limit, memory_limit):
        """Evaluate resource usage efficiency per replica using actual deployment resource configuration."""
        
        # Calculate utilization percentages (much more robust across different deployments)
        cpu_utilization_pct = (cpu_per_replica / cpu_limit) * 100 if cpu_limit > 0 else 0
        
        # Ideal utilization ranges (percentage-based, deployment-agnostic)
        ideal_cpu_util_range = (15, 65)        # 15-65% CPU utilization per replica = efficient
        ideal_memory_pressure_range = (0.1, 2.0)  # Low to moderate GC activity per replica
        
        # CPU efficiency based on utilization percentage
        if ideal_cpu_util_range[0] <= cpu_utilization_pct <= ideal_cpu_util_range[1]:
            cpu_efficiency = 1.0  # Optimal utilization
        else:
            cpu_range_center = np.mean(ideal_cpu_util_range)
            cpu_efficiency = max(0, 1.0 - abs(cpu_utilization_pct - cpu_range_center) / cpu_range_center)
        
        # Memory efficiency based on GC pressure (independent of memory limits for now)
        if ideal_memory_pressure_range[0] <= memory_pressure_per_replica <= ideal_memory_pressure_range[1]:
            memory_efficiency = 1.0
        else:
            memory_range_center = np.mean(ideal_memory_pressure_range) 
            memory_efficiency = max(0, 1.0 - abs(memory_pressure_per_replica - memory_range_center) / memory_range_center)
        
        # Combined efficiency
        efficiency = (cpu_efficiency * 0.7 + memory_efficiency * 0.3)  # CPU weighted higher
        
        # Context-aware bonuses and penalties
        if replicas <= 3 and 10 <= cpu_utilization_pct <= 50:
            efficiency += 0.5  # Reward efficient small deployments
        elif replicas >= 8 and cpu_utilization_pct < 5:
            efficiency -= 0.3  # Penalize potential over-provisioning with very low utilization
        elif cpu_utilization_pct > 90:
            efficiency -= 0.4  # Penalty for dangerous over-utilization
            
        return np.clip(efficiency, -2.0, 2.0)

    resource_score = resource_efficiency_score(cpu_per_replica, memory_pressure_per_replica, current_replicas, cpu_limit, memory_limit)

    # === SYSTEM HEALTH COMPONENT (20%) ===
    # Reward system stability and health

    # HTTP throughput health (consistent request processing)
    http_throughput_health = 1.0 if next_http_requests_rate >= curr_http_requests_rate * 0.9 else \
                            max(0, next_http_requests_rate / max(curr_http_requests_rate, 0.1))
    
    # Response efficiency (reasonable response sizes)
    response_efficiency = 1.0
    if next_response_size_rate > 0 and next_request_count_rate > 0:
        avg_response_size = next_response_size_rate / max(next_request_count_rate, 0.1)
        # Reasonable response size: 1KB - 100KB
        if 1024 <= avg_response_size <= 102400:
            response_efficiency = 1.0
        else:
            response_efficiency = max(0.2, 1.0 - abs(avg_response_size - 10240) / 10240)  # Optimal around 10KB

    health_score = (http_throughput_health + response_efficiency) / 2.0

    # === COST OPTIMIZATION COMPONENT (10%) ===
    # FIXED: Smooth cost function eliminates discontinuities

    def smooth_cost_function(replicas):
        """Smooth cost function prevents learning discontinuities."""
        if replicas <= 3:
            return 2.0  # Reward very efficient sizing
        elif replicas <= 12:
            # Linear decay from 2.0 to 0.0 (smooth transition)
            return 2.0 - (replicas - 3) * (2.0 / 9.0)
        else:
            # Gentle penalty for over-provisioning (no sudden jumps)
            return -0.1 * (replicas - 12)

    cost_score = smooth_cost_function(current_replicas)

    # === LOAD AWARENESS COMPONENT (15%) ===
    # Assess if resource allocation matches actual demand

    def calculate_load_awareness_score(cpu_rate, gc_rate, http_rate, open_fds, replicas):
        """
        Assess if current performance metrics indicate appropriate load for the replica count.
        """
        # Normalize metrics per replica
        cpu_per_replica = cpu_rate / max(replicas, 1)
        gc_per_replica = gc_rate / max(replicas, 1)
        http_per_replica = http_rate / max(replicas, 1)
        fds_per_replica = open_fds / max(replicas, 1)
        
        # Ideal load indicators per replica
        ideal_cpu_load = (0.02, 0.08)      # 2-8% CPU per replica
        ideal_gc_load = (0.1, 1.0)         # Moderate GC activity per replica
        ideal_http_load = (0.5, 5.0)       # 0.5-5 requests per second per replica
        ideal_fds_load = (8, 15)           # 8-15 file descriptors per replica
        
        # Calculate load appropriateness scores
        cpu_load_score = 1.0 if ideal_cpu_load[0] <= cpu_per_replica <= ideal_cpu_load[1] else \
                        max(0, 1.0 - abs(cpu_per_replica - np.mean(ideal_cpu_load)) / np.mean(ideal_cpu_load))
        
        gc_load_score = 1.0 if ideal_gc_load[0] <= gc_per_replica <= ideal_gc_load[1] else \
                       max(0, 1.0 - abs(gc_per_replica - np.mean(ideal_gc_load)) / np.mean(ideal_gc_load))
        
        http_load_score = 1.0 if ideal_http_load[0] <= http_per_replica <= ideal_http_load[1] else \
                         max(0, 1.0 - abs(http_per_replica - np.mean(ideal_http_load)) / np.mean(ideal_http_load))
        
        fds_load_score = 1.0 if ideal_fds_load[0] <= fds_per_replica <= ideal_fds_load[1] else \
                        max(0, 1.0 - abs(fds_per_replica - np.mean(ideal_fds_load)) / np.mean(ideal_fds_load))
        
        # Combined load awareness
        load_score = (cpu_load_score + gc_load_score + http_load_score + fds_load_score) / 4.0
        
        # Penalty for very low utilization across all metrics (indicates over-provisioning)
        if cpu_per_replica < 0.01 and gc_per_replica < 0.05 and http_per_replica < 0.2:
            load_score -= 0.5  # Significant penalty for apparent over-provisioning
        
        return np.clip(load_score, -1.0, 1.0)

    load_awareness_score = calculate_load_awareness_score(
        next_cpu_rate, next_gc_collections_rate, next_http_requests_rate, next_open_fds, current_replicas
    )

    # === ACTION APPROPRIATENESS PENALTY ===
    action_penalty = 0.0
    
    # Define system state using consumer performance metrics
    is_healthy = (next_http_requests_rate > 0.1 and next_cpu_rate < 0.2 and next_open_fds > 5)
    is_overprovisioned = (next_cpu_rate < 0.02 and next_gc_collections_rate < 0.1 and next_http_requests_rate < 0.5)
    is_underprovisioned = (next_cpu_rate > 0.15 or next_gc_collections_rate > 2.0 or next_http_requests_rate > 10.0)
    
    if action == "Scale Down":
        if current_replicas <= 1:
            action_penalty = -1.0  # Prevent scaling below minimum
        elif not is_healthy:
            action_penalty = -0.5  # Light penalty for scaling down unhealthy systems
        elif is_overprovisioned:
            action_penalty = +0.5  # Reward scaling down when over-provisioned
    
    elif action == "Scale Up":
        if is_underprovisioned and is_healthy:
            action_penalty = +0.2  # Light reward for scaling up when under-provisioned
        elif current_replicas >= 12:
            action_penalty = -0.5  # Prevent excessive scaling
        elif is_overprovisioned:
            action_penalty = -0.3  # Discourage scaling up when over-provisioned

    # === FINAL REWARD CALCULATION ===
    if config:
        total_reward = (
                performance_score * config.reward.performance_weight +
                resource_score * config.reward.resource_weight +
                health_score * config.reward.health_weight +
                cost_score * config.reward.cost_weight +
                load_awareness_score * 0.15 +  # Load awareness component (15% weight)
                action_penalty
        )
        
        # Enhanced logging with config values
        logger.info(f"REWARD_BREAKDOWN: performance={performance_score:.2f}({config.reward.performance_weight:.0%}) "
                    f"resource={resource_score:.2f}({config.reward.resource_weight:.0%}) health={health_score:.2f}({config.reward.health_weight:.0%}) "
                    f"cost={cost_score:.2f}({config.reward.cost_weight:.0%}) load_awareness={load_awareness_score:.2f}(15%) "
                    f"action_penalty={action_penalty:.2f} total={total_reward:.2f}")
    else:
        # Fallback to defaults if config not available
        total_reward = (
                performance_score * 0.35 +      # ADJUSTED: Reduced from 40% to make room for load awareness
                resource_score * 0.25 +         # ADJUSTED: Reduced from 30% to make room for load awareness
                health_score * 0.15 +           # ADJUSTED: Reduced from 20% to make room for load awareness
                cost_score * 0.10 +             # UNCHANGED: 10%
                load_awareness_score * 0.15 +   # NEW: Load awareness component (15% weight)
                action_penalty
        )
        
        logger.info(f"REWARD_BREAKDOWN: performance={performance_score:.2f}(35%) "
                    f"resource={resource_score:.2f}(25%) health={health_score:.2f}(15%) "
                    f"cost={cost_score:.2f}(10%) load_awareness={load_awareness_score:.2f}(15%) "
                    f"action_penalty={action_penalty:.2f} total={total_reward:.2f}")

    # Clip to reasonable range
    total_reward = np.clip(total_reward, -10.0, 10.0)

    # Enhanced logging with consumer performance metrics
    logger.info(f"CONSUMER_STATE_ANALYSIS: curr_replicas={current_replicas} "
                f"cpu_rate={curr_cpu_rate:.4f}=>{next_cpu_rate:.4f} "
                f"gc_rate={curr_gc_collections_rate:.4f}=>{next_gc_collections_rate:.4f} "
                f"http_rate={curr_http_requests_rate:.4f}=>{next_http_requests_rate:.4f} "
                f"open_fds={curr_open_fds:.0f}=>{next_open_fds:.0f}")
    
    logger.info(f"PERFORMANCE_EFFICIENCY: cpu_per_replica={next_cpu_rate/max(current_replicas,1):.4f} "
                f"gc_per_replica={next_gc_collections_rate/max(current_replicas,1):.4f} "
                f"http_per_replica={next_http_requests_rate/max(current_replicas,1):.4f} "
                f"fds_per_replica={next_open_fds/max(current_replicas,1):.1f}")

    return total_reward


def calculate_response_efficiency(http_bucket_count, response_size_sum):
    """Calculate response efficiency score (higher is better)."""
    # Normalize metrics and combine for efficiency score
    bucket_score = max(0, 5.0 - http_bucket_count * 0.002)  # Lower bucket count = better
    size_score = max(0, 3.0 - response_size_sum * 0.00001)  # Reasonable response size
    return (bucket_score + size_score) / 2.0


def calculate_cpu_efficiency(current_cpu, next_cpu, replicas):
    """Calculate CPU utilization efficiency."""
    cpu_usage_rate = max(0.1, next_cpu - current_cpu)  # CPU usage in time period
    cpu_per_replica = cpu_usage_rate / max(1, replicas)

    # Optimal CPU usage per replica (not too low, not too high)
    if 0.5 <= cpu_per_replica <= 2.0:  # Sweet spot
        return 2.0
    elif cpu_per_replica < 0.5:  # Under-utilized
        return 1.0 - (0.5 - cpu_per_replica)
    else:  # Over-utilized
        return max(0, 2.0 - (cpu_per_replica - 2.0) * 0.5)


def calculate_health_score(current_scrape, next_scrape, memory_mb, http_buckets):
    """Calculate overall system health score."""
    # Scrape sample health (monitoring effectiveness)
    scrape_health = min(3.0, next_scrape / 100.0)  # 300+ samples = full score

    # System stress indicators
    memory_stress = max(0, 2.0 - max(0, memory_mb - 500) * 0.002)  # Penalty for >500MB
    traffic_stress = max(0, 2.0 - max(0, http_buckets - 100) * 0.001)  # Penalty for high traffic

    return (scrape_health + memory_stress + traffic_stress) / 3.0


def calculate_proactive_bonus(action, current_state, next_state):
    """
    Simple bonus function that doesn't rely on LSTM predictions.
    Rewards sensible scaling decisions based on system state only.
    """
    # Since LSTM is removed, return 0 (no additional bonus)
    return 0.0


def calculate_action_penalty(action, current_replicas, current_state=None, next_state=None):
    """
    BALANCED: Much reduced penalties, focus on rewarding good decisions.
    Addresses: Reduced harsh penalties + appropriate scale-down rewards + load context.
    FIXED: Uses actual selected features (gauges), not cumulative counters.
    """
    penalty = 0.0

    # FIXED: Get load context using actual available metrics from our feature set
    if current_state:
        unavailable_replicas = current_state.get('kube_deployment_status_replicas_unavailable', 0)
        pod_readiness = current_state.get('kube_pod_container_status_ready', 1.0)
        running_containers = current_state.get('kube_pod_container_status_running', 0)
        cpu_limits = current_state.get('kube_pod_container_resource_limits_cpu', 0)
        memory_limits = current_state.get('kube_pod_container_resource_limits_memory', 0)

        memory_mb = memory_limits / 1000000 if memory_limits > 0 else 0

        # Define load context using actual Kubernetes state metrics
        is_no_load = (
                unavailable_replicas == 0 and  # No failing pods
                pod_readiness >= 0.95 and  # High readiness
                cpu_limits <= 1.0 and  # Low CPU allocation
                memory_mb <= 300  # Low memory allocation
        )

        is_low_load = (
                unavailable_replicas == 0 and  # No failing pods
                pod_readiness >= 0.90 and  # Good readiness
                cpu_limits <= 2.0 and  # Moderate CPU
                memory_mb <= 600  # Moderate memory
        )
    else:
        is_no_load = False
        is_low_load = False

    # Scale Down penalties/rewards
    if action == "Scale Down":
        if current_replicas <= 1:
            penalty = -1.0  # REDUCED: was -2.0, still prevent scaling below 1
        elif current_state is not None:
            is_underutilized = check_system_underutilized(current_state, next_state, current_replicas)

            if is_underutilized or is_no_load or is_low_load:
                # REWARD appropriate scale-down - this was the key missing piece!
                if is_no_load:
                    penalty = +2.5  # Strong reward for scaling down with no load
                elif is_low_load:
                    penalty = +2.0  # Good reward for scaling down with low load
                elif current_replicas >= 8:
                    penalty = +1.5  # Reward for reducing over-provisioning
                elif current_replicas >= 5:
                    penalty = +1.0  # Reward for right-sizing
                else:
                    penalty = +0.5  # Light reward for any appropriate scale-down
            else:
                # System is NOT underutilized - light penalty
                penalty = -0.5  # REDUCED: was -1.0, less harsh

    # Scale Up penalties
    elif action == "Scale Up":
        if current_state is not None:
            is_overloaded = check_system_overloaded(current_state, next_state, current_replicas)

            if is_overloaded and not is_no_load:
                # System needs resources and has actual load - minimal penalty
                if current_replicas >= 15:
                    penalty = -1.0  # REDUCED: was -2.0
                else:
                    penalty = 0.0  # Neutral - system needs resources
            else:
                # Unnecessary scaling up - moderate penalties
                if is_no_load:
                    penalty = -1.5  # REDUCED: was much higher, but still discourage waste
                elif current_replicas >= 10:
                    penalty = -1.0  # REDUCED: was -3.0
                elif current_replicas >= 5:
                    penalty = -0.5  # REDUCED: was -1.0
                else:
                    penalty = -0.2  # REDUCED: was -0.5

    # Keep Same - very light penalty for any action (encourage slight bias toward stability)
    elif action != "Keep Same":
        penalty = -0.1  # REDUCED: was -0.2, minimal stability bias

    return penalty


def check_system_underutilized(current_state, next_state, current_replicas):
    """
    FIXED: Check consumer performance metrics to determine if system is underutilized.
    Now uses scientifically selected consumer performance indicators instead of Kubernetes metrics.
    """
    if not current_state:
        return False

    # Extract consumer performance metrics
    cpu_rate = current_state.get('process_cpu_seconds_total_rate', 0.01)
    gc_collections_rate = current_state.get('python_gc_collections_total_rate', 0.1)
    gc_objects_rate = current_state.get('python_gc_objects_collected_total_rate', 1.0)
    http_requests_rate = current_state.get('http_requests_total_rate', 1.0)
    http_duration_rate = current_state.get('http_request_duration_seconds_sum_rate', 0.05)
    open_fds = current_state.get('process_open_fds', 10.0)

    # Calculate per-replica utilization (what matters for scaling decisions)
    cpu_per_replica = cpu_rate / max(1, current_replicas)
    gc_per_replica = (gc_collections_rate + gc_objects_rate) / max(1, current_replicas)
    http_per_replica = http_requests_rate / max(1, current_replicas)
    fds_per_replica = open_fds / max(1, current_replicas)

    # Define underutilization thresholds for consumer performance
    underutilized_conditions = [
        cpu_per_replica < 0.02,        # < 2% CPU per replica = very low utilization
        gc_per_replica < 0.2,          # < 0.2 GC activity per replica = low memory pressure
        http_per_replica < 0.5,        # < 0.5 requests per second per replica = low traffic
        open_fds > 5,                  # System is running (basic health check)
        http_requests_rate > 0.1,      # Some HTTP activity (not completely dead)
        current_replicas > 1           # Don't scale below 1 replica (minimum viable)
    ]

    is_underutilized = all(underutilized_conditions)

    # Enhanced logging to show the consumer performance analysis
    if current_replicas >= 2:
        logger.info(f"UNDERUTILIZATION_CHECK: {sum(underutilized_conditions)}/6 conditions_met "
                    f"cpu_per_replica={cpu_per_replica:.4f} gc_per_replica={gc_per_replica:.4f} "
                    f"http_per_replica={http_per_replica:.4f} fds_per_replica={fds_per_replica:.1f} "
                    f"(total: cpu_rate={cpu_rate:.4f} http_rate={http_requests_rate:.4f} fds={open_fds:.0f} across {current_replicas} replicas) "
                    f"result={'SAFE_TO_SCALE_DOWN' if is_underutilized else 'NOT_SAFE'}")

    return is_underutilized


def check_system_overloaded(current_state, next_state, current_replicas):
    """
    HONEST: Overload detection based on consumer performance metrics.
    Uses clear thresholds for actual consumer pod performance indicators.
    """
    if not current_state:
        return False

    # Extract consumer performance metrics
    cpu_rate = current_state.get('process_cpu_seconds_total_rate', 0.01)
    gc_collections_rate = current_state.get('python_gc_collections_total_rate', 0.1)
    gc_objects_rate = current_state.get('python_gc_objects_collected_total_rate', 1.0)
    http_requests_rate = current_state.get('http_requests_total_rate', 1.0)
    http_duration_rate = current_state.get('http_request_duration_seconds_sum_rate', 0.05)
    open_fds = current_state.get('process_open_fds', 10.0)

    # Calculate performance stress indicators
    total_gc_pressure = gc_collections_rate + gc_objects_rate
    
    # HTTP latency per request (indicating performance degradation)
    latency_per_request = 0
    if http_requests_rate > 0.1:
        latency_per_request = http_duration_rate / http_requests_rate

    # HONEST stress detection (when system actually needs help)
    critical_conditions = [
        cpu_rate > 0.2,                    # > 20% total CPU usage = high load
        total_gc_pressure > 5.0,           # High garbage collection activity = memory pressure
        latency_per_request > 0.1,         # > 100ms average latency = performance degradation
    ]

    # High utilization conditions
    high_utilization_conditions = [
        http_requests_rate > 20.0,         # > 20 requests/sec = high traffic
        open_fds > 50,                     # > 50 file descriptors = high I/O activity
    ]

    # Overloaded if ANY critical condition OR multiple utilization conditions
    is_overloaded = any(critical_conditions) or sum(high_utilization_conditions) >= 2

    if current_replicas <= 15:
        logger.info(
            f"OVERLOAD_CHECK: critical={sum(critical_conditions)}/3 utilization={sum(high_utilization_conditions)}/2 "
            f"cpu_rate={cpu_rate:.4f} gc_pressure={total_gc_pressure:.4f} "
            f"http_rate={http_requests_rate:.4f} latency_per_req={latency_per_request:.4f}s "
            f"open_fds={open_fds:.0f} "
            f"result={'NEEDS_MORE_RESOURCES' if is_overloaded else 'ADEQUATE_RESOURCES'}")

    return is_overloaded


async def log_experience(state: ScalingState, services: ServiceContainer) -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("NODE_START: log_experience")
    logger.info("=" * 60)
    exp = state['experience']

    # Combined approach: immediate training + Redis backup + research logging
    try:
        # 1. Trigger immediate training (primary)
        if services.dqn_trainer:
            await services.dqn_trainer.add_experience_for_training(exp)
            logger.info("EXPERIENCE: queued_for_training")

        # 2. Also log to Redis as backup (for monitoring/debugging)
        if services.redis_client:
            replay_buffer_key = services.config.redis.replay_buffer_key
            experience_json = json.dumps(exp)
            services.redis_client.lpush(replay_buffer_key, experience_json)
            services.redis_client.ltrim(replay_buffer_key, 0, 99)  # Keep last 100 for monitoring
            logger.info("EXPERIENCE: logged_to_redis monitoring_enabled")

        # 3. Add to evaluator for analysis
        if services.evaluator and services.config.enable_evaluation_outputs:
            services.evaluator.add_experience(exp)

        # 4. Update Prometheus experience counter
        DQN_EXPERIENCES_COUNTER.inc()

    except Exception as e:
        logger.error(f"EXPERIENCE: logging_failed error={e}")

    logger.info("=" * 60)
    logger.info("NODE_END: log_experience")
    logger.info("=" * 60)
    return {}


def test_reward_function_accuracy():
    """
    Test the reward function for mathematical accuracy and consistency.
    Validates that the improvements fix the scaling bias and discontinuity issues.
    """
    logger.info("REWARD_TEST: starting_accuracy_validation")
    
    # Test cases: [current_replicas, cpu_total, memory_total, action, expected_reward_sign]
    test_cases = [
        [8, 2.0, 1600000000, "Scale Down", "positive"],  # Over-provisioned
        [2, 4.0, 3200000000, "Scale Up", "positive"],    # Under-provisioned  
        [5, 4.0, 2000000000, "Keep Same", "neutral"],    # Well-sized
        [1, 0.5, 500000000, "Scale Down", "negative"],   # Can't scale below 1
        [15, 3.0, 3000000000, "Scale Up", "negative"],   # Already too many
        [3, 0.6, 450000000, "Scale Down", "positive"],   # Low load, should scale down
        [12, 6.0, 6000000000, "Scale Up", "negative"],   # High resource usage but many replicas
    ]
    
    passed = 0
    failed = 0
    
    for replicas, cpu, memory, action, expected in test_cases:
        # Create test state
        current_state = {
            'kube_deployment_spec_replicas': replicas,
            'kube_pod_container_resource_limits_cpu': cpu,
            'kube_pod_container_resource_limits_memory': memory,
            'kube_deployment_status_replicas_unavailable': 0,
            'kube_pod_container_status_ready': replicas,
            'kube_pod_container_status_running': replicas,
            'kube_deployment_status_observed_generation': 1,
            'node_network_up': 1,
            'kube_pod_container_status_last_terminated_exitcode': 0,
        }
        
        # Next state (simulate action effect)
        next_state = current_state.copy()
        if action == "Scale Up":
            next_state['kube_deployment_spec_replicas'] = replicas + 1
            next_state['kube_pod_container_status_ready'] = replicas + 1
            next_state['kube_pod_container_status_running'] = replicas + 1
        elif action == "Scale Down" and replicas > 1:
            next_state['kube_deployment_spec_replicas'] = replicas - 1
            next_state['kube_pod_container_status_ready'] = replicas - 1
            next_state['kube_pod_container_status_running'] = replicas - 1
        
        try:
            reward = calculate_stable_reward(current_state, next_state, action, replicas)
            
            if expected == "positive":
                actual_sign = "positive" if reward > 0 else "negative" if reward < 0 else "neutral"
            elif expected == "negative":
                actual_sign = "positive" if reward > 0 else "negative" if reward < 0 else "neutral"
            else:  # neutral
                actual_sign = "neutral" if abs(reward) < 0.1 else ("positive" if reward > 0 else "negative")
            
            logger.info(f"REWARD_TEST: {replicas} replicas, {action} -> Reward: {reward:.2f} ({actual_sign})")
            
            if (expected == "positive" and reward > 0) or \
               (expected == "negative" and reward < 0) or \
               (expected == "neutral" and abs(reward) < 0.5):
                logger.info("✅ PASSED")
                passed += 1
            else:
                logger.warning(f"❌ FAILED: Expected {expected}, got {actual_sign}")
                failed += 1
                
        except Exception as e:
            logger.error(f"❌ ERROR: Test failed with exception: {e}")
            failed += 1
    
    logger.info(f"REWARD_TEST: validation_complete passed={passed} failed={failed}")
    return {"passed": passed, "failed": failed, "accuracy": passed / (passed + failed) if (passed + failed) > 0 else 0}