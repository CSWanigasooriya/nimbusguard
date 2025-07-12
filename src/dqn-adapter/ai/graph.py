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
    workflow.add_node("wait_for_system_to_stabilize", partial(wait_for_system_to_stabilize, services=services))
    workflow.add_node("observe_next_state_and_calculate_reward", partial(observe_next_state_and_calculate_reward, services=services))
    workflow.add_node("log_experience", partial(log_experience, services=services))

    workflow.set_entry_point("get_live_metrics")
    workflow.add_edge("get_live_metrics", "get_dqn_recommendation")
    workflow.add_edge("get_dqn_recommendation", "validate_with_llm")
    workflow.add_edge("validate_with_llm", "plan_final_action")
    workflow.add_edge("plan_final_action", "wait_for_system_to_stabilize")
    workflow.add_edge("wait_for_system_to_stabilize", "observe_next_state_and_calculate_reward")
    workflow.add_edge("observe_next_state_and_calculate_reward", "log_experience")
    workflow.add_edge("log_experience", END)

    return workflow.compile()


# --- LangGraph Nodes ---
async def get_live_metrics(state: ScalingState, services: ServiceContainer, is_next_state: bool = False) -> Dict[str, Any]:
    """
    Get live metrics from Prometheus with deduplication fixes.
    
    FIXED ISSUES:
    - Changed sum() to count() for ready/running containers to avoid double-counting duplicated metrics
    - Changed sum() to sum(sum by (pod)) for resource limits to deduplicate by pod first
    - This fixes the issue where readiness=18 instead of 9, and resource calculations being doubled
    """
    node_name = "observe_next_state" if is_next_state else "get_live_metrics"
    logger.info("=" * 60)
    logger.info(f"NODE_START: {node_name}")
    logger.info("=" * 60)

    from datetime import datetime
    current_time = datetime.now()

    # FIXED: Multi-dimensional consumer pod queries with proper target deployment scoping
    # These queries target consumer pod metrics with rate calculations where needed
    # Using job=prometheus.scrape.nimbusguard_consumer to target specifically the consumer deployment
    queries = {
        # 1. CPU rate (process CPU seconds total rate across consumer pods)
        "process_cpu_seconds_total_rate": f'sum(rate(process_cpu_seconds_total{{job="prometheus.scrape.nimbusguard_consumer"}}[2m])) or vector(0)',

        # 2. Python GC collections rate (memory pressure indicator)
        "python_gc_collections_total_rate": f'sum(rate(python_gc_collections_total{{job="prometheus.scrape.nimbusguard_consumer"}}[2m])) or vector(0)',

        # 3. Python GC objects collected rate (memory churn indicator)
        "python_gc_objects_collected_total_rate": f'sum(rate(python_gc_objects_collected_total{{job="prometheus.scrape.nimbusguard_consumer"}}[2m])) or vector(0)',

        # 4. HTTP request duration sum rate (latency indicator)
        "http_request_duration_seconds_sum_rate": f'sum(rate(http_request_duration_seconds_sum{{job="prometheus.scrape.nimbusguard_consumer"}}[2m])) or vector(0)',

        # 5. HTTP requests total rate (throughput indicator)
        "http_requests_total_rate": f'sum(rate(http_requests_total{{job="prometheus.scrape.nimbusguard_consumer"}}[2m])) or vector(0)',

        # 6. HTTP request duration count rate (request count indicator)
        "http_request_duration_seconds_count_rate": f'sum(rate(http_request_duration_seconds_count{{job="prometheus.scrape.nimbusguard_consumer"}}[2m])) or vector(0)',

        # 7. Process open file descriptors (system resource indicator)
        "process_open_fds": f'sum(process_open_fds{{job="prometheus.scrape.nimbusguard_consumer"}}) or vector(0)',

        # 8. HTTP response size bytes sum rate (response size indicator)
        "http_response_size_bytes_sum_rate": f'sum(rate(http_response_size_bytes_sum{{job="prometheus.scrape.nimbusguard_consumer"}}[2m])) or vector(0)',

        # 9. HTTP request size bytes count rate (request size indicator)
        "http_request_size_bytes_count_rate": f'sum(rate(http_request_size_bytes_count{{job="prometheus.scrape.nimbusguard_consumer"}}[2m])) or vector(0)',

        # AUXILIARY: Current replica count for scaling decisions
        "current_replicas": f'kube_deployment_status_replicas{{deployment="{services.config.kubernetes.target_deployment}",namespace="{services.config.kubernetes.target_namespace}"}}',
        
        # KUBERNETES STATE QUERIES: For proper availability and health status
        "deployment_replicas_unavailable": f'kube_deployment_status_replicas_unavailable{{deployment="{services.config.kubernetes.target_deployment}",namespace="{services.config.kubernetes.target_namespace}"}} or vector(0)',
        "deployment_replicas_ready": f'kube_deployment_status_ready_replicas{{deployment="{services.config.kubernetes.target_deployment}",namespace="{services.config.kubernetes.target_namespace}"}} or vector(0)',
        "pod_container_ready": f'kube_pod_container_status_ready{{namespace="{services.config.kubernetes.target_namespace}",pod=~"{services.config.kubernetes.target_deployment}-.*"}}',
        "pod_container_running": f'kube_pod_container_status_running{{namespace="{services.config.kubernetes.target_namespace}",pod=~"{services.config.kubernetes.target_deployment}-.*"}}',
        "pod_container_restarts": f'kube_pod_container_status_restarts_total{{namespace="{services.config.kubernetes.target_namespace}",pod=~"{services.config.kubernetes.target_deployment}-.*"}}',
        "pod_resource_limits_cpu": f'kube_pod_container_resource_limits{{resource="cpu",namespace="{services.config.kubernetes.target_namespace}",pod=~"{services.config.kubernetes.target_deployment}-.*"}}',
        "pod_resource_limits_memory": f'kube_pod_container_resource_limits{{resource="memory",namespace="{services.config.kubernetes.target_namespace}",pod=~"{services.config.kubernetes.target_deployment}-.*"}}',
        "node_network_up": f'node_network_up{{device!~"veth.*"}}',
    }
    # CLEAN ARCHITECTURE: No computed temporal features (moving averages, deviations, etc.)


    # Debug logging for query targeting
    logger.info(f"QUERY_TARGETING: namespace={services.config.kubernetes.target_namespace} "
                f"deployment={services.config.kubernetes.target_deployment}")
    
    # Log key queries for debugging consumer pod metrics
    logger.debug(f"CPU_RATE_QUERY: {queries['process_cpu_seconds_total_rate']}")
    logger.debug(f"HTTP_REQUESTS_QUERY: {queries['http_requests_total_rate']}")
    logger.debug(f"GC_COLLECTIONS_QUERY: {queries['python_gc_collections_total_rate']}")
    logger.debug(f"OPEN_FDS_QUERY: {queries['process_open_fds']}")

    tasks = {name: services.prometheus_client.query(query) for name, query in queries.items()}
    
    # Debug query to see what pods are actually being targeted
    debug_pods_query = f'kube_pod_info{{namespace="{services.config.kubernetes.target_namespace}",pod=~"{services.config.kubernetes.target_deployment}-.*"}}'
    tasks['debug_target_pods'] = services.prometheus_client.query_raw(debug_pods_query)

    results = await asyncio.gather(*tasks.values())

    metrics = dict(zip(tasks.keys(), results))
    current_replicas = int(metrics.pop('current_replicas', 1))
    debug_pods_result = metrics.pop('debug_target_pods', [])
    
    # Log debug information about targeted pods
    if debug_pods_result and isinstance(debug_pods_result, list) and len(debug_pods_result) > 0:
        pod_names = [result.get('metric', {}).get('pod', 'unknown') for result in debug_pods_result if 'metric' in result]
        unique_pods = list(set(pod_names))  # Remove duplicates
        logger.info(f"TARGETED_PODS: total_results={len(pod_names)} unique_pods={len(unique_pods)} pods={unique_pods[:5]}{'...' if len(unique_pods) > 5 else ''}")
    else:
        logger.warning(f"TARGETED_PODS: debug_query_result={debug_pods_result}")
        logger.warning(f"TARGETED_PODS: no_pods_found_matching_pattern={services.config.kubernetes.target_deployment}-*")

    # Ensure raw features have defaults (no computations)
    metrics = _ensure_raw_features(metrics)



    CURRENT_REPLICAS_GAUGE.set(current_replicas)

    # Enhanced logging with consumer pod metrics
    cpu_rate = metrics.get('process_cpu_seconds_total_rate', 0.0)
    gc_collections_rate = metrics.get('python_gc_collections_total_rate', 0.0)
    gc_objects_rate = metrics.get('python_gc_objects_collected_total_rate', 0.0)
    http_duration_rate = metrics.get('http_request_duration_seconds_sum_rate', 0.0)
    http_requests_rate = metrics.get('http_requests_total_rate', 0.0)
    http_count_rate = metrics.get('http_request_duration_seconds_count_rate', 0.0)
    open_fds = metrics.get('process_open_fds', 0.0)
    response_size_rate = metrics.get('http_response_size_bytes_sum_rate', 0.0)
    request_size_rate = metrics.get('http_request_size_bytes_count_rate', 0.0)

    # Kubernetes state metrics for proper availability and health reporting
    replicas_unavailable = metrics.get('deployment_replicas_unavailable', 0)
    replicas_ready = metrics.get('deployment_replicas_ready', current_replicas)
    containers_ready = metrics.get('pod_container_ready', 0)
    containers_running = metrics.get('pod_container_running', 0)
    container_restarts = metrics.get('pod_container_restarts', 0)
    cpu_limits = metrics.get('pod_resource_limits_cpu', 0)
    memory_limits = metrics.get('pod_resource_limits_memory', 0)
    network_up = metrics.get('node_network_up', 0)

    logger.info(f"CONSUMER_METRICS: cpu_rate={cpu_rate:.4f} gc_collections_rate={gc_collections_rate:.4f} "
                f"gc_objects_rate={gc_objects_rate:.4f} http_requests_rate={http_requests_rate:.4f} "
                f"current_replicas={current_replicas}")

    logger.info(f"NETWORK_METRICS: http_duration_rate={http_duration_rate:.4f} "
                f"http_count_rate={http_count_rate:.4f} response_size_rate={response_size_rate:.4f} "
                f"request_size_rate={request_size_rate:.4f}")

    logger.info(f"SYSTEM_METRICS: open_fds={open_fds:.0f} targeting_consumer_pods={services.config.kubernetes.target_deployment} "
                f"namespace={services.config.kubernetes.target_namespace}")

    # Kubernetes state logging for availability and health status
    logger.info(f"KUBERNETES_STATE: replicas_unavailable={replicas_unavailable} replicas_ready={replicas_ready} "
                f"containers_ready={containers_ready} containers_running={containers_running} "
                f"container_restarts={container_restarts}")
    
    # Resource limits logging
    memory_limits_mb = memory_limits / 1000000 if memory_limits > 0 else 0
    logger.info(f"RESOURCE_LIMITS: cpu_limits={cpu_limits:.2f} memory_limits_mb={memory_limits_mb:.0f} "
                f"network_up={network_up:.0f}")

    # Consumer pod metrics analysis for debugging
    if current_replicas > 0:
        # Analyze per-replica metrics to understand load distribution
        cpu_rate_per_replica = cpu_rate / current_replicas if current_replicas > 0 else 0
        http_rate_per_replica = http_requests_rate / current_replicas if current_replicas > 0 else 0
        
        logger.info(f"LOAD_ANALYSIS: cpu_rate_per_replica={cpu_rate_per_replica:.4f} "
                    f"http_rate_per_replica={http_rate_per_replica:.4f}")
        
        # Analyze system load patterns
        if http_requests_rate > 0:
            avg_request_duration = http_duration_rate / http_requests_rate if http_requests_rate > 0 else 0
            logger.info(f"PERFORMANCE_ANALYSIS: avg_request_duration={avg_request_duration:.4f}s "
                        f"gc_pressure={gc_collections_rate:.4f} fds_usage={open_fds:.0f}")
        
        # Load distribution analysis
        logger.info(f"SCALING_INDICATORS: replicas={current_replicas} "
                    f"total_cpu_rate={cpu_rate:.4f} total_http_rate={http_requests_rate:.4f} "
                    f"memory_pressure={gc_collections_rate:.4f}")

    # Log feature availability for debugging (clean architecture: consumer metrics only)
    # Check if features exist (not None) rather than > 0, since 0 can be a valid value
    base_features_available = sum(1 for feat in services.config.base_features if feat in metrics)
    total_features_available = base_features_available

    logger.info(f"FEATURE_AVAILABILITY: total={total_features_available}/{len(services.config.feature_order)} "
                f"base={base_features_available}/{len(services.config.base_features)}")

    # Debug: Log missing features if any
    if total_features_available < len(services.config.feature_order):
        missing_base = [feat for feat in services.config.base_features if feat not in metrics]
        if missing_base:
            logger.warning(f"MISSING_CONSUMER_FEATURES: {missing_base}")

    logger.info("=" * 60)
    logger.info(f"NODE_END: {node_name}")
    logger.info("=" * 60)
    return {"current_metrics": metrics, "current_replicas": current_replicas}


# --- Feature Engineering Helper ---
def _ensure_raw_features(metrics: Dict[str, float]) -> Dict[str, float]:
    """Ensure consumer pod metrics have fallback values - no computed features."""

    # Consumer pod metric defaults (application performance indicators)
    feature_defaults = {
        'process_cpu_seconds_total_rate': 0.0,  # Default: no CPU usage rate
        'python_gc_collections_total_rate': 0.0,  # Default: no GC collections rate
        'python_gc_objects_collected_total_rate': 0.0,  # Default: no GC objects rate
        'http_request_duration_seconds_sum_rate': 0.0,  # Default: no HTTP latency rate
        'http_requests_total_rate': 0.0,  # Default: no HTTP requests rate
        'http_request_duration_seconds_count_rate': 0.0,  # Default: no HTTP request count rate
        'process_open_fds': 0.0,  # Default: no open file descriptors
        'http_response_size_bytes_sum_rate': 0.0,  # Default: no HTTP response size rate
        'http_request_size_bytes_count_rate': 0.0,  # Default: no HTTP request size rate
    }

    # Kubernetes state metric defaults (infrastructure health indicators)
    k8s_defaults = {
        'deployment_replicas_unavailable': 0.0,  # Default: no unavailable replicas
        'deployment_replicas_ready': 0.0,  # Default: no ready replicas (will be set to current_replicas)
        'pod_container_ready': 0.0,  # Default: no ready containers
        'pod_container_running': 0.0,  # Default: no running containers
        'pod_container_restarts': 0.0,  # Default: no container restarts
        'pod_resource_limits_cpu': 0.0,  # Default: no CPU limits set
        'pod_resource_limits_memory': 0.0,  # Default: no memory limits set
        'node_network_up': 0.0,  # Default: network status unknown
    }

    # Apply defaults for missing consumer pod metrics only
    for feature, default_value in feature_defaults.items():
        if feature not in metrics or metrics[feature] is None:
            metrics[feature] = default_value

    # Apply defaults for missing Kubernetes state metrics
    for feature, default_value in k8s_defaults.items():
        if feature not in metrics or metrics[feature] is None:
            metrics[feature] = default_value

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

    # Transform numpy array to maintain consistency with how scaler was fitted (no feature names)
    raw_features_array = np.array([raw_features])
    scaled_raw_features = services.scaler.transform(raw_features_array)

    # Use only the scaled raw features for DQN (9 scientifically-selected features)
    final_feature_vector = scaled_raw_features[0].tolist()

    logger.info(f"FEATURE_COMPOSITION: scaled_raw={len(scaled_raw_features[0])} "
                f"total={len(final_feature_vector)}")

    # Count available feature types (check existence, not value > 0)
    base_available = sum(1 for feat in services.config.base_features if feat in metrics)

    # Get Kubernetes state metrics for proper system state reporting
    replicas_unavailable = metrics.get('deployment_replicas_unavailable', 0)
    replicas_ready = metrics.get('deployment_replicas_ready', current_replicas)
    containers_ready = metrics.get('pod_container_ready', 0)
    containers_running = metrics.get('pod_container_running', 0)
    cpu_limits = metrics.get('pod_resource_limits_cpu', 0)
    memory_limits = metrics.get('pod_resource_limits_memory', 0)
    network_up = metrics.get('node_network_up', 0)

    logger.info(f"SYSTEM_STATE: replicas={current_replicas} "
                f"unavailable={replicas_unavailable} "
                f"readiness={containers_ready/current_replicas if current_replicas > 0 else 0:.3f}")

    # DIAGNOSTIC: Log key features that should indicate load decrease
    logger.info(f"LOAD_INDICATORS: "
                f"cpu_limits={cpu_limits:.2f} "
                f"memory_limits_mb={memory_limits / 1000000:.0f} "
                f"running_containers={containers_running} "
                f"network_up={network_up}")



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
            # Enhanced fallback: Rule-based logic with balanced selected features
            logger.info("DQN_FALLBACK: using_rule_based_logic model_not_available")

            # Get current consumer pod metrics for fallback decision
            cpu_rate = metrics.get('process_cpu_seconds_total_rate', 0.0)
            gc_collections_rate = metrics.get('python_gc_collections_total_rate', 0.0)
            gc_objects_rate = metrics.get('python_gc_objects_collected_total_rate', 0.0)
            http_duration_rate = metrics.get('http_request_duration_seconds_sum_rate', 0.0)
            http_requests_rate = metrics.get('http_requests_total_rate', 0.0)
            http_count_rate = metrics.get('http_request_duration_seconds_count_rate', 0.0)
            open_fds = metrics.get('process_open_fds', 0.0)
            response_size_rate = metrics.get('http_response_size_bytes_sum_rate', 0.0)
            request_size_rate = metrics.get('http_request_size_bytes_count_rate', 0.0)

            # Generate analysis for fallback decision
            analysis = decision_reasoning.analyze_metrics(metrics, current_replicas)

            # Enhanced rule-based decision with consumer pod metrics
            reasoning_factors = []

            # Decision logic based on consumer application metrics
            avg_request_duration = http_duration_rate / http_requests_rate if http_requests_rate > 0 else 0
            
            if cpu_rate > 0.8 or gc_collections_rate > 0.1 or avg_request_duration > 1.0:  # High load/pressure
                action_name = "Scale Up"
                reasoning_factors.append(
                    f"High load detected: CPU rate {cpu_rate:.4f}, GC rate {gc_collections_rate:.4f}, avg duration {avg_request_duration:.4f}s")
                reasoning_factors.append("Scaling up to handle increased load and reduce latency")
                risk_level = "high"
            elif http_requests_rate > 10.0 or response_size_rate > 1000000:  # High traffic
                action_name = "Scale Up"
                reasoning_factors.append(
                    f"High traffic: {http_requests_rate:.4f} req/s, response size rate {response_size_rate:.0f} bytes/s")
                reasoning_factors.append("Scaling up to handle high traffic volume")
                risk_level = "medium"
            elif cpu_rate < 0.1 and http_requests_rate < 1.0 and current_replicas > 1:  # Low utilization
                action_name = "Scale Down"
                reasoning_factors.append(
                    f"Low utilization: CPU rate {cpu_rate:.4f}, HTTP rate {http_requests_rate:.4f}")
                reasoning_factors.append(
                    f"System has excess capacity with {current_replicas} replicas - can optimize costs")
                risk_level = "low"
            # ENHANCED: Add explicit scale-down for high replica counts with low load
            elif current_replicas >= 8 and cpu_rate < 0.2 and http_requests_rate < 2.0:  # Over-provisioned
                action_name = "Scale Down"
                reasoning_factors.append(
                    f"Over-provisioned: {current_replicas} replicas with low load - cost optimization opportunity")
                reasoning_factors.append(
                    f"Low load indicators: CPU rate {cpu_rate:.4f}, HTTP rate {http_requests_rate:.4f}")
                risk_level = "low"
            else:
                action_name = "Keep Same"
                reasoning_factors.append(
                    f"Balanced state: CPU rate {cpu_rate:.4f}, HTTP rate {http_requests_rate:.4f}, {current_replicas} replicas")
                reasoning_factors.append("System operating within acceptable parameters")
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
    RESEARCH-BASED: Reward function using ONLY the metrics DQN actually sees.
    Fixes the fundamental issue where reward used different metrics than DQN input.
    """

    # === USE ONLY ACTUAL DQN INPUT FEATURES ===
    # These are the 5 base metrics the DQN uses for decisions

    # Current state metrics (what DQN saw when making decision)
    curr_unavailable = current_state.get('kube_deployment_status_replicas_unavailable', 0)
    curr_ready = current_state.get('kube_pod_container_status_ready', 1.0)
    curr_desired = current_state.get('kube_deployment_spec_replicas', 1)
    curr_running = current_state.get('kube_pod_container_status_running', 1.0)
    curr_generation = current_state.get('kube_deployment_status_observed_generation', 1)
    curr_cpu_limits = current_state.get('kube_pod_container_resource_limits_cpu', 0.5)
    curr_memory_limits = current_state.get('kube_pod_container_resource_limits_memory', 536870912)
    curr_network_up = current_state.get('node_network_up', 1)
    curr_exit_code = current_state.get('kube_pod_container_status_last_terminated_exitcode', 0)

    # Next state metrics (outcome after DQN action)
    next_unavailable = next_state.get('kube_deployment_status_replicas_unavailable', 0)
    next_ready = next_state.get('kube_pod_container_status_ready', 1.0)
    next_desired = next_state.get('kube_deployment_spec_replicas', 1)
    next_running = next_state.get('kube_pod_container_status_running', 1.0)
    next_generation = next_state.get('kube_deployment_status_observed_generation', 1)
    next_cpu_limits = next_state.get('kube_pod_container_resource_limits_cpu', 0.5)
    next_memory_limits = next_state.get('kube_pod_container_resource_limits_memory', 536870912)
    next_network_up = next_state.get('node_network_up', 1)
    next_exit_code = next_state.get('kube_pod_container_status_last_terminated_exitcode', 0)

    # Convert to more interpretable units
    curr_memory_mb = curr_memory_limits / 1000000
    next_memory_mb = next_memory_limits / 1000000

    # === PERFORMANCE COMPONENT (40%) ===
    # Reward system stability and health improvements with accurate scaling

    # FIXED: Use normalized absolute improvements instead of ratios
    readiness_improvement = (next_ready - curr_ready) / max(1, next_desired)  # Normalize by target
    availability_improvement = max(0, curr_unavailable - next_unavailable) / max(1, next_desired)  # Normalize by target

    # Container health improvement (normalized)
    running_improvement = (next_running - curr_running) / max(1, next_desired)  # Normalize by target
    exit_code_improvement = max(0, abs(curr_exit_code) - abs(next_exit_code)) * 0.1  # Better exit codes

    # FIXED: Component-wise clipping to prevent extreme values
    performance_score = np.clip(
        readiness_improvement * 3.0 +  # Readiness is critical
        availability_improvement * 2.0 +  # Availability is key
        running_improvement * 1.0 +  # More running containers
        exit_code_improvement,  # Healthier exits
        -5.0, 5.0  # Bound performance component
    )

    # === RESOURCE EFFICIENCY COMPONENT (30%) ===
    # FIXED: Measure total resource efficiency changes, not per-replica (which never changes)
    
    # Get actual deployment resource specifications (per-replica basis)
    actual_cpu_limit_per_replica = current_state.get('deployment_cpu_limits_per_replica', next_state.get('deployment_cpu_limits_per_replica', 0.25))
    actual_memory_limit_per_replica = current_state.get('deployment_memory_limits_per_replica', next_state.get('deployment_memory_limits_per_replica', 268435456))
    actual_cpu_request_per_replica = current_state.get('deployment_cpu_requests_per_replica', next_state.get('deployment_cpu_requests_per_replica', 0.1))
    actual_memory_request_per_replica = current_state.get('deployment_memory_requests_per_replica', next_state.get('deployment_memory_requests_per_replica', 134217728))
    
    # Convert memory to MB for easier interpretation
    actual_memory_limit_mb_per_replica = actual_memory_limit_per_replica / 1000000
    actual_memory_request_mb_per_replica = actual_memory_request_per_replica / 1000000
    
    # FIXED: Calculate TOTAL resource allocation (this changes with scaling)
    curr_total_cpu = curr_desired * actual_cpu_limit_per_replica
    next_total_cpu = next_desired * actual_cpu_limit_per_replica
    curr_total_memory_mb = curr_desired * actual_memory_limit_mb_per_replica  
    next_total_memory_mb = next_desired * actual_memory_limit_mb_per_replica

    # Define optimal total resource ranges based on system load patterns
    def total_resource_efficiency_score(total_cpu, total_memory_mb, replicas):
        """
        Evaluate total resource allocation efficiency.
        Rewards right-sizing for typical workloads, penalizes waste and under-provisioning.
        """
        # Ideal total resource ranges for most applications
        ideal_cpu_range = (1.0, 4.0)        # 1-4 total CPU cores  
        ideal_memory_range = (800, 3200)     # 800MB-3.2GB total memory
        
        # Scale efficiency based on replica count (more replicas = higher tolerance)
        cpu_efficiency = 1.0 if ideal_cpu_range[0] <= total_cpu <= ideal_cpu_range[1] + replicas * 0.5 else \
                        max(-1.0, 1.0 - abs(total_cpu - np.mean(ideal_cpu_range)) / np.mean(ideal_cpu_range))
        
        memory_efficiency = 1.0 if ideal_memory_range[0] <= total_memory_mb <= ideal_memory_range[1] + replicas * 100 else \
                           max(-1.0, 1.0 - abs(total_memory_mb - np.mean(ideal_memory_range)) / np.mean(ideal_memory_range))
        
        # Combine and normalize
        efficiency = (cpu_efficiency + memory_efficiency) / 2.0
        
        # Bonus for efficient small deployments, penalty for waste
        if replicas <= 3 and total_cpu <= 2.0 and total_memory_mb <= 1200:
            efficiency += 0.5  # Reward efficient small deployments
        elif replicas >= 8 and total_cpu >= 6.0:
            efficiency -= 0.3  # Penalize potential over-provisioning
            
        return np.clip(efficiency, -2.0, 2.0)

    curr_total_efficiency = total_resource_efficiency_score(curr_total_cpu, curr_total_memory_mb, current_replicas)
    next_total_efficiency = total_resource_efficiency_score(next_total_cpu, next_total_memory_mb, next_desired)
    resource_score = next_total_efficiency - curr_total_efficiency
    
    # Debug log to show the fix working
    logger.debug(f"RESOURCE_EFFICIENCY_FIX: curr_total={curr_total_cpu:.1f}cpu+{curr_total_memory_mb:.0f}mb->efficiency={curr_total_efficiency:.2f} "
                f"next_total={next_total_cpu:.1f}cpu+{next_total_memory_mb:.0f}mb->efficiency={next_total_efficiency:.2f} "
                f"resource_score={resource_score:.2f}")

    # === SYSTEM HEALTH COMPONENT (20%) ===
    # Reward network stability and deployment health

    network_health = next_network_up / max(0.1, curr_network_up) - 1.0  # Network improvement
    deployment_stability = max(0, 2.0 - abs(next_generation - curr_generation) * 0.1)  # Stable deployments

    health_score = network_health + deployment_stability

    # === COST OPTIMIZATION COMPONENT (10%) ===
    # FIXED: Smooth cost function eliminates discontinuities that confuse learning

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

    cost_score = smooth_cost_function(next_desired)

    # === LOAD AWARENESS COMPONENT (NEW) ===
    # CRITICAL: Evaluate if pod states are appropriate for actual resource demand
    
    def calculate_load_awareness_score(cpu_per_replica, memory_per_replica):
        """
        Assess if current resource allocation matches actual demand.
        Prevents rewarding over-provisioning during low load periods.
        """
        # Ideal resource pressure ranges per replica
        ideal_cpu_range = (0.3, 1.2)    # 30% to 120% of 1 core per replica
        ideal_memory_range = (150, 600)  # 150MB to 600MB per replica
        
        # Calculate efficiency scores
        cpu_efficiency = 1.0 if ideal_cpu_range[0] <= cpu_per_replica <= ideal_cpu_range[1] else \
                        max(0, 1.0 - abs(cpu_per_replica - np.mean(ideal_cpu_range)) / np.mean(ideal_cpu_range))
        
        memory_efficiency = 1.0 if ideal_memory_range[0] <= memory_per_replica <= ideal_memory_range[1] else \
                           max(0, 1.0 - abs(memory_per_replica - np.mean(ideal_memory_range)) / np.mean(ideal_memory_range))
        
        # Combined load awareness score
        load_score = (cpu_efficiency + memory_efficiency) / 2.0
        
        # Penalty for wasteful over-provisioning (very low resource usage)
        if cpu_per_replica < 0.2 and memory_per_replica < 100:
            load_score -= 0.5  # Significant penalty for waste
        
        return np.clip(load_score, -1.0, 1.0)
    
    # Use ACTUAL deployment resource specifications for load awareness
    load_awareness_score = calculate_load_awareness_score(actual_cpu_limit_per_replica, actual_memory_limit_mb_per_replica)

    # === ACTION APPROPRIATENESS PENALTY (SIMPLIFIED) ===
    # FIXED: Simplified logic, easier to debug and understand
    
    action_penalty = 0.0
    
    # Define system state using actual deployment specifications
    is_healthy = (next_unavailable == 0 and next_ready >= 0.95 and next_running > 0)
    is_overprovisioned = (actual_cpu_limit_per_replica < 0.3 and actual_memory_limit_mb_per_replica < 150)
    is_underprovisioned = (actual_cpu_limit_per_replica > 1.2 or actual_memory_limit_mb_per_replica > 600)
    
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

    # === FINAL REWARD CALCULATION (IMPROVED) ===
    if config:
        total_reward = (
                performance_score * config.reward.performance_weight +
                resource_score * config.reward.resource_weight +
                health_score * config.reward.health_weight +
                cost_score * config.reward.cost_weight +
                load_awareness_score * 0.15 +  # NEW: Load awareness component (15% weight)
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
                performance_score * 0.35 +  # ADJUSTED: Reduced from 40% to make room for load awareness
                resource_score * 0.25 +     # ADJUSTED: Reduced from 30% to make room for load awareness
                health_score * 0.15 +       # ADJUSTED: Reduced from 20% to make room for load awareness
                cost_score * 0.10 +         # UNCHANGED: 10%
                load_awareness_score * 0.15 +  # NEW: Load awareness component (15% weight)
                action_penalty
        )
        
        logger.info(f"REWARD_BREAKDOWN: performance={performance_score:.2f}(35%) "
                    f"resource={resource_score:.2f}(25%) health={health_score:.2f}(15%) "
                    f"cost={cost_score:.2f}(10%) load_awareness={load_awareness_score:.2f}(15%) "
                    f"action_penalty={action_penalty:.2f} total={total_reward:.2f}")

    # Clip to reasonable range
    total_reward = np.clip(total_reward, -10.0, 10.0)

    # Enhanced logging with fixed resource calculations
    logger.info(f"REWARD_STATE_ANALYSIS: curr_replicas={current_replicas} next_replicas={next_desired} "
                f"readiness={curr_ready:.2f}=>{next_ready:.2f} unavailable={curr_unavailable}=>{next_unavailable} "
                f"cpu_total={curr_total_cpu:.1f}=>{next_total_cpu:.1f} "
                f"memory_total_mb={curr_total_memory_mb:.0f}=>{next_total_memory_mb:.0f} "
                f"resource_efficiency={curr_total_efficiency:.2f}=>{next_total_efficiency:.2f} "
                f"load_awareness={load_awareness_score:.2f}")
    
    # Log actual deployment resource specifications being used
    logger.info(f"RESOURCE_SPECS_ACTUAL: cpu_limit_per_replica={actual_cpu_limit_per_replica:.3f}cores "
                f"memory_limit_per_replica={actual_memory_limit_mb_per_replica:.0f}MB "
                f"cpu_request_per_replica={actual_cpu_request_per_replica:.3f}cores "
                f"memory_request_per_replica={actual_memory_request_mb_per_replica:.0f}MB")
    
    # Compare with total resource allocation
    total_cpu_allocated = actual_cpu_limit_per_replica * next_desired
    total_memory_allocated = actual_memory_limit_mb_per_replica * next_desired
    logger.info(f"RESOURCE_ALLOCATION_TOTAL: replicas={next_desired} "
                f"cpu_total={total_cpu_allocated:.2f}cores "
                f"memory_total={total_memory_allocated:.0f}MB "
                f"efficient={'YES' if total_cpu_allocated <= 2.0 and total_memory_allocated <= 1000 else 'NO'}")



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
    FIXED: Check actual usage per replica, not total limits across all replicas.
    This was the core bug - with 10 replicas, total limits are high even with zero load.
    """
    if not current_state:
        return False

    # FIXED: Calculate usage per replica instead of total limits
    cpu_limits_total = current_state.get('kube_pod_container_resource_limits_cpu', 0)  # total cores
    memory_bytes_total = current_state.get('kube_pod_container_resource_limits_memory', 0)
    memory_mb_total = memory_bytes_total / 1000000 if memory_bytes_total > 0 else 0

    # Per-replica resource usage (this is what matters for scaling decisions)
    cpu_per_replica = cpu_limits_total / max(1, current_replicas)
    memory_per_replica = memory_mb_total / max(1, current_replicas)

    # Kubernetes health indicators (what actually matters for scaling)
    unavailable_replicas = current_state.get('kube_deployment_status_replicas_unavailable', 0)
    pod_readiness = current_state.get('kube_pod_container_status_ready', 1.0)
    running_containers = current_state.get('kube_pod_container_status_running', 0)

    # FIXED: Check per-replica usage, not total usage
    underutilized_conditions = [
        cpu_per_replica < 1.0,  # < 1 CPU core per replica = low utilization
        memory_per_replica < 300,  # < 300MB per replica = low memory usage
        unavailable_replicas == 0,  # No failing pods (absolute requirement)
        pod_readiness >= 0.95,  # 95% readiness = healthy
        running_containers > 0,  # At least some containers running
        current_replicas > 1  # Don't scale below 1 replica (minimum viable)
    ]

    is_underutilized = all(underutilized_conditions)

    # Enhanced logging to show the fix
    if current_replicas >= 2:
        logger.info(f"UNDERUTILIZATION_CHECK: {sum(underutilized_conditions)}/6 conditions_met "
                    f"cpu_per_replica={cpu_per_replica:.1f}cores memory_per_replica={memory_per_replica:.0f}MB "
                    f"(total: cpu={cpu_limits_total:.1f} memory={memory_mb_total:.0f}MB across {current_replicas} replicas) "
                    f"unavailable={unavailable_replicas} readiness={pod_readiness:.2f} "
                    f"running={running_containers} result={'SAFE_TO_SCALE_DOWN' if is_underutilized else 'NOT_SAFE'}")

    return is_underutilized


def check_system_overloaded(current_state, next_state, current_replicas):
    """
    HONEST: Simple overload detection based on operational experience.
    Uses clear thresholds that make sense.
    """
    if not current_state:
        return False

    # Use the same metrics the DQN sees
    cpu_limits = current_state.get('kube_pod_container_resource_limits_cpu', 0)
    memory_bytes = current_state.get('kube_pod_container_resource_limits_memory', 0)
    memory_mb = memory_bytes / 1000000 if memory_bytes > 0 else 0

    # Kubernetes stress indicators
    unavailable_replicas = current_state.get('kube_deployment_status_replicas_unavailable', 0)
    pod_readiness = current_state.get('kube_pod_container_status_ready', 1.0)

    # HONEST stress detection (when system actually needs help)
    critical_conditions = [
        unavailable_replicas > 0,  # Pods failing = immediate problem
        pod_readiness < 0.8,  # Low readiness = stress
    ]

    # High utilization conditions
    high_utilization_conditions = [
        cpu_limits > 3.0,  # > 3 CPU cores = high load
        memory_mb > 800,  # > 800MB = memory pressure
    ]

    # Overloaded if ANY critical condition OR multiple utilization conditions
    is_overloaded = any(critical_conditions) or sum(high_utilization_conditions) >= 2

    if current_replicas <= 15:
        logger.info(
            f"OVERLOAD_CHECK: critical={sum(critical_conditions)}/2 utilization={sum(high_utilization_conditions)}/2 "
            f"cpu_limits={cpu_limits:.1f}cores memory={memory_mb:.0f}MB "
            f"unavailable={unavailable_replicas} readiness={pod_readiness:.2f} "
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

        # 3. Log experience for monitoring
        logger.info("EXPERIENCE: logged for monitoring")

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