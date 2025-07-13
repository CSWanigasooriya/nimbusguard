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
from .rewards import calculate_stable_reward
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
    Get live metrics from Prometheus with enhanced failure detection and proactive scaling.
    
    ENHANCED FOR PROACTIVE SCALING:
    - Detects pod failures and restarts
    - Monitors resource pressure indicators
    - Predicts scaling needs before crashes occur
    - Uses historical failure patterns for better decisions
    """
    node_name = "observe_next_state" if is_next_state else "get_live_metrics"
    logger.info("=" * 60)
    logger.info(f"NODE_START: {node_name}")
    logger.info("=" * 60)

    from datetime import datetime
    current_time = datetime.now()

    # ENHANCED: Add failure detection and resource pressure queries
    queries = {
        # Core consumer metrics (FIXED to compute rates with _rate suffix for DQN compatibility)
        "process_cpu_seconds_total_rate": f'sum(rate(process_cpu_seconds_total{{job="prometheus.scrape.annotated_pods", instance=~".*8000"}}[30s])) or vector(0)',
        "process_resident_memory_bytes": f'sum(process_resident_memory_bytes{{job="prometheus.scrape.annotated_pods", instance=~".*8000"}}) or vector(0)',
        "process_virtual_memory_bytes": f'sum(process_virtual_memory_bytes{{job="prometheus.scrape.annotated_pods", instance=~".*8000"}}) or vector(0)',
        "http_request_duration_seconds_sum_rate": f'sum(rate(http_request_duration_seconds_sum{{job="prometheus.scrape.annotated_pods", instance=~".*8000"}}[30s])) or vector(0)',
        "http_requests_total_rate": f'sum(rate(http_requests_total{{job="prometheus.scrape.annotated_pods", instance=~".*8000"}}[30s])) or vector(0)',
        "http_request_duration_seconds_count_rate": f'sum(rate(http_request_duration_seconds_count{{job="prometheus.scrape.annotated_pods", instance=~".*8000"}}[30s])) or vector(0)',
        "process_open_fds": f'sum(process_open_fds{{job="prometheus.scrape.annotated_pods", instance=~".*8000"}}) or vector(0)',
        "http_response_size_bytes_sum_rate": f'sum(rate(http_response_size_bytes_sum{{job="prometheus.scrape.annotated_pods", instance=~".*8000"}}[30s])) or vector(0)',
        "http_request_size_bytes_count_rate": f'sum(rate(http_request_size_bytes_count{{job="prometheus.scrape.annotated_pods", instance=~".*8000"}}[30s])) or vector(0)',

        # Current replica count for scaling decisions
        "current_replicas": f'kube_deployment_status_replicas{{deployment="{services.config.kubernetes.target_deployment}",namespace="{services.config.kubernetes.target_namespace}"}}',
        
        # ENHANCED: Failure detection and resource pressure indicators
        "deployment_replicas_unavailable": f'kube_deployment_status_replicas_unavailable{{deployment="{services.config.kubernetes.target_deployment}",namespace="{services.config.kubernetes.target_namespace}"}} or vector(0)',
        "deployment_replicas_ready": f'kube_deployment_status_ready_replicas{{deployment="{services.config.kubernetes.target_deployment}",namespace="{services.config.kubernetes.target_namespace}"}} or vector(0)',
        "pod_container_ready": f'kube_pod_container_status_ready{{namespace="{services.config.kubernetes.target_namespace}",pod=~"{services.config.kubernetes.target_deployment}-.*"}}',
        "pod_container_running": f'kube_pod_container_status_running{{namespace="{services.config.kubernetes.target_namespace}",pod=~"{services.config.kubernetes.target_deployment}-.*"}}',
        "pod_container_restarts": f'kube_pod_container_status_restarts_total{{namespace="{services.config.kubernetes.target_namespace}",pod=~"{services.config.kubernetes.target_deployment}-.*"}}',
        
        # PROACTIVE: Resource pressure and failure prediction
        "pod_resource_limits_cpu": f'kube_pod_container_resource_limits{{resource="cpu",namespace="{services.config.kubernetes.target_namespace}",pod=~"{services.config.kubernetes.target_deployment}-.*"}}',
        "pod_resource_limits_memory": f'kube_pod_container_resource_limits{{resource="memory",namespace="{services.config.kubernetes.target_namespace}",pod=~"{services.config.kubernetes.target_deployment}-.*"}}',
        "pod_resource_requests_cpu": f'kube_pod_container_resource_requests{{resource="cpu",namespace="{services.config.kubernetes.target_namespace}",pod=~"{services.config.kubernetes.target_deployment}-.*"}} or vector(0)',
        "pod_resource_requests_memory": f'kube_pod_container_resource_requests{{resource="memory",namespace="{services.config.kubernetes.target_namespace}",pod=~"{services.config.kubernetes.target_deployment}-.*"}} or vector(0)',
        
        # CRITICAL: Failure indicators - recent restarts and exit codes
        "pod_container_restarts_recent": f'increase(kube_pod_container_status_restarts_total{{namespace="{services.config.kubernetes.target_namespace}",pod=~"{services.config.kubernetes.target_deployment}-.*"}}[5m]) or vector(0)',
        "pod_container_last_terminated_exitcode": f'kube_pod_container_status_last_terminated_exitcode{{namespace="{services.config.kubernetes.target_namespace}",pod=~"{services.config.kubernetes.target_deployment}-.*"}} or vector(0)',
        
        # PROACTIVE: Network and cluster health indicators
        "node_network_up": f'node_network_up{{device!~"veth.*"}}',
        "prometheus_scrape_up": f'up{{job="prometheus.scrape.annotated_pods", instance=~".*8000"}}',
        
        # ENHANCED: Historical failure patterns (5min and 15min windows)
        "restarts_5m": f'increase(kube_pod_container_status_restarts_total{{namespace="{services.config.kubernetes.target_namespace}",pod=~"{services.config.kubernetes.target_deployment}-.*"}}[5m]) or vector(0)',
        "restarts_15m": f'increase(kube_pod_container_status_restarts_total{{namespace="{services.config.kubernetes.target_namespace}",pod=~"{services.config.kubernetes.target_deployment}-.*"}}[15m]) or vector(0)',
    }

    # Debug logging for query targeting
    logger.info(f"QUERY_TARGETING: namespace={services.config.kubernetes.target_namespace} "
                f"deployment={services.config.kubernetes.target_deployment}")
    
    tasks = {name: services.prometheus_client.query(query) for name, query in queries.items()}
    
    # Debug query to see what consumer instances are actually being scraped
    debug_consumer_query = f'up{{job="prometheus.scrape.annotated_pods", instance=~".*8000"}}'
    tasks['debug_consumer_instances'] = services.prometheus_client.query_raw(debug_consumer_query)

    results = await asyncio.gather(*tasks.values())

    metrics = dict(zip(tasks.keys(), results))
    current_replicas = int(metrics.pop('current_replicas', 1))
    debug_consumer_result = metrics.pop('debug_consumer_instances', [])
    
    # Log debug information about consumer instances being scraped
    if debug_consumer_result and isinstance(debug_consumer_result, list) and len(debug_consumer_result) > 0:
        consumer_instances = [result.get('metric', {}).get('instance', 'unknown') for result in debug_consumer_result if 'metric' in result]
        up_values = [result.get('value', [None, '0'])[1] for result in debug_consumer_result if 'value' in result]
        logger.info(f"CONSUMER_INSTANCES: total={len(consumer_instances)} instances={consumer_instances} up_status={up_values}")
    else:
        logger.warning(f"CONSUMER_INSTANCES: debug_query_result={debug_consumer_result}")
        logger.warning(f"CONSUMER_INSTANCES: no_consumer_instances_found_for_job=prometheus.scrape.annotated_pods_port_8000")

    # Ensure raw features have defaults (no computations)
    metrics = _ensure_raw_features(metrics)

    # ENHANCED: Failure detection and proactive scaling analysis
    failure_indicators = _analyze_failure_patterns(metrics, current_replicas)
    
    # Add failure indicators to metrics for DQN
    metrics.update(failure_indicators)

    CURRENT_REPLICAS_GAUGE.set(current_replicas)

    # Enhanced logging with consumer pod metrics and failure analysis
    cpu_rate = metrics.get('process_cpu_seconds_total_rate', 0.0)
    resident_memory = metrics.get('process_resident_memory_bytes', 0.0)
    virtual_memory = metrics.get('process_virtual_memory_bytes', 0.0)
    http_duration_rate = metrics.get('http_request_duration_seconds_sum_rate', 0.0)
    http_requests_rate = metrics.get('http_requests_total_rate', 0.0)
    http_count_rate = metrics.get('http_request_duration_seconds_count_rate', 0.0)
    open_fds = metrics.get('process_open_fds', 0.0)
    response_size_rate = metrics.get('http_response_size_bytes_sum_rate', 0.0)
    request_size_rate = metrics.get('http_request_size_bytes_count_rate', 0.0)

    # ENHANCED: Failure and resource pressure indicators
    replicas_unavailable = metrics.get('deployment_replicas_unavailable', 0)
    replicas_ready = metrics.get('deployment_replicas_ready', current_replicas)
    containers_ready = metrics.get('pod_container_ready', 0)
    containers_running = metrics.get('pod_container_running', 0)
    container_restarts = metrics.get('pod_container_restarts', 0)
    restarts_recent = metrics.get('pod_container_restarts_recent', 0)
    restarts_5m = metrics.get('restarts_5m', 0)
    restarts_15m = metrics.get('restarts_15m', 0)
    last_exit_code = metrics.get('pod_container_last_terminated_exitcode', 0)
    
    cpu_limits = metrics.get('pod_resource_limits_cpu', 0)
    memory_limits = metrics.get('pod_resource_limits_memory', 0)
    cpu_requests = metrics.get('pod_resource_requests_cpu', 0)
    memory_requests = metrics.get('pod_resource_requests_memory', 0)
    network_up = metrics.get('node_network_up', 0)
    prometheus_up = metrics.get('prometheus_scrape_up', 0)

    # CRITICAL: Failure pattern analysis
    failure_severity = failure_indicators.get('failure_severity', 'none')
    scaling_urgency = failure_indicators.get('scaling_urgency', 'normal')
    resource_pressure = failure_indicators.get('resource_pressure', 'low')

    logger.info(f"CONSUMER_METRICS: cpu_rate={cpu_rate:.4f} resident_memory={resident_memory:.0f}bytes "
                f"virtual_memory={virtual_memory:.0f}bytes http_requests_rate={http_requests_rate:.4f} "
                f"current_replicas={current_replicas}")

    logger.info(f"NETWORK_METRICS: http_duration_rate={http_duration_rate:.4f} "
                f"http_count_rate={http_count_rate:.4f} response_size_rate={response_size_rate:.4f} "
                f"request_size_rate={request_size_rate:.4f}")

    logger.info(f"SYSTEM_METRICS: open_fds={open_fds:.0f} targeting_consumer_pods={services.config.kubernetes.target_deployment} "
                f"namespace={services.config.kubernetes.target_namespace}")

    # ENHANCED: Failure and health analysis logging
    logger.info(f"FAILURE_ANALYSIS: severity={failure_severity} urgency={scaling_urgency} resource_pressure={resource_pressure}")
    logger.info(f"RESTART_PATTERN: total={container_restarts} recent={restarts_recent} 5m={restarts_5m} 15m={restarts_15m} exit_code={last_exit_code}")
    logger.info(f"AVAILABILITY: unavailable={replicas_unavailable} ready={replicas_ready}/{current_replicas} "
                f"containers_ready={containers_ready} containers_running={containers_running}")
    
    # Resource pressure analysis
    memory_limits_mb = memory_limits / 1000000 if memory_limits > 0 else 0
    memory_requests_mb = memory_requests / 1000000 if memory_requests > 0 else 0
    logger.info(f"RESOURCE_PRESSURE: cpu_limits={cpu_limits:.2f} memory_limits_mb={memory_limits_mb:.0f} "
                f"cpu_requests={cpu_requests:.2f} memory_requests_mb={memory_requests_mb:.0f}")
    
    # Monitoring infrastructure health
    logger.info(f"MONITORING_HEALTH: network_up={network_up:.0f} prometheus_scrape_up={prometheus_up:.0f}")

    # PROACTIVE SCALING INDICATORS
    if failure_severity in ['high', 'critical']:
        logger.warning(f"PROACTIVE_SCALING_NEEDED: failure_severity={failure_severity} restarts_recent={restarts_recent}")
        logger.warning(f"SCALING_RECOMMENDATION: immediate_scale_up_required current_replicas={current_replicas}")
    elif scaling_urgency == 'high':
        logger.warning(f"SCALING_URGENCY: high_urgency_detected resource_pressure={resource_pressure}")
    
    # Resource utilization analysis for proactive scaling
    if current_replicas > 0:
        # Analyze per-replica metrics to understand load distribution
        cpu_rate_per_replica = cpu_rate / current_replicas if current_replicas > 0 else 0
        http_rate_per_replica = http_requests_rate / current_replicas if current_replicas > 0 else 0
        
        logger.info(f"LOAD_ANALYSIS: cpu_rate_per_replica={cpu_rate_per_replica:.4f} "
                    f"http_rate_per_replica={http_rate_per_replica:.4f}")
        
        # Analyze system load patterns
        if http_requests_rate > 0:
            avg_request_duration = http_duration_rate / http_requests_rate if http_requests_rate > 0 else 0
            # Convert memory from bytes to MB for readability
            resident_memory_mb = resident_memory / 1000000 if resident_memory > 0 else 0
            virtual_memory_mb = virtual_memory / 1000000 if virtual_memory > 0 else 0
            logger.info(f"PERFORMANCE_ANALYSIS: avg_request_duration={avg_request_duration:.4f}s "
                        f"resident_memory={resident_memory_mb:.1f}MB virtual_memory={virtual_memory_mb:.1f}MB fds_usage={open_fds:.0f}")
        
        # Load distribution analysis
        logger.info(f"SCALING_INDICATORS: replicas={current_replicas} "
                    f"total_cpu_rate={cpu_rate:.4f} total_http_rate={http_requests_rate:.4f} "
                    f"memory_usage_mb={resident_memory / 1000000:.1f}")

    # Log feature availability for debugging
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
        'process_resident_memory_bytes': 0.0,  # Default: no resident memory usage
        'process_virtual_memory_bytes': 0.0,  # Default: no virtual memory usage
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
        'pod_resource_requests_cpu': 0.0,  # Default: no CPU requests set
        'pod_resource_requests_memory': 0.0,  # Default: no memory requests set
        'node_network_up': 0.0,  # Default: network status unknown
        'prometheus_scrape_up': 0.0,  # Default: prometheus scraping down
        'pod_container_restarts_recent': 0.0,  # Default: no recent restarts
        'pod_container_last_terminated_exitcode': 0.0,  # Default: normal exit
        'restarts_5m': 0.0,  # Default: no restarts in 5 minutes
        'restarts_15m': 0.0,  # Default: no restarts in 15 minutes
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

    # DEBUG: Log what features are expected vs available
    logger.info(f"DQN_FEATURES_DEBUG: expected_features={services.config.base_features}")
    logger.info(f"DQN_FEATURES_DEBUG: available_metrics={list(metrics.keys())[:10]}...")  # First 10 keys
    logger.info(f"DQN_FEATURES_DEBUG: raw_feature_values={raw_features}")
    
    # Create pandas DataFrame with feature names to match how scaler was fitted
    import pandas as pd
    raw_features_df = pd.DataFrame([raw_features], columns=services.config.base_features)
    scaled_raw_features = services.scaler.transform(raw_features_df)

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

                # ENHANCED: Proactive scaling based on failure indicators
            failure_severity = metrics.get('failure_severity', 'none')
            scaling_urgency = metrics.get('scaling_urgency', 'normal')
            resource_pressure = metrics.get('resource_pressure', 'low')
            proactive_scale_recommended = metrics.get('proactive_scale_recommended', False)
            failure_score = metrics.get('failure_score', 0.0)
            failure_reasons = metrics.get('failure_reasons', [])

            # PROACTIVE OVERRIDE: If system is failing, override DQN to scale up immediately
            if proactive_scale_recommended and current_replicas < 10:
                logger.warning(f"PROACTIVE_OVERRIDE: failure_detected={failure_severity} overriding_dqn_to_scale_up")
                logger.warning(f"FAILURE_REASONS: {', '.join(failure_reasons)}")
                action_index = 2  # Force Scale Up
                action_name = "Scale Up"
                exploration_type = "proactive_intervention"
                
                # Log the override decision
                logger.warning(f"PROACTIVE_SCALING: forced_scale_up failure_score={failure_score:.2f} "
                              f"urgency={scaling_urgency} current_replicas={current_replicas}")
            else:
                # Normal DQN decision logic
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

                action_map = {0: "Scale Down", 1: "Keep Same", 2: "Scale Up"}
                action_name = action_map.get(action_index, "Unknown")

            # Update epsilon and decision count through ServiceContainer
            updated_epsilon = services.update_epsilon(services.config.dqn.epsilon_decay, services.config.dqn.epsilon_end)
            decision_count = services.increment_decision_count()

            # Use DQN decision directly
            dqn_action_name = action_name

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

            # Log failure analysis for all decisions
            logger.info(f"FAILURE_ANALYSIS_DQN: severity={failure_severity} urgency={scaling_urgency} "
                        f"resource_pressure={resource_pressure} score={failure_score:.2f}")
            if failure_reasons:
                logger.info(f"FAILURE_INDICATORS: {', '.join(failure_reasons[:3])}")  # Log top 3 reasons

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
            # Enhanced fallback: Rule-based logic with proactive failure detection
            logger.info("DQN_FALLBACK: using_rule_based_logic model_not_available")

            # Get current consumer pod metrics for fallback decision
            cpu_rate = metrics.get('process_cpu_seconds_total_rate', 0.0)
            resident_memory = metrics.get('process_resident_memory_bytes', 0.0)
            virtual_memory = metrics.get('process_virtual_memory_bytes', 0.0)
            http_duration_rate = metrics.get('http_request_duration_seconds_sum_rate', 0.0)
            http_requests_rate = metrics.get('http_requests_total_rate', 0.0)
            http_count_rate = metrics.get('http_request_duration_seconds_count_rate', 0.0)
            open_fds = metrics.get('process_open_fds', 0.0)
            response_size_rate = metrics.get('http_response_size_bytes_sum_rate', 0.0)
            request_size_rate = metrics.get('http_request_size_bytes_count_rate', 0.0)

            # ENHANCED: Get failure indicators for proactive fallback scaling
            failure_severity = metrics.get('failure_severity', 'none')
            scaling_urgency = metrics.get('scaling_urgency', 'normal')
            resource_pressure = metrics.get('resource_pressure', 'low')
            proactive_scale_recommended = metrics.get('proactive_scale_recommended', False)
            failure_score = metrics.get('failure_score', 0.0)
            failure_reasons = metrics.get('failure_reasons', [])

            # Generate analysis for fallback decision
            analysis = decision_reasoning.analyze_metrics(metrics, current_replicas)

            # Enhanced rule-based decision with consumer pod metrics and failure detection
            reasoning_factors = []

            # Calculate metrics for decision logic
            avg_request_duration = http_duration_rate / http_requests_rate if http_requests_rate > 0 else 0
            
            # Memory pressure indicators (convert to MB for thresholds)
            resident_memory_mb = resident_memory / 1000000 if resident_memory > 0 else 0
            virtual_memory_mb = virtual_memory / 1000000 if virtual_memory > 0 else 0

            # PROACTIVE: Check for failure conditions first (highest priority)
            if proactive_scale_recommended and current_replicas < 10:
                action_name = "Scale Up"
                reasoning_factors.append(f"PROACTIVE SCALING: System failure detected - {failure_severity} severity")
                reasoning_factors.append(f"Failure indicators: {', '.join(failure_reasons[:2])}")
                reasoning_factors.append(f"Scaling urgency: {scaling_urgency} - immediate action required")
                risk_level = "high"
                logger.warning(f"FALLBACK_PROACTIVE: failure_detected={failure_severity} forcing_scale_up")
            # Decision logic based on consumer application metrics
            elif cpu_rate > 0.8 or avg_request_duration > 1.0 or resident_memory_mb > 500:  # High load/pressure
                action_name = "Scale Up"
                reasoning_factors.append(
                    f"High load detected: CPU rate {cpu_rate:.4f}, avg duration {avg_request_duration:.4f}s, memory {resident_memory_mb:.1f}MB")
                reasoning_factors.append("Scaling up to handle increased load and reduce latency")
                risk_level = "high"
            elif http_requests_rate > 10.0 or response_size_rate > 1000000:  # High traffic
                action_name = "Scale Up"
                reasoning_factors.append(
                    f"High traffic: {http_requests_rate:.4f} req/s, response size rate {response_size_rate:.0f} bytes/s")
                reasoning_factors.append("Scaling up to handle high traffic volume")
                risk_level = "medium"
            elif cpu_rate < 0.1 and http_requests_rate < 1.0 and resident_memory_mb < 100 and current_replicas > 1:  # Low utilization
                action_name = "Scale Down"
                reasoning_factors.append(
                    f"Low utilization: CPU rate {cpu_rate:.4f}, HTTP rate {http_requests_rate:.4f}, memory {resident_memory_mb:.1f}MB")
                reasoning_factors.append(
                    f"System has excess capacity with {current_replicas} replicas - can optimize costs")
                risk_level = "low"
            # ENHANCED: Add explicit scale-down for high replica counts with low load
            elif current_replicas >= 8 and cpu_rate < 0.2 and http_requests_rate < 2.0 and resident_memory_mb < 200:  # Over-provisioned
                action_name = "Scale Down"
                reasoning_factors.append(
                    f"Over-provisioned: {current_replicas} replicas with low load - cost optimization opportunity")
                reasoning_factors.append(
                    f"Low load indicators: CPU rate {cpu_rate:.4f}, HTTP rate {http_requests_rate:.4f}, memory {resident_memory_mb:.1f}MB")
                risk_level = "low"
            else:
                action_name = "Keep Same"
                reasoning_factors.append(
                    f"Balanced state: CPU rate {cpu_rate:.4f}, HTTP rate {http_requests_rate:.4f}, memory {resident_memory_mb:.1f}MB, {current_replicas} replicas")
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

1. **RUNAWAY SCALING**: Scaling to >50 replicas without clear justification
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


def _analyze_failure_patterns(metrics: Dict[str, float], current_replicas: int) -> Dict[str, Any]:
    """
    Analyze failure patterns and resource pressure to enable proactive scaling.
    
    Returns:
        Dict with failure indicators:
        - failure_severity: none, low, medium, high, critical
        - scaling_urgency: normal, medium, high, critical
        - resource_pressure: low, medium, high, critical
        - failure_score: 0.0-1.0 (higher = more likely to need scaling)
    """
    
    # Get failure indicators
    restarts_recent = metrics.get('pod_container_restarts_recent', 0)
    restarts_5m = metrics.get('restarts_5m', 0)
    restarts_15m = metrics.get('restarts_15m', 0)
    last_exit_code = metrics.get('pod_container_last_terminated_exitcode', 0)
    replicas_unavailable = metrics.get('deployment_replicas_unavailable', 0)
    containers_ready = metrics.get('pod_container_ready', 0)
    containers_running = metrics.get('pod_container_running', 0)
    prometheus_up = metrics.get('prometheus_scrape_up', 0)
    
    # Resource pressure indicators
    cpu_limits = metrics.get('pod_resource_limits_cpu', 0)
    memory_limits = metrics.get('pod_resource_limits_memory', 0)
    cpu_requests = metrics.get('pod_resource_requests_cpu', 0)
    memory_requests = metrics.get('pod_resource_requests_memory', 0)
    
    # Calculate failure severity
    failure_score = 0.0
    failure_reasons = []
    
    # Recent restart patterns (high weight)
    if restarts_recent > 0:
        failure_score += 0.4
        failure_reasons.append(f"recent_restarts={restarts_recent}")
    
    if restarts_5m > 2:  # More than 2 restarts in 5 minutes
        failure_score += 0.3
        failure_reasons.append(f"frequent_restarts_5m={restarts_5m}")
    
    if restarts_15m > 5:  # More than 5 restarts in 15 minutes
        failure_score += 0.2
        failure_reasons.append(f"excessive_restarts_15m={restarts_15m}")
    
    # Exit code analysis
    if last_exit_code in [137, 143]:  # OOMKilled or SIGTERM
        failure_score += 0.3
        failure_reasons.append(f"oom_killed_exit_code={last_exit_code}")
    elif last_exit_code != 0 and last_exit_code != -1:
        failure_score += 0.2
        failure_reasons.append(f"abnormal_exit_code={last_exit_code}")
    
    # Availability issues
    if replicas_unavailable > 0:
        failure_score += 0.3
        failure_reasons.append(f"unavailable_replicas={replicas_unavailable}")
    
    if containers_ready < current_replicas:
        failure_score += 0.2
        failure_reasons.append(f"containers_not_ready={containers_ready}/{current_replicas}")
    
    if containers_running < current_replicas:
        failure_score += 0.2
        failure_reasons.append(f"containers_not_running={containers_running}/{current_replicas}")
    
    # Monitoring issues
    if prometheus_up == 0:
        failure_score += 0.4
        failure_reasons.append("prometheus_scrape_down")
    
    # Determine failure severity
    if failure_score >= 0.8:
        failure_severity = 'critical'
    elif failure_score >= 0.6:
        failure_severity = 'high'
    elif failure_score >= 0.4:
        failure_severity = 'medium'
    elif failure_score >= 0.2:
        failure_severity = 'low'
    else:
        failure_severity = 'none'
    
    # Calculate scaling urgency
    scaling_urgency = 'normal'
    if failure_severity in ['critical', 'high']:
        scaling_urgency = 'critical'
    elif failure_severity == 'medium':
        scaling_urgency = 'high'
    elif failure_severity == 'low':
        scaling_urgency = 'medium'
    
    # Analyze resource pressure
    resource_pressure = 'low'
    if memory_limits > 0 and memory_limits < 2000000000:  # Less than 2GB
        resource_pressure = 'high'
        failure_reasons.append(f"low_memory_limit={memory_limits/1000000:.0f}MB")
    elif cpu_limits > 0 and cpu_limits < 1.0:  # Less than 1 CPU
        resource_pressure = 'medium'
        failure_reasons.append(f"low_cpu_limit={cpu_limits:.2f}")
    
    # Special case: If we have only 1 replica and it's failing, urgency is critical
    if current_replicas == 1 and failure_severity in ['medium', 'high', 'critical']:
        scaling_urgency = 'critical'
        failure_reasons.append("single_replica_failing")
    
    return {
        'failure_severity': failure_severity,
        'scaling_urgency': scaling_urgency,
        'resource_pressure': resource_pressure,
        'failure_score': failure_score,
        'failure_reasons': failure_reasons,
        'proactive_scale_recommended': failure_severity in ['medium', 'high', 'critical'] or scaling_urgency in ['high', 'critical']
    }


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
        [50, 3.0, 3000000000, "Scale Up", "negative"],   # Already too many
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