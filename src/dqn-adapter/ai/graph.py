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

async def log_experience(state: ScalingState, services: ServiceContainer) -> Dict[str, Any]:
    """Log the complete experience and send to trainer for learning."""
    logger.info("=" * 60)
    logger.info("NODE_START: log_experience")
    logger.info("=" * 60)
    
    experience = state.get('experience')
    if experience and 'reward' in experience:
        # Send experience to DQN trainer
        if services.dqn_trainer:
            await services.dqn_trainer.add_experience_for_training(experience)
            logger.info(f"EXPERIENCE_LOGGED: action={experience.get('action')} reward={experience.get('reward'):.3f}")
        else:
            logger.warning("EXPERIENCE_TRAINER_MISSING: dqn_trainer_not_available")
    else:
        logger.warning("EXPERIENCE_MISSING: no_valid_experience_to_log")
    
    logger.info("=" * 60)
    logger.info("NODE_END: log_experience")
    logger.info("=" * 60)
    return {}


def create_graph(services: ServiceContainer):
    """
    Create the LangGraph workflow with dependency injection.
    🔧 STABILIZED WORKFLOW: Wait for system stabilization before reward calculation.
    
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

    # STABILIZED WORKFLOW: Wait for system stabilization before reward calculation
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
    
    if not services.scaler or not services.dqn_model:
        logger.error("DQN_PREDICTION: model_or_scaler_not_loaded")
        return {"error": "DQN_MODEL_OR_SCALER_NOT_LOADED"}

    metrics = state['current_metrics'].copy()
    current_replicas = state['current_replicas']

    # Scale the 9 scientifically-selected base features from Prometheus
    raw_features = [metrics.get(feat, 0.0) for feat in services.config.base_features]

    # Transform with feature names to match how scaler was fitted
    import pandas as pd
    raw_features_df = pd.DataFrame([raw_features], columns=services.config.base_features)
    scaled_raw_features = services.scaler.transform(raw_features_df)
    final_feature_vector = scaled_raw_features[0].tolist()

    logger.info(f"FEATURE_COMPOSITION: features={len(final_feature_vector)} replicas={current_replicas}")

    # DQN Model Inference
    current_epsilon = services.get_epsilon()
    device = next(services.dqn_model.parameters()).device
    input_tensor = torch.FloatTensor([final_feature_vector]).to(device)

    services.dqn_model.eval()
    with torch.no_grad():
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        q_values = services.dqn_model(input_tensor).cpu().numpy().flatten()

    # Pure epsilon-greedy exploration without hardcoded overrides
    # Let the LLM safety validator handle all resource pressure scenarios intelligently
    if random.random() < current_epsilon:
        action_index = random.randint(0, 2)
        exploration_type = "exploration"
        logger.info(f"DQN_EXPLORATION: random_action epsilon={current_epsilon:.3f}")
    else:
        action_index = np.argmax(q_values)
        exploration_type = "exploitation"
        logger.info(f"DQN_EXPLOITATION: best_q_value epsilon={current_epsilon:.3f}")

    # Update epsilon and decision count
    updated_epsilon = services.update_epsilon(services.config.dqn.epsilon_decay, services.config.dqn.epsilon_end)
    decision_count = services.increment_decision_count()

    action_map = {0: "Scale Down", 1: "Keep Same", 2: "Scale Up"}
    action_name = action_map.get(action_index, "Unknown")

    # Generate explanation
    explanation = decision_reasoning.explain_dqn_decision(
        metrics=metrics,
        q_values=q_values.tolist(),
        action_name=action_name,
        exploration_type=exploration_type,
        epsilon=updated_epsilon,
        current_replicas=current_replicas
    )

    # Log decision with resource context for LLM safety validator
    confidence = explanation['confidence_metrics']['decision_confidence']
    risk_level = explanation['risk_assessment']
    
    # Calculate resource context for logging (informational only)
    cpu_rate = metrics.get('process_cpu_seconds_total_rate', 0.01)
    cpu_per_replica = cpu_rate / max(current_replicas, 1)
    memory_pressure = metrics.get('python_gc_collections_total_rate', 0.1) + metrics.get('python_gc_objects_collected_total_rate', 1.0)
    http_load = metrics.get('http_requests_total_rate', 1.0) / max(current_replicas, 1)

    logger.info(f"DQN_DECISION: action={action_name} confidence={confidence} risk={risk_level}")
    logger.info(f"RESOURCE_CONTEXT: cpu_per_replica={cpu_per_replica:.4f} memory_pressure={memory_pressure:.2f} http_load_per_replica={http_load:.2f}")
    logger.info(f"DQN_Q_VALUES: ScaleDown={q_values[0]:.3f} KeepSame={q_values[1]:.3f} ScaleUp={q_values[2]:.3f}")
    logger.info(f"LLM_SAFETY_VALIDATOR: will_analyze_resource_pressure_and_override_if_unsafe")

    # Update Prometheus metrics
    DQN_DECISIONS_COUNTER.inc()
    DQN_EPSILON_GAUGE.set(updated_epsilon)

    confidence_numeric = {'high': 1.0, 'medium': 0.5, 'low': 0.0}.get(confidence.lower(), 0.5)
    DQN_DECISION_CONFIDENCE_GAUGE.set(confidence_numeric)
    DQN_Q_VALUE_SCALE_DOWN_GAUGE.set(q_values[0])
    DQN_Q_VALUE_KEEP_SAME_GAUGE.set(q_values[1])
    DQN_Q_VALUE_SCALE_UP_GAUGE.set(q_values[2])

    # Update action counters
    if action_name == "Scale Up":
        DQN_ACTION_SCALE_UP_COUNTER.inc()
    elif action_name == "Scale Down":
        DQN_ACTION_SCALE_DOWN_COUNTER.inc()
    else:
        DQN_ACTION_KEEP_SAME_COUNTER.inc()

    # Update exploration/exploitation counters
    if exploration_type == "exploration":
        DQN_EXPLORATION_COUNTER.inc()
    else:
        DQN_EXPLOITATION_COUNTER.inc()

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
    prompt = f"""You are an INTELLIGENT SAFETY MONITOR for Kubernetes autoscaling. Your role is to detect RESOURCE PRESSURE scenarios and override unsafe DQN decisions using real-time cluster analysis.

🎯 **PRIMARY MISSION**: Detect resource pressure (CPU throttling, memory pressure, I/O bottlenecks, network issues) and override unsafe scaling decisions.

📊 **DQN DECISION TO EVALUATE**:
- Recommended Action: {action_name}
- Current Replicas: {state['current_replicas']}
- DQN Confidence: {dqn_confidence}
- DQN Risk Assessment: {dqn_risk}

📈 **SYSTEM METRICS CONTEXT**:
- Current Replicas: {state['current_replicas']}
- CPU Limits: {metrics.get('kube_pod_container_resource_limits_cpu', 0.5):.2f} cores per pod
- Memory Limits: {metrics.get('kube_pod_container_resource_limits_memory', 1073741824)/1000000:.0f} MB per pod
- CPU Usage Rate: {metrics.get('process_cpu_seconds_total_rate', 0.01):.4f}/sec
- Memory Pressure (GC): {metrics.get('python_gc_collections_total_rate', 0.1):.2f}/sec
- HTTP Load: {metrics.get('http_requests_total_rate', 1.0):.2f} req/sec
- I/O Load: {metrics.get('process_open_fds', 10):.0f} open file descriptors

🔍 **CRITICAL RESOURCE PRESSURE DETECTION**:

**MANDATORY CLUSTER VERIFICATION**: You MUST use MCP Kubernetes tools to verify actual resource usage and detect pressure scenarios:

1. **Resource Usage Analysis**: `mcp_kubernetes_pods_top` - Check actual CPU/memory usage vs limits
2. **Deployment Health**: `mcp_kubernetes_resources_get` - Verify deployment status and replica health  
3. **System Events**: `mcp_kubernetes_events_list` - Look for resource pressure warnings (OOMKilled, CPU throttling, etc.)
4. **Pod Status**: `mcp_kubernetes_pods_list` - Check for restart patterns indicating resource issues

🚨 **RESOURCE PRESSURE OVERRIDE SCENARIOS** (Block or Force Alternative):

**CPU THROTTLING**:
- If pods are hitting CPU limits (usage ≥ 95% of limit) → Force "Scale Up" if DQN suggests "Keep Same" or "Scale Down"
- If CPU per replica is very high → Override "Scale Down" decisions

**MEMORY PRESSURE**:
- If memory usage ≥ 90% of limit → Force "Scale Up" 
- If OOMKilled events detected → Block "Scale Down" decisions
- If high GC pressure (>5/sec per replica) → Override conservative decisions

**I/O BOTTLENECKS**:
- If file descriptor usage is high (>80% of system limit) → Force "Scale Up"
- If network payload per replica is excessive → Override "Keep Same"

**LOAD DISTRIBUTION**:
- If HTTP requests per replica >5/sec → Force "Scale Up" 
- If response times degrading → Block "Scale Down"

**SYSTEM INSTABILITY**:
- If frequent pod restarts → Block "Scale Down"
- If deployment shows errors → Force conservative scaling

🛡️ **DECISION OVERRIDE LOGIC**:

1. **FORCE SCALE UP**: When resource pressure detected but DQN suggests "Keep Same" or "Scale Down"
2. **BLOCK SCALE DOWN**: When system shows stress but DQN suggests reducing resources  
3. **APPROVE**: When no resource pressure detected or DQN decision aligns with resource needs
4. **FORCE SCALE DOWN**: When massive over-provisioning detected (rare)

🎯 **TARGET DEPLOYMENT**:
- Namespace: {services.config.kubernetes.target_namespace}
- Deployment: {services.config.kubernetes.target_deployment}

⚠️ **RESPONSE FORMAT**: Respond with ONLY valid JSON. No markdown, no explanations outside JSON.

**Example Override (Force Scale Up)**:
{{
    "approved": false,
    "confidence": "high", 
    "reasoning": "RESOURCE PRESSURE OVERRIDE: mcp_kubernetes_pods_top shows CPU usage at 98% of limits. Forcing Scale Up instead of DQN's 'Keep Same' recommendation.",
    "safety_risk": "high",
    "extreme_factors": ["cpu_throttling_detected", "resource_pressure_override"],
    "alternative_suggestion": "Scale Up",
    "override_action": "Scale Up",
    "cluster_check_performed": true,
    "tool_findings": ["CPU usage 495m/500m (99%)", "Memory pressure detected", "No recent OOM events"]
}}

**Example Approval (No Issues)**:
{{
    "approved": true,
    "confidence": "high",
    "reasoning": "No resource pressure detected. CPU usage 150m/500m (30%), memory stable. DQN decision appropriate.",
    "safety_risk": "none", 
    "extreme_factors": [],
    "cluster_check_performed": true,
    "tool_findings": ["All pods healthy", "Resource usage normal", "No pressure indicators"]
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
    override_action = llm_validation.get('override_action', None)  # LLM can suggest alternative action

    # Calculate new replica count based on DQN decision or LLM override
    if not llm_approved and override_action:
        # LLM RESOURCE PRESSURE OVERRIDE: Use LLM's alternative action
        logger.warning(f"RESOURCE_PRESSURE_OVERRIDE: llm_suggests={override_action} instead_of_dqn={action_name}")
        action_name = override_action  # Override DQN decision with LLM recommendation
        
    new_replicas = current_replicas
    if action_name == 'Scale Up':
        new_replicas += 1
    elif action_name == 'Scale Down':
        new_replicas -= 1

    # Apply safety constraints
    final_decision = max(1, min(20, new_replicas))  # Never go below 1 or above 20 replicas

    # Create comprehensive decision explanation AFTER final_decision is known
    decision_explanation = {
        'timestamp': datetime.now().isoformat(),
        'decision_pipeline': {
            'dqn_recommendation': {
                'action': dqn_prediction.get('action_name', 'Keep Same'),
                'confidence': dqn_confidence,
                'risk_assessment': dqn_risk
            },
            'llm_validation': {
                'approved': llm_approved,
                'confidence': llm_confidence,
                'reasoning_summary': llm_reasoning,
                'override_action': override_action
            },
            'final_decision': {
                'from_replicas': current_replicas,
                'to_replicas': final_decision,
                'action_executed': 'Scale Up' if final_decision > current_replicas else 'Scale Down' if final_decision < current_replicas else 'Keep Same',
                'decision_source': 'llm_override' if override_action else 'dqn_approved'
            }
        },
        'decision_factors': [],
        'risk_mitigation': [],
        'expected_outcomes': []
    }

    # Add decision factors based on override result
    if not llm_approved and override_action:
        decision_explanation['decision_factors'].append(
            f"RESOURCE PRESSURE OVERRIDE: LLM detected resource pressure and overrode DQN's '{dqn_prediction.get('action_name', 'Keep Same')}' with '{override_action}'")
        decision_explanation['risk_mitigation'].append("CRITICAL: Resource pressure scenario handled by intelligent safety override")
    elif not llm_approved:
        decision_explanation['decision_factors'].append(
            "SAFETY BLOCK: Extreme scaling decision blocked - maintaining current replica count")
        decision_explanation['risk_mitigation'].append("CRITICAL: Dangerous scaling prevented by safety monitor")
    else:
        decision_explanation['decision_factors'].append("Safety monitor found no resource pressure - DQN decision approved")

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

    # Calculate reward using LLM-based evaluation only
    logger.info("REWARD_CALCULATION: using_llm_based_evaluation")
    reward = await calculate_llm_reward(current_state_metrics, next_state_metrics, action, 
                                      current_replicas, services)

    experience['reward'] = reward
    experience['next_state'] = {**next_state_metrics, 'current_replicas': current_replicas}

    # Update LLM reward metric
    DQN_REWARD_TOTAL_GAUGE.set(reward)

    logger.info("=" * 60)
    logger.info("NODE_END: observe_next_state_and_calculate_reward")
    logger.info("=" * 60)
    return {"experience": experience}


def _extract_comprehensive_metrics(current_state: Dict[str, Any], next_state: Dict[str, Any], current_replicas: int) -> Dict[str, Any]:
    """
    Extract and analyze comprehensive metrics for intelligent reward calculation.
    
    Returns:
        Dictionary containing analyzed metrics and formatted output for LLM
    """
    
    # Core Performance Metrics
    curr_cpu = current_state.get('process_cpu_seconds_total_rate', 0.01)
    next_cpu = next_state.get('process_cpu_seconds_total_rate', 0.01)
    cpu_change = ((next_cpu - curr_cpu) / max(curr_cpu, 0.001)) * 100
    
    # Memory Pressure Indicators
    curr_gc_collections = current_state.get('python_gc_collections_total_rate', 0.1)
    next_gc_collections = next_state.get('python_gc_collections_total_rate', 0.1)
    curr_gc_objects = current_state.get('python_gc_objects_collected_total_rate', 1.0)
    next_gc_objects = next_state.get('python_gc_objects_collected_total_rate', 1.0)
    
    gc_pressure_curr = curr_gc_collections + (curr_gc_objects * 0.1)
    gc_pressure_next = next_gc_collections + (next_gc_objects * 0.1)
    gc_change = ((gc_pressure_next - gc_pressure_curr) / max(gc_pressure_curr, 0.001)) * 100
    
    # HTTP Performance Metrics
    curr_http_rate = current_state.get('http_requests_total_rate', 1.0)
    next_http_rate = next_state.get('http_requests_total_rate', 1.0)
    curr_duration_sum = current_state.get('http_request_duration_seconds_sum_rate', 0.05)
    next_duration_sum = next_state.get('http_request_duration_seconds_sum_rate', 0.05)
    curr_duration_count = current_state.get('http_request_duration_seconds_count_rate', 1.0)
    next_duration_count = next_state.get('http_request_duration_seconds_count_rate', 1.0)
    
    # Calculate average response time
    curr_avg_response = curr_duration_sum / max(curr_duration_count, 0.001)
    next_avg_response = next_duration_sum / max(next_duration_count, 0.001)
    response_time_change = ((next_avg_response - curr_avg_response) / max(curr_avg_response, 0.001)) * 100
    
    # Throughput analysis
    throughput_change = ((next_http_rate - curr_http_rate) / max(curr_http_rate, 0.001)) * 100
    
    # I/O and Network Analysis
    curr_fds = current_state.get('process_open_fds', 10.0)
    next_fds = next_state.get('process_open_fds', 10.0)
    curr_response_size = current_state.get('http_response_size_bytes_sum_rate', 1024.0)
    next_response_size = next_state.get('http_response_size_bytes_sum_rate', 1024.0)
    curr_request_count = current_state.get('http_request_size_bytes_count_rate', 1.0)
    next_request_count = next_state.get('http_request_size_bytes_count_rate', 1.0)
    
    # Resource Efficiency Calculations
    cpu_per_replica_curr = curr_cpu / max(current_replicas, 1)
    cpu_per_replica_next = next_cpu / max(current_replicas, 1)
    
    requests_per_replica_curr = curr_http_rate / max(current_replicas, 1)
    requests_per_replica_next = next_http_rate / max(current_replicas, 1)
    
    # System Health Indicators
    system_load_curr = (curr_cpu * 0.4) + (gc_pressure_curr * 0.3) + (curr_fds / 100 * 0.3)
    system_load_next = (next_cpu * 0.4) + (gc_pressure_next * 0.3) + (next_fds / 100 * 0.3)
    
    # Trend Analysis
    performance_trend = "stable"
    if response_time_change < -5:
        performance_trend = "improving"
    elif response_time_change > 10:
        performance_trend = "degrading"
    
    efficiency_trend = "stable"
    if cpu_change < -10 and throughput_change >= 0:
        efficiency_trend = "optimizing"
    elif cpu_change > 15 and throughput_change < 0:
        efficiency_trend = "wasting"
    
    # Format comprehensive analysis for LLM
    formatted_metrics = f"""
**PERFORMANCE METRICS:**
- CPU Rate: {curr_cpu:.4f} → {next_cpu:.4f} /sec ({cpu_change:+.1f}%)
- Response Time: {curr_avg_response:.4f} → {next_avg_response:.4f} sec ({response_time_change:+.1f}%)
- Throughput: {curr_http_rate:.2f} → {next_http_rate:.2f} req/sec ({throughput_change:+.1f}%)

**MEMORY & STABILITY:**
- GC Pressure: {gc_pressure_curr:.4f} → {gc_pressure_next:.4f} ({gc_change:+.1f}%)
- File Descriptors: {curr_fds:.0f} → {next_fds:.0f}
- System Load Index: {system_load_curr:.3f} → {system_load_next:.3f}

**EFFICIENCY METRICS:**
- CPU per Replica: {cpu_per_replica_curr:.4f} → {cpu_per_replica_next:.4f}
- Requests per Replica: {requests_per_replica_curr:.2f} → {requests_per_replica_next:.2f}
- Network Efficiency: {curr_response_size/max(curr_request_count,0.001):.0f} → {next_response_size/max(next_request_count,0.001):.0f} bytes/req

**CRITICAL SCALING SIGNALS:**
- CPU Throttling Risk: {'🚨 HIGH (CPU >{0.08:.3f}/sec per replica)' if cpu_per_replica_next > 0.08 else '🟡 MEDIUM' if cpu_per_replica_next > 0.05 else '✅ LOW'}
- Memory Pressure: {'🚨 HIGH (GC >{2.0:.1f}/sec per replica)' if gc_pressure_next/max(current_replicas,1) > 2.0 else '🟡 MEDIUM' if gc_pressure_next/max(current_replicas,1) > 1.0 else '✅ LOW'}
- Load Distribution: {'🚨 HIGH (>{3.0:.1f} req/sec per replica)' if requests_per_replica_next > 3.0 else '🟡 MEDIUM' if requests_per_replica_next > 1.5 else '✅ LOW'}

**TREND ANALYSIS:**
- Performance: {performance_trend}
- Efficiency: {efficiency_trend}
- Load Pattern: {'increasing' if throughput_change > 5 else 'decreasing' if throughput_change < -5 else 'stable'}

**RESOURCE CONTEXT:**
- Current Replicas: {current_replicas}
- Resource Utilization: {'High' if system_load_next > 2.0 else 'Medium' if system_load_next > 1.0 else 'Low'}
- Scaling Pressure: {'Up' if system_load_next > system_load_curr * 1.2 else 'Down' if system_load_next < system_load_curr * 0.8 else 'Neutral'}
"""
    
    return {
        'formatted_metrics': formatted_metrics,
        'performance_trend': performance_trend,
        'efficiency_trend': efficiency_trend,
        'cpu_change': cpu_change,
        'response_time_change': response_time_change,
        'throughput_change': throughput_change,
        'gc_change': gc_change,
        'system_load_curr': system_load_curr,
        'system_load_next': system_load_next,
        'requests_per_replica_curr': requests_per_replica_curr,
        'requests_per_replica_next': requests_per_replica_next
    }


async def calculate_llm_reward(current_state: Dict[str, Any], next_state: Dict[str, Any], 
                             action: str, current_replicas: int, services: ServiceContainer) -> float:
    """
    🧠 INTELLIGENT BALANCED REWARD CALCULATION: Advanced multi-dimensional reward evaluation.
    
    Features:
    - Balanced evaluation across all actions (Scale Up, Scale Down, Keep Same)
    - Multi-dimensional analysis (Performance, Efficiency, Stability, Cost)
    - Trend analysis and pattern recognition
    - Action-specific reward logic
    - Real-time cluster verification
    """
    
    # Only require the LLM agent to be available for reward calculation
    if not services.validator_agent:
        logger.error("LLM_REWARD: llm_required_but_not_available - Cannot calculate intelligent rewards without LLM!")
        raise ValueError("LLM agent must be available for reward calculation. Provide OPENAI_API_KEY and ensure LLM is initialized.")
    
    # Extract comprehensive metrics for analysis
    metrics_analysis = _extract_comprehensive_metrics(current_state, next_state, current_replicas)
    
    # Create advanced reward evaluation prompt
    prompt = f"""You are an EXPERT AI AUTOSCALING REWARD EVALUATOR with deep knowledge of Kubernetes performance optimization.

🎯 MISSION: Calculate a balanced reward (-5.0 to +5.0) for action "{action}" using multi-dimensional analysis.

📊 SYSTEM CONTEXT:
- Action: {action}
- Replicas: {current_replicas}
- Deployment: {services.config.kubernetes.target_deployment}
- Namespace: {services.config.kubernetes.target_namespace}

📈 COMPREHENSIVE METRICS ANALYSIS:
{metrics_analysis['formatted_metrics']}

🚨 CRITICAL SCALING SIGNALS TO PRIORITIZE:

**OVER-PROVISIONING DETECTION (PRIORITY #1):**
- If replicas >15 AND CPU rate <0.01/sec per replica AND HTTP load <0.5 req/sec per replica: SEVERE over-provisioning, Scale Up should get -4 to -5
- If replicas >10 AND CPU rate <0.02/sec per replica AND HTTP load <1.0 req/sec per replica: Over-provisioning, Scale Up should get -3 to -4
- If CPU rate <0.005/sec per replica AND HTTP load <0.3 req/sec per replica: Scale Down opportunity, Scale Up should be negative

**CPU THROTTLING DETECTION:**
- If CPU rate >0.08/sec per replica AND current replicas=1: STRONG scale up signal
- If CPU rate >0.12/sec per replica: CRITICAL scale up signal (resource starvation)
- If CPU rate <0.02/sec per replica: Consider scale down opportunity

**MEMORY PRESSURE DETECTION:**
- If GC pressure >2.0/sec per replica: Memory constrained, scale up needed
- If GC pressure >4.0/sec per replica: CRITICAL memory pressure

**LOAD DISTRIBUTION:**
- If HTTP requests >3.0/sec per replica: High load, scale up beneficial
- If HTTP requests >5.0/sec per replica: CRITICAL load, scale up essential

🔍 MULTI-DIMENSIONAL EVALUATION FRAMEWORK:

**1. PERFORMANCE DIMENSION (30%)**
- Response Time: HTTP latency trends (CRITICAL if degrading)
- Throughput: Request processing capability
- Resource Efficiency: CPU/Memory utilization patterns (PRIORITIZE CPU throttling)

**2. STABILITY DIMENSION (25%)**  
- System Health: Error rates, GC pressure
- Predictability: Metric variance and stability
- Resilience: Ability to handle load fluctuations

**3. EFFICIENCY DIMENSION (25%)**
- Resource Optimization: Right-sizing effectiveness
- Cost Efficiency: Performance per resource unit
- Waste Minimization: Avoiding over/under-provisioning

**4. ADAPTABILITY DIMENSION (20%)**
- Proactive vs Reactive: Anticipating vs responding to issues
- Context Awareness: Matching action to current conditions
- Learning Opportunity: Value for model improvement

🎯 ACTION-SPECIFIC BALANCED EVALUATION WITH CPU FOCUS:

**SCALE UP REWARDS (STRICT OVER-PROVISIONING DETECTION):**
+4 to +5: Prevented CPU throttling/resource starvation, optimal timing, clear demand increase
+3 to +4: Good performance improvement, addressed resource constraints  
+1 to +2: Reasonable resource increase, moderate benefit
+0 to +1: Minor benefit, acceptable resource trade-off
-1 to -2: Premature scaling, minimal performance gain
-3 to -4: OVER-PROVISIONING - scaling up with low CPU (<0.02/sec per replica) and low HTTP load (<1 req/sec per replica)
-4 to -5: SEVERE OVER-PROVISIONING - scaling up with very low load and already high replica count (>15 replicas)

**SCALE DOWN REWARDS:**
+4 to +5: Excellent cost optimization, maintained performance, intelligent right-sizing
+2 to +3: Good efficiency gain, stable performance, no resource pressure
+0 to +1: Reasonable optimization, minor efficiency improvement
-1 to -2: Performance risk, rushed decision
-3 to -5: Performance degradation, created resource pressure, under-provisioning

**KEEP SAME REWARDS:**
+4 to +5: Perfect stability maintenance, optimal current state, no action needed
+2 to +3: Good decision to maintain, system in acceptable state, no resource pressure
+0 to +1: Reasonable choice, no clear alternative
-1 to -2: Missed optimization opportunity, mild resource pressure ignored
-3 to -5: Failed to address CPU throttling/resource starvation, poor pattern recognition

🛠️ MANDATORY CLUSTER VERIFICATION:

Use MCP tools to verify actual system state:

1. **Deployment Health Check:**
   `mcp_kubernetes_resources_get` (apiVersion: apps/v1, kind: Deployment, name: {services.config.kubernetes.target_deployment}, namespace: {services.config.kubernetes.target_namespace})

2. **Resource Consumption Analysis:**
   `mcp_kubernetes_pods_top` for actual resource usage patterns

3. **System Events Review:**
   `mcp_kubernetes_events_list` (namespace: {services.config.kubernetes.target_namespace}) for issues/warnings

⚠️ Respond with ONLY a valid, strictly parsable JSON object. All property names and string values MUST use double quotes. Do not use single quotes. Do not include any text or explanation outside the JSON.

📊 REQUIRED RESPONSE FORMAT:

{{
    "reward": <float -5.0 to +5.0>,
    "dimensional_scores": {{
        "performance": <float 0.0 to 1.0>,
        "stability": <float 0.0 to 1.0>, 
        "efficiency": <float 0.0 to 1.0>,
        "adaptability": <float 0.0 to 1.0>
    }},
    "action_specific_analysis": {{
        "action_appropriateness": "<excellent|good|fair|poor>",
        "timing_quality": "<perfect|good|acceptable|poor>",
        "context_match": "<excellent|good|fair|poor>"
    }},
    "trend_analysis": {{
        "performance_trend": "<improving|stable|degrading>",
        "resource_trend": "<optimizing|stable|wasting>",
        "load_pattern": "<increasing|stable|decreasing>"
    }},
    "reasoning": "<detailed multi-dimensional analysis>",
    "cluster_findings": ["<specific MCP tool findings>"],
    "balance_justification": "<explanation of how reward balances across dimensions>",
    "cpu_throttling_detected": <boolean>,
    "scaling_urgency": "<none|low|medium|high|critical>"
}}

CRITICAL: Ensure rewards are balanced across all actions. Each action type should have equal opportunity for high rewards when appropriate. However, PRIORITIZE CPU throttling and resource starvation as strong scaling signals."""

    try:
        logger.info(f"LLM_REWARD: starting_evaluation action={action}")
        
        # Invoke LLM agent with MCP tools
        response = await services.validator_agent.ainvoke({
            "messages": [{"role": "user", "content": prompt}]
        })
        
        last_message = response['messages'][-1].content
        logger.info(f"LLM_REWARD: response_received chars={len(last_message)}")
        
        # Parse JSON response
        import re
        json_match = re.search(r'\{.*\}', last_message, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            
            reward = float(result.get('reward', 0.0))
            reward = np.clip(reward, -5.0, 5.0)  # Safety bounds
            
            # Extract dimensional scores for analysis
            dimensional_scores = result.get('dimensional_scores', {})
            action_analysis = result.get('action_specific_analysis', {})
            trend_analysis = result.get('trend_analysis', {})
            
            # Enhanced logging with multi-dimensional analysis
            logger.info(f"LLM_REWARD: calculated_reward={reward:.3f}")
            logger.info(f"LLM_REWARD: performance={dimensional_scores.get('performance', 0.0):.2f} "
                       f"stability={dimensional_scores.get('stability', 0.0):.2f} "
                       f"efficiency={dimensional_scores.get('efficiency', 0.0):.2f} "
                       f"adaptability={dimensional_scores.get('adaptability', 0.0):.2f}")
            
            logger.info(f"LLM_ACTION_ANALYSIS: appropriateness={action_analysis.get('action_appropriateness', 'unknown')} "
                       f"timing={action_analysis.get('timing_quality', 'unknown')} "
                       f"context_match={action_analysis.get('context_match', 'unknown')}")
            
            logger.info(f"LLM_TRENDS: performance={trend_analysis.get('performance_trend', 'unknown')} "
                       f"resource={trend_analysis.get('resource_trend', 'unknown')} "
                       f"load={trend_analysis.get('load_pattern', 'unknown')}")
            
            logger.info(f"LLM_REASONING: {result.get('reasoning', 'No reasoning provided')[:150]}...")
            
            balance_justification = result.get('balance_justification', '')
            if balance_justification:
                logger.info(f"LLM_BALANCE: {balance_justification[:100]}...")
            
            cluster_findings = result.get('cluster_findings', [])
            for i, finding in enumerate(cluster_findings[:3]):
                logger.info(f"LLM_CLUSTER_FINDING_{i+1}: {finding}")
            
            # Log reward distribution for balance analysis
            if reward > 3.0:
                logger.info(f"LLM_REWARD_ANALYSIS: high_positive_reward action={action} reward={reward:.3f}")
            elif reward < -3.0:
                logger.info(f"LLM_REWARD_ANALYSIS: high_negative_reward action={action} reward={reward:.3f}")
            elif -1.0 <= reward <= 1.0:
                logger.info(f"LLM_REWARD_ANALYSIS: neutral_reward action={action} reward={reward:.3f}")
                
            return reward
        else:
            logger.error("LLM_REWARD: json_parsing_failed - LLM required for reward calculation")
            raise ValueError(f"LLM reward calculation failed - invalid JSON response: {last_message[:200]}")
                
    except Exception as e:
        logger.error(f"LLM_REWARD: calculation_failed error={e} - LLM required for reward calculation")
        raise ValueError(f"LLM reward calculation failed: {e}")