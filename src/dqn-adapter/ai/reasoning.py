import logging
from typing import Dict, Any, List
from datetime import datetime

# --- Explainable AI Helper Functions ---
class DecisionReasoning:
    """Comprehensive decision reasoning and explanation system."""

    def __init__(self):
        self.decision_history = []
        self.reasoning_logger = logging.getLogger("AI_Reasoning")

        # Set reasoning log level - default to INFO
        self.reasoning_logger.setLevel(logging.INFO)

    def analyze_metrics(self, metrics: Dict[str, float], current_replicas: int) -> Dict[str, Any]:
        """Analyze current metrics and provide detailed insights."""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'current_replicas': current_replicas,
            'raw_metrics': metrics.copy(),
            'insights': {},
            'risk_factors': [],
            'performance_indicators': {}
        }

        # Analyze key performance indicators using consumer pod metrics
        cpu_rate = metrics.get('process_cpu_seconds_total_rate', 0.0)
        resident_memory = metrics.get('process_resident_memory_bytes', 0.0)
        virtual_memory = metrics.get('process_virtual_memory_bytes', 0.0)
        http_duration_rate = metrics.get('http_request_duration_seconds_sum_rate', 0.0)
        http_requests_rate = metrics.get('http_requests_total_rate', 0.0)
        http_count_rate = metrics.get('http_request_duration_seconds_count_rate', 0.0)
        open_fds = metrics.get('process_open_fds', 0.0)
        response_size_rate = metrics.get('http_response_size_bytes_sum_rate', 0.0)
        request_size_rate = metrics.get('http_request_size_bytes_count_rate', 0.0)

        # Calculate derived metrics for analysis
        avg_request_duration = http_duration_rate / http_requests_rate if http_requests_rate > 0 else 0
        
        # CPU performance analysis
        if cpu_rate > 0.8:
            analysis['insights']['cpu'] = "HIGH CPU USAGE"
            analysis['risk_factors'].append(f"CPU rate {cpu_rate:.4f} exceeds 0.8 - high load")
            analysis['performance_indicators']['cpu_severity'] = 'critical'
        elif cpu_rate > 0.4:
            analysis['insights']['cpu'] = "ELEVATED CPU USAGE"
            analysis['risk_factors'].append(f"CPU rate {cpu_rate:.4f} approaching high load")
            analysis['performance_indicators']['cpu_severity'] = 'warning'
        else:
            analysis['insights']['cpu'] = "CPU USAGE NORMAL"
            analysis['performance_indicators']['cpu_severity'] = 'normal'

        # Memory usage analysis (using actual memory metrics)
        resident_memory_mb = resident_memory / 1000000 if resident_memory > 0 else 0
        virtual_memory_mb = virtual_memory / 1000000 if virtual_memory > 0 else 0
        
        if resident_memory_mb > 500:  # High memory usage threshold
            analysis['insights']['memory'] = "HIGH MEMORY USAGE"
            analysis['risk_factors'].append(f"Resident memory {resident_memory_mb:.1f}MB - high memory usage")
            analysis['performance_indicators']['memory_severity'] = 'critical'
        elif resident_memory_mb > 200:  # Moderate memory usage threshold
            analysis['insights']['memory'] = "ELEVATED MEMORY USAGE"
            analysis['risk_factors'].append(f"Resident memory {resident_memory_mb:.1f}MB - moderate memory usage")
            analysis['performance_indicators']['memory_severity'] = 'warning'
        else:
            analysis['insights']['memory'] = "MEMORY USAGE NORMAL"
            analysis['performance_indicators']['memory_severity'] = 'normal'

        # Network/HTTP performance analysis
        if avg_request_duration > 1.0:
            analysis['insights']['latency'] = "HIGH LATENCY"
            analysis['risk_factors'].append(f"Average request duration {avg_request_duration:.4f}s exceeds 1s")
            analysis['performance_indicators']['latency_severity'] = 'critical'
        elif avg_request_duration > 0.5:
            analysis['insights']['latency'] = "ELEVATED LATENCY"
            analysis['risk_factors'].append(f"Average request duration {avg_request_duration:.4f}s approaching 1s")
            analysis['performance_indicators']['latency_severity'] = 'warning'
        else:
            analysis['insights']['latency'] = "LATENCY NORMAL"
            analysis['performance_indicators']['latency_severity'] = 'normal'

        # Throughput analysis
        if http_requests_rate > 10.0:
            analysis['insights']['throughput'] = "HIGH THROUGHPUT"
            analysis['risk_factors'].append(f"HTTP requests rate {http_requests_rate:.4f} req/s - high traffic")
            analysis['performance_indicators']['throughput_severity'] = 'warning'
        elif http_requests_rate < 0.1:
            analysis['insights']['throughput'] = "LOW THROUGHPUT"
            analysis['risk_factors'].append(f"HTTP requests rate {http_requests_rate:.4f} req/s - low traffic")
            analysis['performance_indicators']['throughput_severity'] = 'info'
        else:
            analysis['insights']['throughput'] = "THROUGHPUT NORMAL"
            analysis['performance_indicators']['throughput_severity'] = 'normal'

        # I/O analysis
        if open_fds > 1000:
            analysis['insights']['io'] = "HIGH FD USAGE"
            analysis['risk_factors'].append(f"Open file descriptors {open_fds:.0f} - high I/O load")
            analysis['performance_indicators']['io_severity'] = 'warning'
        else:
            analysis['insights']['io'] = "FD USAGE NORMAL"
            analysis['performance_indicators']['io_severity'] = 'normal'

        return analysis

    def explain_dqn_decision(self, metrics: Dict[str, float], q_values: List[float],
                             action_name: str, exploration_type: str, epsilon: float,
                             current_replicas: int) -> Dict[str, Any]:
        """Provide detailed explanation of DQN decision."""
        explanation = {
            'timestamp': datetime.now().isoformat(),
            'decision_type': 'DQN_RECOMMENDATION',
            'recommended_action': action_name,
            'exploration_strategy': exploration_type,
            'confidence_metrics': {},
            'reasoning_factors': [],
            'model_analysis': {},
            'risk_assessment': 'low'
        }

        # Q-value analysis
        action_map = {0: "Scale Down", 1: "Keep Same", 2: "Scale Up"}
        q_value_analysis = {}
        for i, (action, q_val) in enumerate(zip(action_map.values(), q_values)):
            q_value_analysis[action] = {
                'q_value': q_val,
                'confidence': 'high' if q_val > 0.7 else 'medium' if q_val > 0.3 else 'low'
            }

        explanation['model_analysis']['q_values'] = q_value_analysis

        # Confidence calculation
        max_q = max(q_values)
        second_max_q = sorted(q_values, reverse=True)[1] if len(q_values) > 1 else 0
        confidence_gap = max_q - second_max_q

        explanation['confidence_metrics'] = {
            'max_q_value': max_q,
            'confidence_gap': confidence_gap,
            'decision_confidence': 'high' if confidence_gap > 0.3 else 'medium' if confidence_gap > 0.1 else 'low',
            'epsilon': epsilon,
            'exploration_probability': epsilon
        }

        # Reasoning factors based on metrics
        analysis = self.analyze_metrics(metrics, current_replicas)
        
        # Get Kubernetes state metrics for better availability and health reporting
        replicas_unavailable = metrics.get('deployment_replicas_unavailable', 0)
        replicas_ready = metrics.get('deployment_replicas_ready', current_replicas)
        containers_ready = metrics.get('pod_container_ready', 0)
        containers_running = metrics.get('pod_container_running', 0)
        container_restarts = metrics.get('pod_container_restarts', 0)
        
        # Determine availability status
        if replicas_unavailable > 0:
            availability_status = f"PODS UNAVAILABLE ({replicas_unavailable})"
        elif replicas_ready < current_replicas:
            availability_status = f"PODS NOT READY ({replicas_ready}/{current_replicas})"
        else:
            availability_status = "ALL PODS AVAILABLE"
        
        # Determine container health status
        if container_restarts > 5:
            health_status = f"HIGH RESTART COUNT ({container_restarts})"
        elif containers_running < current_replicas:
            health_status = f"CONTAINERS NOT RUNNING ({containers_running}/{current_replicas})"
        else:
            health_status = "ALL CONTAINERS HEALTHY"
        
        explanation['reasoning_factors'] = [
            f"Availability status: {availability_status}",
            f"Container health: {health_status}",
            f"Current system state: {len(analysis['risk_factors'])} risk factors detected",
            f"Decision confidence: {explanation['confidence_metrics']['decision_confidence']}",
            f"Exploration strategy: {exploration_type} (epsilon={epsilon:.3f})"
        ]

        # Risk assessment
        critical_risks = len([r for r in analysis['risk_factors'] if 'critical' in r.lower() or 'exceeds' in r.lower()])
        if critical_risks > 0:
            explanation['risk_assessment'] = 'high'
        elif len(analysis['risk_factors']) > 0:
            explanation['risk_assessment'] = 'medium'

        # Add reasoning based on action
        if action_name == 'Scale Up':
            explanation['reasoning_factors'].append(
                "SCALE UP reasoning: System may be under-provisioned, scaling up to meet demand")
        elif action_name == 'Scale Down':
            explanation['reasoning_factors'].append(
                "SCALE DOWN reasoning: System has excess capacity and can operate efficiently with fewer resources")
        else:
            explanation['reasoning_factors'].append(
                "KEEP SAME reasoning: System is operating within acceptable parameters")

        explanation['reasoning_factors'] = sorted(explanation['reasoning_factors'])
        return explanation

    def log_decision_reasoning(self, explanation: Dict[str, Any], metrics: Dict[str, float],
                               q_values: List[float] = None) -> None:
        """Log detailed decision reasoning in a structured, regex-friendly format."""

        self.reasoning_logger.info("AI_REASONING: analysis_start")
        self.reasoning_logger.info(f"AI_REASONING: timestamp={datetime.now().isoformat()}")
        self.reasoning_logger.info(f"AI_REASONING: recommended_action={explanation['recommended_action']}")
        self.reasoning_logger.info(f"AI_REASONING: exploration_strategy={explanation['exploration_strategy']}")
        self.reasoning_logger.info(f"AI_REASONING: risk_assessment={explanation['risk_assessment'].upper()}")
        self.reasoning_logger.info(
            f"AI_REASONING: decision_confidence={explanation['confidence_metrics']['decision_confidence'].upper()} confidence_gap={explanation['confidence_metrics']['confidence_gap']:.3f}")
        self.reasoning_logger.info(f"AI_REASONING: exploration_rate={explanation['confidence_metrics']['epsilon']:.3f}")

        # Q-Value analysis (if available)
        if q_values and len(q_values) >= 3:
            q_confidences = [self.get_q_value_confidence(q) for q in q_values]
            self.reasoning_logger.info(
                f"AI_REASONING: q_value_scale_down={q_values[0]:.3f} confidence={q_confidences[0]}")
            self.reasoning_logger.info(
                f"AI_REASONING: q_value_keep_same={q_values[1]:.3f} confidence={q_confidences[1]}")
            self.reasoning_logger.info(
                f"AI_REASONING: q_value_scale_up={q_values[2]:.3f} confidence={q_confidences[2]}")
        else:
            self.reasoning_logger.info("AI_REASONING: q_values=unavailable fallback_mode=true")

        # Key reasoning factors
        for i, factor in enumerate(explanation['reasoning_factors']):
            clean_factor = factor.lstrip('•').strip()
            self.reasoning_logger.info(f"AI_REASONING: factor_{i + 1}={clean_factor}")

        # Key metrics at time of decision (actual consumer pod metrics being used)
        self.reasoning_logger.info(
            f"AI_REASONING: raw_feature name=cpu_rate value={metrics.get('process_cpu_seconds_total_rate', 'N/A')}")
        
        # Convert memory values to MB for better readability
        resident_memory_raw = metrics.get('process_resident_memory_bytes', 0)
        virtual_memory_raw = metrics.get('process_virtual_memory_bytes', 0)
        resident_memory_mb = resident_memory_raw / 1000000 if resident_memory_raw > 0 else 0
        virtual_memory_mb = virtual_memory_raw / 1000000 if virtual_memory_raw > 0 else 0
        
        self.reasoning_logger.info(
            f"AI_REASONING: raw_feature name=resident_memory_mb value={resident_memory_mb:.1f}MB")
        self.reasoning_logger.info(
            f"AI_REASONING: raw_feature name=virtual_memory_mb value={virtual_memory_mb:.1f}MB")
        self.reasoning_logger.info(
            f"AI_REASONING: raw_feature name=http_duration_rate value={metrics.get('http_request_duration_seconds_sum_rate', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: raw_feature name=http_requests_rate value={metrics.get('http_requests_total_rate', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: raw_feature name=http_count_rate value={metrics.get('http_request_duration_seconds_count_rate', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: raw_feature name=open_fds value={metrics.get('process_open_fds', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: raw_feature name=response_size_rate value={metrics.get('http_response_size_bytes_sum_rate', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: raw_feature name=request_size_rate value={metrics.get('http_request_size_bytes_count_rate', 'N/A')}")

        # Log Kubernetes state metrics for availability and health context
        self.reasoning_logger.info(
            f"AI_REASONING: k8s_metric name=replicas_unavailable value={metrics.get('deployment_replicas_unavailable', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: k8s_metric name=replicas_ready value={metrics.get('deployment_replicas_ready', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: k8s_metric name=containers_ready value={metrics.get('pod_container_ready', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: k8s_metric name=containers_running value={metrics.get('pod_container_running', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: k8s_metric name=container_restarts value={metrics.get('pod_container_restarts', 'N/A')}")

        self.reasoning_logger.info("AI_REASONING: analysis_end")

    def get_q_value_confidence(self, q_value: float) -> str:
        """Get confidence level based on Q-value magnitude."""
        if abs(q_value) > 5: return "high"
        if abs(q_value) > 2: return "medium"
        return "low"

    def create_audit_trail(self, explanation: Dict[str, Any], final_decision: int, llm_validation: Dict[str, Any]) -> \
    Dict[str, Any]:
        """Create a structured audit trail for the decision."""
        import uuid

        audit_trail = {
            'decision_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'dqn_explanation': explanation,
            'llm_validation': llm_validation,
            'final_decision': final_decision,
            'explainable': True,
            'auditable': True,
            'reversible': True
        }

        # Store in decision history (keep last 100)
        self.decision_history.append(audit_trail)
        if len(self.decision_history) > 100:
            self.decision_history.pop(0)

        return audit_trail
