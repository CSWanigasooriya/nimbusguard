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
        """Analyze current metrics and provide detailed insights using consumer performance features."""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'current_replicas': current_replicas,
            'raw_metrics': metrics.copy(),
            'insights': {},
            'risk_factors': [],
            'performance_indicators': {}
        }

        # Analyze consumer performance indicators using the 9 scientifically-selected features
        cpu_rate = metrics.get('process_cpu_seconds_total_rate', 0.01)
        gc_collections_rate = metrics.get('python_gc_collections_total_rate', 0.1)
        gc_objects_rate = metrics.get('python_gc_objects_collected_total_rate', 1.0)
        http_duration_rate = metrics.get('http_request_duration_seconds_sum_rate', 0.05)
        http_requests_rate = metrics.get('http_requests_total_rate', 1.0)
        http_count_rate = metrics.get('http_request_duration_seconds_count_rate', 1.0)
        open_fds = metrics.get('process_open_fds', 10.0)
        response_size_rate = metrics.get('http_response_size_bytes_sum_rate', 1024.0)
        request_count_rate = metrics.get('http_request_size_bytes_count_rate', 1.0)

        # Calculate derived metrics
        cpu_per_replica = cpu_rate / max(current_replicas, 1)
        gc_pressure = gc_collections_rate + gc_objects_rate
        gc_per_replica = gc_pressure / max(current_replicas, 1)
        http_per_replica = http_requests_rate / max(current_replicas, 1)
        latency_per_request = http_duration_rate / max(http_requests_rate, 0.1)
        fds_per_replica = open_fds / max(current_replicas, 1)

        # CPU performance analysis
        if cpu_per_replica > 0.1:
            analysis['insights']['cpu'] = "HIGH CPU UTILIZATION"
            analysis['risk_factors'].append(f"CPU rate per replica {cpu_per_replica:.4f}/sec indicates high load")
            analysis['performance_indicators']['cpu_severity'] = 'critical'
        elif cpu_per_replica > 0.05:
            analysis['insights']['cpu'] = "ELEVATED CPU UTILIZATION"
            analysis['risk_factors'].append(f"CPU rate per replica {cpu_per_replica:.4f}/sec approaching limits")
            analysis['performance_indicators']['cpu_severity'] = 'warning'
        else:
            analysis['insights']['cpu'] = "CPU UTILIZATION NORMAL"
            analysis['performance_indicators']['cpu_severity'] = 'normal'

        # Memory pressure analysis (garbage collection activity)
        if gc_per_replica > 2.0:
            analysis['insights']['memory'] = "HIGH MEMORY PRESSURE"
            analysis['risk_factors'].append(f"GC activity {gc_per_replica:.2f}/sec per replica indicates memory pressure")
            analysis['performance_indicators']['memory_severity'] = 'critical'
        elif gc_per_replica > 1.0:
            analysis['insights']['memory'] = "ELEVATED MEMORY PRESSURE"
            analysis['risk_factors'].append(f"GC activity {gc_per_replica:.2f}/sec per replica approaching limits")
            analysis['performance_indicators']['memory_severity'] = 'warning'
        else:
            analysis['insights']['memory'] = "MEMORY PRESSURE NORMAL"
            analysis['performance_indicators']['memory_severity'] = 'normal'

        # HTTP performance analysis
        if latency_per_request > 0.1:
            analysis['insights']['http'] = "HIGH HTTP LATENCY"
            analysis['risk_factors'].append(f"Average latency {latency_per_request:.3f}s per request indicates performance issues")
            analysis['performance_indicators']['http_severity'] = 'critical'
        elif latency_per_request > 0.05:
            analysis['insights']['http'] = "ELEVATED HTTP LATENCY"
            analysis['risk_factors'].append(f"Average latency {latency_per_request:.3f}s per request approaching limits")
            analysis['performance_indicators']['http_severity'] = 'warning'
        else:
            analysis['insights']['http'] = "HTTP PERFORMANCE NORMAL"
            analysis['performance_indicators']['http_severity'] = 'normal'

        # Load distribution analysis
        if http_per_replica > 5.0:
            analysis['insights']['load'] = "HIGH LOAD PER REPLICA"
            analysis['risk_factors'].append(f"HTTP requests {http_per_replica:.2f}/sec per replica indicates high load")
            analysis['performance_indicators']['load_severity'] = 'critical'
        elif http_per_replica > 2.0:
            analysis['insights']['load'] = "ELEVATED LOAD PER REPLICA"
            analysis['risk_factors'].append(f"HTTP requests {http_per_replica:.2f}/sec per replica approaching limits")
            analysis['performance_indicators']['load_severity'] = 'warning'
        else:
            analysis['insights']['load'] = "LOAD DISTRIBUTION NORMAL"
            analysis['performance_indicators']['load_severity'] = 'normal'

        # I/O resource analysis
        if fds_per_replica > 20:
            analysis['insights']['io'] = "HIGH I/O LOAD"
            analysis['risk_factors'].append(f"File descriptors {fds_per_replica:.1f} per replica indicates high I/O activity")
            analysis['performance_indicators']['io_severity'] = 'critical'
        elif fds_per_replica > 15:
            analysis['insights']['io'] = "ELEVATED I/O LOAD"
            analysis['risk_factors'].append(f"File descriptors {fds_per_replica:.1f} per replica approaching limits")
            analysis['performance_indicators']['io_severity'] = 'warning'
        else:
            analysis['insights']['io'] = "I/O LOAD NORMAL"
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

        # Reasoning factors based on consumer performance metrics
        analysis = self.analyze_metrics(metrics, current_replicas)
        explanation['reasoning_factors'] = [
            f"Current system state: {len(analysis['risk_factors'])} risk factors detected",
            f"CPU performance: {analysis['insights'].get('cpu', 'unknown')}",
            f"Memory pressure: {analysis['insights'].get('memory', 'unknown')}",
            f"HTTP performance: {analysis['insights'].get('http', 'unknown')}",
            f"Load distribution: {analysis['insights'].get('load', 'unknown')}",
            f"I/O load: {analysis['insights'].get('io', 'unknown')}",
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

        # Log actual consumer performance features being used (9 scientifically-selected features)
        self.reasoning_logger.info(
            f"AI_REASONING: consumer_feature name=cpu_rate value={metrics.get('process_cpu_seconds_total_rate', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: consumer_feature name=gc_collections_rate value={metrics.get('python_gc_collections_total_rate', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: consumer_feature name=gc_objects_rate value={metrics.get('python_gc_objects_collected_total_rate', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: consumer_feature name=http_duration_rate value={metrics.get('http_request_duration_seconds_sum_rate', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: consumer_feature name=http_requests_rate value={metrics.get('http_requests_total_rate', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: consumer_feature name=http_count_rate value={metrics.get('http_request_duration_seconds_count_rate', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: consumer_feature name=open_fds value={metrics.get('process_open_fds', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: consumer_feature name=response_size_rate value={metrics.get('http_response_size_bytes_sum_rate', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: consumer_feature name=request_count_rate value={metrics.get('http_request_size_bytes_count_rate', 'N/A')}")

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
