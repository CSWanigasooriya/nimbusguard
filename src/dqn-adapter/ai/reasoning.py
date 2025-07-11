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

        # Analyze key performance indicators using actual Kubernetes metrics
        unavailable_replicas = metrics.get('kube_deployment_status_replicas_unavailable', 0)
        pods_ready = metrics.get('kube_pod_container_status_ready', 0)
        desired_replicas = metrics.get('kube_deployment_spec_replicas', 1)
        cpu_limits = metrics.get('kube_pod_container_resource_limits_cpu', 0)
        memory_limits = metrics.get('kube_pod_container_resource_limits_memory', 0)
        containers_running = metrics.get('kube_pod_container_status_running', 0)
        exit_code = metrics.get('kube_pod_container_status_last_terminated_exitcode', 0)

        memory_mb = memory_limits / 1000000  # Convert to MB

        # Availability analysis (replica readiness)
        if unavailable_replicas > 0:
            analysis['insights']['availability'] = "PODS UNAVAILABLE"
            analysis['risk_factors'].append(
                f"Unavailable replicas {unavailable_replicas:.0f} - service degradation")
            analysis['performance_indicators']['availability_severity'] = 'critical'
        elif pods_ready < desired_replicas:
            analysis['insights']['availability'] = "PODS NOT READY"
            analysis['risk_factors'].append(
                f"Only {pods_ready:.0f}/{desired_replicas:.0f} pods ready - potential instability")
            analysis['performance_indicators']['availability_severity'] = 'warning'
        else:
            analysis['insights']['availability'] = "PODS HEALTHY"
            analysis['performance_indicators']['availability_severity'] = 'normal'

        # Memory analysis (resource limits)
        if memory_mb > 2000:
            analysis['insights']['memory'] = "HIGH MEMORY LIMITS"
            analysis['risk_factors'].append(f"Memory limits {memory_mb:.1f}MB exceeds 2GB")
            analysis['performance_indicators']['memory_severity'] = 'critical'
        elif memory_mb > 1000:
            analysis['insights']['memory'] = "ELEVATED MEMORY LIMITS"
            analysis['risk_factors'].append(f"Memory limits {memory_mb:.1f}MB approaching 2GB")
            analysis['performance_indicators']['memory_severity'] = 'warning'
        else:
            analysis['insights']['memory'] = "MEMORY LIMITS NORMAL"
            analysis['performance_indicators']['memory_severity'] = 'normal'

        # Container health analysis
        if containers_running < desired_replicas:
            analysis['insights']['health'] = "CONTAINERS NOT RUNNING"
            analysis['risk_factors'].append(f"Only {containers_running:.0f}/{desired_replicas:.0f} containers running")
            analysis['performance_indicators']['health_severity'] = 'critical'
        elif exit_code == 137:  # SIGKILL/OOMKilled - predictable system response
            analysis['insights']['health'] = "OOM KILLED"
            analysis['risk_factors'].append(f"Container OOMKilled (exit code 137) - memory limit exceeded")
            analysis['performance_indicators']['health_severity'] = 'warning'
        elif exit_code not in [0, 130, 143]:  # 0=success, 130=SIGINT, 143=SIGTERM (normal)
            analysis['insights']['health'] = "ABNORMAL EXIT CODE"
            analysis['risk_factors'].append(f"Last exit code {exit_code:.0f} indicates potential issues")
            analysis['performance_indicators']['health_severity'] = 'warning'
        else:
            analysis['insights']['health'] = "CONTAINERS HEALTHY"
            analysis['performance_indicators']['health_severity'] = 'normal'

        # CPU analysis (resource limits)
        if cpu_limits > 2.0:
            analysis['insights']['cpu'] = "HIGH CPU LIMITS"
            analysis['risk_factors'].append(f"CPU limits {cpu_limits:.1f} cores exceeds 2.0")
            analysis['performance_indicators']['cpu_severity'] = 'critical'
        elif cpu_limits > 1.0:
            analysis['insights']['cpu'] = "ELEVATED CPU LIMITS"
            analysis['risk_factors'].append(f"CPU limits {cpu_limits:.1f} cores approaching 2.0")
            analysis['performance_indicators']['cpu_severity'] = 'warning'
        else:
            analysis['insights']['cpu'] = "CPU LIMITS NORMAL"
            analysis['performance_indicators']['cpu_severity'] = 'normal'

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
        explanation['reasoning_factors'] = [
            f"Current system state: {len(analysis['risk_factors'])} risk factors detected",
            f"Availability status: {analysis['insights'].get('availability', 'unknown')}",
            f"Memory status: {analysis['insights'].get('memory', 'unknown')}",
            f"Container health: {analysis['insights'].get('health', 'unknown')}",
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

        # Key metrics at time of decision (actual Kubernetes metrics being used)
        self.reasoning_logger.info(
            f"AI_REASONING: raw_feature name=unavailable_replicas value={metrics.get('kube_deployment_status_replicas_unavailable', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: raw_feature name=pods_ready value={metrics.get('kube_pod_container_status_ready', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: raw_feature name=desired_replicas value={metrics.get('kube_deployment_spec_replicas', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: raw_feature name=cpu_limits value={metrics.get('kube_pod_container_resource_limits_cpu', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: raw_feature name=memory_limits value={metrics.get('kube_pod_container_resource_limits_memory', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: raw_feature name=containers_running value={metrics.get('kube_pod_container_status_running', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: raw_feature name=deployment_generation value={metrics.get('kube_deployment_status_observed_generation', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: raw_feature name=network_up value={metrics.get('node_network_up', 'N/A')}")
        self.reasoning_logger.info(
            f"AI_REASONING: raw_feature name=exit_code value={metrics.get('kube_pod_container_status_last_terminated_exitcode', 'N/A')}")

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
