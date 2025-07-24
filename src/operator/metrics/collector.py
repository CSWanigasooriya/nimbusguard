"""
Metrics collection and monitoring for the operator.
"""

import logging
from typing import Dict, Any, Optional

from prometheus_client import Counter, Gauge, Histogram, generate_latest

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and exposes Prometheus metrics for the operator."""

    def __init__(self, metrics_config):
        self.config = metrics_config
        self.logger = logger

        # Initialize Prometheus metrics
        self._init_metrics()

        self.logger.info("Metrics collector initialized")

    def _init_metrics(self):
        """Initialize all Prometheus metrics."""

        # Core scaling metrics (matching dashboard expectations)
        self.current_replicas = Gauge(
            'nimbusguard_current_replicas',
            'Current number of replicas'
        )

        self.dqn_desired_replicas = Gauge(
            'nimbusguard_dqn_desired_replicas',
            'Desired number of replicas from DQN'
        )

        self.scaling_decisions_total = Counter(
            'nimbusguard_scaling_decisions_total',
            'Total number of scaling decisions made',
            ['action', 'reason']
        )

        # DQN Core Metrics
        self.dqn_training_loss = Gauge(
            'dqn_training_loss',
            'Current DQN training loss'
        )

        self.dqn_epsilon_value = Gauge(
            'dqn_epsilon_value',
            'Current DQN exploration epsilon value'
        )

        self.dqn_exploration_actions_total = Counter(
            'dqn_exploration_actions_total',
            'Total number of exploration actions taken'
        )

        self.dqn_exploitation_actions_total = Counter(
            'dqn_exploitation_actions_total',
            'Total number of exploitation actions taken'
        )

        self.dqn_replay_buffer_size = Gauge(
            'dqn_replay_buffer_size',
            'Current replay buffer size'
        )

        self.dqn_experiences_added_total = Counter(
            'dqn_experiences_added_total',
            'Total experiences added to replay buffer'
        )

        self.dqn_training_steps_total = Counter(
            'dqn_training_steps_total',
            'Total DQN training steps'
        )

        # DQN Action Distribution
        self.dqn_action_scale_up_total = Counter(
            'dqn_action_scale_up_total',
            'Total scale up actions chosen by DQN'
        )

        self.dqn_action_scale_down_total = Counter(
            'dqn_action_scale_down_total',
            'Total scale down actions chosen by DQN'
        )

        self.dqn_action_keep_same_total = Counter(
            'dqn_action_keep_same_total',
            'Total keep same actions chosen by DQN'
        )

        # LSTM forecast metrics
        self.lstm_forecast_cpu_rate = Gauge(
            'lstm_forecast_cpu_rate',
            'LSTM predicted CPU rate for next step'
        )

        self.lstm_forecast_memory_bytes = Gauge(
            'lstm_forecast_memory_bytes', 
            'LSTM predicted memory bytes for next step'
        )

        # DQN Q-Value Distribution
        self.dqn_q_value_scale_up = Gauge(
            'dqn_q_value_scale_up',
            'Q-value for scale up action'
        )

        self.dqn_q_value_scale_down = Gauge(
            'dqn_q_value_scale_down',
            'Q-value for scale down action'
        )

        self.dqn_q_value_keep_same = Gauge(
            'dqn_q_value_keep_same',
            'Q-value for keep same action'
        )

        # Reward Analysis
        self.dqn_reward_total = Gauge(
            'dqn_reward_total',
            'Total accumulated DQN reward'
        )

        # DQN Performance Metrics
        self.dqn_decision_confidence_avg = Gauge(
            'dqn_decision_confidence_avg',
            'Average DQN decision confidence'
        )

        self.dqn_decisions_total = Counter(
            'dqn_decisions_total',
            'Total DQN decisions made'
        )

        # System metrics
        self.scaling_duration = Histogram(
            'nimbusguard_scaling_duration_seconds',
            'Time taken for scaling operations',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )

        # Forecasting metrics
        self.forecast_horizon_seconds = Gauge(
            'nimbusguard_forecast_horizon_seconds',
            'Forecast horizon in seconds'
        )

        # System health metrics
        self.system_health_score = Gauge(
            'nimbusguard_system_health_score',
            'Overall system health score (0-1)'
        )

        self.component_status = Gauge(
            'nimbusguard_component_status',
            'Status of operator components (1=healthy, 0=unhealthy)',
            ['component']
        )

        self.decision_workflow_duration = Histogram(
            'nimbusguard_decision_workflow_duration_seconds',
            'Time taken for complete decision workflow',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )

        # Performance metrics
        self.prometheus_query_duration = Histogram(
            'nimbusguard_prometheus_query_duration_seconds',
            'Duration of Prometheus queries',
            ['query_type']
        )

        self.model_save_duration = Histogram(
            'nimbusguard_model_save_duration_seconds',
            'Duration of model save operations'
        )

        # Error tracking
        self.errors_total = Counter(
            'nimbusguard_errors_total',
            'Total number of errors by component',
            ['component', 'error_type']
        )

        # Initialize counters to 0 to ensure they appear in metrics
        self._initialize_counters()

    def _initialize_counters(self):
        """Initialize all counters to 0 so they appear in metrics."""
        # Initialize action counters
        for action in ['scale_up', 'scale_down', 'keep_same']:
            getattr(self, f'dqn_action_{action}_total')._value._value = 0

        # Initialize exploration counters
        self.dqn_exploration_actions_total._value._value = 0
        self.dqn_exploitation_actions_total._value._value = 0

        # Initialize other counters
        self.dqn_experiences_added_total._value._value = 0
        self.dqn_training_steps_total._value._value = 0
        self.dqn_decisions_total._value._value = 0

    async def update_scaling_metrics(self, action: str, current_replicas: int,
                                     desired_replicas: int, reason: str = "DQN"):
        """Update scaling-related metrics."""
        try:
            self.scaling_decisions_total.labels(action=action, reason=reason).inc()
            self.current_replicas.set(current_replicas)
            self.dqn_desired_replicas.set(desired_replicas)

            # Update action-specific counters
            if action == "scale_up":
                self.dqn_action_scale_up_total.inc()
            elif action == "scale_down":
                self.dqn_action_scale_down_total.inc()
            else:  # keep_same
                self.dqn_action_keep_same_total.inc()

            self.logger.debug(f"Updated scaling metrics: {action}, {current_replicas}→{desired_replicas}")

        except Exception as e:
            self.logger.error(f"Failed to update scaling metrics: {e}")

    async def update_dqn_metrics(self, training_stats: Dict[str, Any],
                                 q_values: Optional[Dict[str, float]] = None,
                                 decision_confidence: Optional[float] = None,
                                 is_exploration: Optional[bool] = None):
        """Update DQN-related metrics."""
        try:
            # Update training metrics
            if 'loss' in training_stats:
                self.dqn_training_loss.set(training_stats['loss'])

            if 'epsilon' in training_stats:
                self.dqn_epsilon_value.set(training_stats['epsilon'])

            if 'buffer_size' in training_stats:
                self.dqn_replay_buffer_size.set(training_stats['buffer_size'])

            if 'training_steps' in training_stats:
                # Note: Counters increment, so we track the delta
                current_steps = self.dqn_training_steps_total._value._value
                new_steps = training_stats['training_steps']
                if new_steps > current_steps:
                    for _ in range(int(new_steps - current_steps)):
                        self.dqn_training_steps_total.inc()

            # Update Q-values
            if q_values:
                for action, value in q_values.items():
                    self.dqn_q_values.labels(action=action).set(value)

                # Update specific Q-value metrics
                if 'scale_up' in q_values:
                    self.dqn_q_value_scale_up.set(q_values['scale_up'])
                if 'scale_down' in q_values:
                    self.dqn_q_value_scale_down.set(q_values['scale_down'])
                if 'keep_same' in q_values:
                    self.dqn_q_value_keep_same.set(q_values['keep_same'])

            # Update decision confidence
            if decision_confidence is not None:
                self.dqn_decision_confidence_avg.set(decision_confidence)
                self.dqn_decisions_total.inc()

            # Update exploration/exploitation
            if is_exploration is not None:
                if is_exploration:
                    self.dqn_exploration_actions_total.inc()
                else:
                    self.dqn_exploitation_actions_total.inc()

            self.logger.debug("Updated DQN metrics")

        except Exception as e:
            self.logger.error(f"Failed to update DQN metrics: {e}")

    async def update_reward_metrics(self, reward_value: float, total_reward: Optional[float] = None):
        """Update reward analysis metrics."""
        try:
            if total_reward is not None:
                self.dqn_reward_total.set(total_reward)
            else:
                # Accumulate reward
                current_total = self.dqn_reward_total._value._value
                self.dqn_reward_total.set(current_total + reward_value)

            self.logger.debug(f"Updated reward metrics: {reward_value}")

        except Exception as e:
            self.logger.error(f"Failed to update reward metrics: {e}")

    async def update_experience_metrics(self, experiences_added: int = 1):
        """Update experience replay metrics."""
        try:
            for _ in range(experiences_added):
                self.dqn_experiences_added_total.inc()

            self.logger.debug(f"Updated experience metrics: +{experiences_added}")

        except Exception as e:
            self.logger.error(f"Failed to update experience metrics: {e}")

    async def update_forecast_metrics(self, forecast_result: Dict[str, Any]):
        """Update LSTM forecast metrics with real predictions."""
        try:
            if 'predicted_metrics' in forecast_result:
                predicted = forecast_result['predicted_metrics']
                
                # Update CPU forecast
                if 'process_cpu_seconds_total_rate' in predicted:
                    self.lstm_forecast_cpu_rate.set(predicted['process_cpu_seconds_total_rate'])
                
                # Update memory forecast  
                if 'process_resident_memory_bytes' in predicted:
                    self.lstm_forecast_memory_bytes.set(predicted['process_resident_memory_bytes'])

            # Update horizon metric
            if 'horizon_seconds' in forecast_result:
                self.forecast_horizon_seconds.set(forecast_result['horizon_seconds'])

            self.logger.debug("Updated LSTM forecast metrics")

        except Exception as e:
            self.logger.error(f"Failed to update forecast metrics: {e}")

    async def update_component_status(self, component: str, is_healthy: bool):
        """Update component health status."""
        try:
            self.component_status.labels(component=component).set(1 if is_healthy else 0)

        except Exception as e:
            self.logger.error(f"Failed to update component status: {e}")

    async def record_error(self, component: str, error_type: str):
        """Record an error occurrence."""
        try:
            self.errors_total.labels(component=component, error_type=error_type).inc()

        except Exception as e:
            self.logger.error(f"Failed to record error metric: {e}")

    async def record_decision_duration(self, duration_seconds: float):
        """Record decision workflow duration."""
        try:
            self.decision_workflow_duration.observe(duration_seconds)

        except Exception as e:
            self.logger.error(f"Failed to record decision duration: {e}")

    async def record_prometheus_query_duration(self, query_type: str, duration_seconds: float):
        """Record Prometheus query duration."""
        try:
            self.prometheus_query_duration.labels(query_type=query_type).observe(duration_seconds)

        except Exception as e:
            self.logger.error(f"Failed to record Prometheus query duration: {e}")

    async def record_model_save_duration(self, duration_seconds: float):
        """Record model save duration."""
        try:
            self.model_save_duration.observe(duration_seconds)

        except Exception as e:
            self.logger.error(f"Failed to record model save duration: {e}")

    async def update_replica_metrics(self, old_replicas: int, new_replicas: int):
        """Update replica change metrics."""
        try:
            self.current_replicas.set(new_replicas)

            if old_replicas != new_replicas:
                action = "scale_up" if new_replicas > old_replicas else "scale_down"
                self.scaling_decisions_total.labels(action=action, reason="external").inc()

        except Exception as e:
            self.logger.error(f"Failed to update replica metrics: {e}")

    async def update_decision_metrics(self, workflow_result: Dict[str, Any]):
        """Update metrics from workflow decision result."""
        try:
            # Extract metrics from workflow result
            if 'final_decision' in workflow_result:
                final_decision = workflow_result['final_decision']

                if 'replicas' in final_decision:
                    self.dqn_desired_replicas.set(final_decision['replicas'])

                if 'confidence' in final_decision:
                    self.dqn_decision_confidence_avg.set(final_decision['confidence'])
                    self.dqn_decisions_total.inc()

                if 'action' in final_decision:
                    action = final_decision['action']
                    if action == "scale_up":
                        self.dqn_action_scale_up_total.inc()
                    elif action == "scale_down":
                        self.dqn_action_scale_down_total.inc()
                    else:
                        self.dqn_action_keep_same_total.inc()

            # Update system health based on workflow success
            workflow_success = workflow_result.get('success', False)
            self.system_health_score.set(1.0 if workflow_success else 0.5)

            # Record decision
            self.dqn_decisions_total.inc()

        except Exception as e:
            self.logger.error(f"Failed to update decision metrics: {e}")

    async def get_prometheus_metrics(self) -> str:
        """Get all metrics in Prometheus format."""
        try:
            return generate_latest().decode('utf-8')
        except Exception as e:
            self.logger.error(f"Failed to generate Prometheus metrics: {e}")
            return "# Error generating metrics\n"

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        try:
            return {
                'current_replicas': self.current_replicas._value._value,
                'dqn_desired_replicas': self.dqn_desired_replicas._value._value,
                'dqn_epsilon_value': self.dqn_epsilon_value._value._value,
                'dqn_replay_buffer_size': self.dqn_replay_buffer_size._value._value,
                'system_health_score': self.system_health_score._value._value,
                'dqn_training_loss': self.dqn_training_loss._value._value,
                'dqn_decisions_total': self.dqn_decisions_total._value._value
            }
        except Exception as e:
            self.logger.error(f"Failed to get metrics summary: {e}")
            return {}
