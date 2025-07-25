"""
Prometheus metrics collector for NimbusGuard operator.
"""

import logging
from typing import Dict, Optional
from prometheus_client import (
    Gauge, Counter, Histogram, Summary, CollectorRegistry, 
    start_http_server, REGISTRY
)

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and exposes Prometheus metrics for the NimbusGuard operator."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics collector.
        
        Args:
            registry: Prometheus registry to use, defaults to global registry
        """
        self.registry = registry or REGISTRY
        
        # Initialize all metrics
        self._init_dqn_metrics()
        self._init_forecasting_metrics()
        self._init_system_metrics()
        
        logger.info("Metrics collector initialized")
    
    def _init_dqn_metrics(self):
        """Initialize DQN-related metrics."""
        
        # DQN State and Decision Metrics
        self.dqn_desired_replicas = Gauge(
            'nimbusguard_dqn_desired_replicas',
            'Number of replicas desired by DQN agent',
            registry=self.registry
        )
        
        self.current_replicas = Gauge(
            'nimbusguard_current_replicas',
            'Current number of replicas in deployment',
            registry=self.registry
        )
        
        # DQN Training Metrics
        self.dqn_training_loss = Gauge(
            'dqn_training_loss',
            'Current DQN training loss',
            registry=self.registry
        )
        
        self.dqn_epsilon_value = Gauge(
            'dqn_epsilon_value',
            'Current epsilon value for exploration',
            registry=self.registry
        )
        
        # DQN Action Counters
        self.dqn_exploration_actions = Counter(
            'dqn_exploration_actions_total',
            'Total number of exploration actions taken',
            registry=self.registry
        )
        
        self.dqn_exploitation_actions = Counter(
            'dqn_exploitation_actions_total',
            'Total number of exploitation actions taken',
            registry=self.registry
        )
        
        # DQN Experience Replay Metrics
        self.dqn_replay_buffer_size = Gauge(
            'dqn_replay_buffer_size',
            'Current size of experience replay buffer',
            registry=self.registry
        )
        
        self.dqn_experiences_added = Counter(
            'dqn_experiences_added_total',
            'Total number of experiences added to replay buffer',
            registry=self.registry
        )
        
        self.dqn_training_steps = Counter(
            'dqn_training_steps_total',
            'Total number of DQN training steps completed',
            registry=self.registry
        )
        
        # DQN Action Distribution
        self.dqn_action_scale_up = Counter(
            'dqn_action_scale_up_total',
            'Total number of scale up actions',
            registry=self.registry
        )
        
        self.dqn_action_scale_down = Counter(
            'dqn_action_scale_down_total',
            'Total number of scale down actions',
            registry=self.registry
        )
        
        self.dqn_action_keep_same = Counter(
            'dqn_action_keep_same_total',
            'Total number of keep same actions',
            registry=self.registry
        )
        
        # DQN Q-Values
        self.dqn_q_value_scale_up = Gauge(
            'dqn_q_value_scale_up',
            'Q-value for scale up action',
            registry=self.registry
        )
        
        self.dqn_q_value_scale_down = Gauge(
            'dqn_q_value_scale_down',
            'Q-value for scale down action',
            registry=self.registry
        )
        
        self.dqn_q_value_keep_same = Gauge(
            'dqn_q_value_keep_same',
            'Q-value for keep same action',
            registry=self.registry
        )
        
        # DQN Reward Metrics
        self.dqn_reward_total = Gauge(
            'dqn_reward_total',
            'Total cumulative reward',
            registry=self.registry
        )
        
        self.dqn_decision_confidence = Gauge(
            'dqn_decision_confidence_avg',
            'Average confidence of DQN decisions',
            registry=self.registry
        )
        
        self.dqn_decisions = Counter(
            'dqn_decisions_total',
            'Total number of DQN decisions made',
            registry=self.registry
        )
    
    def _init_forecasting_metrics(self):
        """Initialize forecasting-related metrics."""
        
        # AI Forecast Results
        self.gru_forecast_cpu_rate = Gauge(
            'gru_forecast_cpu_rate',
            'GRU predicted CPU rate for next interval',
            registry=self.registry
        )
        
        self.lstm_forecast_memory_bytes = Gauge(
            'lstm_forecast_memory_bytes',
            'LSTM predicted memory usage in bytes for next interval',
            registry=self.registry
        )
        
        # Forecasting Performance
        self.forecasting_prediction_time = Histogram(
            'forecasting_prediction_time_seconds',
            'Time taken to generate forecasts',
            registry=self.registry
        )
        
        self.forecasting_errors = Counter(
            'forecasting_errors_total',
            'Total number of forecasting errors',
            ['model_type'],  # cpu or memory
            registry=self.registry
        )
    
    def _init_system_metrics(self):
        """Initialize system and workflow metrics."""
        
        # Workflow Performance
        self.workflow_execution_time = Histogram(
            'workflow_execution_time_seconds',
            'Time taken to execute complete scaling workflow',
            registry=self.registry
        )
        
        self.workflow_errors = Counter(
            'workflow_errors_total',
            'Total number of workflow execution errors',
            ['error_type'],
            registry=self.registry
        )
        
        # Prometheus Metrics Collection
        self.prometheus_fetch_time = Histogram(
            'prometheus_fetch_time_seconds',
            'Time taken to fetch metrics from Prometheus',
            registry=self.registry
        )
        
        self.prometheus_fetch_errors = Counter(
            'prometheus_fetch_errors_total',
            'Total number of Prometheus fetch errors',
            registry=self.registry
        )
        
        # Scaling Actions
        self.scaling_actions_executed = Counter(
            'scaling_actions_executed_total',
            'Total number of scaling actions executed',
            ['action'],  # scale_up, scale_down, no_action
            registry=self.registry
        )
        
        self.scaling_validation_failures = Counter(
            'scaling_validation_failures_total',
            'Total number of scaling validation failures',
            ['reason'],
            registry=self.registry
        )
    
    # DQN Metric Updates
    def update_dqn_decision(self, action: str, desired_replicas: int, current_replicas: int, 
                           confidence: float, q_values: Optional[Dict[str, float]] = None):
        """Update DQN decision metrics."""
        self.dqn_desired_replicas.set(desired_replicas)
        self.current_replicas.set(current_replicas)
        self.dqn_decision_confidence.set(confidence)
        self.dqn_decisions.inc()
        
        # Update action counters
        if action == "scale_up":
            self.dqn_action_scale_up.inc()
        elif action == "scale_down":
            self.dqn_action_scale_down.inc()
        else:
            self.dqn_action_keep_same.inc()
        
        # Update Q-values if provided
        if q_values:
            self.dqn_q_value_scale_up.set(q_values.get('scale_up', 0))
            self.dqn_q_value_scale_down.set(q_values.get('scale_down', 0))
            self.dqn_q_value_keep_same.set(q_values.get('keep_same', 0))
    
    def update_dqn_training(self, loss: float, epsilon: float, buffer_size: int):
        """Update DQN training metrics."""
        self.dqn_training_loss.set(loss)
        self.dqn_epsilon_value.set(epsilon)
        self.dqn_replay_buffer_size.set(buffer_size)
        self.dqn_training_steps.inc()
    
    def record_dqn_action(self, is_exploration: bool):
        """Record whether action was exploration or exploitation."""
        if is_exploration:
            self.dqn_exploration_actions.inc()
        else:
            self.dqn_exploitation_actions.inc()
    
    def add_experience(self):
        """Record addition of experience to replay buffer."""
        self.dqn_experiences_added.inc()
    
    def update_reward(self, reward: float):
        """Update reward metrics."""
        current_total = self.dqn_reward_total._value._value
        self.dqn_reward_total.set(current_total + reward)
    
    # Forecasting Metric Updates
    def update_forecasts(self, cpu_forecast: Optional[float], memory_forecast: Optional[float]):
        """Update forecasting predictions."""
        if cpu_forecast is not None:
            self.gru_forecast_cpu_rate.set(cpu_forecast)
        
        if memory_forecast is not None:
            # Convert MB to bytes for dashboard consistency
            memory_bytes = memory_forecast * 1024 * 1024 if memory_forecast else 0
            self.lstm_forecast_memory_bytes.set(memory_bytes)
    
    def record_forecasting_error(self, model_type: str):
        """Record forecasting error."""
        self.forecasting_errors.labels(model_type=model_type).inc()
    
    # System Metric Updates
    def record_workflow_execution(self, duration: float, success: bool, error_type: str = None):
        """Record workflow execution metrics."""
        self.workflow_execution_time.observe(duration)
        
        if not success and error_type:
            self.workflow_errors.labels(error_type=error_type).inc()
    
    def record_prometheus_fetch(self, duration: float, success: bool):
        """Record Prometheus fetch metrics."""
        self.prometheus_fetch_time.observe(duration)
        
        if not success:
            self.prometheus_fetch_errors.inc()
    
    def record_scaling_action(self, action: str, executed: bool, validation_failure_reason: str = None):
        """Record scaling action metrics."""
        if executed:
            self.scaling_actions_executed.labels(action=action).inc()
        
        if validation_failure_reason:
            self.scaling_validation_failures.labels(reason=validation_failure_reason).inc()
    
    def start_metrics_server(self, port: int = 8080):
        """Start HTTP server to serve metrics."""
        try:
            start_http_server(port, registry=self.registry)
            logger.info(f"Metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise


# Global metrics collector instance
metrics = MetricsCollector() 