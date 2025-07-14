"""
Simple configuration system using environment variables.
"""

import os


class PrometheusConfig:
    """Prometheus configuration."""

    def __init__(self):
        self.url = os.getenv("PROMETHEUS_URL", "http://prometheus.nimbusguard.svc:9090")
        self.timeout = int(os.getenv("PROMETHEUS_TIMEOUT", "30"))


class MinIOConfig:
    """MinIO configuration."""

    def __init__(self):
        self.endpoint = os.getenv("MINIO_ENDPOINT", "http://minio.nimbusguard.svc:9000")
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")


class ForecastingConfig:
    """LSTM forecasting configuration."""

    def __init__(self):
        self.enabled = os.getenv("FORECASTING_ENABLED", "true").lower() == "true"
        self.lookback_minutes = int(os.getenv("FORECASTING_LOOKBACK_MINUTES", "60"))  # Increased for better training
        self.forecast_horizon_minutes = int(
            os.getenv("FORECASTING_HORIZON_MINUTES", "5"))  # Shorter horizon for better accuracy
        self.hidden_units = int(os.getenv("LSTM_HIDDEN_UNITS", "32"))  # Smaller model for limited data
        self.lstm_hidden_size = int(os.getenv("LSTM_HIDDEN_UNITS", "32"))  # Alias for compatibility
        self.lstm_num_layers = int(os.getenv("LSTM_NUM_LAYERS", "1"))  # Single layer for small datasets
        self.sequence_length = int(os.getenv("LSTM_SEQUENCE_LENGTH", "15"))  # Longer sequences for better patterns
        self.retrain_interval_minutes = int(os.getenv("LSTM_RETRAIN_INTERVAL_MINUTES", "30"))  # More frequent training

        # New parameters for improved training
        self.min_training_samples = int(os.getenv("LSTM_MIN_TRAINING_SAMPLES", "20"))  # Minimum data for training
        self.validation_split = float(os.getenv("LSTM_VALIDATION_SPLIT", "0.2"))  # 20% for validation
        self.early_stopping_patience = int(os.getenv("LSTM_EARLY_STOPPING_PATIENCE", "10"))  # Early stopping
        self.learning_rate = float(os.getenv("LSTM_LEARNING_RATE", "0.001"))  # Learning rate
        self.batch_size = int(os.getenv("LSTM_BATCH_SIZE", "8"))  # Smaller batch size for limited data
        self.max_epochs = int(os.getenv("LSTM_MAX_EPOCHS", "100"))  # More epochs for better training


class ScalingConfig:
    """Scaling configuration."""

    def __init__(self):
        self.target_deployment = os.getenv("TARGET_DEPLOYMENT", "consumer")
        self.target_namespace = os.getenv("TARGET_NAMESPACE", "nimbusguard")
        self.min_replicas = int(os.getenv("MIN_REPLICAS", "1"))
        self.max_replicas = int(os.getenv("MAX_REPLICAS", "10"))

        # DQN configuration
        self.dqn_hidden_dims = [int(x) for x in os.getenv("DQN_HIDDEN_DIMS", "64,32").split(",")]
        self.dqn_learning_rate = float(os.getenv("DQN_LEARNING_RATE", "0.001"))
        self.dqn_gamma = float(os.getenv("DQN_GAMMA", "0.95"))
        self.dqn_epsilon_start = float(os.getenv("DQN_EPSILON_START", "0.3"))
        self.dqn_epsilon_end = float(os.getenv("DQN_EPSILON_END", "0.01"))
        self.dqn_epsilon_decay = float(
            os.getenv("DQN_EPSILON_DECAY", "0.9995"))  # Much slower decay: ~2000 steps to reach ~0.15
        self.dqn_batch_size = int(os.getenv("DQN_BATCH_SIZE", "16"))
        self.dqn_memory_capacity = int(os.getenv("DQN_MEMORY_CAPACITY", "10000"))

        # Training loop configuration
        self.stabilization_period_seconds = int(os.getenv("STABILIZATION_PERIOD_SECONDS", "30"))

        # Consumer-focused features based on HPA baseline analysis
        # These features directly correlate with scaling decisions
        self.selected_features = [
            'http_request_duration_seconds_sum_rate',  # PRIMARY: Request latency (35-542s spikes)
            'http_request_duration_seconds_count_rate',  # Request rate (actual workload)
            'process_cpu_seconds_total_rate',  # CPU per pod (0.48-15.51s spikes)
            'process_resident_memory_bytes',  # Memory per pod (61-197MB spikes)
            'http_requests_total_process_rate',  # Actual /process endpoint requests
            'http_response_size_bytes_sum_rate',  # Response throughput
            'process_open_fds',  # Connection pressure
            'kube_pod_container_resource_limits_cpu',  # Resource constraints
            'http_server_active_connections'  # Active connections (if available)
        ]


class ServerConfig:
    """HTTP server configuration."""

    def __init__(self):
        self.port = int(os.getenv("SERVER_PORT", "8080"))
        self.health_port = int(os.getenv("HEALTH_PORT", "8081"))


class LoggingConfig:
    """Logging configuration."""

    def __init__(self):
        self.level = os.getenv("LOG_LEVEL", "INFO")


class MetricsConfig:
    """Metrics configuration."""

    def __init__(self):
        self.enabled = os.getenv("METRICS_ENABLED", "true").lower() == "true"


class RewardConfig:
    """Reward system configuration."""

    def __init__(self):
        # Single weight parameter - current gets this weight, forecast gets (1 - weight)
        self.current_weight = float(os.getenv("REWARD_CURRENT_WEIGHT", "0.5"))
        self.forecast_weight = round(1.0 - self.current_weight, 10)  # Round to avoid floating-point precision issues


class Config:
    """Main configuration object."""

    def __init__(self):
        self.prometheus = PrometheusConfig()
        self.minio = MinIOConfig()
        self.forecasting = ForecastingConfig()
        self.scaling = ScalingConfig()
        self.server = ServerConfig()
        self.logging = LoggingConfig()
        self.metrics = MetricsConfig()
        self.reward = RewardConfig()


def load_config() -> Config:
    """Load configuration from environment variables."""
    return Config()
