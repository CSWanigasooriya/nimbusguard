"""
Simple configuration system using environment variables.
"""

import os
from typing import List

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
        self.lookback_minutes = int(os.getenv("FORECASTING_LOOKBACK_MINUTES", "30"))
        self.forecast_horizon_minutes = int(os.getenv("FORECASTING_HORIZON_MINUTES", "10"))
        self.hidden_units = int(os.getenv("LSTM_HIDDEN_UNITS", "64"))
        self.lstm_hidden_size = int(os.getenv("LSTM_HIDDEN_UNITS", "64"))  # Alias for compatibility
        self.lstm_num_layers = int(os.getenv("LSTM_NUM_LAYERS", "2"))
        self.sequence_length = int(os.getenv("LSTM_SEQUENCE_LENGTH", "10"))
        self.retrain_interval_minutes = int(os.getenv("LSTM_RETRAIN_INTERVAL_MINUTES", "60"))

class ScalingConfig:
    """Scaling configuration."""
    def __init__(self):
        self.target_deployment = os.getenv("TARGET_DEPLOYMENT", "consumer")
        self.target_namespace = os.getenv("TARGET_NAMESPACE", "nimbusguard")
        self.min_replicas = int(os.getenv("MIN_REPLICAS", "1"))
        self.max_replicas = int(os.getenv("MAX_REPLICAS", "10"))
        self.decision_interval = int(os.getenv("DECISION_INTERVAL", "30"))
        
        # DQN configuration
        self.dqn_hidden_dims = [int(x) for x in os.getenv("DQN_HIDDEN_DIMS", "64,32").split(",")]
        self.dqn_learning_rate = float(os.getenv("DQN_LEARNING_RATE", "0.001"))
        self.dqn_gamma = float(os.getenv("DQN_GAMMA", "0.95"))
        self.dqn_epsilon_start = float(os.getenv("DQN_EPSILON_START", "0.3"))
        self.dqn_epsilon_end = float(os.getenv("DQN_EPSILON_END", "0.01"))
        self.dqn_epsilon_decay = float(os.getenv("DQN_EPSILON_DECAY", "0.9995"))  # Much slower decay: ~2000 steps to reach ~0.15
        self.dqn_batch_size = int(os.getenv("DQN_BATCH_SIZE", "16"))
        self.dqn_memory_capacity = int(os.getenv("DQN_MEMORY_CAPACITY", "10000"))
        
        # The 9 scientifically selected features from your existing system
        self.selected_features = [
            'process_cpu_seconds_total_rate',
            'process_resident_memory_bytes',
            'process_virtual_memory_bytes',
            'http_request_duration_seconds_sum_rate',
            'http_requests_total_rate',
            'http_request_duration_seconds_count_rate',
            'process_open_fds',
            'http_response_size_bytes_sum_rate',
            'http_request_size_bytes_count_rate'
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

def load_config() -> Config:
    """Load configuration from environment variables."""
    return Config() 