"""
Configuration system using environment variables for the NimbusGuard operator.
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
    """Dual-model forecasting configuration for GRU (CPU) and LSTM (Memory) models."""

    def __init__(self):
        self.enabled = os.getenv("FORECASTING_ENABLED", "true").lower() == "true"

        # General forecasting parameters
        self.lookback_minutes = int(os.getenv("FORECASTING_LOOKBACK_MINUTES", "15"))
        self.forecast_horizon_seconds = int(os.getenv("FORECASTING_HORIZON_SECONDS", "15"))

        # CPU Model Configuration (GRU-based)
        self.cpu_model_type = os.getenv("CPU_MODEL_TYPE", "gru")
        self.cpu_sequence_length = int(os.getenv("CPU_SEQUENCE_LENGTH", "8"))
        self.cpu_features_count = int(os.getenv("CPU_FEATURES_COUNT", "9"))
        self.cpu_prediction_steps = int(os.getenv("CPU_PREDICTION_STEPS", "1"))

        # Memory Model Configuration (LSTM-based)
        self.memory_model_type = os.getenv("MEMORY_MODEL_TYPE", "lstm")
        self.memory_sequence_length = int(os.getenv("MEMORY_SEQUENCE_LENGTH", "4"))
        self.memory_features_count = int(os.getenv("MEMORY_FEATURES_COUNT", "4"))
        self.memory_prediction_steps = int(os.getenv("MEMORY_PREDICTION_STEPS", "10"))

        # Legacy support for old LSTM_SEQUENCE_LENGTH variable
        self.sequence_length = max(self.cpu_sequence_length, self.memory_sequence_length)

        # Features will be set by Config.__init__ from scaling.selected_features
        self.selected_features: list[str] = []


class ScalingConfig:
    """Scaling configuration."""

    def __init__(self):
        self.target_deployment = os.getenv("TARGET_DEPLOYMENT", "consumer")
        self.target_namespace = os.getenv("TARGET_NAMESPACE", "nimbusguard")

        # DQN configuration
        self.dqn_hidden_dims = [int(x) for x in os.getenv("DQN_HIDDEN_DIMS", "64,32").split(",")]
        self.dqn_learning_rate = float(os.getenv("DQN_LEARNING_RATE", "0.001"))  # Increased for faster training
        self.dqn_gamma = float(os.getenv("DQN_GAMMA", "0.95"))
        self.dqn_epsilon_start = float(os.getenv("DQN_EPSILON_START", "0.9"))
        self.dqn_epsilon_end = float(os.getenv("DQN_EPSILON_END", "0.01"))
        self.dqn_batch_size = int(os.getenv("DQN_BATCH_SIZE", "32"))
        self.dqn_memory_capacity = int(os.getenv("DQN_MEMORY_CAPACITY", "10000"))

        # Additional DQN configuration (previously hardcoded)
        self.dqn_epsilon_decay = float(os.getenv("DQN_EPSILON_DECAY", "0.995"))
        self.dqn_min_replay_size = int(os.getenv("DQN_MIN_REPLAY_SIZE", "100"))
        self.dqn_target_update_frequency = int(os.getenv("DQN_TARGET_UPDATE_FREQUENCY", "100"))

        # Training loop configuration
        self.stabilization_period_seconds = int(os.getenv("STABILIZATION_PERIOD_SECONDS", "15"))

        # Consumer-focused features based on HPA baseline analysis
        # These features directly correlate with scaling decisions
        self.selected_features = [
            'process_cpu_seconds_total_rate',     # CPU per pod (0.48-15.51s spikes)
            'process_resident_memory_bytes',      # Memory per pod (61-197MB spikes)  
            'kube_deployment_status_replicas',    # Current replica count (essential for scaling decisions)
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



class AIConfig:
    def __init__(self):
        self.model_name = os.getenv("AI_MODEL", "gpt-4-turbo")
        self.temperature = float(os.getenv("AI_TEMPERATURE", "0.1"))
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.mcp_server_url = os.getenv("MCP_SERVER_URL", "http://mcp-server.nimbusguard.svc:8080")
        self.enable_llm_validation = os.getenv("ENABLE_LLM_VALIDATION", "false").lower() == "true"


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
        self.ai = AIConfig()

        # Share feature list between scaling and forecasting configs
        self.forecasting.selected_features = self.scaling.selected_features


def load_config() -> Config:
    """Load configuration from environment variables."""
    return Config()
