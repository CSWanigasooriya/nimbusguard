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
    """Minimal forecasting configuration for using a pre-trained LSTM model."""

    def __init__(self):
        self.enabled = os.getenv("FORECASTING_ENABLED", "true").lower() == "true"

        # How much history to fetch from Prometheus for the model input
        self.lookback_minutes = int(os.getenv("FORECASTING_LOOKBACK_MINUTES", "6"))

        # Horizon: 1 step ahead = 15 seconds (since LSTM was trained on 15s intervals)
        self.forecast_horizon_seconds = int(os.getenv("FORECASTING_HORIZON_SECONDS", "15"))

        # Number of data points (timesteps) fed into the model: 24 timesteps × 15s each = 360s lookback
        self.sequence_length = int(os.getenv("LSTM_SEQUENCE_LENGTH", "24"))

        # Features will be set by Config.__init__ from scaling.selected_features
        self.selected_features: list[str] = []


class ScalingConfig:
    """Scaling configuration."""

    def __init__(self):
        self.target_deployment = os.getenv("TARGET_DEPLOYMENT", "consumer")
        self.target_namespace = os.getenv("TARGET_NAMESPACE", "nimbusguard")
        self.min_replicas = int(os.getenv("MIN_REPLICAS", "1"))
        self.max_replicas = int(os.getenv("MAX_REPLICAS", "50"))

        # DQN configuration
        self.dqn_hidden_dims = [int(x) for x in os.getenv("DQN_HIDDEN_DIMS", "64,32").split(",")]
        self.dqn_learning_rate = float(os.getenv("DQN_LEARNING_RATE", "0.001"))
        self.dqn_gamma = float(os.getenv("DQN_GAMMA", "0.95"))
        self.dqn_epsilon_start = float(os.getenv("DQN_EPSILON_START", "0.2"))
        self.dqn_epsilon_end = float(os.getenv("DQN_EPSILON_END", "0.01"))
        self.dqn_batch_size = int(os.getenv("DQN_BATCH_SIZE", "16"))
        self.dqn_memory_capacity = int(os.getenv("DQN_MEMORY_CAPACITY", "10000"))

        # Training loop configuration
        self.stabilization_period_seconds = int(os.getenv("STABILIZATION_PERIOD_SECONDS", "15"))

        # Consumer-focused features based on HPA baseline analysis
        # These features directly correlate with scaling decisions
        self.selected_features = [
            'process_cpu_seconds_total_rate',  # CPU per pod (0.48-15.51s spikes)
            'process_resident_memory_bytes',  # Memory per pod (61-197MB spikes)
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
        self.reward = RewardConfig()
        self.ai = AIConfig()

        # Share feature list between scaling and forecasting configs
        self.forecasting.selected_features = self.scaling.selected_features


def load_config() -> Config:
    """Load configuration from environment variables."""
    return Config()
