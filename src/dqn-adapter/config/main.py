"""
Main configuration for DQN Adapter.
Combines all component-specific configurations.
"""

from typing import List, Optional
from pydantic import ConfigDict, Field, AliasChoices
from pydantic_settings import BaseSettings
from collections import namedtuple

# Import component-specific configurations
from .dqn_config import DQNConfig
from .reward_config import RewardConfig
from .infrastructure_config import PrometheusConfig, MinIOConfig, RedisConfig, KubernetesConfig, ServerConfig
from .ai_config import AIConfig, LogLevel


class DQNAdapterConfig(BaseSettings):
    """Main configuration class that combines all component configurations."""
    
    # === GLOBAL SETTINGS ===
    log_level: LogLevel = Field(default=LogLevel.INFO, validation_alias=AliasChoices("LOG_LEVEL", "log_level"))
    save_interval_seconds: int = Field(default=300, validation_alias=AliasChoices("SAVE_INTERVAL_SECONDS", "save_interval_seconds"))
    
    # === FEATURE CONFIGURATION ===
    base_features: List[str] = Field(default=[
        'process_cpu_seconds_total_rate',         # Rate of CPU usage (computed from counter)
        'process_resident_memory_bytes',          # Gauge - memory usage (unchanged)
        'process_virtual_memory_bytes',           # Gauge - virtual memory (unchanged)
        'http_request_duration_seconds_sum_rate', # Rate of total duration (computed from counter)
        'http_requests_total_rate',               # Rate of total requests (computed from counter)
        'http_request_duration_seconds_count_rate', # Rate of request count (computed from counter)
        'process_open_fds',                       # Gauge - file descriptors (unchanged)
        'http_response_size_bytes_sum_rate',      # Rate of response size (computed from counter)
        'http_request_size_bytes_count_rate',     # Rate of request size count (computed from counter)
    ])
    

    
    # === COMPONENT CONFIGURATIONS ===
    # These are not initialized with default_factory to allow proper env var loading
    dqn: DQNConfig = Field(default=None)
    reward: RewardConfig = Field(default=None)
    prometheus: PrometheusConfig = Field(default=None)
    minio: MinIOConfig = Field(default=None)
    redis: RedisConfig = Field(default=None)
    kubernetes: KubernetesConfig = Field(default=None)
    server: ServerConfig = Field(default=None)
    ai: AIConfig = Field(default=None)
    
    def __init__(self, **data):
        """Initialize with component configurations."""
        # Initialize component configs first
        if 'dqn' not in data or data['dqn'] is None:
            data['dqn'] = DQNConfig()
        if 'reward' not in data or data['reward'] is None:
            data['reward'] = RewardConfig()
        if 'prometheus' not in data or data['prometheus'] is None:
            data['prometheus'] = PrometheusConfig()
        if 'minio' not in data or data['minio'] is None:
            data['minio'] = MinIOConfig()
        if 'redis' not in data or data['redis'] is None:
            data['redis'] = RedisConfig()
        if 'kubernetes' not in data or data['kubernetes'] is None:
            data['kubernetes'] = KubernetesConfig()
        if 'server' not in data or data['server'] is None:
            data['server'] = ServerConfig()
        if 'ai' not in data or data['ai'] is None:
            data['ai'] = AIConfig()
            
        super().__init__(**data)
    
    # === COMPUTED PROPERTIES ===
    @property
    def all_features(self) -> List[str]:
        """Get all features (base only)."""
        return self.base_features
    
    @property
    def feature_count(self) -> int:
        """Get total feature count."""
        return len(self.all_features)
    
    @property
    def feature_order(self) -> List[str]:
        """Get feature order for compatibility with constants.FEATURE_ORDER."""
        return self.all_features
    
    def validate_config(self) -> None:
        """Validate the entire configuration."""
        if len(self.base_features) == 0:
            raise ValueError("At least one base feature must be configured")
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_assignment=True,
        extra="forbid"
    )


# Experience tuple for replay buffer
TrainingExperience = namedtuple('TrainingExperience', ['state', 'action', 'reward', 'next_state', 'done'])


def load_config() -> DQNAdapterConfig:
    """
    Load and validate configuration from environment variables.
    
    Returns:
        DQNAdapterConfig: Validated configuration object
        
    Raises:
        ValidationError: If configuration is invalid
    """
    try:
        config = DQNAdapterConfig()
        config.validate_config()
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration: {e}") from e


# Global configuration instance
_config: Optional[DQNAdapterConfig] = None


def get_config() -> DQNAdapterConfig:
    """
    Get the global configuration instance.
    
    Returns:
        DQNAdapterConfig: The global configuration
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config 