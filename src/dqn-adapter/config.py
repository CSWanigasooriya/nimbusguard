"""
Configuration module - backwards compatibility import.

This module imports everything from the config package for backwards compatibility.
"""

# Import everything from the config package
from config import *

# Maintain backwards compatibility
__all__ = [
    'DQNAdapterConfig',
    'TrainingExperience', 
    'load_config',
    'get_config',
    'DQNConfig',

    'RewardConfig',
    'PrometheusConfig',
    'MinIOConfig',
    'RedisConfig',
    'KubernetesConfig',
    'ServerConfig',
    'AIConfig',
    'LogLevel'
]
