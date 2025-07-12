"""
DQN-specific configuration.
"""

from typing import List
from pydantic import ConfigDict, Field, field_validator, AliasChoices
from pydantic_settings import BaseSettings


class DQNConfig(BaseSettings):
    """DQN algorithm configuration."""
    
    # Architecture
    hidden_dims_str: str = Field(default="64,32", validation_alias=AliasChoices("DQN_HIDDEN_DIMS", "hidden_dims_str"))
    
    # Training parameters - Optimized for Kubernetes autoscaling
    epsilon_start: float = Field(default=0.15, validation_alias=AliasChoices("EPSILON_START", "epsilon_start"))
    epsilon_end: float = Field(default=0.01, validation_alias=AliasChoices("EPSILON_END", "epsilon_end"))
    epsilon_decay: float = Field(default=0.98, validation_alias=AliasChoices("EPSILON_DECAY", "epsilon_decay"))
    gamma: float = Field(default=0.95, validation_alias=AliasChoices("GAMMA", "gamma"))
    learning_rate: float = Field(default=0.0005, validation_alias=AliasChoices("LEARNING_RATE", "learning_rate"))
    
    # Memory and batching - Optimized for fast learning from negative rewards
    memory_capacity: int = Field(default=50000, validation_alias=AliasChoices("MEMORY_CAPACITY", "memory_capacity"))
    batch_size: int = Field(default=8, validation_alias=AliasChoices("BATCH_SIZE", "batch_size"))
    min_batch_size: int = Field(default=4, validation_alias=AliasChoices("MIN_BATCH_SIZE", "min_batch_size"))
    target_batch_size: int = Field(default=16, validation_alias=AliasChoices("TARGET_BATCH_SIZE", "target_batch_size"))
    target_update_interval: int = Field(default=500, validation_alias=AliasChoices("TARGET_UPDATE_INTERVAL", "target_update_interval"))
    
    # Epsilon management
    reset_epsilon_on_load: bool = Field(default=True, validation_alias=AliasChoices("RESET_EPSILON_ON_LOAD", "reset_epsilon_on_load"))
    
    @property
    def hidden_dims(self) -> List[int]:
        """Parse DQN hidden dimensions from string."""
        try:
            return [int(x.strip()) for x in self.hidden_dims_str.split(',') if x.strip()]
        except ValueError as e:
            raise ValueError(f"Invalid format for DQN_HIDDEN_DIMS: {self.hidden_dims_str}. Expected comma-separated integers.") from e
    
    @field_validator('epsilon_start', 'epsilon_end', 'epsilon_decay', 'gamma')
    @classmethod
    def validate_probability_range(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Probability values must be between 0.0 and 1.0')
        return v
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    ) 