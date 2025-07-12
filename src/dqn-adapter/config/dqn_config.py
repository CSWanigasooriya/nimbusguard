"""
DQN-specific configuration.
"""

from typing import List
from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings


class DQNConfig(BaseSettings):
    """DQN algorithm configuration."""
    
    # Architecture
    hidden_dims_str: str = Field(default="64,32", env="DQN_HIDDEN_DIMS")
    
    # Training parameters  
    epsilon_start: float = Field(default=1.0, env="EPSILON_START")  # Start with 100% exploration
    epsilon_end: float = Field(default=0.05, env="EPSILON_END")     # End with 5% exploration
    epsilon_decay: float = Field(default=0.995, env="EPSILON_DECAY") # Decay 0.5% per decision
    gamma: float = Field(default=0.99, env="GAMMA")
    learning_rate: float = Field(default=1e-4, env="LEARNING_RATE")
    
    # Memory and batching
    memory_capacity: int = Field(default=50000, env="MEMORY_CAPACITY")
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    min_batch_size: int = Field(default=8, env="MIN_BATCH_SIZE")
    target_batch_size: int = Field(default=64, env="TARGET_BATCH_SIZE")
    target_update_interval: int = Field(default=1000, env="TARGET_UPDATE_INTERVAL")
    
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