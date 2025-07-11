"""
Reward function configuration.
"""

from pydantic import ConfigDict, Field, model_validator, AliasChoices
from pydantic_settings import BaseSettings


class RewardConfig(BaseSettings):
    """Reward function configuration."""
    
    # Component weights
    performance_weight: float = Field(default=0.40, validation_alias=AliasChoices("REWARD_PERFORMANCE_WEIGHT", "performance_weight"))
    resource_weight: float = Field(default=0.30, validation_alias=AliasChoices("REWARD_RESOURCE_WEIGHT", "resource_weight"))
    health_weight: float = Field(default=0.20, validation_alias=AliasChoices("REWARD_HEALTH_WEIGHT", "health_weight"))
    cost_weight: float = Field(default=0.10, validation_alias=AliasChoices("REWARD_COST_WEIGHT", "cost_weight"))
    
    # Additional reward parameters
    latency_weight: float = Field(default=10.0, validation_alias=AliasChoices("REWARD_LATENCY_WEIGHT", "latency_weight"))
    replica_cost: float = Field(default=0.1, validation_alias=AliasChoices("REWARD_REPLICA_COST", "replica_cost"))
    enable_improved_rewards: bool = Field(default=True, validation_alias=AliasChoices("ENABLE_IMPROVED_REWARDS", "enable_improved_rewards"))
    
    @model_validator(mode='after')
    def validate_weights_sum(self):
        total = (self.performance_weight + 
                self.resource_weight + 
                self.health_weight + 
                self.cost_weight)
        if abs(total - 1.0) > 0.01:
            raise ValueError('Reward weights must sum to 1.0')
        return self
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    ) 