"""
Reward function configuration.
"""

from pydantic import ConfigDict, Field, model_validator, AliasChoices
from pydantic_settings import BaseSettings


class RewardConfig(BaseSettings):
    """Reward function configuration."""
    
    # Component weights (must sum to 1.0 for appropriateness + proactive components)
    performance_weight: float = Field(default=0.35, validation_alias=AliasChoices("REWARD_PERFORMANCE_WEIGHT", "performance_weight"))
    resource_weight: float = Field(default=0.15, validation_alias=AliasChoices("REWARD_RESOURCE_WEIGHT", "resource_weight"))
    health_weight: float = Field(default=0.10, validation_alias=AliasChoices("REWARD_HEALTH_WEIGHT", "health_weight"))
    cost_weight: float = Field(default=0.10, validation_alias=AliasChoices("REWARD_COST_WEIGHT", "cost_weight"))
    
    # Additional reward parameters
    latency_weight: float = Field(default=10.0, validation_alias=AliasChoices("REWARD_LATENCY_WEIGHT", "latency_weight"))
    replica_cost: float = Field(default=0.1, validation_alias=AliasChoices("REWARD_REPLICA_COST", "replica_cost"))
    enable_improved_rewards: bool = Field(default=True, validation_alias=AliasChoices("ENABLE_IMPROVED_REWARDS", "enable_improved_rewards"))
    
    @model_validator(mode='after')
    def validate_weights_sum(self):
        # These weights are for configurable components only
        # Appropriateness (35%) and proactive (5%) are hardcoded in rewards.py
        # So these should sum to 60% (35% perf + 15% resource + 10% cost = 60%)
        total = (self.performance_weight + 
                self.resource_weight + 
                self.cost_weight)  # Removed health_weight as it's not used
        expected_total = 0.60  # 60% for configurable components
        if abs(total - expected_total) > 0.01:
            raise ValueError(f'Configurable reward weights must sum to {expected_total:.0%} (performance + resource + cost). Appropriateness (35%) and proactive (5%) are hardcoded.')
        return self
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    ) 