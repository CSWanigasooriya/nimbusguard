"""
AI and LLM configuration.
"""

from typing import Optional
from enum import Enum
from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings


class LogLevel(str, Enum):
    """Valid log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class AIConfig(BaseSettings):
    """AI and LLM configuration."""
    
    # LLM settings
    model: str = Field(default="gpt-4-turbo", env="AI_MODEL")
    temperature: float = Field(default=0.1, env="AI_TEMPERATURE")
    
    # Separate controls for LLM features
    enable_llm_validation: bool = Field(default=False, env="ENABLE_LLM_VALIDATION")
    enable_llm_rewards: bool = Field(default=True, env="ENABLE_LLM_REWARDS")
    
    @field_validator('enable_llm_validation', 'enable_llm_rewards', mode='before')
    @classmethod
    def validate_llm_booleans(cls, v):
        """Parse string boolean values correctly."""
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes', 'on')
        return bool(v)
    
    mcp_server_url: Optional[str] = Field(default=None, env="MCP_SERVER_URL")
    
    # Reasoning and logging
    enable_detailed_reasoning: bool = Field(default=True, env="ENABLE_DETAILED_REASONING")
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        if v < 0.0 or v > 2.0:
            raise ValueError('Temperature must be between 0.0 and 2.0') 
        return v
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    ) 