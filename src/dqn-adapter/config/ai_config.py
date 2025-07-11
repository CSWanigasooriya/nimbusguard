"""
AI and LLM configuration.
"""

from typing import Optional
from enum import Enum
from pydantic import ConfigDict, Field, field_validator, AliasChoices
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
    model: str = Field(default="gpt-4-turbo", validation_alias=AliasChoices("AI_MODEL", "ai_model"))
    temperature: float = Field(default=0.1, validation_alias=AliasChoices("AI_TEMPERATURE", "ai_temperature"))
    enable_llm_validation: bool = Field(default=False, validation_alias=AliasChoices("ENABLE_LLM_VALIDATION", "enable_llm_validation"))
    mcp_server_url: Optional[str] = Field(default=None, validation_alias=AliasChoices("MCP_SERVER_URL", "mcp_server_url"))
    
    # Reasoning and logging
    enable_detailed_reasoning: bool = Field(default=True, validation_alias=AliasChoices("ENABLE_DETAILED_REASONING", "enable_detailed_reasoning"))
    reasoning_log_level: LogLevel = Field(default=LogLevel.INFO, validation_alias=AliasChoices("REASONING_LOG_LEVEL", "reasoning_log_level"))
    
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