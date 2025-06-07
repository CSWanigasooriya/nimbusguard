"""
Configuration Management for NimbusGuard LangGraph Operator

This module handles loading and managing configuration for all agents,
including OpenAI API keys, environment variables, and agent-specific settings.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from string import Template

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


class ConfigManager:
    """
    Manages configuration for NimbusGuard LangGraph Operator.
    
    Handles:
    - Loading YAML configuration files
    - Environment variable substitution  
    - OpenAI API key management
    - Agent-specific configuration access
    - Validation of required settings
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config: Dict[str, Any] = {}
        self._load_config()
        self._setup_openai()
        
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        current_dir = Path(__file__).parent
        return str(current_dir / "agent_config.yaml")
    
    def _load_config(self) -> None:
        """Load configuration from YAML file with environment variable substitution."""
        try:
            with open(self.config_path, 'r') as file:
                config_template = file.read()
            
            # Substitute environment variables
            template = Template(config_template)
            config_str = template.safe_substitute(os.environ)
            
            # Parse YAML
            self.config = yaml.safe_load(config_str)
            
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")
    
    def _setup_openai(self) -> None:
        """Setup OpenAI configuration and validate API key."""
        openai_config = self.config.get("openai", {})
        
        # Get API key from config or environment
        api_key = openai_config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or configure in agent_config.yaml"
            )
        
        # Set environment variables for langchain
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Set optional OpenAI configurations
        if openai_config.get("base_url"):
            os.environ["OPENAI_BASE_URL"] = openai_config["base_url"]
            
        if openai_config.get("organization"):
            os.environ["OPENAI_ORG_ID"] = openai_config["organization"]
        
        logger.info("OpenAI configuration initialized")
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific agent.
        
        Args:
            agent_name: Name of the agent (supervisor, decision_agent, etc.)
            
        Returns:
            Dictionary containing agent configuration
            
        Raises:
            ConfigurationError: If agent configuration not found
        """
        agents_config = self.config.get("agents", {})
        agent_config = agents_config.get(agent_name)
        
        if not agent_config:
            raise ConfigurationError(f"Configuration not found for agent: {agent_name}")
        
        # Add global OpenAI settings to agent config
        openai_config = self.config.get("openai", {})
        agent_config["openai"] = openai_config
        
        return agent_config
    
    def get_system_prompt(self, agent_name: str) -> str:
        """
        Get system prompt for a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            System prompt string
            
        Raises:
            ConfigurationError: If prompt not found
        """
        agent_config = self.get_agent_config(agent_name)
        prompt = agent_config.get("system_prompt")
        
        if not prompt:
            raise ConfigurationError(f"System prompt not found for agent: {agent_name}")
        
        return prompt.strip()
    
    def get_scaling_config(self) -> Dict[str, Any]:
        """Get scaling configuration."""
        return self.config.get("scaling", {})
    
    def get_mcp_config(self) -> Dict[str, Any]:
        """Get MCP integration configuration."""
        return self.config.get("mcp_integration", {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get("logging", {})
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment configuration."""
        return self.config.get("environment", {})
    
    def validate_configuration(self) -> None:
        """
        Validate that all required configuration is present.
        
        Raises:
            ConfigurationError: If required configuration is missing
        """
        required_agents = ["supervisor", "state_observer", "decision_agent", "action_executor", "reward_calculator"]
        
        for agent in required_agents:
            agent_config = self.get_agent_config(agent)
            
            # Check required fields
            required_fields = ["model", "temperature", "max_tokens", "timeout", "system_prompt"]
            for field in required_fields:
                if field not in agent_config:
                    raise ConfigurationError(f"Missing required field '{field}' for agent '{agent}'")
        
        # Validate OpenAI configuration
        if not os.getenv("OPENAI_API_KEY"):
            raise ConfigurationError("OpenAI API key not configured")
        
        logger.info("Configuration validation passed")
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self._load_config()
        self._setup_openai()
        logger.info("Configuration reloaded")


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get global configuration manager instance.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
        _config_manager.validate_configuration()
    
    return _config_manager


def get_agent_config(agent_name: str) -> Dict[str, Any]:
    """
    Convenience function to get agent configuration.
    
    Args:
        agent_name: Name of the agent
        
    Returns:
        Agent configuration dictionary
    """
    return get_config_manager().get_agent_config(agent_name)


def get_system_prompt(agent_name: str) -> str:
    """
    Convenience function to get agent system prompt.
    
    Args:
        agent_name: Name of the agent
        
    Returns:
        System prompt string
    """
    return get_config_manager().get_system_prompt(agent_name)


# Setup logging based on configuration when module is imported
def _setup_logging():
    """Setup logging based on configuration."""
    try:
        config_manager = get_config_manager()
        logging_config = config_manager.get_logging_config()
        
        level = logging_config.get("level", "INFO")
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info(f"Logging configured at level: {level}")
        
    except Exception as e:
        # Fallback logging setup
        logging.basicConfig(level=logging.INFO)
        logger.warning(f"Could not setup logging from config: {e}")


# Initialize logging when module is imported
_setup_logging() 