"""
Service container for dependency injection.
Eliminates global state and provides clean dependency management.
"""

from typing import Optional, Any
from dataclasses import dataclass

from data.prometheus import PrometheusClient
from config import DQNAdapterConfig


@dataclass
class ServiceContainer:
    """Container for all services used by the DQN adapter."""
    
    # Configuration
    config: DQNAdapterConfig
    
    # Core clients
    prometheus_client: Optional[PrometheusClient] = None
    redis_client: Optional[Any] = None
    minio_client: Optional[Any] = None
    
    # AI/ML models
    scaler: Optional[Any] = None
    dqn_model: Optional[Any] = None
    validator_agent: Optional[Any] = None  # For LLM safety validation
    reward_agent: Optional[Any] = None     # For LLM reward calculation
    dqn_trainer: Optional[Any] = None
    
    # Utilities

    llm: Optional[Any] = None
    
    # DQN state management (eliminates global variables)
    current_epsilon: float = None  # Will be initialized from config
    decision_count: int = 0
    
    def __post_init__(self):
        """Initialize epsilon from configuration after dataclass creation."""
        if self.current_epsilon is None:
            self.current_epsilon = self.config.dqn.epsilon_start
    
    def get_epsilon(self) -> float:
        """Get current epsilon value for exploration."""
        return self.current_epsilon
    
    def update_epsilon(self, decay_rate: float, min_epsilon: float) -> float:
        """Update epsilon with decay and return new value."""
        previous_epsilon = self.current_epsilon
        self.current_epsilon = max(min_epsilon, self.current_epsilon * decay_rate)
        
        # Log epsilon decay for monitoring (every 10 decisions to avoid spam)
        if self.decision_count % 10 == 0:
            import logging
            logger = logging.getLogger("ServiceContainer")
            logger.info(f"EPSILON_DECAY: decision={self.decision_count} epsilon={previous_epsilon:.4f}→{self.current_epsilon:.4f} decay_rate={decay_rate}")
        
        return self.current_epsilon
    
    def increment_decision_count(self) -> int:
        """Increment and return decision count."""
        self.decision_count += 1
        return self.decision_count
    
    def get_health_status(self) -> dict:
        """Get comprehensive health status of all services."""
        health = {
            "overall_status": "healthy",
            "services": {},
            "warnings": [],
            "errors": []
        }
        
        # Check core services
        if self.prometheus_client is None:
            health["services"]["prometheus"] = "missing"
            health["errors"].append("Prometheus client not initialized")
        else:
            health["services"]["prometheus"] = "healthy"
            
        if self.scaler is None:
            health["services"]["scaler"] = "missing"
            health["errors"].append("Feature scaler not loaded")
        else:
            health["services"]["scaler"] = "healthy"
            
        if self.dqn_model is None:
            health["services"]["dqn_model"] = "missing"
            health["errors"].append("DQN model not loaded")
        else:
            health["services"]["dqn_model"] = "healthy"
            
        # Check optional services
        health["services"]["validator_agent"] = "healthy" if self.validator_agent else "disabled"
        health["services"]["redis"] = "healthy" if self.redis_client else "missing"
        health["services"]["minio"] = "healthy" if self.minio_client else "missing"
        # Evaluator removed - not required
            
        # Overall status assessment
        if health["errors"]:
            health["overall_status"] = "unhealthy"
        elif health["warnings"]:
            health["overall_status"] = "degraded"
            
        return health
    
    def is_ready(self) -> bool:
        """Check if core services are initialized."""
        return (
            self.prometheus_client is not None and
            self.scaler is not None and
            self.dqn_model is not None
        )
    
    def get_missing_services(self) -> list[str]:
        """Get list of missing core services."""
        missing = []
        if self.prometheus_client is None:
            missing.append("prometheus_client")
        if self.scaler is None:
            missing.append("scaler")
        if self.dqn_model is None:
            missing.append("dqn_model")
        return missing 