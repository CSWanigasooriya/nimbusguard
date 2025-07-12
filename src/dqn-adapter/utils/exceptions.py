"""
Custom exception classes for DQN Adapter.
Provides specific error types for better error handling and debugging.
"""

from typing import Any, Dict, Optional


class DQNAdapterError(Exception):
    """Base exception class for DQN Adapter."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
    
    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class ConfigurationError(DQNAdapterError):
    """Raised when configuration is invalid or missing."""
    pass


class PrometheusError(DQNAdapterError):
    """Raised when Prometheus operations fail."""
    pass


class DQNModelError(DQNAdapterError):
    """Raised when DQN model operations fail."""
    pass





class TrainingError(DQNAdapterError):
    """Raised when training operations fail."""
    pass


class ScalingError(DQNAdapterError):
    """Raised when scaling operations fail."""
    pass


class ValidationError(DQNAdapterError):
    """Raised when validation operations fail."""
    pass


class StorageError(DQNAdapterError):
    """Raised when storage operations (MinIO/Redis) fail."""
    pass


class FeatureExtractionError(DQNAdapterError):
    """Raised when feature extraction fails."""
    pass


class RewardCalculationError(DQNAdapterError):
    """Raised when reward calculation fails."""
    pass


class KubernetesError(DQNAdapterError):
    """Raised when Kubernetes operations fail."""
    pass


class HTTPServerError(DQNAdapterError):
    """Raised when HTTP server operations fail."""
    pass


class ExternalServiceError(DQNAdapterError):
    """Raised when external service calls fail."""
    pass


# Context manager for error handling
class ErrorContext:
    """Context manager for consistent error handling."""
    
    def __init__(self, operation: str, component: str = "DQN-Adapter"):
        self.operation = operation
        self.component = component
        self.context = {}
    
    def add_context(self, **kwargs) -> 'ErrorContext':
        """Add context information."""
        self.context.update(kwargs)
        return self
    
    def __enter__(self) -> 'ErrorContext':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return
        
        # Add operation context to the exception if it's our custom type
        if isinstance(exc_val, DQNAdapterError):
            exc_val.context.update({
                'operation': self.operation,
                'component': self.component,
                **self.context
            })
        
        # Re-raise the exception
        return False 