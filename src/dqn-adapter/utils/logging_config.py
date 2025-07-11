"""
Structured logging configuration for DQN Adapter.
Provides consistent logging patterns and structured output.
"""

import json
import logging
import logging.config
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from config import LogLevel


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as simplified JSON."""
        
        # Simplified log structure with only essential fields
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add exception information if present (this is critical for debugging)
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        return json.dumps(log_entry, default=str)


class ComponentLogger:
    """Component-specific logger with consistent patterns."""
    
    def __init__(self, component: str):
        self.component = component
        self.logger = logging.getLogger(f"DQN-Adapter.{component}")
    
    def _log_with_context(self, level: int, message: str, **context) -> None:
        """Log message with additional context."""
        extra = {
            "component": self.component,
            **context
        }
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **context) -> None:
        """Log debug message."""
        self._log_with_context(logging.DEBUG, message, **context)
    
    def info(self, message: str, **context) -> None:
        """Log info message."""
        self._log_with_context(logging.INFO, message, **context)
    
    def warning(self, message: str, **context) -> None:
        """Log warning message."""
        self._log_with_context(logging.WARNING, message, **context)
    
    def error(self, message: str, **context) -> None:
        """Log error message."""
        self._log_with_context(logging.ERROR, message, **context)
    
    def critical(self, message: str, **context) -> None:
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, message, **context)
    
    def operation_start(self, operation: str, **context) -> None:
        """Log operation start."""
        self.info(f"OPERATION_START: {operation}", operation=operation, **context)
    
    def operation_end(self, operation: str, duration: Optional[float] = None, **context) -> None:
        """Log operation end."""
        context_data = {"operation": operation, **context}
        if duration is not None:
            context_data["duration_seconds"] = duration
        self.info(f"OPERATION_END: {operation}", **context_data)
    
    def operation_error(self, operation: str, error: Exception, **context) -> None:
        """Log operation error."""
        self.error(
            f"OPERATION_ERROR: {operation} failed",
            operation=operation,
            error_type=type(error).__name__,
            error_message=str(error),
            **context
        )
    
    def metric_update(self, metric_name: str, value: Any, **context) -> None:
        """Log metric update."""
        self.debug(
            f"METRIC_UPDATE: {metric_name}={value}",
            metric_name=metric_name,
            metric_value=value,
            **context
        )
    
    def decision_log(self, decision: str, confidence: float, **context) -> None:
        """Log decision with confidence."""
        self.info(
            f"DECISION: {decision}",
            decision=decision,
            confidence=confidence,
            **context
        )
    
    def performance_log(self, operation: str, duration: float, **context) -> None:
        """Log performance metrics."""
        self.info(
            f"PERFORMANCE: {operation} took {duration:.3f}s",
            operation=operation,
            duration_seconds=duration,
            **context
        )


class DQNAdapterLogger:
    """Central logging manager for DQN Adapter."""
    
    def __init__(self, log_level: LogLevel = LogLevel.INFO, use_structured: bool = True):
        self.log_level = log_level
        self.use_structured = use_structured
        self.component_loggers: Dict[str, ComponentLogger] = {}
        self.setup_logging()
    
    def setup_logging(self) -> None:
        """Configure logging for the entire application."""
        
        # Clear existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        if self.use_structured:
            # Use structured JSON formatter
            formatter = StructuredFormatter()
        else:
            # Use simple formatter for development
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger.setLevel(getattr(logging, self.log_level.value))
        root_logger.addHandler(console_handler)
        
        # Configure specific loggers
        self._configure_external_loggers()
    
    def _configure_external_loggers(self) -> None:
        """Configure external library loggers."""
        
        # Suppress verbose external library logs
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("aiohttp").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        
        # Configure kopf logger
        logging.getLogger("kopf").setLevel(logging.INFO)
        
        # Configure torch logger
        logging.getLogger("torch").setLevel(logging.WARNING)
        
        # Configure transformers logger
        logging.getLogger("transformers").setLevel(logging.WARNING)
    
    def get_logger(self, component: str) -> ComponentLogger:
        """Get component-specific logger."""
        if component not in self.component_loggers:
            self.component_loggers[component] = ComponentLogger(component)
        return self.component_loggers[component]
    
    def configure_for_production(self) -> None:
        """Configure logging for production environment."""
        self.use_structured = True
        self.log_level = LogLevel.INFO
        self.setup_logging()
    
    def configure_for_development(self) -> None:
        """Configure logging for development environment."""
        self.use_structured = False
        self.log_level = LogLevel.DEBUG
        self.setup_logging()


# Global logger instance
_logger_manager: Optional[DQNAdapterLogger] = None


def setup_logging(log_level: LogLevel = LogLevel.INFO, use_structured: bool = True) -> None:
    """Setup global logging configuration."""
    global _logger_manager
    _logger_manager = DQNAdapterLogger(log_level, use_structured)


def get_logger(component: str) -> ComponentLogger:
    """Get component-specific logger."""
    if _logger_manager is None:
        setup_logging()
    return _logger_manager.get_logger(component)


def configure_for_production() -> None:
    """Configure logging for production."""
    if _logger_manager is None:
        setup_logging()
    _logger_manager.configure_for_production()


def configure_for_development() -> None:
    """Configure logging for development."""
    if _logger_manager is None:
        setup_logging()
    _logger_manager.configure_for_development()


# Context manager for operation logging
class OperationLogger:
    """Context manager for logging operations with timing."""
    
    def __init__(self, logger: ComponentLogger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self) -> 'OperationLogger':
        self.start_time = datetime.utcnow()
        self.logger.operation_start(self.operation, **self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.utcnow() - self.start_time).total_seconds()
            
            if exc_type is None:
                self.logger.operation_end(self.operation, duration, **self.context)
            else:
                self.logger.operation_error(self.operation, exc_val, **self.context)
        
        # Don't suppress exceptions
        return False 