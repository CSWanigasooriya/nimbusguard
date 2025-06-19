import json
import logging
import os
from datetime import datetime

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.event_endpoints import router as event_router
from api.health_endpoints import router as health_router
from api.metrics_endpoints import router as metrics_router, metrics_middleware
from api.workload_endpoints import router as workload_router
from consumers.scaling_event_consumer import ScalingEventConsumer
from tracing import setup_tracing

# Setup tracing first
setup_tracing()


class JSONFormatter(logging.Formatter):
    """Enhanced JSON formatter for structured logging with trace correlation"""

    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "service": "nimbusguard-consumer-workload"
        }

        # Add trace correlation if available
        try:
            from opentelemetry import trace
            span = trace.get_current_span()
            if span and span.is_recording():
                span_context = span.get_span_context()
                log_record["trace_id"] = f"{span_context.trace_id:032x}"
                log_record["span_id"] = f"{span_context.span_id:016x}"
        except Exception:
            pass

        # Add extra fields if they exist
        if hasattr(record, 'extra'):
            log_record.update(record.extra)

        # Add exception info if it exists
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)

        # Add business context for log-based metrics
        message_lower = record.getMessage().lower()
        if any(keyword in message_lower for keyword in ['error', 'failed', 'exception']):
            log_record['log_level_numeric'] = 40  # ERROR
            log_record['is_error'] = True
        elif any(keyword in message_lower for keyword in ['warning', 'warn']):
            log_record['log_level_numeric'] = 30  # WARNING
            log_record['is_warning'] = True
        else:
            log_record['log_level_numeric'] = 20  # INFO
            log_record['is_error'] = False

        # Add business metrics context
        if any(keyword in message_lower for keyword in ['transaction', 'business', 'user', 'order']):
            log_record['is_business_event'] = True
            if 'success' in message_lower or 'completed' in message_lower:
                log_record['business_status'] = 'success'
            elif 'failed' in message_lower or 'error' in message_lower:
                log_record['business_status'] = 'error'

        return json.dumps(log_record)


# Configure enhanced logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()

# Enhanced log file configuration
log_file_path = os.getenv("LOG_FILE_PATH", "/tmp/app.log")
try:
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    file_handler = logging.FileHandler(log_file_path)

    # Create formatters and add it to handlers
    json_formatter = JSONFormatter()
    file_handler.setFormatter(json_formatter)
    console_handler.setFormatter(json_formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Enhanced logging configured", extra={
        "log_file": log_file_path,
        "json_format": True,
        "trace_correlation": True
    })
except (OSError, PermissionError) as e:
    json_formatter = JSONFormatter()
    console_handler.setFormatter(json_formatter)
    logger.addHandler(console_handler)
    logger.warning(f"Could not set up file logging ({e}), using console only")

# Enhanced startup logging for DQN observability
logger.info("Starting NimbusGuard Consumer Workload with DQN metrics", extra={
    "version": "0.2.0",
    "environment": os.getenv("ENVIRONMENT", "development"),
    "cpu_count": os.cpu_count(),
    "memory_total": os.getenv("MEMORY_TOTAL", "unknown"),
    "otel_endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
    "service_name": os.getenv("OTEL_SERVICE_NAME", "nimbusguard-consumer-workload"),
    "dqn_metrics_enabled": True,
    "trace_correlation_enabled": True
})

app = FastAPI(
    title="NimbusGuard Consumer Workload",
    description="Enhanced workload generator with DQN observability features",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add metrics middleware for automatic HTTP instrumentation
app.middleware("http")(metrics_middleware)

# Include routers
app.include_router(workload_router)
app.include_router(event_router)
app.include_router(metrics_router)
app.include_router(health_router)

scaling_event_consumer = ScalingEventConsumer()


@app.on_event("startup")
async def startup_event():
    """Enhanced startup with DQN metrics initialization"""
    try:
        # Start Kafka consumer
        scaling_event_consumer.start()

        # Log successful startup for business metrics
        logger.info("Application startup completed successfully", extra={
            "event_type": "startup",
            "is_business_event": True,
            "business_status": "success",
            "kafka_consumer_started": True,
            "metrics_endpoint_enabled": True,
            "health_checks_enabled": True
        })

    except Exception as e:
        logger.error("Application startup failed", extra={
            "event_type": "startup",
            "is_business_event": True,
            "business_status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e)
        })
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Enhanced shutdown with proper cleanup logging"""
    try:
        scaling_event_consumer.stop()

        logger.info("Application shutdown completed successfully", extra={
            "event_type": "shutdown",
            "is_business_event": True,
            "business_status": "success",
            "kafka_consumer_stopped": True
        })

    except Exception as e:
        logger.error("Application shutdown encountered errors", extra={
            "event_type": "shutdown",
            "is_business_event": True,
            "business_status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e)
        })


# Enhanced health endpoint with DQN context
@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "nimbusguard-consumer-workload",
        "version": "0.2.0",
        "status": "running",
        "dqn_metrics_enabled": True,
        "endpoints": {
            "metrics": "/metrics",
            "health": "/health",
            "docs": "/docs",
            "workload": "/api/v1/workload"
        }
    }


# Error handler for better error metrics
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for enhanced error logging"""

    logger.error("Unhandled exception occurred", extra={
        "event_type": "unhandled_exception",
        "is_error": True,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "request_method": request.method,
        "request_path": str(request.url.path),
        "request_query": str(request.url.query) if request.url.query else None
    })

    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "status_code": 500
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,  # Disable reload for production metrics stability
        log_level="info",
        access_log=True
    )