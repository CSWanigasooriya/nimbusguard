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
    """Root endpoint with service dashboard"""
    import psutil
    import os
    
    # Get system information
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    cpu_percent = process.cpu_percent()
    
    # Get scaling consumer status
    consumer_status = "running" if scaling_event_consumer._running else "stopped"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>NimbusGuard Consumer Workload</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                background: rgba(255,255,255,0.1); 
                padding: 30px; 
                border-radius: 15px; 
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            }}
            h1 {{ 
                color: #fff; 
                text-align: center; 
                margin-bottom: 30px; 
                font-size: 2.5em;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}
            .status-grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                gap: 20px; 
                margin-bottom: 30px; 
            }}
            .status-card {{ 
                background: rgba(255,255,255,0.15); 
                padding: 20px; 
                border-radius: 10px; 
                border: 1px solid rgba(255,255,255,0.2);
            }}
            .status-card h3 {{ 
                margin-top: 0; 
                color: #fff; 
                border-bottom: 2px solid rgba(255,255,255,0.3);
                padding-bottom: 10px;
            }}
            .metric {{ 
                display: flex; 
                justify-content: space-between; 
                margin: 10px 0; 
                padding: 5px 0;
            }}
            .metric-value {{ 
                font-weight: bold; 
                color: #4CAF50; 
            }}
            .endpoints {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                gap: 15px; 
                margin-top: 30px; 
            }}
            .endpoint-link {{ 
                display: block; 
                padding: 15px; 
                background: rgba(255,255,255,0.2); 
                color: white; 
                text-decoration: none; 
                border-radius: 8px; 
                text-align: center; 
                transition: all 0.3s ease;
                border: 1px solid rgba(255,255,255,0.3);
            }}
            .endpoint-link:hover {{ 
                background: rgba(255,255,255,0.3); 
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }}
            .status-indicator {{ 
                display: inline-block; 
                width: 12px; 
                height: 12px; 
                border-radius: 50%; 
                margin-right: 8px; 
            }}
            .status-running {{ background-color: #4CAF50; }}
            .status-stopped {{ background-color: #f44336; }}
            .refresh-btn {{
                background: rgba(255,255,255,0.2);
                color: white;
                border: 1px solid rgba(255,255,255,0.3);
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                margin: 10px 0;
            }}
            .refresh-btn:hover {{
                background: rgba(255,255,255,0.3);
            }}
        </style>
        <script>
            function refreshPage() {{
                location.reload();
            }}
            // Auto-refresh every 30 seconds
            setTimeout(refreshPage, 30000);
        </script>
    </head>
    <body>
        <div class="container">
            <h1>🚀 NimbusGuard Consumer Workload</h1>
            
            <button class="refresh-btn" onclick="refreshPage()">🔄 Refresh Status</button>
            
            <div class="status-grid">
                <div class="status-card">
                    <h3>📊 Service Status</h3>
                    <div class="metric">
                        <span>Service:</span>
                        <span class="metric-value">
                            <span class="status-indicator status-running"></span>Running
                        </span>
                    </div>
                    <div class="metric">
                        <span>Version:</span>
                        <span class="metric-value">0.2.0</span>
                    </div>
                    <div class="metric">
                        <span>DQN Metrics:</span>
                        <span class="metric-value">✅ Enabled</span>
                    </div>
                    <div class="metric">
                        <span>Kafka Consumer:</span>
                        <span class="metric-value">
                            <span class="status-indicator status-{'running' if consumer_status == 'running' else 'stopped'}"></span>{consumer_status.title()}
                        </span>
                    </div>
                </div>
                
                <div class="status-card">
                    <h3>💻 System Metrics</h3>
                    <div class="metric">
                        <span>CPU Usage:</span>
                        <span class="metric-value">{cpu_percent:.1f}%</span>
                    </div>
                    <div class="metric">
                        <span>Memory (RSS):</span>
                        <span class="metric-value">{memory_info.rss / 1024 / 1024:.1f} MB</span>
                    </div>
                    <div class="metric">
                        <span>Process ID:</span>
                        <span class="metric-value">{os.getpid()}</span>
                    </div>
                    <div class="metric">
                        <span>CPU Cores:</span>
                        <span class="metric-value">{os.cpu_count()}</span>
                    </div>
                </div>
                
                <div class="status-card">
                    <h3>🔧 Features</h3>
                    <div class="metric">
                        <span>CPU Workload:</span>
                        <span class="metric-value">✅ Available</span>
                    </div>
                    <div class="metric">
                        <span>Memory Workload:</span>
                        <span class="metric-value">✅ Available</span>
                    </div>
                    <div class="metric">
                        <span>Event Processing:</span>
                        <span class="metric-value">✅ Available</span>
                    </div>
                    <div class="metric">
                        <span>Prometheus Metrics:</span>
                        <span class="metric-value">✅ Available</span>
                    </div>
                </div>
            </div>
            
            <h2>🔗 Available Endpoints</h2>
            <div class="endpoints">
                <a href="/docs" class="endpoint-link">
                    📚 API Documentation
                    <br><small>Interactive Swagger UI</small>
                </a>
                <a href="/metrics" class="endpoint-link">
                    📈 Prometheus Metrics
                    <br><small>System & DQN metrics</small>
                </a>
                <a href="/health" class="endpoint-link">
                    ❤️ Health Check
                    <br><small>Service health status</small>
                </a>
                <a href="/api/v1/workload/cpu/status" class="endpoint-link">
                    🖥️ CPU Workload Status
                    <br><small>Current CPU load status</small>
                </a>
                <a href="/api/v1/workload/memory/status" class="endpoint-link">
                    🧠 Memory Workload Status
                    <br><small>Current memory usage</small>
                </a>
                <a href="/api/v1/events/consumers/status" class="endpoint-link">
                    📨 Event Consumers
                    <br><small>Kafka consumer status</small>
                </a>
            </div>
            
            <div style="text-align: center; margin-top: 30px; opacity: 0.7;">
                <small>Auto-refreshes every 30 seconds • Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
            </div>
        </div>
    </body>
    </html>
    """
    
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_content)


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