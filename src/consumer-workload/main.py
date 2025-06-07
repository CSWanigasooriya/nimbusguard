from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import json
import os
from typing import Dict, Any
from datetime import datetime

from api.workload_endpoints import router as workload_router
from api.event_endpoints import router as event_router
from consumers.scaling_event_consumer import ScalingEventConsumer

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if they exist
        if hasattr(record, 'extra'):
            log_record.update(record.extra)
            
        # Add exception info if it exists
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_record)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler("app.log")
console_handler = logging.StreamHandler()

# Create formatters and add it to handlers
json_formatter = JSONFormatter()
file_handler.setFormatter(json_formatter)
console_handler.setFormatter(json_formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Log startup information
logger.info("Starting NimbusGuard Consumer Workload", extra={
    "version": "0.1.0",
    "environment": os.getenv("ENVIRONMENT", "development"),
    "cpu_count": os.cpu_count(),
    "memory_total": os.getenv("MEMORY_TOTAL", "unknown")
})

app = FastAPI(
    title="NimbusGuard Consumer Workload",
    description="A workload generator and consumer for NimbusGuard",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(workload_router)
app.include_router(event_router)

scaling_event_consumer = ScalingEventConsumer()

@app.on_event("startup")
def start_kafka_consumer():
    scaling_event_consumer.start()

@app.on_event("shutdown")
def stop_kafka_consumer():
    scaling_event_consumer.stop()

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    logger.info("Health check requested", extra={"endpoint": "/health"})
    return {"status": "healthy"}

@app.get("/ready")
async def readiness_check() -> Dict[str, str]:
    """Readiness check endpoint"""
    return {"status": "ready"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    ) 