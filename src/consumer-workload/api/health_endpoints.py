"""
Health check endpoints for Kubernetes probes
"""

from fastapi import APIRouter
from pydantic import BaseModel
import psutil
import os

router = APIRouter()

class HealthStatus(BaseModel):
    status: str
    service: str
    version: str = "1.0.0"
    uptime_seconds: float
    memory_usage_mb: float
    cpu_percent: float

# Track startup time
import time
startup_time = time.time()

@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "service": "nimbusguard-consumer-workload",
        "timestamp": time.time()
    }

@router.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe endpoint"""
    try:
        # Check if the application is ready to serve traffic
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        uptime = time.time() - startup_time
        
        return HealthStatus(
            status="ready",
            service="nimbusguard-consumer-workload",
            uptime_seconds=uptime,
            memory_usage_mb=memory_info.rss / 1024 / 1024,
            cpu_percent=process.cpu_percent()
        )
    except Exception as e:
        return {
            "status": "not_ready",
            "error": str(e)
        }

@router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe endpoint"""
    try:
        # Basic liveness check - ensure the process is responding
        return {
            "status": "alive",
            "service": "nimbusguard-consumer-workload",
            "pid": os.getpid(),
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "dead",
            "error": str(e)
        } 