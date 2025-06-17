import logging
import os
import threading
import time
from typing import Optional

import psutil
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from workload.cpu_workload import CPUWorkloadGenerator
from workload.memory_workload import MemoryWorkloadGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/workload", tags=["workload"])

# Initialize workload generators
cpu_generator = CPUWorkloadGenerator()
memory_generator = MemoryWorkloadGenerator()


class WorkloadRequest(BaseModel):
    intensity: float = Field(..., ge=0, le=100, description="Target usage percentage (0-100)")
    duration: int = Field(..., gt=0, description="Duration in seconds")


class WorkloadResponse(BaseModel):
    status: str
    current_usage: float
    target_usage: float
    remaining_time: Optional[float] = None


def generate_cpu_workload_thread(intensity: float, duration: int):
    """Generate CPU workload in a background thread"""
    try:
        cpu_generator.generate_load(intensity, duration)
    except Exception as e:
        logger.error(f"Error in CPU workload thread: {str(e)}")


def generate_memory_workload_thread(intensity: float, duration: int):
    """Generate memory workload in a background thread"""
    try:
        memory_generator.generate_load(intensity, duration)
    except Exception as e:
        logger.error(f"Error in memory workload thread: {str(e)}")


@router.post("/cpu", response_model=WorkloadResponse)
async def generate_cpu_workload(request: WorkloadRequest):
    """Generate CPU workload"""
    if cpu_generator.is_running:
        raise HTTPException(status_code=400, detail="CPU workload already running")

    # Start workload in background thread
    thread = threading.Thread(
        target=generate_cpu_workload_thread,
        args=(request.intensity, request.duration)
    )
    thread.daemon = True
    thread.start()

    # Brief wait to ensure workload thread has started
    import time
    for _ in range(10):  # Wait up to 1 second
        if cpu_generator.is_running:
            break
        time.sleep(0.1)

    return WorkloadResponse(
        status="running",
        current_usage=cpu_generator.current_usage,
        target_usage=cpu_generator.target_usage,
        remaining_time=cpu_generator.remaining_time
    )


@router.post("/memory", response_model=WorkloadResponse)
async def generate_memory_workload(request: WorkloadRequest):
    """Generate memory workload"""
    if memory_generator.is_running:
        raise HTTPException(status_code=400, detail="Memory workload already running")

    # Start workload in background thread
    thread = threading.Thread(
        target=generate_memory_workload_thread,
        args=(request.intensity, request.duration)
    )
    thread.daemon = True
    thread.start()

    # Brief wait to ensure workload thread has started
    for _ in range(10):  # Wait up to 1 second
        if memory_generator.is_running:
            break
        time.sleep(0.1)

    return WorkloadResponse(
        status="running",
        current_usage=memory_generator.current_usage,
        target_usage=memory_generator.target_usage,
        remaining_time=memory_generator.remaining_time
    )


@router.get("/cpu/status", response_model=WorkloadResponse)
async def get_cpu_status():
    """Get CPU workload status"""
    status = "running" if cpu_generator.is_running else "idle"
    target_usage = cpu_generator.target_usage if cpu_generator.is_running else 0.0
    remaining_time = cpu_generator.remaining_time if cpu_generator.is_running else 0.0

    # Get CPU usage of the current process
    process = psutil.Process(os.getpid())
    current_usage = process.cpu_percent(interval=1.0)

    # Normalize CPU usage by the number of CPU cores
    num_cores = os.cpu_count() or 1
    normalized_usage = current_usage / num_cores

    return WorkloadResponse(
        status=status,
        current_usage=normalized_usage,
        target_usage=target_usage,
        remaining_time=remaining_time
    )


@router.get("/memory/status", response_model=WorkloadResponse)
async def get_memory_status():
    """Get memory workload status"""
    # Force check remaining time and cleanup if needed
    remaining_time = memory_generator.remaining_time
    is_running = memory_generator.is_running

    logger.info("Memory status check", extra={
        "is_running": is_running,
        "remaining_time": remaining_time,
        "target_usage": memory_generator.target_usage,
        "current_usage": memory_generator.current_usage,
        "start_time": memory_generator._start_time,
        "duration": memory_generator.duration,
        "elapsed": time.time() - memory_generator._start_time if memory_generator._start_time > 0 else 0
    })

    # If not running or no remaining time, ensure cleanup
    if not is_running or remaining_time <= 0:
        logger.info("Stopping memory workload", extra={
            "reason": "no_remaining_time" if remaining_time <= 0 else "not_running"
        })
        memory_generator.stop()
        status = "idle"
        target_usage = 0.0
        remaining_time = 0.0
    else:
        status = "running"
        target_usage = memory_generator.target_usage

    return WorkloadResponse(
        status=status,
        current_usage=memory_generator.current_usage,
        target_usage=target_usage,
        remaining_time=remaining_time
    )


@router.post("/cpu/stop")
async def stop_cpu_workload():
    """Stop CPU workload"""
    cpu_generator.stop()
    return {"status": "stopped"}


@router.post("/memory/stop")
async def stop_memory_workload():
    """Stop memory workload"""
    memory_generator.stop()
    return {"status": "stopped"}
