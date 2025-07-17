import asyncio
import os
import random
import time
import math
import threading
from typing import Dict, List

from fastapi import FastAPI, BackgroundTasks, HTTPException
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_fastapi_instrumentator import Instrumentator


# Initialize OpenTelemetry tracing
def setup_tracing():
    resource = Resource(attributes={
        SERVICE_NAME: "nimbusguard-consumer",
        "service.version": "1.0.0",
        "service.namespace": "nimbusguard"
    })

    trace.set_tracer_provider(TracerProvider(resource=resource))
    tracer = trace.get_tracer(__name__)

    otlp_exporter = OTLPSpanExporter(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://alloy:4318/v1/traces"),
        headers={}
    )

    span_processor = BatchSpanProcessor(otlp_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)

    return tracer


tracer = setup_tracing()

app = FastAPI(
    title="NimbusGuard Consumer",
    description="A consumer service with INTENSE resource usage patterns for HPA testing",
    version="1.0.0"
)

# Initialize instrumentation
Instrumentator().instrument(app).expose(app)
FastAPIInstrumentor.instrument_app(app)

# INTENSIVE resource configuration for reliable HPA scaling
RESOURCE_CONFIG = {
    "target_cpu_utilization": 0.8,  # 80% CPU utilization to guarantee scaling per request
    "total_duration": 15,           # Duration for which CPU and newly allocated memory are active per request
    "cpu_burst_cycles": 1000000,    # 1M operations per burst for CPU workload
    "memory_increment_mb_per_request": 50, # Each request adds this much memory cumulatively
    "memory_chunk_size_mb": 10      # Internal chunk size for memory allocation
}

# Global list to hold allocated memory chunks for cumulative memory usage
# This will cause the service's memory footprint to grow over time
allocated_memory_chunks: List[bytearray] = []
global_memory_lock = threading.Lock() # Lock for accessing allocated_memory_chunks

# Processing stats
processing_stats = {
    "total_requests": 0,
    "active_requests": 0,
    "completed_requests": 0,
    "average_duration": 0,
    "peak_cpu_percent": 0,
    "current_allocated_memory_mb": 0, # Tracks cumulative memory
    "peak_memory_mb": 0, # Peak observed cumulative memory
}

# Thread-safe counter for active requests
stats_lock = threading.Lock()


def intensive_cpu_workload():
    """
    INTENSIVE CPU workload that actually consumes CPU consistently for the configured duration.
    No sleeping - pure CPU burning for sustained resource pressure.
    Each concurrent request will add to the overall CPU utilization.
    """
    with tracer.start_as_current_span("intensive_cpu_workload") as span:
        start_time = time.time()
        total_operations = 0
        
        span.set_attribute("target_cpu_utilization", RESOURCE_CONFIG["target_cpu_utilization"])
        span.set_attribute("total_duration", RESOURCE_CONFIG["total_duration"])
        
        # Continuous CPU burning for the entire duration
        while time.time() - start_time < RESOURCE_CONFIG["total_duration"]:
            # Burst of intensive operations
            for _ in range(RESOURCE_CONFIG["cpu_burst_cycles"]):
                # Mix of CPU-intensive operations (NO SLEEP!)
                x = random.random() * 1000
                result = math.sin(x) * math.cos(x)
                result = math.sqrt(abs(result))
                result = math.log(result + 1) if result > 0 else 0
                result = math.pow(result, 2)
                
                # Additional CPU work - string operations
                temp_str = str(result) * 100
                temp_str = temp_str.upper().lower()
                
                # Numeric operations
                for i in range(10):
                    temp_num = result * i + math.pi
                    temp_num = temp_num / (i + 1)
                
                total_operations += 1
                
                # Brief check if we should continue (every 10k operations)
                if total_operations % 10000 == 0:
                    if time.time() - start_time >= RESOURCE_CONFIG["total_duration"]:
                        break
        
        end_time = time.time()
        actual_duration = end_time - start_time
        
        # Update peak CPU stats (estimated)
        estimated_cpu_usage = min(100, RESOURCE_CONFIG["target_cpu_utilization"] * 100)
        with stats_lock:
            if estimated_cpu_usage > processing_stats["peak_cpu_percent"]:
                processing_stats["peak_cpu_percent"] = estimated_cpu_usage
        
        span.set_attribute("total_operations", total_operations)
        span.set_attribute("actual_duration", actual_duration)
        span.set_attribute("estimated_cpu_usage", estimated_cpu_usage)
        span.set_attribute("operations_per_second", total_operations / actual_duration)
        
        return {
            "total_operations": total_operations,
            "actual_duration": actual_duration,
            "estimated_cpu_usage": estimated_cpu_usage,
            "operations_per_second": total_operations / actual_duration
        }


def intensive_memory_workload():
    """
    INTENSIVE memory workload that cumulatively allocates memory.
    Each call to this function will add `memory_increment_mb_per_request` to the global memory pool.
    The allocated memory is kept active for the `total_duration` of the request.
    """
    with tracer.start_as_current_span("intensive_memory_workload") as span:
        start_time = time.time()
        
        memory_to_add_mb = RESOURCE_CONFIG["memory_increment_mb_per_request"]
        chunk_size_mb = RESOURCE_CONFIG["memory_chunk_size_mb"]
        chunks_to_add = memory_to_add_mb // chunk_size_mb
        
        newly_allocated_chunks = []

        with global_memory_lock:
            for i in range(chunks_to_add):
                # Create chunk of actual data (not just references)
                chunk = bytearray(chunk_size_mb * 1024 * 1024)
                
                # Fill with actual data to prevent optimization
                for j in range(0, len(chunk), 1024):
                    chunk[j:j+1024] = b'x' * 1024
                
                allocated_memory_chunks.append(chunk)
                newly_allocated_chunks.append(chunk) # Keep track of newly added for this request
                
                # Update current allocated memory stats
                processing_stats["current_allocated_memory_mb"] += chunk_size_mb
                if processing_stats["current_allocated_memory_mb"] > processing_stats["peak_memory_mb"]:
                    processing_stats["peak_memory_mb"] = processing_stats["current_allocated_memory_mb"]

        span.set_attribute("memory_added_this_request_mb", memory_to_add_mb)
        span.set_attribute("total_allocated_memory_mb", processing_stats["current_allocated_memory_mb"])
        
        # Keep ALL allocated memory (global_memory_chunks) actively used for the full duration
        # This ensures the HPA sees the cumulative memory usage
        while time.time() - start_time < RESOURCE_CONFIG["total_duration"]:
            # Actively access a random subset of the global memory chunks to prevent swapping/optimization
            with global_memory_lock:
                if allocated_memory_chunks:
                    # Access a few random chunks to keep them hot
                    for _ in range(min(5, len(allocated_memory_chunks))): # Access up to 5 chunks
                        chunk_to_access = random.choice(allocated_memory_chunks)
                        # Modify some bytes to keep memory active
                        for j in range(0, len(chunk_to_access), 10000):
                            chunk_to_access[j] = (chunk_to_access[j] + 1) % 256
            
            # Brief pause to prevent 100% CPU usage in memory thread
            time.sleep(0.01)
        
        end_time = time.time()
        actual_duration = end_time - start_time
        
        span.set_attribute("actual_duration", actual_duration)
        
        return {
            "memory_added_this_request_mb": memory_to_add_mb,
            "total_allocated_memory_mb": processing_stats["current_allocated_memory_mb"],
            "chunks_added_this_request": chunks_to_add,
            "actual_duration": actual_duration,
            "estimated_memory_usage": f"{memory_to_add_mb}MB added, total {processing_stats['current_allocated_memory_mb']}MB"
        }


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to NimbusGuard Consumer with INTENSIVE Resource Usage!"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    """Readiness check endpoint"""
    return {"status": "ready"}


@app.post("/process")
async def process_load(
    background_tasks: BackgroundTasks,
    async_mode: bool = False
):
    """
    INTENSIVE processing endpoint that GUARANTEES resource usage for HPA scaling.
    
    This endpoint will:
    - BURN CPU at 80% utilization for `total_duration` seconds per request.
    - CUMULATIVELY ALLOCATE `memory_increment_mb_per_request` memory per request.
      This memory is retained by the service, causing its overall footprint to grow.
    - Use sustained resource pressure to trigger HPA scaling.
    """
    
    with tracer.start_as_current_span("process_load_endpoint") as span:
        task_id = f"task_{int(time.time())}_{random.randint(1000, 9999)}"
        span.set_attribute("task_id", task_id)
        span.set_attribute("async_mode", async_mode)
        
        # Update stats
        with stats_lock:
            processing_stats["total_requests"] += 1
            processing_stats["active_requests"] += 1
        
        if async_mode:
            # Run in background
            background_tasks.add_task(run_intensive_workload, task_id)
            
            return {
                "status": "started",
                "task_id": task_id,
                "message": "INTENSIVE resource workload started in background",
                "resource_pattern": {
                    "cpu_utilization_per_request": f"{RESOURCE_CONFIG['target_cpu_utilization']*100}% for {RESOURCE_CONFIG['total_duration']}s",
                    "memory_increment_per_request_mb": RESOURCE_CONFIG["memory_increment_mb_per_request"],
                    "total_duration_seconds": RESOURCE_CONFIG["total_duration"],
                    "warning": "This will consume significant CPU and cumulatively increase memory!"
                },
                "estimated_completion": time.time() + RESOURCE_CONFIG["total_duration"]
            }
        else:
            # Run synchronously
            result = await run_intensive_workload(task_id)
            return result


async def run_intensive_workload(task_id: str):
    """Run the INTENSIVE CPU and cumulative memory workload"""
    start_time = time.time()
    
    try:
        # Run CPU and memory workloads in parallel - both intensive!
        cpu_task = asyncio.create_task(asyncio.to_thread(intensive_cpu_workload))
        memory_task = asyncio.create_task(asyncio.to_thread(intensive_memory_workload))
        
        # Wait for both tasks to complete
        cpu_result, memory_result = await asyncio.gather(cpu_task, memory_task)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Update stats
        with stats_lock:
            processing_stats["active_requests"] -= 1
            processing_stats["completed_requests"] += 1
            
            # Update average duration
            total_completed = processing_stats["completed_requests"]
            current_avg = processing_stats["average_duration"]
            processing_stats["average_duration"] = (
                (current_avg * (total_completed - 1) + processing_time) / total_completed
            )
        
        return {
            "status": "completed",
            "task_id": task_id,
            "processing_time_seconds": round(processing_time, 2),
            "resource_usage": {
                "cpu_workload": cpu_result,
                "memory_workload": memory_result,
                "config": RESOURCE_CONFIG,
                "current_total_memory_allocated_mb": processing_stats["current_allocated_memory_mb"]
            },
            "hpa_impact": {
                "cpu_burned": f"{RESOURCE_CONFIG['target_cpu_utilization']*100}% for {processing_time:.1f}s",
                "memory_allocated_cumulatively": f"{memory_result['memory_added_this_request_mb']}MB added, total {processing_stats['current_allocated_memory_mb']}MB",
                "scaling_trigger": "This load should trigger HPA scaling!"
            }
        }
        
    except Exception as e:
        # Update stats on error
        with stats_lock:
            processing_stats["active_requests"] -= 1
        
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {e}")


@app.get("/process/status")
async def process_status():
    """Get current processing status"""
    with stats_lock:
        current_stats = processing_stats.copy()
    
    return {
        "resource_config": RESOURCE_CONFIG,
        "processing_stats": current_stats,
        "system_monitoring": {
            "note": "Real-time system monitoring disabled (no psutil)",
            "workload_impact": "CPU and memory workloads still run intensively"
        },
        "kubernetes_limits": {
            "memory_request": "512Mi",
            "memory_limit": "1Gi",
            "cpu_request": "200m",
            "cpu_limit": "500m"
        },
        "hpa_scaling_expectations": {
            "cpu_threshold": "70% of 200m = 140m",
            "memory_threshold": "80% of 512Mi = ~410MB",
            "intensive_workload_impact": {
                "single_request_cpu": f"{RESOURCE_CONFIG['target_cpu_utilization']*100}% CPU ({RESOURCE_CONFIG['target_cpu_utilization']*200:.0f}m) for {RESOURCE_CONFIG['total_duration']}s",
                "single_request_memory_increment": f"{RESOURCE_CONFIG['memory_increment_mb_per_request']}MB cumulatively added",
                "scaling_behavior": "Each request adds to cumulative memory and contributes to CPU load, designed to guarantee HPA triggering"
            }
        }
    }

@app.post("/process/clear-memory")
async def clear_memory():
    """Endpoint to clear all cumulatively allocated memory."""
    with global_memory_lock:
        global allocated_memory_chunks
        allocated_memory_chunks = []
        processing_stats["current_allocated_memory_mb"] = 0
    return {"status": "success", "message": "All allocated memory cleared."}


@app.get("/process/config")
async def get_config():
    """Get current resource configuration"""
    return {
        "resource_config": RESOURCE_CONFIG,
        "description": {
            "intensive_pattern": "Each request burns CPU and cumulatively increases memory for guaranteed HPA scaling",
            "no_sleeping": "CPU workload runs continuously without sleep statements",
            "sustained_pressure": "Memory allocation is kept active for the full duration and accumulates across requests",
            "hpa_optimized": "Resource usage designed to trigger HPA scaling reliably",
            "scaling_guarantee": "Each request adds to the overall load, ensuring HPA scaling kicks in as thresholds are met"
        }
    }


@app.put("/process/config")
async def update_config(
    target_cpu_utilization: float = None,
    total_duration: int = None,
    cpu_burst_cycles: int = None,
    memory_increment_mb_per_request: int = None, # Changed parameter name
    memory_chunk_size_mb: int = None # New parameter
):
    """Update resource configuration"""
    changes = {}
    
    if target_cpu_utilization is not None:
        RESOURCE_CONFIG["target_cpu_utilization"] = target_cpu_utilization
        changes["target_cpu_utilization"] = target_cpu_utilization
    
    if total_duration is not None:
        RESOURCE_CONFIG["total_duration"] = total_duration
        changes["total_duration"] = total_duration
    
    if cpu_burst_cycles is not None:
        RESOURCE_CONFIG["cpu_burst_cycles"] = cpu_burst_cycles
        changes["cpu_burst_cycles"] = cpu_burst_cycles
    
    if memory_increment_mb_per_request is not None: # Changed parameter name
        RESOURCE_CONFIG["memory_increment_mb_per_request"] = memory_increment_mb_per_request
        changes["memory_increment_mb_per_request"] = memory_increment_mb_per_request
    
    if memory_chunk_size_mb is not None: # New parameter update
        RESOURCE_CONFIG["memory_chunk_size_mb"] = memory_chunk_size_mb
        changes["memory_chunk_size_mb"] = memory_chunk_size_mb

    return {
        "status": "updated",
        "changes": changes,
        "current_config": RESOURCE_CONFIG,
        "warning": "New configuration will take effect on next request"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
