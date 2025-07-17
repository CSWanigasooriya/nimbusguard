import asyncio
import os
import random
import time
import math
import threading
from typing import Dict, List

from fastapi import FastAPI, BackgroundTasks
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor
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
    description="A consumer service with fixed resource usage patterns",
    version="1.0.0"
)

# Initialize instrumentation
Instrumentator().instrument(app).expose(app)
FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()
URLLib3Instrumentor().instrument()

# Fixed resource configuration optimized for high-concurrency load testing
# Designed for 10-30 concurrent requests with multiple pods
RESOURCE_CONFIG = {
    "target_memory_mb": 80,   # Target ~80MB (allows 5-6 requests per pod before hitting limits)
    "target_cpu_cores": 0.04, # Target ~40m (allows 5-6 requests per pod before hitting limits)
    "ramp_up_duration": 10,   # 10 seconds to reach target
    "sustain_duration": 20,   # 20 seconds at target
    "ramp_down_duration": 5,  # 5 seconds to ramp down
}

# Processing stats
processing_stats = {
    "total_requests": 0,
    "active_requests": 0,
    "completed_requests": 0,
    "average_duration": 0,
    "peak_memory_mb": 0,
    "peak_cpu_usage": 0,
}

# Thread-safe counter for active requests
stats_lock = threading.Lock()


def fixed_cpu_workload():
    """
    Fixed CPU workload that gradually increases to target CPU usage
    Uses mathematical calculations to consume CPU predictably
    """
    with tracer.start_as_current_span("fixed_cpu_workload") as span:
        total_duration = (RESOURCE_CONFIG["ramp_up_duration"] + 
                         RESOURCE_CONFIG["sustain_duration"] + 
                         RESOURCE_CONFIG["ramp_down_duration"])
        
        span.set_attribute("total_duration", total_duration)
        span.set_attribute("target_cpu_cores", RESOURCE_CONFIG["target_cpu_cores"])
        
        start_time = time.time()
        calculations_done = 0
        
        while time.time() - start_time < total_duration:
            current_time = time.time() - start_time
            
            # Calculate target CPU intensity based on phase
            if current_time < RESOURCE_CONFIG["ramp_up_duration"]:
                # Ramp up phase
                progress = current_time / RESOURCE_CONFIG["ramp_up_duration"]
                cpu_intensity = progress * RESOURCE_CONFIG["target_cpu_cores"]
                phase = "ramp_up"
            elif current_time < RESOURCE_CONFIG["ramp_up_duration"] + RESOURCE_CONFIG["sustain_duration"]:
                # Sustain phase
                cpu_intensity = RESOURCE_CONFIG["target_cpu_cores"]
                phase = "sustain"
            else:
                # Ramp down phase
                ramp_down_start = RESOURCE_CONFIG["ramp_up_duration"] + RESOURCE_CONFIG["sustain_duration"]
                progress = (current_time - ramp_down_start) / RESOURCE_CONFIG["ramp_down_duration"]
                cpu_intensity = RESOURCE_CONFIG["target_cpu_cores"] * (1 - progress)
                phase = "ramp_down"
            
            span.set_attribute("current_phase", phase)
            span.set_attribute("cpu_intensity", cpu_intensity)
            
            # Adjust work based on CPU intensity
            work_cycles = int(cpu_intensity * 100000)  # Scale work to CPU target
            
            # Do mathematical work to consume CPU
            for _ in range(work_cycles):
                # Mix of different CPU operations
                result = math.sin(random.random()) * math.cos(random.random())
                result = math.sqrt(abs(result * 1000))
                result = math.log(result + 1) if result > 0 else 0
                calculations_done += 1
            
            # Brief pause between cycles (realistic for I/O or coordination)
            time.sleep(0.1)
        
        span.set_attribute("total_calculations", calculations_done)
        return calculations_done


def fixed_memory_workload():
    """
    Fixed memory workload that gradually increases to target memory usage
    Uses data structures that predictably consume memory
    """
    with tracer.start_as_current_span("fixed_memory_workload") as span:
        total_duration = (RESOURCE_CONFIG["ramp_up_duration"] + 
                         RESOURCE_CONFIG["sustain_duration"] + 
                         RESOURCE_CONFIG["ramp_down_duration"])
        
        span.set_attribute("total_duration", total_duration)
        span.set_attribute("target_memory_mb", RESOURCE_CONFIG["target_memory_mb"])
        
        # Memory storage - will be automatically GC'd when function exits
        memory_data: List[List[str]] = []
        start_time = time.time()
        
        # Calculate total memory needed upfront (more realistic)
        target_memory_mb = RESOURCE_CONFIG["target_memory_mb"]
        total_chunks_needed = target_memory_mb  # 1MB per chunk
        
        # Allocate memory in phases based on timing
        while time.time() - start_time < total_duration:
            current_time = time.time() - start_time
            
            # Calculate how much memory we should have allocated by now
            if current_time < RESOURCE_CONFIG["ramp_up_duration"]:
                # Ramp up phase
                progress = current_time / RESOURCE_CONFIG["ramp_up_duration"]
                target_chunks = int(progress * total_chunks_needed)
                phase = "ramp_up"
            elif current_time < RESOURCE_CONFIG["ramp_up_duration"] + RESOURCE_CONFIG["sustain_duration"]:
                # Sustain phase
                target_chunks = total_chunks_needed
                phase = "sustain"
            else:
                # Ramp down phase - but don't actively free memory (unrealistic)
                # Real apps don't monitor and free memory dynamically
                target_chunks = total_chunks_needed
                phase = "ramp_down"
            
            span.set_attribute("current_phase", phase)
            span.set_attribute("target_chunks", target_chunks)
            
            # Allocate memory if we need more (realistic behavior)
            while len(memory_data) < target_chunks:
                # Create 1MB chunks (realistic data processing)
                chunk = ['x' * 1024 for _ in range(1024)]  # 1MB of data
                memory_data.append(chunk)
            
            # Update peak memory tracking
            current_memory_mb = len(memory_data)
            with stats_lock:
                if current_memory_mb > processing_stats["peak_memory_mb"]:
                    processing_stats["peak_memory_mb"] = current_memory_mb
            
            # Process/work with the data (realistic pause)
            time.sleep(0.1)
        
        # Memory will be automatically garbage collected when function exits
        data_chunks = len(memory_data)
        estimated_mb = sum(len(chunk) for chunk in memory_data) / (1024 * 1024)
        
        span.set_attribute("final_data_chunks", data_chunks)
        span.set_attribute("estimated_final_mb", estimated_mb)
        
        return data_chunks


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to NimbusGuard Consumer with Fixed Resources!"}


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
    Process endpoint optimized for high-concurrency load testing (10-30+ requests)
    
    This endpoint will:
    - Gradually ramp up CPU usage to ~40m per request over 10 seconds
    - Gradually ramp up memory usage to ~80MB per request over 10 seconds  
    - Sustain usage for 20 seconds 
    - Ramp down over 5 seconds
    - Total duration: ~35 seconds
    
    High-Concurrency Load Testing:
    - 1-3 requests: Below HPA thresholds (no scaling)
    - 4-5 requests: Triggers HPA scaling (CPU > 140m, Memory > 410MB)
    - 5-6 requests per pod: Approaches pod limits (200m CPU, 512MB memory)
    - 10-30+ requests: Distributed across multiple pods via HPA
    - Each pod can handle ~5-6 concurrent requests before needing to scale
    """
    
    with tracer.start_as_current_span("process_load_endpoint") as span:
        task_id = f"task_{int(time.time())}_{random.randint(1000, 9999)}"
        span.set_attribute("task_id", task_id)
        span.set_attribute("async_mode", async_mode)
        
        # Update stats
        with stats_lock:
            processing_stats["total_requests"] += 1
            processing_stats["active_requests"] += 1
        
        total_duration = (RESOURCE_CONFIG["ramp_up_duration"] + 
                         RESOURCE_CONFIG["sustain_duration"] + 
                         RESOURCE_CONFIG["ramp_down_duration"])
        
        if async_mode:
            # Run in background
            background_tasks.add_task(run_fixed_workload, task_id)
            
            return {
                "status": "started",
                "task_id": task_id,
                "message": "Fixed resource workload started in background",
                "resource_pattern": {
                    "target_memory_mb": RESOURCE_CONFIG["target_memory_mb"],
                    "target_cpu_cores": RESOURCE_CONFIG["target_cpu_cores"],
                    "total_duration_seconds": total_duration,
                    "phases": {
                        "ramp_up": RESOURCE_CONFIG["ramp_up_duration"],
                        "sustain": RESOURCE_CONFIG["sustain_duration"],
                        "ramp_down": RESOURCE_CONFIG["ramp_down_duration"]
                    }
                },
                "estimated_completion": time.time() + total_duration
            }
        else:
            # Run synchronously
            result = await run_fixed_workload(task_id)
            return result


async def run_fixed_workload(task_id: str):
    """Run the fixed CPU and memory workload"""
    start_time = time.time()
    
    try:
        # Run CPU and memory workloads in parallel
        cpu_task = asyncio.create_task(asyncio.to_thread(fixed_cpu_workload))
        memory_task = asyncio.create_task(asyncio.to_thread(fixed_memory_workload))
        
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
                "cpu_calculations": cpu_result,
                "memory_chunks_created": memory_result,
                "target_memory_mb": RESOURCE_CONFIG["target_memory_mb"],
                "target_cpu_cores": RESOURCE_CONFIG["target_cpu_cores"]
            },
            "phases_completed": {
                "ramp_up": RESOURCE_CONFIG["ramp_up_duration"],
                "sustain": RESOURCE_CONFIG["sustain_duration"], 
                "ramp_down": RESOURCE_CONFIG["ramp_down_duration"]
            }
        }
        
    except Exception as e:
        # Update stats on error
        with stats_lock:
            processing_stats["active_requests"] -= 1
        
        raise e


@app.get("/process/status")
async def process_status():
    """Get current processing status"""
    with stats_lock:
        current_stats = processing_stats.copy()
    
    return {
        "resource_config": RESOURCE_CONFIG,
        "processing_stats": current_stats,
        "kubernetes_limits": {
            "memory_request": "512Mi",
            "memory_limit": "1Gi", 
            "cpu_request": "200m",
            "cpu_limit": "500m"
        },
        "resource_utilization": {
            "memory_target_vs_request": f"{RESOURCE_CONFIG['target_memory_mb']}MB / 512MB",
            "cpu_target_vs_request": f"{RESOURCE_CONFIG['target_cpu_cores']} / 0.2 cores",
            "memory_utilization_percent": round((RESOURCE_CONFIG["target_memory_mb"] / 512) * 100, 1),
            "cpu_utilization_percent": round((RESOURCE_CONFIG["target_cpu_cores"] / 0.2) * 100, 1),
            "hpa_scaling_behavior": {
                "cpu_threshold": "70% of 200m = 140m (0.14 cores)",
                "memory_threshold": "80% of 512Mi = ~410MB",
                "requests_to_trigger_hpa": {
                    "cpu_scaling": max(1, round(0.14 / RESOURCE_CONFIG['target_cpu_cores'])),
                    "memory_scaling": max(1, round(410 / RESOURCE_CONFIG['target_memory_mb']))
                },
                "requests_per_pod_limit": {
                    "cpu_limit": max(1, round(0.2 / RESOURCE_CONFIG['target_cpu_cores'])),
                    "memory_limit": max(1, round(512 / RESOURCE_CONFIG['target_memory_mb']))
                },
                "high_concurrency_scenario": {
                    "10_requests": "2 pods needed (5 requests per pod)",
                    "20_requests": "4 pods needed (5 requests per pod)",
                    "30_requests": "6 pods needed (5 requests per pod)"
                }
            }
        }
    }


@app.get("/process/config")
async def get_config():
    """Get current resource configuration"""
    return {
        "resource_config": RESOURCE_CONFIG,
        "description": {
            "fixed_pattern": "Each request follows a predictable resource usage pattern optimized for high-concurrency load testing",
            "ramp_up": "Gradually increases CPU and memory usage to small, predictable levels",
            "sustain": "Maintains consistent resource usage allowing multiple requests per pod", 
            "ramp_down": "Gradually decreases resource usage to test scale-down behavior",
            "auto_gc": "Python automatically garbage collects memory when functions exit",
            "high_concurrency": "Designed for 10-30+ concurrent requests distributed across multiple pods via HPA",
            "pod_capacity": "Each pod can handle ~5-6 concurrent requests before approaching resource limits"
        }
    }


@app.put("/process/config")
async def update_config(
    target_memory_mb: int = None,
    target_cpu_cores: float = None,
    ramp_up_duration: int = None,
    sustain_duration: int = None,
    ramp_down_duration: int = None
):
    """Update resource configuration (if needed for testing)"""
    changes = {}
    
    if target_memory_mb is not None:
        RESOURCE_CONFIG["target_memory_mb"] = target_memory_mb
        changes["target_memory_mb"] = RESOURCE_CONFIG["target_memory_mb"]
    
    if target_cpu_cores is not None:
        RESOURCE_CONFIG["target_cpu_cores"] = target_cpu_cores
        changes["target_cpu_cores"] = RESOURCE_CONFIG["target_cpu_cores"]
    
    if ramp_up_duration is not None:
        RESOURCE_CONFIG["ramp_up_duration"] = ramp_up_duration
        changes["ramp_up_duration"] = ramp_up_duration
    
    if sustain_duration is not None:
        RESOURCE_CONFIG["sustain_duration"] = sustain_duration
        changes["sustain_duration"] = sustain_duration
    
    if ramp_down_duration is not None:
        RESOURCE_CONFIG["ramp_down_duration"] = ramp_down_duration
        changes["ramp_down_duration"] = ramp_down_duration
    
    return {
        "status": "updated",
        "changes": changes,
        "current_config": RESOURCE_CONFIG
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)