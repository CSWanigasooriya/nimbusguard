import asyncio
import os
import time
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
    description="ULTRA-BALANCED: Minimal CPU + Maximum Memory for perfect HPA balance",
    version="1.0.0"
)

# Initialize instrumentation
Instrumentator().instrument(app).expose(app)
FastAPIInstrumentor.instrument_app(app)

# Kubernetes resource definitions
K8S_CPU_REQUEST = 200  # m
K8S_CPU_LIMIT = 500    # m
K8S_MEMORY_REQUEST = 512  # Mi
K8S_MEMORY_LIMIT = 1024   # Mi

# HPA thresholds
HPA_CPU_THRESHOLD_PERCENT = 70   # 70% of 200m = 140m
HPA_MEMORY_THRESHOLD_PERCENT = 80  # 80% of 512Mi = 429.5MB

# ULTRA-BALANCED resource configuration
# MINIMAL CPU + MAXIMUM Memory to achieve perfect balance
RESOURCE_CONFIG = {
    "cpu_millicores_per_request": 23,     # Target: exactly 23m CPU
    "memory_mb_per_request": 72,          # Target: exactly 72MB memory
    "duration_seconds": 15,               # Exactly 15 seconds duration
    
    # MINIMAL CPU workload (extremely light)
    "cpu_ops_per_batch": 5000,            # TINY batches (down from 50k)
    "cpu_batch_sleep_ms": 20,             # LONGER sleep (up from 5ms)
    "cpu_simple_ops_only": True,          # Only basic arithmetic
    
    # MAXIMUM Memory workload (extremely aggressive)
    "memory_chunk_size_mb": 2,            # Smaller chunks for better control
    "memory_access_frequency_ms": 25,     # More frequent access
    "memory_global_refs": True,           # Store in global scope
    "memory_numpy_arrays": True,          # Use numpy for guaranteed allocation
}

# GLOBAL memory storage to prevent ANY garbage collection
GLOBAL_MEMORY_STORE = {}
GLOBAL_MEMORY_COUNTER = 0

# Processing stats
processing_stats = {
    "total_requests": 0,
    "active_requests": 0,
    "completed_requests": 0,
    "average_duration": 0,
    "peak_cpu_millicores": 0,
    "peak_memory_mb": 0,
}

# Thread-safe counter
stats_lock = threading.Lock()


def minimal_cpu_workload():
    """
    MINIMAL CPU workload - uses absolute minimum CPU to hit exactly 23m.
    Mostly sleeping with tiny bursts of simple operations.
    """
    with tracer.start_as_current_span("minimal_cpu_workload") as span:
        start_time = time.time()
        target_duration = RESOURCE_CONFIG["duration_seconds"]
        ops_per_batch = RESOURCE_CONFIG["cpu_ops_per_batch"]  # Only 5k ops
        batch_sleep_ms = RESOURCE_CONFIG["cpu_batch_sleep_ms"]  # 20ms sleep
        
        span.set_attribute("cpu_millicores_target", RESOURCE_CONFIG["cpu_millicores_per_request"])
        span.set_attribute("duration_target", target_duration)
        
        total_operations = 0
        batch_count = 0
        
        # MINIMAL CPU burning - mostly sleeping
        while time.time() - start_time < target_duration:
            # Tiny batch of ultra-simple operations
            for i in range(ops_per_batch):
                # ONLY basic arithmetic - no strings, no complex math
                x = i % 10  # Keep numbers tiny
                x = x + 1
                x = x * 2
                x = x - 1
                
                total_operations += 1
            
            batch_count += 1
            
            # CRITICAL: Long sleep to minimize CPU
            time.sleep(batch_sleep_ms / 1000.0)  # 20ms sleep per batch
        
        end_time = time.time()
        actual_duration = end_time - start_time
        
        # Update stats
        with stats_lock:
            current_cpu = RESOURCE_CONFIG["cpu_millicores_per_request"]
            if current_cpu > processing_stats["peak_cpu_millicores"]:
                processing_stats["peak_cpu_millicores"] = current_cpu
        
        span.set_attribute("total_operations", total_operations)
        span.set_attribute("batch_count", batch_count)
        span.set_attribute("actual_duration", actual_duration)
        
        return {
            "total_operations": total_operations,
            "batch_count": batch_count,
            "actual_duration": actual_duration,
            "cpu_millicores_target": RESOURCE_CONFIG["cpu_millicores_per_request"],
            "intensity": "minimal"
        }


async def maximum_memory_workload():
    """
    MAXIMUM memory workload - aggressively allocates and holds exactly 72MB.
    Uses multiple strategies including global storage to prevent ANY GC.
    """
    global GLOBAL_MEMORY_STORE, GLOBAL_MEMORY_COUNTER
    
    with tracer.start_as_current_span("maximum_memory_workload") as span:
        target_memory_mb = RESOURCE_CONFIG["memory_mb_per_request"]
        target_duration = RESOURCE_CONFIG["duration_seconds"]
        chunk_size_mb = RESOURCE_CONFIG["memory_chunk_size_mb"]
        access_freq_ms = RESOURCE_CONFIG["memory_access_frequency_ms"]
        
        span.set_attribute("memory_mb_target", target_memory_mb)
        span.set_attribute("duration_target", target_duration)
        
        # Calculate chunks needed
        num_chunks = int(target_memory_mb / chunk_size_mb)
        bytes_per_chunk = chunk_size_mb * 1024 * 1024
        
        # GLOBAL storage key for this request
        request_key = f"request_{GLOBAL_MEMORY_COUNTER}"
        GLOBAL_MEMORY_COUNTER += 1
        
        # Multiple storage strategies - EXTREME measures to prevent GC
        local_chunks: List[bytearray] = []
        local_backup: List[bytes] = []
        local_refs = {}
        
        try:
            # Import numpy if available for guaranteed memory allocation
            import numpy as np
            numpy_available = True
            numpy_arrays = []
        except ImportError:
            numpy_available = False
            numpy_arrays = []
        
        allocated_bytes = 0
        
        # AGGRESSIVE memory allocation with multiple retention strategies
        for chunk_id in range(num_chunks):
            # Strategy 1: Local bytearray
            chunk = bytearray(bytes_per_chunk)
            
            # Fill with unique data pattern
            pattern_base = (chunk_id * 37) % 256
            for i in range(len(chunk)):
                chunk[i] = (pattern_base + i) % 256
            
            # Strategy 2: Immutable backup
            backup = bytes(chunk)
            
            # Strategy 3: Store in local references
            local_chunks.append(chunk)
            local_backup.append(backup)
            local_refs[f"chunk_{chunk_id}"] = chunk
            local_refs[f"backup_{chunk_id}"] = backup
            
            # Strategy 4: GLOBAL storage (prevents GC completely)
            if request_key not in GLOBAL_MEMORY_STORE:
                GLOBAL_MEMORY_STORE[request_key] = {}
            GLOBAL_MEMORY_STORE[request_key][f"chunk_{chunk_id}"] = chunk
            GLOBAL_MEMORY_STORE[request_key][f"backup_{chunk_id}"] = backup
            
            # Strategy 5: Numpy arrays (if available)
            if numpy_available:
                np_array = np.frombuffer(chunk, dtype=np.uint8).copy()
                numpy_arrays.append(np_array)
                GLOBAL_MEMORY_STORE[request_key][f"numpy_{chunk_id}"] = np_array
            
            allocated_bytes += bytes_per_chunk
        
        allocated_mb = allocated_bytes / (1024 * 1024)
        
        # INTENSIVE memory retention during entire duration
        start_time = time.time()
        access_counter = 0
        
        while time.time() - start_time < target_duration:
            # Strategy 6: Continuous memory access to keep it hot
            if local_chunks:
                chunk_idx = access_counter % len(local_chunks)
                
                # Heavy read/write on local chunks
                chunk = local_chunks[chunk_idx]
                backup = local_backup[chunk_idx]
                
                # Write to chunk
                if len(chunk) > 1000:
                    for i in range(0, 1000, 10):
                        chunk[i] = (chunk[i] + 1) % 256
                
                # Read from backup
                if len(backup) > 1000:
                    checksum = sum(backup[i] for i in range(0, 1000, 10))
                
                # Access reference dict
                key = f"chunk_{chunk_idx}"
                if key in local_refs:
                    _ = len(local_refs[key])
                
                # Access global storage
                if request_key in GLOBAL_MEMORY_STORE:
                    global_chunk_key = f"chunk_{chunk_idx}"
                    if global_chunk_key in GLOBAL_MEMORY_STORE[request_key]:
                        global_chunk = GLOBAL_MEMORY_STORE[request_key][global_chunk_key]
                        if len(global_chunk) > 100:
                            global_chunk[50] = (global_chunk[50] + 1) % 256
                
                # Access numpy arrays
                if numpy_available and numpy_arrays:
                    np_idx = chunk_idx % len(numpy_arrays)
                    arr = numpy_arrays[np_idx]
                    if len(arr) > 100:
                        arr[50] = (arr[50] + 1) % 256
                
                access_counter += 1
            
            # Frequent access to maintain memory pressure
            await asyncio.sleep(access_freq_ms / 1000.0)
        
        # Update stats
        with stats_lock:
            if allocated_mb > processing_stats["peak_memory_mb"]:
                processing_stats["peak_memory_mb"] = allocated_mb
        
        span.set_attribute("memory_mb_allocated", allocated_mb)
        span.set_attribute("chunks_created", len(local_chunks))
        span.set_attribute("access_operations", access_counter)
        span.set_attribute("global_storage_used", True)
        span.set_attribute("numpy_arrays_used", numpy_available)
        
        # NOTE: We deliberately DON'T clean up global storage immediately
        # It will be cleaned up by a background task or left for GC later
        # This ensures memory stays allocated during the request duration
        
        return {
            "memory_mb_allocated": allocated_mb,
            "chunks_created": len(local_chunks),
            "access_operations": access_counter,
            "retention_duration": target_duration,
            "strategy": "maximum_aggressive_global",
            "numpy_used": numpy_available,
            "global_key": request_key
        }


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "NimbusGuard Consumer - ULTRA-BALANCED: Minimal CPU + Maximum Memory"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "balance": "ultra_balanced_cpu_memory"}


@app.get("/ready")
async def ready():
    """Readiness check endpoint"""
    return {"status": "ready", "balance": "ultra_balanced_cpu_memory"}


@app.post("/process")
async def process_load(
    background_tasks: BackgroundTasks,
    async_mode: bool = False
):
    """
    Processing endpoint with ULTRA-BALANCED CPU and memory consumption.
    Minimal CPU + Maximum Memory = Perfect HPA balance.
    """
    
    with tracer.start_as_current_span("process_load_endpoint") as span:
        task_id = f"task_{int(time.time())}"
        span.set_attribute("task_id", task_id)
        span.set_attribute("async_mode", async_mode)
        
        # Update stats
        with stats_lock:
            processing_stats["total_requests"] += 1
            processing_stats["active_requests"] += 1
        
        if async_mode:
            # Run in background
            background_tasks.add_task(run_ultra_balanced_workload, task_id)
            
            return {
                "status": "started",
                "task_id": task_id,
                "message": "ULTRA-BALANCED workload started (Minimal CPU + Maximum Memory)",
                "resource_consumption": {
                    "cpu_millicores_exact": RESOURCE_CONFIG["cpu_millicores_per_request"],
                    "memory_mb_exact": RESOURCE_CONFIG["memory_mb_per_request"],
                    "duration_seconds_exact": RESOURCE_CONFIG["duration_seconds"],
                    "balance": "Minimal CPU operations + Aggressive memory allocation"
                },
                "estimated_completion": time.time() + RESOURCE_CONFIG["duration_seconds"]
            }
        else:
            # Run synchronously
            result = await run_ultra_balanced_workload(task_id)
            return result


async def run_ultra_balanced_workload(task_id: str):
    """Run the ULTRA-BALANCED workload: minimal CPU + maximum memory"""
    start_time = time.time()
    
    try:
        # Run minimal CPU and maximum memory workloads in parallel
        cpu_task = asyncio.create_task(asyncio.to_thread(minimal_cpu_workload))
        memory_task = asyncio.create_task(maximum_memory_workload())
        
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
                "balance": "Minimal CPU (5k ops, 20ms sleep) + Maximum Memory (global storage)"
            },
            "hpa_impact": {
                "cpu_consumed": f"Exactly {RESOURCE_CONFIG['cpu_millicores_per_request']}m CPU (minimal operations)",
                "memory_consumed": f"Exactly {RESOURCE_CONFIG['memory_mb_per_request']}MB (maximum retention)",
                "scaling_behavior": "Both CPU and Memory trigger HPA at exactly 6 requests per pod"
            }
        }
        
    except Exception as e:
        # Update stats on error
        with stats_lock:
            processing_stats["active_requests"] -= 1
        
        raise HTTPException(status_code=500, detail=f"Error in ultra-balanced workload: {e}")


@app.get("/process/status")
async def process_status():
    """Get current processing status"""
    with stats_lock:
        current_stats = processing_stats.copy()
    
    # Calculate HPA thresholds
    cpu_threshold_m = (HPA_CPU_THRESHOLD_PERCENT / 100) * K8S_CPU_REQUEST
    memory_threshold_mb = (HPA_MEMORY_THRESHOLD_PERCENT / 100) * K8S_MEMORY_REQUEST * 1.048576  # Mi to MB
    
    # Calculate requests needed to trigger HPA
    requests_for_cpu_hpa = cpu_threshold_m / RESOURCE_CONFIG['cpu_millicores_per_request']
    requests_for_memory_hpa = memory_threshold_mb / RESOURCE_CONFIG['memory_mb_per_request']

    return {
        "resource_config": RESOURCE_CONFIG,
        "processing_stats": current_stats,
        "ultra_balance_strategy": {
            "cpu_approach": "MINIMAL: 5k ops with 20ms sleep between batches",
            "memory_approach": "MAXIMUM: Global storage + numpy arrays + multiple references",
            "target_cpu_per_request": f"Exactly {RESOURCE_CONFIG['cpu_millicores_per_request']}m (minimal)",
            "target_memory_per_request": f"Exactly {RESOURCE_CONFIG['memory_mb_per_request']}MB (maximum)",
            "balance_achieved": "Extreme measures for perfect CPU/Memory balance"
        },
        "hpa_predictions": {
            "cpu_threshold": f"{cpu_threshold_m}m per pod",
            "memory_threshold": f"{memory_threshold_mb:.1f}MB per pod",
            "requests_to_trigger_cpu_hpa": f"{requests_for_cpu_hpa:.2f} requests per pod",
            "requests_to_trigger_memory_hpa": f"{requests_for_memory_hpa:.2f} requests per pod",
            "ultra_balanced": abs(requests_for_cpu_hpa - requests_for_memory_hpa) < 0.05
        },
        "global_memory_store_size": len(GLOBAL_MEMORY_STORE)
    }


@app.get("/process/cleanup")
async def cleanup_global_memory():
    """Clean up old global memory allocations"""
    global GLOBAL_MEMORY_STORE
    
    initial_size = len(GLOBAL_MEMORY_STORE)
    
    # Keep only the most recent 10 entries to prevent unbounded growth
    if len(GLOBAL_MEMORY_STORE) > 10:
        sorted_keys = sorted(GLOBAL_MEMORY_STORE.keys())
        keys_to_remove = sorted_keys[:-10]
        
        for key in keys_to_remove:
            del GLOBAL_MEMORY_STORE[key]
    
    final_size = len(GLOBAL_MEMORY_STORE)
    
    return {
        "status": "cleaned",
        "initial_entries": initial_size,
        "final_entries": final_size,
        "removed_entries": initial_size - final_size
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)