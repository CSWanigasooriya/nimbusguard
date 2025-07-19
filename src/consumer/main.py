import asyncio
import os
import time
import threading
import hashlib
import math
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
        "service.version": "3.1.0", # Version updated
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
    description="Deterministic HPA Consumer - Predictable CPU and Memory consumption",
    version="3.1.0" # Version updated
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
HPA_MEMORY_THRESHOLD_PERCENT = 80  # 80% of 512Mi = 410Mi (approx)

# DETERMINISTIC resource configuration - ULTRA-MINIMAL CPU INTENSITY
RESOURCE_CONFIG = {
    "cpu_operations_per_request": 200,     # Extremely reduced from 1,000
    "memory_mb_per_request": 40,           # Kept the same
    "duration_seconds": 15,                # Kept the same
    
    # Deterministic CPU workload parameters - ULTRA-MINIMAL
    "fibonacci_calculations": 1,           # Already minimal
    "matrix_multiplications": 1,           # Already minimal
    
    # Deterministic memory parameters
    "memory_chunk_size_kb": 1024,          # 1MB chunks
    "memory_pattern_seed": 42,             # Deterministic pattern seed
    "memory_access_cycles": 1000,          # Fixed number of memory access cycles
}

# Global memory storage to prevent garbage collection
ACTIVE_ALLOCATIONS = {}
ALLOCATION_COUNTER = 0
ALLOCATION_LOCK = threading.Lock()

# Processing stats
processing_stats = {
    "total_requests": 0,
    "active_requests": 0,
    "completed_requests": 0,
    "average_duration": 0,
    "total_cpu_operations": 0,
    "total_memory_allocated_mb": 0,
}

stats_lock = threading.Lock()


def deterministic_cpu_work(request_id: int) -> Dict:
    """
    Performs EXACTLY the same CPU work every time.
    Uses deterministic algorithms with fixed operation counts.
    """
    start_time = time.time()
    operations_completed = 0
    

    

    
    # Phase 3: Fibonacci calculations (deterministic recursion)
    fibonacci_results = []
    for i in range(RESOURCE_CONFIG["fibonacci_calculations"]):
        n = 20 + (i % 10)  # Calculate fibonacci for numbers 20-29
        fib_result = calculate_fibonacci_iterative(n)
        fibonacci_results.append(fib_result)
        operations_completed += fib_result  # Count iterations as operations
    

    
    # Phase 5: Small matrix operations (deterministic)
    matrix_results = []
    for i in range(RESOURCE_CONFIG["matrix_multiplications"]):
        # Create deterministic 3x3 matrices
        seed = request_id * 1000 + i
        matrix_a = create_deterministic_matrix(seed)
        matrix_b = create_deterministic_matrix(seed + 1)
        result_matrix = multiply_3x3_matrices(matrix_a, matrix_b)
        matrix_results.append(result_matrix)
        operations_completed += 27  # 3x3 matrix multiplication operations
    
    end_time = time.time()
    actual_duration = end_time - start_time
    
    return {
        "operations_completed": operations_completed,
        "fibonacci_calculations": len(fibonacci_results),
        "matrix_multiplications": len(matrix_results),
        "cpu_duration_seconds": actual_duration,
        "deterministic": True,
        "expected_operations": RESOURCE_CONFIG["cpu_operations_per_request"]
    }


def calculate_fibonacci_iterative(n: int) -> int:
    """Calculate fibonacci number iteratively (deterministic)"""
    if n <= 1:
        return n
    
    a, b = 0, 1
    operations = 0
    for _ in range(2, n + 1):
        a, b = b, a + b
        operations += 1
    return operations





def create_deterministic_matrix(seed: int) -> List[List[float]]:
    """Create a deterministic 3x3 matrix based on seed"""
    matrix = []
    for i in range(3):
        row = []
        for j in range(3):
            # Deterministic value generation
            value = ((seed + i * 10 + j) % 100) / 10.0
            row.append(value)
        matrix.append(row)
    return matrix


def multiply_3x3_matrices(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Multiply two 3x3 matrices (deterministic)"""
    result = [[0.0 for _ in range(3)] for _ in range(3)]
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] += a[i][k] * b[k][j]
    
    return result


def deterministic_memory_allocation(size_mb: int, allocation_id: str, request_id: int) -> Dict:
    """
    Allocates EXACTLY the specified memory with deterministic patterns.
    Uses fixed patterns to ensure consistent memory usage.
    """
    global ACTIVE_ALLOCATIONS
    
    size_bytes = size_mb * 1024 * 1024
    chunk_size_bytes = RESOURCE_CONFIG["memory_chunk_size_kb"] * 1024  # 1MB chunks
    num_chunks = size_mb  # 1 chunk per MB
    
    # Deterministic pattern generation
    seed = RESOURCE_CONFIG["memory_pattern_seed"] + request_id
    
    chunks = []
    total_allocated = 0
    
    for chunk_idx in range(num_chunks):
        # Create chunk with deterministic pattern
        chunk = bytearray(chunk_size_bytes)
        
        # Fill with deterministic pattern based on chunk index and seed
        pattern_base = (seed + chunk_idx * 17) % 256
        
        for byte_idx in range(0, chunk_size_bytes, 4096):  # Every 4KB page
            # Deterministic pattern: use chunk_idx and byte_idx for reproducible data
            pattern_value = (pattern_base + byte_idx // 4096) % 256
            end_idx = min(byte_idx + 4096, chunk_size_bytes)
            
            # Fill the 4KB page with pattern
            for i in range(byte_idx, end_idx):
                chunk[i] = (pattern_value + (i % 256)) % 256
        
        chunks.append(chunk)
        total_allocated += len(chunk)
    
    # Store in global scope to prevent garbage collection
    with ALLOCATION_LOCK:
        ACTIVE_ALLOCATIONS[allocation_id] = {
            'chunks': chunks,
            'size_mb': size_mb,
            'allocated_time': time.time(),
            'request_id': request_id,
            'pattern_seed': seed,
            'total_bytes': total_allocated
        }
    
    return {
        "allocated_mb": total_allocated / (1024 * 1024),
        "chunks_count": len(chunks),
        "pattern_seed": seed,
        "total_bytes": total_allocated,
        "deterministic": True
    }


def deterministic_memory_access(allocation_id: str, duration_seconds: float) -> Dict:
    """
    Performs EXACTLY the same memory access pattern every time.
    Keeps memory active with deterministic read/write operations.
    """
    access_operations = 0
    cycles_completed = 0
    
    start_time = time.time()
    
    with ALLOCATION_LOCK:
        if allocation_id not in ACTIVE_ALLOCATIONS:
            return {"error": "allocation_not_found"}
        
        allocation = ACTIVE_ALLOCATIONS[allocation_id]
        chunks = allocation['chunks']
        seed = allocation['pattern_seed']
    
    # Deterministic access pattern
    while time.time() - start_time < duration_seconds:
        for cycle in range(min(RESOURCE_CONFIG["memory_access_cycles"], 100)):  # Limit per iteration
            for chunk_idx, chunk in enumerate(chunks):
                if len(chunk) > 4096:  # Ensure chunk is large enough
                    # Deterministic access positions
                    pos1 = ((seed + cycle * 13 + chunk_idx * 7) % (len(chunk) - 1000))
                    pos2 = pos1 + 500
                    pos3 = pos1 + 1000
                    
                    # Deterministic read operations
                    val1 = chunk[pos1]
                    val2 = chunk[pos2]
                    val3 = chunk[pos3]
                    
                    # Deterministic write operations
                    chunk[pos1] = (val1 + cycle) % 256
                    chunk[pos2] = (val2 + cycle * 2) % 256
                    chunk[pos3] = (val3 + cycle * 3) % 256
                    
                    access_operations += 6  # 3 reads + 3 writes
                
                cycles_completed += 1
                
                # Check time periodically to avoid overrunning
                if cycles_completed % 1000 == 0:
                    if time.time() - start_time >= duration_seconds:
                        break
            
            if time.time() - start_time >= duration_seconds:
                break
        
        # Small sleep to prevent excessive CPU usage from memory operations
        time.sleep(0.01)
    
    actual_duration = time.time() - start_time
    
    return {
        "access_operations": access_operations,
        "cycles_completed": cycles_completed,
        "actual_duration": actual_duration,
        "deterministic": True
    }


async def run_deterministic_workload(task_id: str) -> Dict:
    """Run deterministic CPU and memory workload"""
    global ALLOCATION_COUNTER
    
    start_time = time.time()
    request_id = int(task_id.split('_')[-1]) if '_' in task_id else hash(task_id) % 10000
    
    with ALLOCATION_LOCK:
        ALLOCATION_COUNTER += 1
        allocation_id = f"alloc_{ALLOCATION_COUNTER}_{task_id}"
    
    try:
        # Phase 1: Deterministic CPU work
        cpu_result = await asyncio.get_event_loop().run_in_executor(
            None, 
            deterministic_cpu_work,
            request_id
        )
        
        # Phase 2: Deterministic memory allocation
        memory_alloc_result = await asyncio.get_event_loop().run_in_executor(
            None,
            deterministic_memory_allocation,
            RESOURCE_CONFIG["memory_mb_per_request"],
            allocation_id,
            request_id
        )
        
        # Phase 3: Deterministic memory access for specified duration
        memory_access_result = await asyncio.get_event_loop().run_in_executor(
            None,
            deterministic_memory_access,
            allocation_id,
            RESOURCE_CONFIG["duration_seconds"]
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Update stats
        with stats_lock:
            processing_stats["active_requests"] -= 1
            processing_stats["completed_requests"] += 1
            processing_stats["total_cpu_operations"] += cpu_result.get("operations_completed", 0)
            processing_stats["total_memory_allocated_mb"] += memory_alloc_result.get("allocated_mb", 0)
            
            # Update average duration
            total_completed = processing_stats["completed_requests"]
            current_avg = processing_stats["average_duration"]
            if total_completed > 0:
                processing_stats["average_duration"] = (
                    (current_avg * (total_completed - 1) + processing_time) / total_completed
                )
        
        return {
            "status": "completed",
            "task_id": task_id,
            "request_id": request_id,
            "processing_time_seconds": round(processing_time, 3),
            "deterministic_workload": {
                "cpu_result": cpu_result,
                "memory_allocation": memory_alloc_result,
                "memory_access": memory_access_result,
                "guaranteed_resources": {
                    "cpu_operations": RESOURCE_CONFIG["cpu_operations_per_request"],
                    "memory_mb": RESOURCE_CONFIG["memory_mb_per_request"],
                    "duration_seconds": RESOURCE_CONFIG["duration_seconds"]
                }
            },
            "allocation_id": allocation_id,
            "deterministic": True
        }
        
    except Exception as e:
        with stats_lock:
            processing_stats["active_requests"] -= 1
        raise HTTPException(status_code=500, detail=f"Error in deterministic workload: {e}")


def cleanup_old_allocations():
    """Clean up allocations that have exceeded their duration"""
    global ACTIVE_ALLOCATIONS
    
    current_time = time.time()
    to_remove = []
    
    with ALLOCATION_LOCK:
        for allocation_id, allocation in ACTIVE_ALLOCATIONS.items():
            if current_time - allocation['allocated_time'] > RESOURCE_CONFIG["duration_seconds"] + 5:
                to_remove.append(allocation_id)
        
        for allocation_id in to_remove:
            del ACTIVE_ALLOCATIONS[allocation_id]
    
    return len(to_remove)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "NimbusGuard Consumer - Deterministic HPA Testing", 
        "version": "3.1.0",
        "deterministic": True
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    with ALLOCATION_LOCK:
        active_allocations = len(ACTIVE_ALLOCATIONS)
        total_allocated_mb = sum(alloc['size_mb'] for alloc in ACTIVE_ALLOCATIONS.values())
    
    return {
        "status": "healthy",
        "active_allocations": active_allocations,
        "total_allocated_mb": total_allocated_mb,
        "deterministic": True
    }


@app.get("/ready")
async def ready():
    """Readiness check endpoint"""
    return {"status": "ready", "deterministic": True}


@app.post("/process")
async def process_load(background_tasks: BackgroundTasks, async_mode: bool = False):
    """
    Processing endpoint that consumes a predictable amount of CPU and Memory.
    
    DETERMINISTIC GUARANTEES (Ultra-Minimal CPU Intensity):
    - CPU: ~1m per request -> Need 140 concurrent requests to trigger HPA
    - Memory: 40MB per request -> Need 11 concurrent requests to trigger HPA
    """
    
    with tracer.start_as_current_span("deterministic_process_endpoint") as span:
        task_id = f"task_{int(time.time() * 1000000)}"  # Microsecond precision
        span.set_attribute("task_id", task_id)
        span.set_attribute("async_mode", async_mode)
        span.set_attribute("deterministic", True)
        
        # Update stats
        with stats_lock:
            processing_stats["total_requests"] += 1
            processing_stats["active_requests"] += 1
        
        # Clean up old allocations
        cleanup_old_allocations()
        
        if async_mode:
            # Run in background
            background_tasks.add_task(run_deterministic_workload, task_id)
            
            return {
                "status": "started",
                "task_id": task_id,
                "message": "Deterministic workload started",
                "guaranteed_resource_consumption": {
                    "cpu_operations": RESOURCE_CONFIG["cpu_operations_per_request"],
                    "memory_mb": RESOURCE_CONFIG["memory_mb_per_request"],
                    "duration_seconds": RESOURCE_CONFIG["duration_seconds"]
                },
                "deterministic": True,
                "estimated_completion": time.time() + RESOURCE_CONFIG["duration_seconds"]
            }
        else:
            # Run synchronously
            result = await run_deterministic_workload(task_id)
            return result


@app.get("/process/status")
async def process_status():
    """Get current processing status with deterministic guarantees"""
    with stats_lock:
        current_stats = processing_stats.copy()
    
    # Calculate HPA thresholds
    cpu_threshold_m = (HPA_CPU_THRESHOLD_PERCENT / 100) * K8S_CPU_REQUEST
    memory_threshold_mb = (HPA_MEMORY_THRESHOLD_PERCENT / 100) * K8S_MEMORY_REQUEST
    
    # Estimate CPU usage (rough approximation)
    estimated_cpu_per_request = 1  # millicores (ultra-minimal)
    requests_for_cpu_hpa = cpu_threshold_m / estimated_cpu_per_request
    requests_for_memory_hpa = memory_threshold_mb / RESOURCE_CONFIG['memory_mb_per_request']
    
    with ALLOCATION_LOCK:
        active_allocations_count = len(ACTIVE_ALLOCATIONS)
        total_allocated_mb = sum(alloc['size_mb'] for alloc in ACTIVE_ALLOCATIONS.values())

    return {
        "deterministic_config": RESOURCE_CONFIG,
        "processing_stats": current_stats,
        "current_allocations": {
            "count": active_allocations_count,
            "total_mb": total_allocated_mb
        },
        "deterministic_guarantees": {
            "cpu_operations_per_request": RESOURCE_CONFIG["cpu_operations_per_request"],
            "memory_mb_per_request": RESOURCE_CONFIG["memory_mb_per_request"],
            "duration_seconds": RESOURCE_CONFIG["duration_seconds"],
            "reproducible": True,
            "consistent_load": True
        },
        "hpa_thresholds": {
            "cpu_threshold_m": cpu_threshold_m,
            "memory_threshold_mb": round(memory_threshold_mb, 1),
            "requests_to_trigger_cpu_hpa": round(requests_for_cpu_hpa, 2),
            "requests_to_trigger_memory_hpa": round(requests_for_memory_hpa, 2),
            "recommended_concurrent_requests_for_hpa": math.ceil(max(requests_for_cpu_hpa, requests_for_memory_hpa))
        },
        "testing_instructions": {
            "deterministic_load": "Each request performs exactly the same work",
            "memory_dominant": "Memory usage will be the primary scaling trigger",
            "to_trigger_hpa": f"Send {math.ceil(max(requests_for_cpu_hpa, requests_for_memory_hpa))} concurrent requests",
            "example_curl": f"for i in {{1..12}}; do curl -X POST http://localhost:8000/process?async_mode=true & done; wait"
        }
    }


@app.get("/process/cleanup")
async def cleanup_memory():
    """Force cleanup of memory allocations"""
    cleaned = cleanup_old_allocations()
    
    with stats_lock:
        current_stats = processing_stats.copy()
    
    with ALLOCATION_LOCK:
        remaining_allocations = len(ACTIVE_ALLOCATIONS)
        total_allocated_mb = sum(alloc['size_mb'] for alloc in ACTIVE_ALLOCATIONS.values())
    
    return {
        "status": "cleaned",
        "allocations_removed": cleaned,
        "remaining_allocations": remaining_allocations,
        "total_allocated_mb": total_allocated_mb,
        "deterministic": True
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
