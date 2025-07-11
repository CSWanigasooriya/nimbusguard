import os
import time
import math
import random
import asyncio
import gc  # Add garbage collection import
from typing import Optional, Dict, Any
from fastapi import FastAPI, Query, BackgroundTasks
from prometheus_fastapi_instrumentator import Instrumentator
from collections import defaultdict
import threading
from datetime import datetime, timedelta

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor

# Initialize OpenTelemetry tracing
def setup_tracing():
    # Configure the tracer provider with service information
    resource = Resource(attributes={
        SERVICE_NAME: "nimbusguard-consumer",
        "service.version": "1.0.0",
        "service.namespace": "nimbusguard"
    })
    
    # Set up trace provider
    trace.set_tracer_provider(TracerProvider(resource=resource))
    tracer = trace.get_tracer(__name__)
    
    # Configure OTLP exporter (sends traces to Alloy)
    otlp_exporter = OTLPSpanExporter(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://alloy:4318/v1/traces"),
        headers={}
    )
    
    # Add span processor
    span_processor = BatchSpanProcessor(otlp_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    return tracer

# Set up tracing before creating the FastAPI app
tracer = setup_tracing()

# Create FastAPI instance
app = FastAPI(
    title="NimbusGuard Consumer",
    description="A consumer service for NimbusGuard",
    version="1.0.0"
)

# Initialize Prometheus metrics instrumentation
Instrumentator().instrument(app).expose(app)

# Auto-instrument FastAPI for tracing (captures all HTTP requests automatically)
FastAPIInstrumentor.instrument_app(app)

# Auto-instrument requests library for outbound HTTP calls
RequestsInstrumentor().instrument()

# Auto-instrument urllib3 for HTTP client calls
URLLib3Instrumentor().instrument()

# Real-world memory management configuration
MEMORY_CONFIG = {
    "auto_cleanup_enabled": True,
    "default_retention_seconds": int(os.getenv("MEMORY_RETENTION_SECONDS", "60")),  # 1 minute default
    "cleanup_interval_seconds": int(os.getenv("MEMORY_CLEANUP_INTERVAL", "30")),   # Check every 30s
    "max_memory_objects": int(os.getenv("MAX_MEMORY_OBJECTS", "100")),             # Limit memory objects
    "immediate_cleanup_mode": os.getenv("IMMEDIATE_CLEANUP", "true").lower() == "true"  # Real-world default
}

# Memory store with automatic cleanup (like real applications)
memory_store = {}
memory_stats = {
    "total_processed": 0,
    "current_objects": 0,
    "total_cleaned": 0,
    "memory_freed_mb": 0
}

# Background cleanup task
cleanup_task = None
cleanup_lock = threading.Lock()

async def automatic_memory_cleanup():
    """Background task that automatically cleans up old memory objects like real-world apps"""
    while True:
        try:
            current_time = time.time()
            retention_seconds = MEMORY_CONFIG["default_retention_seconds"]
            
            with cleanup_lock:
                old_keys = []
                freed_mb = 0
                
                for key, obj in memory_store.items():
                    age = current_time - obj["timestamp"]
                    if age > retention_seconds:
                        old_keys.append(key)
                        freed_mb += obj.get("size_mb", 0)
                
                # Remove old objects
                for key in old_keys:
                    del memory_store[key]
                
                if old_keys:
                    # Update stats
                    memory_stats["current_objects"] = len(memory_store)
                    memory_stats["total_cleaned"] += len(old_keys)
                    memory_stats["memory_freed_mb"] += freed_mb
                    
                    # Trigger garbage collection
                    gc.collect()
                    
                    print(f"Auto-cleanup: Removed {len(old_keys)} objects, freed {freed_mb}MB")
            
        except Exception as e:
            print(f"Auto-cleanup error: {e}")
        
        # Wait for next cleanup cycle
        await asyncio.sleep(MEMORY_CONFIG["cleanup_interval_seconds"])

async def start_background_cleanup():
    """Start the background cleanup task"""
    global cleanup_task
    if MEMORY_CONFIG["auto_cleanup_enabled"] and cleanup_task is None:
        cleanup_task = asyncio.create_task(automatic_memory_cleanup())
        print(f"Started automatic memory cleanup (retention: {MEMORY_CONFIG['default_retention_seconds']}s)")

@app.on_event("startup")
async def startup_event():
    """Initialize background tasks on startup"""
    await start_background_cleanup()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    global cleanup_task
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

@app.get("/")
async def root():
    """Root endpoint that returns a welcome message"""
    return {"message": "Welcome to NimbusGuard Consumer!"}

@app.get("/health")
async def health():
    """Liveness probe endpoint"""
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    """Readiness probe endpoint"""
    return {"status": "ready"}

# In-memory storage to simulate memory usage
# memory_store = [] # This line is no longer needed as memory_store is now a dict

def cpu_intensive_task(intensity: int = 5, duration: int = 10):
    """
    CPU-intensive task that calculates prime numbers
    Args:
        intensity: Complexity level (1-10, higher = more CPU usage)
        duration: How long to run the task in seconds
    """
    with tracer.start_as_current_span("cpu_intensive_task") as span:
        span.set_attribute("intensity", intensity)
        span.set_attribute("duration", duration)
        
        start_time = time.time()
        prime_count = 0
        
        # Calculate primes up to a number based on intensity
        max_number = intensity * 10000
        
        while time.time() - start_time < duration:
            # Find primes using trial division (CPU intensive)
            for num in range(2, max_number):
                is_prime = True
                for i in range(2, int(math.sqrt(num)) + 1):
                    if num % i == 0:
                        is_prime = False
                        break
                if is_prime:
                    prime_count += 1
                    
                # Check if duration exceeded
                if time.time() - start_time >= duration:
                    break
        
        span.set_attribute("primes_found", prime_count)
        return prime_count

def memory_intensive_task(size_mb: int = 50, duration: int = 10, task_id: str = None, immediate_cleanup: bool = None):
    """
    Memory-intensive task that allocates and manipulates data
    Args:
        size_mb: Amount of memory to allocate in MB
        duration: How long to keep the memory allocated
        task_id: Unique task identifier
        immediate_cleanup: Whether to cleanup immediately after processing (real-world behavior)
    """
    with tracer.start_as_current_span("memory_intensive_task") as span:
        span.set_attribute("size_mb", size_mb)
        span.set_attribute("duration", duration)
        span.set_attribute("immediate_cleanup", immediate_cleanup or MEMORY_CONFIG["immediate_cleanup_mode"])
        
        # More efficient memory allocation - single data structure instead of duplicate
        chunk_size = size_mb * 1024 * 1024
        
        # Create a single large data structure (not both list AND dict)
        large_data = []
        
        # Fill with random data - more memory efficient
        chunk_count = chunk_size // 1000  # Number of 1KB chunks
        for i in range(chunk_count):
            # Create data in smaller chunks to avoid memory spikes
            random_data = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=1000))
            large_data.append(random_data)
        
        # Store metadata with memory object (simplified structure)
        memory_object = {
            'data': large_data,
            'size_mb': size_mb,
            'timestamp': time.time(),
            'chunk_count': len(large_data),
            'task_id': task_id or f"task_{int(time.time())}"
        }
        
        # Real-world behavior: immediate cleanup vs. temporary storage
        if immediate_cleanup if immediate_cleanup is not None else MEMORY_CONFIG["immediate_cleanup_mode"]:
            # Real-world mode: process data and immediately discard (like most applications)
            time.sleep(duration)  # Simulate processing time
            
            # Data would normally be processed here, then discarded
            # Simulate garbage collection by clearing references
            large_data.clear()
            memory_object.clear()
            
            # Update stats
            memory_stats["total_processed"] += 1
            
            span.set_attribute("cleanup_mode", "immediate")
            print(f"Processed and immediately freed {size_mb}MB (real-world mode)")
            
        else:
            # Testing mode: store temporarily for autoscaling testing
            with cleanup_lock:
                # Check memory limits
                if len(memory_store) >= MEMORY_CONFIG["max_memory_objects"]:
                    # Remove oldest object to make space
                    oldest_key = min(memory_store.keys(), key=lambda k: memory_store[k]["timestamp"])
                    old_obj = memory_store.pop(oldest_key)
                    memory_stats["total_cleaned"] += 1
                    memory_stats["memory_freed_mb"] += old_obj.get("size_mb", 0)
                
                # Store the memory object
                memory_store[memory_object['task_id']] = memory_object
                memory_stats["total_processed"] += 1
                memory_stats["current_objects"] = len(memory_store)
            
            # Keep the memory allocated for the specified duration
            time.sleep(duration)
            
            span.set_attribute("cleanup_mode", "temporary_storage")
        
        # Calculate actual memory usage for tracing
        actual_size = sum(len(chunk) for chunk in memory_object.get('data', []))
        span.set_attribute("actual_bytes_allocated", actual_size)
        span.set_attribute("chunks_created", len(memory_object.get('data', [])))
        
        return len(memory_object.get('data', []))

@app.post("/process")
async def process_load(
    background_tasks: BackgroundTasks,
    cpu_intensity: int = Query(5, ge=1, le=10, description="CPU intensity level (1-10)"),
    memory_size: int = Query(50, ge=10, le=500, description="Memory to allocate in MB"),
    duration: int = Query(10, ge=1, le=60, description="Processing duration in seconds"),
    async_mode: bool = Query(False, description="Run in background (async)"),
    immediate_cleanup: bool = Query(None, description="Cleanup immediately after processing (real-world mode)")
):
    """
    Endpoint to simulate realistic processing load for testing autoscaling
    
    This endpoint will:
    - Consume CPU by calculating prime numbers
    - Consume memory by allocating large data structures
    - Allow configuration of intensity and duration
    - Support both sync and async processing
    - Use real-world memory management by default
    """
    
    with tracer.start_as_current_span("process_load_endpoint") as span:
        span.set_attribute("cpu_intensity", cpu_intensity)
        span.set_attribute("memory_size", memory_size)
        span.set_attribute("duration", duration)
        span.set_attribute("async_mode", async_mode)
        
        task_id = f"task_{int(time.time())}_{random.randint(1000, 9999)}"
        span.set_attribute("task_id", task_id)
        
        # Use immediate cleanup by default (real-world behavior)
        cleanup_mode = immediate_cleanup if immediate_cleanup is not None else MEMORY_CONFIG["immediate_cleanup_mode"]
        span.set_attribute("cleanup_mode", cleanup_mode)
        
        if async_mode:
            # Run in background
            background_tasks.add_task(
                cpu_intensive_task, 
                intensity=cpu_intensity, 
                duration=duration
            )
            background_tasks.add_task(
                memory_intensive_task, 
                size_mb=memory_size, 
                duration=duration,
                task_id=task_id,
                immediate_cleanup=cleanup_mode
            )
            
            return {
                "status": "started",
                "task_id": task_id,
                "message": f"Background processing started for {duration}s",
                "cpu_intensity": cpu_intensity,
                "memory_size_mb": memory_size,
                "cleanup_mode": "immediate" if cleanup_mode else "temporary",
                "estimated_completion": time.time() + duration
            }
        else:
            # Run synchronously
            start_time = time.time()
            
            # Run CPU and memory tasks in parallel using asyncio
            cpu_task = asyncio.create_task(
                asyncio.to_thread(cpu_intensive_task, cpu_intensity, duration)
            )
            memory_task = asyncio.create_task(
                asyncio.to_thread(memory_intensive_task, memory_size, duration, task_id, cleanup_mode)
            )
            
            # Wait for both tasks to complete
            primes_found, memory_objects = await asyncio.gather(cpu_task, memory_task)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            return {
                "status": "completed",
                "task_id": task_id,
                "processing_time_seconds": round(processing_time, 2),
                "cpu_work": {
                    "intensity": cpu_intensity,
                    "primes_found": primes_found
                },
                "memory_work": {
                    "size_mb": memory_size,
                    "objects_created": memory_objects,
                    "cleanup_mode": "immediate" if cleanup_mode else "temporary"
                },
                "memory_store_size": len(memory_store)
            }

@app.get("/process/status")
async def process_status():
    """Get current processing status and memory usage"""
    with cleanup_lock:
        current_memory_objects = list(memory_store.values())
    
    # Calculate more accurate memory estimation
    total_estimated_mb = sum(item.get("size_mb", 50) for item in current_memory_objects)
    
    return {
        "memory_management": {
            "auto_cleanup_enabled": MEMORY_CONFIG["auto_cleanup_enabled"],
            "retention_seconds": MEMORY_CONFIG["default_retention_seconds"],
            "cleanup_interval": MEMORY_CONFIG["cleanup_interval_seconds"],
            "max_objects": MEMORY_CONFIG["max_memory_objects"],
            "immediate_cleanup_mode": MEMORY_CONFIG["immediate_cleanup_mode"]
        },
        "current_status": {
            "memory_store_objects": len(current_memory_objects),
            "memory_store_size_estimate_mb": total_estimated_mb,
        },
        "statistics": memory_stats.copy(),
        "recent_tasks": [
            {
                "task_id": item["task_id"],
                "timestamp": item["timestamp"],
                "age_seconds": round(time.time() - item["timestamp"], 2),
                "data_chunks": len(item["data"]),
                "size_mb": item.get("size_mb", "unknown"),
                "chunk_count": item.get("chunk_count", len(item["data"]))
            }
            for item in current_memory_objects[-5:]  # Last 5 tasks
        ]
    }

@app.delete("/process/cleanup")
async def cleanup_memory():
    """Clean up memory store to free up memory with explicit garbage collection"""
    with cleanup_lock:
        old_size = len(memory_store)
        
        # Calculate total memory before cleanup
        total_mb_before = sum(item.get("size_mb", 50) for item in memory_store.values())
        
        # Clear the memory store
        memory_store.clear()
        
        # Update stats
        memory_stats["current_objects"] = 0
        memory_stats["total_cleaned"] += old_size
        memory_stats["memory_freed_mb"] += total_mb_before
    
    # Explicitly trigger garbage collection to free memory faster
    collected = gc.collect()
    
    return {
        "status": "cleaned",
        "objects_removed": old_size,
        "estimated_mb_freed": total_mb_before,
        "gc_objects_collected": collected,
        "message": f"Memory store cleared, {total_mb_before}MB freed, GC collected {collected} objects"
    }

@app.delete("/process/cleanup-old")
async def cleanup_old_memory(max_age_seconds: int = Query(300, description="Max age in seconds for memory objects")):
    """Clean up memory objects older than specified age"""
    current_time = time.time()
    
    with cleanup_lock:
        old_objects = {}
        remaining_objects = {}
        
        for key, item in memory_store.items():
            if current_time - item["timestamp"] > max_age_seconds:
                old_objects[key] = item
            else:
                remaining_objects[key] = item
        
        # Update memory store
        memory_store.clear()
        memory_store.update(remaining_objects)
        
        # Calculate freed memory
        freed_mb = sum(item.get("size_mb", 50) for item in old_objects.values())
        objects_removed = len(old_objects)
        
        # Update stats
        memory_stats["current_objects"] = len(memory_store)
        memory_stats["total_cleaned"] += objects_removed
        memory_stats["memory_freed_mb"] += freed_mb
    
    # Trigger garbage collection if we removed objects
    collected = 0
    if objects_removed > 0:
        collected = gc.collect()
    
    return {
        "status": "cleaned",
        "objects_removed": objects_removed,
        "objects_remaining": len(memory_store),
        "estimated_mb_freed": freed_mb,
        "gc_objects_collected": collected,
        "max_age_seconds": max_age_seconds,
        "message": f"Removed {objects_removed} old objects, freed {freed_mb}MB"
    }

@app.get("/process/config")
async def get_memory_config():
    """Get current memory management configuration"""
    return {
        "memory_config": MEMORY_CONFIG.copy(),
        "statistics": memory_stats.copy(),
        "description": {
            "immediate_cleanup_mode": "Real-world mode: processes data and immediately frees memory",
            "temporary_storage_mode": "Testing mode: keeps memory allocated for autoscaling testing",
            "auto_cleanup": "Background task automatically cleans up old memory objects"
        }
    }

@app.post("/process/config")
async def update_memory_config(
    immediate_cleanup: bool = Query(None, description="Enable immediate cleanup mode"),
    retention_seconds: int = Query(None, description="Memory retention time in seconds"),
    auto_cleanup: bool = Query(None, description="Enable automatic cleanup")
):
    """Update memory management configuration"""
    global cleanup_task
    
    changes = {}
    
    if immediate_cleanup is not None:
        MEMORY_CONFIG["immediate_cleanup_mode"] = immediate_cleanup
        changes["immediate_cleanup_mode"] = immediate_cleanup
    
    if retention_seconds is not None:
        MEMORY_CONFIG["default_retention_seconds"] = retention_seconds
        changes["retention_seconds"] = retention_seconds
    
    if auto_cleanup is not None:
        MEMORY_CONFIG["auto_cleanup_enabled"] = auto_cleanup
        changes["auto_cleanup_enabled"] = auto_cleanup
        
        # Restart cleanup task if needed
        if auto_cleanup and cleanup_task is None:
            await start_background_cleanup()
        elif not auto_cleanup and cleanup_task:
            cleanup_task.cancel()
            cleanup_task = None
    
    return {
        "status": "updated",
        "changes": changes,
        "current_config": MEMORY_CONFIG.copy()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 