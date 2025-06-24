import os
import time
import math
import random
import asyncio
from typing import Optional
from fastapi import FastAPI, Query, BackgroundTasks
from prometheus_fastapi_instrumentator import Instrumentator

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
memory_store = []

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

def memory_intensive_task(size_mb: int = 50, duration: int = 10):
    """
    Memory-intensive task that allocates and manipulates data
    Args:
        size_mb: Amount of memory to allocate in MB
        duration: How long to keep the memory allocated
    """
    with tracer.start_as_current_span("memory_intensive_task") as span:
        span.set_attribute("size_mb", size_mb)
        span.set_attribute("duration", duration)
        
        # Allocate memory (1 MB = ~1 million chars)
        chunk_size = size_mb * 1024 * 1024
        
        # Create large data structures
        large_list = []
        large_dict = {}
        
        # Fill with random data
        for i in range(0, chunk_size, 1000):  # Create chunks of 1000 bytes
            random_data = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=1000))
            large_list.append(random_data)
            large_dict[i] = random_data
        
        # Store in global memory store to prevent garbage collection
        memory_store.append({
            'list': large_list,
            'dict': large_dict,
            'timestamp': time.time()
        })
        
        # Keep the memory allocated for the specified duration
        time.sleep(duration)
        
        # Simulate some work with the allocated memory
        total_length = sum(len(item) for item in large_list)
        span.set_attribute("total_data_length", total_length)
        
        return len(large_list)

@app.post("/process")
async def process_load(
    background_tasks: BackgroundTasks,
    cpu_intensity: int = Query(5, ge=1, le=10, description="CPU intensity level (1-10)"),
    memory_size: int = Query(50, ge=10, le=500, description="Memory to allocate in MB"),
    duration: int = Query(10, ge=1, le=60, description="Processing duration in seconds"),
    async_mode: bool = Query(False, description="Run in background (async)")
):
    """
    Endpoint to simulate realistic processing load for testing autoscaling
    
    This endpoint will:
    - Consume CPU by calculating prime numbers
    - Consume memory by allocating large data structures
    - Allow configuration of intensity and duration
    - Support both sync and async processing
    """
    
    with tracer.start_as_current_span("process_load_endpoint") as span:
        span.set_attribute("cpu_intensity", cpu_intensity)
        span.set_attribute("memory_size", memory_size)
        span.set_attribute("duration", duration)
        span.set_attribute("async_mode", async_mode)
        
        task_id = f"task_{int(time.time())}_{random.randint(1000, 9999)}"
        span.set_attribute("task_id", task_id)
        
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
                duration=duration
            )
            
            return {
                "status": "started",
                "task_id": task_id,
                "message": f"Background processing started for {duration}s",
                "cpu_intensity": cpu_intensity,
                "memory_size_mb": memory_size,
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
                asyncio.to_thread(memory_intensive_task, memory_size, duration)
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
                    "objects_created": memory_objects
                },
                "memory_store_size": len(memory_store)
            }

@app.get("/process/status")
async def process_status():
    """Get current processing status and memory usage"""
    return {
        "memory_store_objects": len(memory_store),
        "memory_store_size_estimate_mb": len(memory_store) * 50,  # Rough estimate
        "recent_tasks": [
            {
                "timestamp": item["timestamp"],
                "age_seconds": round(time.time() - item["timestamp"], 2),
                "list_size": len(item["list"]),
                "dict_size": len(item["dict"])
            }
            for item in memory_store[-5:]  # Last 5 tasks
        ]
    }

@app.delete("/process/cleanup")
async def cleanup_memory():
    """Clean up memory store to free up memory"""
    global memory_store
    old_size = len(memory_store)
    memory_store.clear()
    
    return {
        "status": "cleaned",
        "objects_removed": old_size,
        "message": "Memory store cleared"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 