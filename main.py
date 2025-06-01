from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import logging
import time
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("nimbusguard")

# Create FastAPI instance
app = FastAPI(
    title="NimbusGuard Heavy Processing API",
    description="Heavy CPU-intensive processing API for load testing",
    version="1.0.0"
)

# Request model
class ProcessRequest(BaseModel):
    data: Dict[str, Any]
    request_id: Optional[str] = None

# Response model
class ProcessResponse(BaseModel):
    status: str
    request_id: str
    processed_at: str
    processing_time_ms: float
    operations_performed: int
    result: Dict[str, Any]

# Request counter for monitoring
request_counter = 0

@app.middleware("http")
async def log_requests(request: Request, call_next):
    global request_counter
    request_counter += 1
    
    start_time = time.time()
    
    # Log incoming request
    logger.info(f"Request #{request_counter} - {request.method} {request.url.path} - Client: {request.client.host}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    # Log response
    logger.info(f"Request #{request_counter} completed - Status: {response.status_code} - Time: {process_time:.3f}s")
    
    return response

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {
        "message": "NimbusGuard Heavy Processing API",
        "status": "running",
        "total_requests": request_counter,
        "endpoint": "/process - Heavy CPU-intensive processing",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "uptime": "running",
        "total_requests": request_counter,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/process", response_model=ProcessResponse)
async def process_heavy_data(request: ProcessRequest):
    start_time = time.time()
    request_id = request.request_id or f"heavy_{request_counter}_{int(time.time())}"
    
    logger.info(f"Heavy processing request {request_id} - Input data size: {len(request.data)} fields")
    
    try:
        result = {}
        operations_count = 0
        
        # CPU-intensive computational work
        logger.info(f"Starting heavy computation for request {request_id}")
        
        # Phase 1: Mathematical computations
        for i in range(5000):  # Increased from 2000
            result[f"computed_{i}"] = sum(range(i, i+500))  # Increased range
            operations_count += 1
        
        # Phase 2: String processing
        for i in range(1000):  # Increased from 500
            text = f"processing_string_{i}_" * 100  # Increased multiplication
            result[f"processed_text_{i}"] = text.upper().replace("_", "-").encode().decode()
            operations_count += 1
        
        # Phase 3: Data transformation
        for key, value in request.data.items():
            if isinstance(value, str):
                # Heavy string operations
                processed_value = value.upper().lower().title()
                for _ in range(1000):  # Increased from 100
                    processed_value = processed_value.encode().decode().replace(" ", "_").replace("_", " ")
                result[f"heavy_processed_{key}"] = processed_value
            elif isinstance(value, (int, float)):
                # Mathematical operations
                processed_value = value
                for _ in range(5000):  # Increased from 1000
                    processed_value = (processed_value * 1.5) / 1.5
                    processed_value = pow(processed_value, 2)
                    processed_value = pow(processed_value, 0.5)
                result[f"heavy_calculated_{key}"] = round(processed_value, 6)
            elif isinstance(value, list):
                # List processing
                processed_list = []
                for item in value:
                    for _ in range(500):  # Increased from 50
                        processed_item = str(item).upper().lower().encode().decode()
                        processed_item = processed_item * 10
                    processed_list.append(processed_item)
                result[f"heavy_list_{key}"] = processed_list
            else:
                # Generic processing
                processed_value = str(value)
                for _ in range(1000):  # Increased from 200
                    processed_value = processed_value[::-1]
                result[f"heavy_generic_{key}"] = processed_value
            
            operations_count += 1
        
        # Phase 4: Additional heavy computation
        matrix_size = 500  # Increased from 100
        matrix_result = []
        for i in range(matrix_size):
            row = []
            for j in range(matrix_size):
                # More complex matrix computation
                value = sum((k * (i + j + k) ** 2) for k in range(20))  # Increased complexity
                row.append(value % 10000)
            matrix_result.append(sum(row))
            operations_count += 1
        
        result["matrix_computation"] = {
            "rows_processed": len(matrix_result),
            "total_sum": sum(matrix_result),
            "average": sum(matrix_result) / len(matrix_result) if matrix_result else 0
        }
        
        # Phase 5: Memory intensive operations
        large_data = []
        for i in range(1000):
            large_data.append({
                f"key_{i}": [j for j in range(50)],
                f"data_{i}": f"large_string_{'x' * 100}_{i}"
            })
            operations_count += 1
        
        result["memory_operations"] = {
            "objects_created": len(large_data),
            "memory_footprint_estimate": len(str(large_data))
        }
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Heavy processing completed for request {request_id} - Operations: {operations_count}, Time: {processing_time:.2f}ms")
        
        return ProcessResponse(
            status="heavy_processing_complete",
            request_id=request_id,
            processed_at=datetime.now().isoformat(),
            processing_time_ms=round(processing_time, 2),
            operations_performed=operations_count,
            result={
                "input_fields_processed": len(request.data),
                "computational_results": len([k for k in result.keys() if "computed" in k]),
                "string_operations": len([k for k in result.keys() if "text" in k]),
                "data_transformations": len([k for k in result.keys() if "processed" in k]),
                "matrix_computation": result["matrix_computation"],
                "memory_operations": result["memory_operations"],
                "total_result_keys": len(result)
            }
        )
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error(f"Error in heavy processing {request_id}: {str(e)} - Time: {processing_time:.2f}ms")
        raise HTTPException(status_code=500, detail=f"Heavy processing failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    logger.info("Stats requested")
    return {
        "total_requests": request_counter,
        "status": "running",
        "api_type": "Heavy Processing Only",
        "endpoint": "/process - CPU and memory intensive operations",
        "operations_per_request": "~4000+ computational operations",
        "timestamp": datetime.now().isoformat()
    }

# Run the application
if __name__ == "__main__":
    logger.info("Starting NimbusGuard Heavy Processing API...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
