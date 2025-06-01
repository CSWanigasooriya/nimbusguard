# NimbusGuard Heavy Processing API

A CPU and memory-intensive FastAPI application designed specifically for heavy load testing and performance monitoring. This service performs extensive computational operations to stress-test systems and generate meaningful logs for analysis.

## 🚀 Quick Start

1. **Activate virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

4. **Access the API:**
   - API: http://localhost:8000
   - Interactive Docs: http://localhost:8000/docs
   - Stats: http://localhost:8000/stats

## 🔥 Heavy Processing Endpoint

### `/process` - Intensive CPU & Memory Operations

This endpoint performs **4000+ computational operations** per request:

- **Mathematical computations**: 2000 iterative calculations
- **String processing**: 500 text transformation operations  
- **Data transformations**: Input data processing with 100-1000 operations per field
- **Matrix operations**: 100x100 matrix computations
- **Memory operations**: Creation of 1000 large objects

**Expected processing time**: 2-10 seconds per request depending on system performance.

## 🧪 Testing the API

### Quick Test
```bash
# Make executable and run
chmod +x quick_test.sh
./quick_test.sh
```

### Load Testing
```bash
# Interactive load test
python test_api.py

# Follow prompts to configure:
# - Number of threads (recommended: 3-5)
# - Requests per thread (recommended: 5-10)
```

### Manual Testing
```bash
curl -X POST "http://localhost:8000/process" \
     -H "Content-Type: application/json" \
     -d '{
       "data": {
         "user_id": 123,
         "dataset": [1,2,3,4,5],
         "text_data": "Heavy processing test",
         "operation_type": "extreme_compute"
       },
       "request_id": "manual_test_001"
     }'
```

## 📊 Monitoring Endpoints

- **`GET /health`** - Health check with request count
- **`GET /stats`** - API statistics and performance info
- **`GET /`** - Basic API information

## 📝 Logging and Performance Analysis

### Log Files
- **`app.log`** - Comprehensive request/response logs
- **Console output** - Real-time monitoring

### Log Details Include:
- Request ID and processing phases
- Input data size and complexity
- Processing time per phase (ms precision)
- Operations count per request
- Memory usage estimates
- Success/failure tracking with error details

### Example Log Output:
```
2025-06-01 10:30:15 - nimbusguard - INFO - Request #1 - POST /process - Client: 127.0.0.1
2025-06-01 10:30:15 - nimbusguard - INFO - Heavy processing request heavy_1 - Input data size: 6 fields
2025-06-01 10:30:15 - nimbusguard - INFO - Starting heavy computation for request heavy_1
2025-06-01 10:30:18 - nimbusguard - INFO - Heavy processing completed for request heavy_1 - Operations: 4247, Time: 3240.85ms
2025-06-01 10:30:18 - nimbusguard - INFO - Request #1 completed - Status: 200 - Time: 3.245s
```

## 🎯 Load Testing Strategies

### Recommended Test Scenarios

**Light Load Test:**
- 3 threads, 5 requests each
- Duration: ~2-3 minutes
- Good for baseline performance

**Medium Load Test:**
- 5 threads, 8 requests each  
- Duration: ~5-8 minutes
- Tests concurrent processing

**Heavy Load Test:**
- 8 threads, 10 requests each
- Duration: ~10-15 minutes
- Stress test for system limits

### Performance Metrics to Monitor

1. **Processing Time**: Target <5 seconds per request
2. **Operations Per Second**: Measure computational throughput  
3. **Memory Usage**: Monitor system memory during tests
4. **CPU Utilization**: Expect high CPU usage (60-90%)
5. **Success Rate**: Target >95% success rate
6. **Queue Times**: Monitor request queuing under load

## 📈 Response Format

```json
{
  "status": "heavy_processing_complete",
  "request_id": "heavy_123_1717234567",
  "processed_at": "2025-06-01T10:30:15.123456",
  "processing_time_ms": 3240.85,
  "operations_performed": 4247,
  "result": {
    "input_fields_processed": 6,
    "computational_results": 2000,
    "string_operations": 500,
    "data_transformations": 6,
    "matrix_computation": {...},
    "memory_operations": {...},
    "total_result_keys": 2506
  }
}
```

## 🔧 Advanced Load Testing

### Using External Tools

**Apache Bench:**
```bash
# Create test data file
echo '{"data":{"test":"heavy load"},"request_id":"ab_test"}' > heavy_data.json

# Run load test
ab -n 50 -c 5 -p heavy_data.json -T application/json http://localhost:8000/process
```

**wrk with Lua script:**
```bash
# Heavy processing load test
wrk -t4 -c8 -d60s --script=heavy_post.lua http://localhost:8000/process
```

### Performance Analysis Commands

```bash
# Count total requests
grep "Heavy processing completed" app.log | wc -l

# Average processing time
grep "Heavy processing completed" app.log | \
  awk '{print $(NF-1)}' | sed 's/ms//' | \
  awk '{sum+=$1; count++} END {print "Average:", sum/count, "ms"}'

# Find slowest requests  
grep "Heavy processing completed" app.log | \
  sort -k12 -n | tail -5

# Operations per second
grep "Heavy processing completed" app.log | \
  awk '{print $10}' | sed 's/,//' | \
  awk '{sum+=$1; count++} END {print "Avg ops:", sum/count}'
```

## ⚡ System Requirements

**Minimum:**
- 2 CPU cores
- 4GB RAM
- Python 3.8+

**Recommended for Load Testing:**
- 4+ CPU cores  
- 8GB+ RAM
- SSD storage for faster I/O

## 🎯 Use Cases

- **API Stress Testing**: Validate system performance under heavy computational load
- **Infrastructure Testing**: Test deployment scaling and resource allocation
- **Performance Benchmarking**: Establish baseline metrics for optimization
- **Monitoring Validation**: Test logging, alerting, and observability systems
- **CI/CD Integration**: Automated performance regression testing

This heavy processing API is designed to push your system to its limits and provide detailed insights into performance characteristics under computational stress.
