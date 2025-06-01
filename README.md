# NimbusGuard Heavy Processing API

A CPU and memory-intensive FastAPI application designed specifically for heavy load testing and performance monitoring. This service performs extensive computational operations to stress-test systems and generate meaningful logs for analysis.

## 🚀 Quick Start

### Local Development

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

### Docker Deployment

1. **Build the Docker image:**
   ```bash
   docker build -t nimbusguard:latest .
   ```

2. **Run with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

### Kubernetes Deployment

The Kubernetes configurations are maintained in a separate repository as a Git submodule.

1. **Initialize the k8s submodule:**
   ```bash
   git submodule update --init --recursive
   ```

2. **Deploy to Kubernetes:**
   ```bash
   kubectl apply -f k8s-nimbusguard/deployment.yaml
   kubectl apply -f k8s-nimbusguard/service.yaml
   ```

3. **Verify the deployment:**
   ```bash
   kubectl get deployments
   kubectl get pods
   kubectl get services
   ```

## 🌐 Access Points

- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
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

## 📊 Monitoring Endpoints

- **`GET /health`** - Health check with request count
- **`GET /stats`** - API statistics and performance info
- **`GET /`** - Basic API information

## 📝 Logging and Performance Analysis

### Log Files
- **`app.log`** - Comprehensive request/response logs (mounted in Docker/K8s)
- **Console output** - Real-time monitoring

### Log Details Include:
- Request ID and processing phases
- Input data size and complexity
- Processing time per phase (ms precision)
- Operations count per request
- Memory usage estimates
- Success/failure tracking with error details

## ⚡ System Requirements

### Local Development
- Python 3.8+
- 4GB RAM minimum
- 2 CPU cores minimum

### Docker/Kubernetes Requirements
As configured in deployment files:
- CPU limit: 2 cores
- Memory limit: 2GB
- CPU request: 500m
- Memory request: 512Mi

## 🔧 Configuration

### Environment Variables
- `PYTHONUNBUFFERED=1`: Ensures Python output is sent straight to terminal
- `PYTHONDONTWRITEBYTECODE=1`: Prevents Python from writing pyc files
- `WORKERS=4`: Number of worker processes

### Resource Limits
Docker and Kubernetes configurations include:
- CPU limits and requests
- Memory limits and requests
- Health check configurations
- Logging configurations

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

## 🎯 Use Cases

- **API Stress Testing**: Validate system performance under heavy computational load
- **Infrastructure Testing**: Test deployment scaling and resource allocation
- **Performance Benchmarking**: Establish baseline metrics for optimization
- **Monitoring Validation**: Test logging, alerting, and observability systems
- **CI/CD Integration**: Automated performance regression testing

## 📦 Project Structure

```
nimbusguard/
├── .git/
├── .venv/
├── k8s-nimbusguard/          # Kubernetes configurations (submodule)
│   ├── deployment.yaml
│   ├── service.yaml
│   └── README.md
├── app.log
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── .dockerignore
├── main.py
├── .gitignore
└── README.md
```

This heavy processing API is designed to push your system to its limits and provide detailed insights into performance characteristics under computational stress. It supports deployment through Docker and Kubernetes for scalability and ease of management.
