# NimbusGuard Intelligent Scaling Operator

A Kubernetes operator for intelligent auto-scaling using AI-driven decision making.

## Project Structure

The operator has been refactored into a modular structure for better maintainability:

```
nimbusguard-operator/
├── nimbusguard/                    # Main package
│   ├── __init__.py                 # Package initialization
│   ├── config.py                   # Configuration and logging setup
│   ├── metrics.py                  # Prometheus metrics definitions
│   ├── health.py                   # Health check functions
│   ├── operator.py                 # Main operator implementation
│   ├── handlers.py                 # Kopf event handlers
│   ├── clients/                    # External service clients
│   │   ├── __init__.py
│   │   └── prometheus.py           # Prometheus client
│   └── engines/                    # Business logic engines
│       ├── __init__.py
│       └── decision.py             # Decision engine
├── main.py                         # Entry point
├── pyproject.toml                  # Modern Python packaging
├── requirements.txt                # Legacy requirements (reference)
├── Makefile                        # Development commands
├── Dockerfile                      # Container image
├── .dockerignore                   # Docker build exclusions
├── README.md                       # This file
└── MIGRATION.md                    # Migration guide
```

## Components

### Core Components

- **`config.py`**: Centralized configuration management, logging setup, and health status tracking
- **`operator.py`**: Main operator class containing the core scaling logic
- **`handlers.py`**: Kopf event handlers for Kubernetes resource lifecycle management
- **`metrics.py`**: Prometheus metrics definitions and server management
- **`health.py`**: Health check functions for liveness and readiness probes

### Client Modules

- **`clients/prometheus.py`**: Prometheus client for metrics collection with error handling and health monitoring

### Engine Modules

- **`engines/decision.py`**: Decision engine for making intelligent scaling decisions based on metrics

## Quick Start

### Prerequisites

- Python 3.8+
- Docker (for containerized deployment)
- Kubernetes cluster access
- Prometheus (optional, for metrics collection)

### Development Setup

```bash
# 1. Install dependencies
make dev-install

# 2. Set environment variables (optional)
export LOG_LEVEL=DEBUG
export OPENAI_API_KEY=your-key-here  # Optional for AI features
export METRICS_PORT=8000

# 3. Run locally
make run
```

### Docker Deployment

```bash
# Build the image
make docker-build

# Run in container
make docker-run
```

**Note**: The operator automatically installs required CRDs (Custom Resource Definitions) on startup using `kubecrd`. **No manual CRD application needed** - the `crd.yaml` files have been removed since kubecrd handles CRD installation programmatically!

### Production Deployment

```bash
# Build for production
make prod-build

# Deploy to Kubernetes
kubectl apply -f kubernetes-manifests/
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `METRICS_PORT` | `8000` | Port for Prometheus metrics server |
| `OPENAI_API_KEY` | None | OpenAI API key for AI-powered decisions (optional) |

### Custom Resource Definition

The operator works with `IntelligentScaling` CRDs:

```yaml
apiVersion: nimbusguard.io/v1alpha1
kind: IntelligentScaling
metadata:
  name: example-scaling
  namespace: default
spec:
  namespace: default
  target_labels:
    app: my-app
  min_replicas: 1
  max_replicas: 10
  metrics_config:
    prometheus_url: "http://prometheus:9090"
    metrics:
      - query: "rate(http_requests_total[5m])"
        threshold: 10
        condition: "gt"
```

## Monitoring and Observability

### Endpoints

- **Metrics**: `http://localhost:8000/metrics` - Prometheus metrics
- **Health**: `http://localhost:8080/healthz` - Health check endpoint
- **Ready**: `http://localhost:8080/ready` - Readiness check endpoint

### Available Metrics

- `nimbusguard_scaling_operations_total`: Total scaling operations by type and namespace
- `nimbusguard_current_replicas`: Current replicas per deployment
- `nimbusguard_decisions_total`: Total decisions made by action type
- `nimbusguard_operator_health`: Health status per component (0=unhealthy, 1=healthy)

### Health Components

The operator monitors health of these components:
- `prometheus`: Prometheus connectivity
- `kubernetes`: Kubernetes API connectivity
- `openai`: OpenAI API availability (if configured)
- `decision_engine`: Decision engine functionality

## Development

### Available Make Commands

```bash
make help              # Show all available commands
make dev-install       # Install in development mode
make run              # Run operator locally
make docker-build     # Build Docker image
make docker-run       # Run in Docker container
make clean            # Clean build artifacts
make setup            # Full setup from scratch
make check-deps       # Check if required tools are installed
```

### Adding New Components

#### Adding a New Client

1. Create `nimbusguard/clients/myclient.py`:
```python
class MyClient:
    async def get_data(self):
        # Implementation
        pass
```

2. Import and use in `operator.py`

#### Adding a New Engine

1. Create `nimbusguard/engines/myengine.py`:
```python
class MyEngine:
    def process(self, data):
        # Implementation
        pass
```

2. Integrate with decision-making process

#### Adding New Metrics

1. Add to `nimbusguard/metrics.py`:
```python
MY_METRIC = Counter('my_metric_total', 'Description', ['label'])
```

2. Use in your components:
```python
from nimbusguard.metrics import MY_METRIC
MY_METRIC.labels(label='value').inc()
```

### Testing

Currently, the project structure is set up for testing but tests are not yet implemented. To add tests:

1. Create a `tests/` directory
2. Add test files like `test_operator.py`, `test_decision_engine.py`
3. Install pytest: `pip install pytest`
4. Run tests: `pytest tests/`

### Code Quality

For code quality improvements:

```bash
# Install tools
pip install black flake8 isort

# Format code
black nimbusguard/
isort nimbusguard/

# Lint code
flake8 nimbusguard/
```

## Troubleshooting

### Common Issues

#### Docker Build Fails
```bash
# Clean and rebuild
make clean
make docker-build
```

#### Permission Errors
Ensure your user has access to:
- Docker daemon
- Kubernetes cluster
- Write permissions in the project directory

#### Import Errors
```bash
# Ensure package is installed
make dev-install

# Check Python path
export PYTHONPATH=/path/to/nimbusguard-operator:$PYTHONPATH
```

#### Kubernetes Connection Issues
```bash
# Check kubeconfig
kubectl cluster-info

# Verify CRD exists
kubectl get crd intelligentscaling.nimbusguard.io
```

### Debug Mode

Run with debug logging:
```bash
export LOG_LEVEL=DEBUG
make run
```

### Health Check

Check operator health:
```bash
# Local
curl http://localhost:8080/healthz

# In Kubernetes
kubectl get pods -l app=nimbusguard-operator
kubectl logs -l app=nimbusguard-operator
```

## Migration from Monolithic Version

See [MIGRATION.md](MIGRATION.md) for detailed migration instructions from the original monolithic `nimbusguard_operator.py`.

## Features

- **🏗️ Modular Architecture**: Clean separation of concerns with dedicated modules for different responsibilities
- **📊 Comprehensive Monitoring**: Prometheus metrics and health checks
- **🔄 Intelligent Scaling**: AI-powered decision making (when configured)
- **🛡️ Robust Error Handling**: Component-level health tracking
- **📦 Easy Deployment**: Docker and Kubernetes ready
- **🔧 Extensible Design**: Simple to add new components
- **⚡ kubecrd Integration**: CRDs defined as Python dataclasses with automatic installation
- **🎯 Type Safety**: Full type checking and validation for all resources

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests (when testing framework is set up)
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs with `LOG_LEVEL=DEBUG`
3. Check health endpoints
4. Open an issue in the repository
