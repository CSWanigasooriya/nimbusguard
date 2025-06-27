# NimbusGuard - Kubernetes Event-Driven Autoscaling

A cloud-native application demonstrating KEDA-based autoscaling with comprehensive monitoring.

## 🚀 Quick Start

### Prerequisites
- Docker Desktop with Kubernetes enabled
- `make`, `helm`, `kubectl` installed

### Development Setup

**One-command deployment:**
```bash
make dev
```

This command will:
1. 🔨 Build all Docker images
2. 🔍 Check if KEDA is installed (auto-install if missing)
3. 🚀 Deploy the complete stack to your cluster

**First-time setup (install tools):**
```bash
make setup  # Install/update all required tools
make dev    # Deploy the application
```

### What Gets Deployed

- **Consumer Service**: FastAPI application with load simulation endpoints
- **KEDA Autoscaling**: Event-driven autoscaling based on Prometheus metrics  
- **Monitoring Stack**: Prometheus, Grafana, Loki, Tempo, Alloy
- **Observability**: Beyla for automatic instrumentation

### Access Services

```bash
make forward    # Port forward all services
```

- **Consumer**: http://localhost:8000
- **Prometheus**: http://localhost:9090  
- **Grafana**: http://localhost:3000 (admin/admin)

### Load Testing

```bash
make load-test-light    # Quick validation
make load-test-medium   # Moderate scaling  
make load-test-heavy    # Immediate scaling
make load-status        # Check autoscaling status
```

## 🎯 Key Features

- **Auto-KEDA Installation**: No manual setup required
- **Intelligent Autoscaling**: CPU, memory, and HTTP metrics-based scaling
- **Full Observability**: Traces, metrics, and logs collection
- **Load Testing Suite**: Multiple test scenarios for validation
- **Production Ready**: Separate overlays for dev/prod environments

## 📋 Available Commands

```bash
make help           # Show all available commands
make dev            # Build and deploy to development  
make clean          # Delete all resources
make keda-install   # Manually install KEDA
make load-status    # View autoscaling status
```

## 🔧 Architecture

The application uses **KEDA ScaledObjects** to automatically scale the consumer service based on:
- HTTP request rate from Prometheus
- Average request duration 
- Python garbage collection metrics

**Monitoring pipeline**: Beyla → Alloy → Prometheus → KEDA → HPA → Kubernetes

---

*Ready to scale! 🚀* 