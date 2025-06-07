# NimbusGuard: AI-Powered Kubernetes Resilience Platform

NimbusGuard is an AI-powered Kubernetes resilience platform that delivers proactive scaling and autonomous recovery using reinforcement learning and LangGraph workflows.

## 🚀 Quick Start

### Prerequisites
- Docker Desktop with Kubernetes enabled
- Python 3.11+
- kubectl configured for your local cluster
- OpenAI API key (for AI operator)

### One-Command Setup

```bash
# Clone the repository
git clone https://github.com/chamathwanigasooriya/nimbusguard.git
cd nimbusguard

# Start complete Kubernetes development environment
make k8s-dev
```

This single command will:
- Build all Docker images with optimized caching
- Deploy all services to Kubernetes
- Set up OpenAI API key for the AI operator
- Configure port forwarding for all services
- Provide access URLs for all components

## 📋 Available Commands

### Development Commands
```bash
make help          # Show all available commands
make dev           # Docker development environment
make k8s-dev       # Kubernetes development (recommended)
```

### Monitoring Commands
```bash
make status        # Show system status with health checks
make forward       # Setup persistent port forwarding only
```

### Cleanup Commands
```bash
make clean         # Clean Docker environment
make k8s-clean     # Clean Kubernetes environment
```

## 🌐 Service Access Points

After running `make k8s-dev` or `make status`, access services at:

- **📊 Grafana**: http://localhost:3000 (admin/nimbusguard)
- **🎯 Consumer Workload**: http://localhost:8080
- **🚀 Load Generator**: http://localhost:8081
- **🧠 AI Operator**: http://localhost:8082
- **📈 Prometheus**: http://localhost:9090

## 🧪 Testing Scaling Scenarios

### 1. Generate CPU Workload
```bash
curl -X POST "http://localhost:8080/api/v1/workload/cpu/start" \
  -H "Content-Type: application/json" \
  -d '{"intensity": 80, "duration": 300}'
```

### 2. Trigger AI Scaling Decision
```bash
curl -X POST "http://localhost:8080/api/v1/events/trigger" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "high_cpu_usage",
    "service": "consumer-workload",
    "value": 90
  }'
```

### 3. Generate Load Patterns
```bash
curl -X POST "http://localhost:8081/load/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "pattern": "spike",
    "duration": 300,
    "target": "http",
    "intensity": 75
  }'
```

### 4. Monitor AI Decisions
```bash
# Watch AI operator logs
kubectl logs -f deployment/langgraph-operator -n nimbusguard

# Check system status
make status
```

## 🏗️ Architecture Overview

NimbusGuard transforms traditional reactive Kubernetes scaling into intelligent, proactive resource management by:

- Learning from workload patterns using Q-learning reinforcement learning
- Predicting resource needs before demand spikes occur
- Making autonomous scaling decisions based on real-time cluster state
- Providing self-healing capabilities for scaling failures and anomalies
- Delivering comprehensive observability with full tracing and metrics

### Core Components

- **Kafka**: Event streaming platform for real-time data pipelines
- **Load Generator**: Configurable service that generates realistic load patterns
- **Consumer Workload**: FastAPI application that generates resource workloads and consumes scaling events
- **LangGraph Operator**: Custom Kubernetes operator orchestrating multi-agent AI system
- **KEDA**: Event-driven autoscaler for Kubernetes workloads

## 🤖 AI Agents Implementation

### State Observer Agent
- Collects real-time cluster metrics via Prometheus
- Monitors pod CPU, memory, network I/O, and request rates
- Provides normalized state vectors for the Q-learning agent

### Decision Agent (Q-Learning)
- Implements epsilon-greedy Q-learning with learning rate 0.1
- State space: CPU utilization, memory utilization, request rate, pod count
- Action space: scale up/down (1-10 replicas), no action
- Reward function: Optimizes for resource efficiency and SLA compliance

### Action Executor Agent
- Executes scaling decisions via Kubernetes API
- Updates deployment replicas and HPA configurations
- Records action outcomes for reward calculation

### Reward Calculator Agent
- Calculates rewards based on resource efficiency, SLA compliance, and stability
- Provides feedback for Q-learning policy updates
- Tracks scaling effectiveness over time windows

## 📊 Observability Features

### Real-Time Monitoring
- **System Health**: All services monitored with health checks
- **AI Decisions**: Real-time tracking of AI scaling decisions
- **Performance Metrics**: Resource utilization, response times, throughput
- **Learning Progress**: Q-learning performance and model accuracy

### Dashboards & Alerts
- **Grafana Dashboards**: Pre-configured dashboards for system overview
- **Prometheus Metrics**: Comprehensive metrics collection
- **Distributed Tracing**: End-to-end request tracing with Tempo
- **Log Aggregation**: Centralized logging with Loki

## 🛠️ Development Features

### Optimized Build System
- **Base Image Caching**: Shared dependencies for 4x faster builds
- **Zero Bandwidth Waste**: Eliminates repeated package downloads
- **Parallel Builds**: Concurrent image building for speed

### Simplified Workflow
- **One-Command Setup**: Complete environment with `make k8s-dev`
- **Health Monitoring**: Built-in health checks for all services
- **Port Management**: Automatic port forwarding with conflict resolution

## 📁 Project Structure

```
nimbusguard/
├── Makefile                             # Simplified development commands
├── docker-compose.yml                   # Docker development setup
├── docker/
│   └── base.Dockerfile                  # Optimized base image for fast builds
│
├── src/
│   ├── consumer-workload/               # FastAPI workload and event consumer
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── main.py                      # FastAPI server entry point
│   │   └── api/                         # REST API endpoints
│   │
│   ├── load-generator/                  # Load generation service
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── main.py                      # Load generation logic
│   │
│   └── langgraph-operator/              # AI-powered Kubernetes operator
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── nimbusguard_operator.py      # Main operator controller
│       ├── agents/                      # AI agent implementations
│       ├── ml_models/                   # Machine learning models
│       ├── workflows/                   # LangGraph workflow definitions
│       └── mlflow_integration/          # MLflow experiment tracking
│
├── kubernetes-manifests/                # Kubernetes deployment manifests
│   ├── base/                           # Core platform components
│   └── components/                     # Additional components
│       ├── observability/              # Prometheus, Grafana, Tempo, Loki
│       └── keda/                       # KEDA autoscaling
│
└── monitoring/                         # Monitoring configurations
    ├── dashboards/                     # Grafana dashboards
    └── alerts/                         # Alerting rules
```

## ⚙️ Configuration

### Q-Learning Parameters
```yaml
learning_rate: 0.1
discount_factor: 0.95
epsilon_start: 1.0
epsilon_end: 0.01
epsilon_decay: 0.995
```

### Scaling Parameters
```yaml
min_replicas: 1
max_replicas: 50
cooldown_period: 300
target_cpu_utilization: 70
```

## 🔧 Technology Stack

### Core Technologies
- **Python 3.11**: Primary development language
- **FastAPI**: High-performance web framework
- **LangGraph**: Multi-agent AI workflow orchestration
- **Kubernetes**: Container orchestration platform
- **Docker**: Containerization and development

### AI/ML Libraries
- **NumPy & Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning utilities
- **OpenAI API**: Large language model integration
- **Kopf**: Kubernetes operator framework

### Event Streaming
- **Apache Kafka**: Event streaming platform
- **kafka-python**: Python Kafka client

### Observability Stack
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **Tempo**: Distributed tracing
- **Loki**: Log aggregation
- **OpenTelemetry**: Observability instrumentation

### Infrastructure
- **KEDA**: Event-driven autoscaling
- **Kubernetes HPA**: Horizontal Pod Autoscaler
- **MLflow**: ML experiment tracking and model registry

## 🔍 Troubleshooting

### Common Issues

#### Port Forward Failures
```bash
# Check if ports are in use
make status

# Restart port forwarding
pkill -f "kubectl port-forward.*nimbusguard"
make forward
```

#### Pod Not Ready
```bash
# Check pod status
kubectl get pods -n nimbusguard

# Check pod logs
kubectl logs -n nimbusguard <pod-name>

# Restart deployment
kubectl rollout restart deployment/<deployment-name> -n nimbusguard
```

#### OpenAI API Issues
```bash
# Update OpenAI API key
make k8s-dev  # Will prompt for new key

# Check operator logs
kubectl logs -f deployment/langgraph-operator -n nimbusguard
```

### Getting Help
```bash
make help          # Show all available commands
make status        # Check system health
kubectl get pods -n nimbusguard  # Check pod status
```

## 🎯 System Status

The system is fully operational with the following features:

### ✅ Working Features
- **One-Command Setup**: `make k8s-dev` deploys entire system
- **Health Monitoring**: All services have working health checks
- **Port Forwarding**: Automatic port management with conflict resolution
- **AI Operator**: Fully functional with OpenAI integration
- **Observability**: Complete monitoring stack with Prometheus, Grafana, Tempo, Loki
- **Build Optimization**: 4x faster builds with zero bandwidth waste

### 🔧 Development Workflow
1. **Start**: `make k8s-dev` - Complete environment setup
2. **Monitor**: `make status` - Check system health
3. **Test**: Use provided curl commands to test scaling scenarios
4. **Debug**: `kubectl logs -f deployment/langgraph-operator -n nimbusguard`
5. **Cleanup**: `make k8s-clean` - Remove all resources

### 📈 Performance Metrics
- **Build Time**: ~4 seconds (with base image cache)
- **Startup Time**: ~30 seconds for complete system
- **Health Check**: All services respond in <1 second
- **Port Forwards**: 5 services automatically configured

## 📄 License

This project is licensed under the MIT License.

## 🔗 Related Projects

- [KEDA](https://keda.sh/) - Kubernetes Event-driven Autoscaling
- [LangGraph](https://github.com/langchain-ai/langgraph) - Multi-agent AI workflows
- [Prometheus](https://prometheus.io/) - Monitoring and alerting
- [MLflow](https://mlflow.org/) - ML lifecycle management
- [OpenTelemetry](https://opentelemetry.io/) - Observability framework