# NimbusGuard: AI-Powered Kubernetes Resilience Platform

NimbusGuard is an AI-powered Kubernetes resilience platform that delivers proactive scaling and autonomous recovery using reinforcement learning and LangGraph workflows.

## Project Overview

NimbusGuard transforms traditional reactive Kubernetes scaling into intelligent, proactive resource management by:

- Learning from workload patterns using Q-learning reinforcement learning
- Predicting resource needs before demand spikes occur using LSTM models
- Making autonomous scaling decisions based on real-time cluster state
- Providing self-healing capabilities for scaling failures and anomalies
- Delivering comprehensive observability with full tracing and metrics

## Architecture Components

### Event-Driven Infrastructure

- **Kafka**: Event streaming platform for real-time data pipelines.
- **Load Generator**: Configurable service that generates realistic load. It can send HTTP traffic to the Consumer Workload App or trigger scaling decisions by sending events via REST or Kafka.
- **Consumer Workload Application**: A unified FastAPI application that both generates resource workloads (CPU/memory) and consumes scaling events from Kafka topics or a REST endpoint to trigger the AI decision engine.
- **KEDA**: Event-driven autoscaler that can scale the Consumer Workload App based on Kafka queue length or other metrics.

### Core AI Platform

- **LangGraph Operator**: Custom Kubernetes operator orchestrating a multi-agent AI system for decision-making.
- **Q-Learning Engine**: Reinforcement learning for dynamic scaling decisions with an epsilon-greedy exploration strategy.
- **LSTM Predictor**: Time series forecasting for proactive resource management.
- **Multi-Agent System**: Comprises a State Observer, Decision Agent, Action Executor, and Reward Calculator.

## Project Structure

```
nimbusguard/
├── README.md
├── .gitignore
├── .dockerignore
├── Makefile
├── docker-compose.yml
├── skaffold.yaml
│
├── src/
│   ├── consumer-workload/               # Unified FastAPI workload and event consumer app
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── main.py                      # FastAPI server entry point
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── workload_endpoints.py    # /workload/cpu, /memory
│   │   │   ├── event_endpoints.py       # /events/trigger for REST-based scaling
│   │   │   ├── metrics_endpoints.py     # /metrics for Prometheus
│   │   │   └── health_endpoints.py      # /health, /ready
│   │   ├── consumers/                   # Kafka consumer logic
│   │   │   ├── __init__.py
│   │   │   └── scaling_event_consumer.py # Consumes scaling events from Kafka
│   │   ├── workload_generators/
│   │   │   ├── __init__.py
│   │   │   ├── cpu_workload.py
│   │   │   └── memory_workload.py
│   │   ├── instrumentation/
│   │   │   ├── __init__.py
│   │   │   ├── otel_config.py
│   │   │   └── metrics_collector.py
│   │   └── config/
│   │       └── app_config.yaml
│   │
│   ├── load-generator/                  # Event-driven load generation service
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── main.py
│   │   ├── generators/
│   │   │   ├── http_load_generator.py
│   │   │   └── kafka_event_generator.py
│   │   │  
│   │   └── config/
│   │       └── load_patterns.yaml
│   │
│   └── langgraph-operator/              # AI-powered Kubernetes operator
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── operator.py                  # Main operator controller
│       ├── agents/                      # AI agent implementations
│       │   ├── __init__.py
│       │   ├── state_observer.py        # Cluster state observation agent
│       │   ├── decision_agent.py        # Q-learning scaling decisions
│       │   ├── action_executor.py       # Kubernetes scaling actions
│       │   └── reward_calculator.py     # Reward calculation for RL
│       ├── ml_models/                   # Machine learning models
│       │   ├── __init__.py
│       │   ├── q_learning.py            # Q-learning implementation
│       │   ├── lstm_predictor.py        # LSTM time series prediction
│       │   └── anomaly_detector.py      # Statistical anomaly detection
│       ├── workflows/                   # LangGraph workflow definitions
│       │   ├── __init__.py
│       │   ├── scaling_workflow.py      # Main scaling decision workflow
│       │   ├── monitoring_workflow.py   # Continuous monitoring workflow
│       │   └── recovery_workflow.py     # Autonomous recovery workflow
│       ├── mcp_integration/             # MCP server integration
│       │   ├── __init__.py
│       │   ├── mcp_manager.py           # MCP connection management
│       │   ├── kubernetes_client.py     # mcp-server-kubernetes integration
│       │   └── prometheus_client.py     # mcp-server-prometheus integration
│       ├── instrumentation/
│       │   ├── __init__.py
│       │   ├── operator_metrics.py      # Operator-specific metrics
│       │   ├── decision_tracing.py      # AI decision tracing
│       │   └── learning_metrics.py      # ML training metrics
│       └── config/
│           ├── mcp_config.yaml          # MCP server configurations
│           ├── agent_config.yaml        # AI agent settings
│           └── ml_config.yaml           # ML model hyperparameters
│
├── kubernetes-manifests/                # Kubernetes deployment manifests
│   ├── base/                           # Base Kubernetes manifests
│   │   ├── kustomization.yaml
│   │   ├── namespace.yaml              # nimbusguard namespace
│   │   ├── kafka.yaml                  # Kafka broker deployment + service
│   │   ├── zookeeper.yaml              # Zookeeper for Kafka
│   │   ├── consumer-workload.yaml      # Merged workload + consumer deployment
│   │   ├── load-generator.yaml         # Load generator deployment
│   │   ├── langgraph-operator.yaml     # LangGraph operator deployment
│   │   ├── mcp-kubernetes.yaml         # MCP Kubernetes server
│   │   ├── mcp-prometheus.yaml         # MCP Prometheus server
│   │   └── rbac.yaml                   # RBAC permissions
│   └── components/                     # Additional components
│       ├── observability/              # Observability stack
│       │   ├── kustomization.yaml
│       │   ├── prometheus.yaml
│       │   ├── grafana.yaml
│       │   ├── tempo.yaml
│       │   ├── loki.yaml
│       │   └── opentelemetry.yaml
│       └── keda/                       # KEDA autoscaling
│           ├── kustomization.yaml
│           ├── keda-operator.yaml
│           └── scaled-objects.yaml
│
├── monitoring/                         # Monitoring configurations
│   ├── alerts/
│   │   ├── nimbusguard-alerts.yaml     # Custom alerting rules
│   │   ├── scaling-alerts.yaml         # Scaling-specific alerts
│   │   └── ai-model-alerts.yaml        # AI model performance alerts
│   ├── dashboards/                     # Grafana dashboard JSON files
│   │   ├── nimbusguard-overview.json   # Main overview dashboard
│   │   ├── ai-decision-tracking.json   # AI decision analysis
│   │   ├── workload-performance.json   # Workload metrics dashboard
│   │   └── learning-metrics.json       # ML model performance
│   └── queries/                        # PromQL query examples
│       ├── scaling-metrics.promql      # Scaling-related queries
│       ├── ai-performance.promql       # AI performance queries
│       └── workload-analysis.promql    # Workload analysis queries
│
├── scripts/                            # Utility and deployment scripts
│   ├── setup/
│   │   ├── setup-local-cluster.sh      # Kind cluster setup
│   │   ├── verify-cluster-context.sh   # Verify kubectl context
│   │   ├── install-dependencies.sh     # Install required components
│   │   ├── deploy-observability.sh     # Deploy monitoring stack
│   │   └── setup-development.sh        # Complete dev environment
│   ├── operations/
│   │   ├── deploy-nimbusguard.sh       # Deploy main platform
│   │   ├── deploy-argocd.sh            # Deploy ArgoCD for GitOps (Phase 2)
│   │   └── health-check.sh             # System health validation
│   └── utilities/
│       ├── port-forward.sh             # Port forwarding for development
│       ├── logs.sh                     # Centralized log collection
│       └── cleanup.sh                  # Resource cleanup
│
├── ml-training/                        # Machine learning pipeline
│   ├── data/
│   │   ├── training-data/              # Historical training datasets
│   │   └── model-artifacts/            # Trained model files
│   ├── notebooks/                      # Jupyter analysis notebooks
│   │   ├── data-exploration.ipynb
│   │   ├── q-learning-analysis.ipynb
│   │   └── lstm-development.ipynb
│   └── training/
│       ├── train-q-learning.py
│       └── train-lstm.py
│
├── tests/                              # Test suite
│   ├── unit/
│   │   ├── test_consumer_workload.py
│   │   └── test_q_learning.py
│   ├── integration/
│   │   ├── test_scaling_workflow.py
│   │   └── test_ai_agents.py
│   └── e2e/
│       └── test_complete_scaling.py
│
└── .github/                            # CI/CD workflows
    └── workflows/
        ├── ci.yaml                     # Continuous Integration
        ├── cd.yaml                     # Continuous Deployment
        └── ml-training.yaml            # ML model training pipeline
```

## Additional ML Platform Integration

### MLflow Integration (Phase 1) - Add to src/langgraph-operator/mlflow_integration/
```
├── __init__.py
├── experiment_tracker.py    # MLflow experiment management
├── model_registry.py        # Model versioning and registry
└── artifact_logger.py       # Log training artifacts and metrics
```

### ArgoCD & Kubeflow Components (Phase 2/3) - Add to kubernetes-manifests/components/argocd/
```
├── kustomization.yaml
├── argocd-server.yaml
└── argocd-application.yaml
```

### kubeflow/ (Phase 3 - Optional)
```
├── kustomization.yaml
└── kubeflow-pipelines.yaml
```

## Development Roadmap & Implementation Phases

The project is designed to be implemented in three distinct phases, starting with a foundational AI system and progressively adding advanced capabilities.

### Phase 1: Foundation & Core AI (Current Priority)

This phase focuses on building the essential components for a functional AI-driven scaling system.

#### Infrastructure Setup:
✅ Deploy a baseline observability stack (Prometheus, Grafana).
✅ Install KEDA for event-driven autoscaling.
✅ Configure Kafka and Zookeeper for event streaming.
✅ Set up basic Kubernetes RBAC and custom resource definitions.

#### Core Application Development:
🔥 Unified Consumer Workload App: Develop the FastAPI application that both serves a resource-intensive workload and consumes scaling events via REST and Kafka.
🔥 Load Generator Service: Create a service to generate configurable HTTP load and trigger scaling events.

#### AI & Operator Development:
🔥 LangGraph Operator: Build the initial Kubernetes operator.
🔥 Basic Q-Learning Agent: Implement a Q-learning agent with a simple state-action space for decision-making.
🔥 Core Agent Workflow: Create a single-agent LangGraph workflow (State Observer -> Decision Agent -> Action Executor).

#### ML Lifecycle:
🔥 MLflow Integration: Integrate MLflow for experiment tracking, model registry, and artifact logging.

### Phase 2: Advanced AI & GitOps Automation (Future Work)

This phase enhances the AI's intelligence, predictive power, and automates the deployment lifecycle using GitOps.

#### GitOps & Deployment Automation:
🔮 ArgoCD Integration: Implement GitOps for continuous deployment of applications and AI model configurations.
🔮 Automated Model Deployment: Build CI/CD pipelines in GitHub Actions to automatically train and deploy new models via ArgoCD.

#### Advanced AI Models:
🔮 Deep Q-Networks (DQN): Upgrade the reinforcement learning model from tabular Q-learning to a neural network-based DQN for handling more complex state spaces.
🔮 LSTM Time Series Prediction: Develop and integrate a 3-layer LSTM model for proactive, 5-minute-ahead resource prediction.
🔮 Advanced Anomaly Detection: Implement statistical and ML-based outlier detection to identify and react to unusual workload patterns.

#### Enhanced Intelligence & Workflows:
🔮 Predictive Scaling: Combine LSTM predictions with Q-learning decisions for a hybrid proactive-reactive scaling strategy.
🔮 Multi-Agent Coordination: Evolve the LangGraph workflow to support sophisticated communication and decision-making between multiple specialized agents.

### Phase 3: Production Hardening & Enterprise Scale (Long-Term Vision)

This phase focuses on making the platform robust, self-sufficient, and ready for production-grade, enterprise environments.

#### Enterprise & Production Features:
🔮 Self-Healing Workflows: Design autonomous recovery workflows in LangGraph to handle scaling failures, pod crashes, or infrastructure anomalies.
🔮 Policy Adaptation & Auto-Tuning: Introduce real-time hyperparameter tuning for the AI models based on performance feedback.
🔮 Multi-Cluster Federation: Extend the operator's capability to manage and learn from workloads across multiple Kubernetes clusters.

#### Advanced MLOps (Optional Upgrade):
🔮 Kubeflow Migration: Optionally migrate the ML lifecycle management from MLflow to Kubeflow to leverage its powerful pipeline orchestration and distributed training capabilities for enterprise-scale needs.

#### Research & Validation:
🔮 Performance Benchmarking: Conduct rigorous A/B testing to compare the performance and cost-efficiency of NimbusGuard against traditional HPA/VPA.
🔮 Publish Findings: Author and publish a research paper detailing the novel reinforcement learning approach to Kubernetes autoscaling.

**Legend:**
✅ = Existing infrastructure to leverage.
🔥 = Current phase custom implementation (novel research).
🔮 = Future phase implementation.

## Quick Start

### Prerequisites
- Docker Desktop with Kubernetes enabled
- Python 3.11+
- kubectl configured for your local cluster
- Git & Helm 3.x

### Setup and Deployment
```bash
# Clone the repository
git clone https://github.com/chamathwanigasooriya/nimbusguard.git
cd nimbusguard

# Deploy with hot-reload for development
skaffold dev
```

### Accessing Services
```bash
# Port forward all services
./scripts/utilities/port-forward.sh

# Access points:
# - Grafana: http://localhost:3000
# - Prometheus: http://localhost:9090
# - Consumer Workload App: http://localhost:8080
# - Load Generator: http://localhost:8081
# - MLflow UI: http://localhost:5000
```

### Testing Scaling Scenarios
```bash
# 1. Generate a direct CPU workload on the consumer app
curl -X POST "http://localhost:8080/workload/cpu?intensity=80&duration=300"

# 2. Trigger a scaling event via the consumer app's REST endpoint
curl -X POST "http://localhost:8080/events/trigger" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "high_cpu_usage",
    "service": "consumer-workload",
    "value": 90
  }'

# 3. Use the Load Generator to create a sustained event stream (e.g., to Kafka)
curl -X POST "http://localhost:8081/load/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "pattern": "spike",
    "duration": 300,
    "target": "kafka"
  }'

# Monitor the AI operator's decisions
kubectl logs -f deployment/langgraph-operator -n nimbusguard
```

## AI Agents Implementation

### State Observer Agent
- Collects real-time cluster metrics via MCP Prometheus server.
- Monitors pod CPU, memory, network I/O, and request rates.
- Aggregates node-level resource availability and utilization.
- Provides normalized state vectors for the Q-learning agent.

### Decision Agent (Q-Learning)
- Implements epsilon-greedy Q-learning with learning rate 0.1.
- State space: CPU utilization, memory utilization, request rate, pod count.
- Action space: scale up (1-10 replicas), scale down (1-10 replicas), no action.
- Reward function: -1 for resource waste, +10 for optimal utilization, -5 for SLA violations.

### LSTM Predictor Agent
- Uses 3-layer LSTM with 128 hidden units per layer.
- Predicts CPU and memory utilization 5 minutes ahead.
- Trained on historical metrics with a 60-minute sliding window.
- Provides proactive scaling recommendations to Decision Agent.

### Action Executor Agent
- Executes scaling decisions via Kubernetes API.
- Updates KEDA ScaledObjects for event-driven scaling.
- Modifies HPA configurations for traditional horizontal scaling.
- Records action outcomes for reward calculation.

### Reward Calculator Agent
- Calculates rewards based on: resource efficiency (40%), SLA compliance (40%), stability (20%).
- Tracks scaling effectiveness over 10-minute windows.
- Provides feedback for Q-learning policy updates.
- Logs reward signals for performance analysis.

## Observability Features

### Metrics Collection
- Workload Metrics: CPU/memory usage, request latency, throughput.
- AI Decision Metrics: Decision confidence, action success rate, learning progress.
- Scaling Metrics: Scaling events, replica changes, resource utilization.
- System Metrics: Overall platform health, error rates, performance.

### Tracing
- End-to-end tracing of scaling decisions through LangGraph workflows.
- AI agent interaction spans with decision rationale.
- MCP server communication tracing.
- Performance bottleneck identification.

### Dashboards
- NimbusGuard Overview: Platform status, scaling events, resource utilization.
- AI Decision Tracking: Real-time AI decisions, confidence scores, learning curves.
- Workload Performance: Application metrics, response times, error rates.
- Learning Metrics: Q-learning performance, LSTM accuracy, exploration rates.

## Development Commands

```bash
# Development workflow
make dev                # Start development with Skaffold
make test               # Run full test suite
make lint               # Run code linting and formatting
make build              # Build Docker images locally

# Deployment
make deploy             # Deploy base platform
make deploy-full        # Deploy with observability and KEDA
make clean              # Clean up all resources

# Monitoring
make logs               # Tail application logs
make metrics            # Open Prometheus metrics
make dashboard          # Open Grafana dashboards
make health             # Check system health

# AI/ML Development
make train-models       # Train Q-learning and LSTM models
make validate-models    # Run model validation tests
make export-data        # Export training data for analysis

# MLflow Operations (Phase 1)
make deploy-mlflow      # Deploy MLflow tracking server
make mlflow-ui          # Open MLflow web UI
make track-experiment   # Start experiment tracking

# ArgoCD Operations (Phase 2)
make deploy-argocd      # Deploy ArgoCD server
make argocd-ui          # Open ArgoCD dashboard
```

## Event-Driven Architecture Flow

With the unified application, the event flow is more direct:

```
Load Generator → (REST API / Kafka Events) → Consumer Workload App → AI Decision Engine
       ↓                                           ↓                    ↓
  HTTP Requests -----------------------------------→ (Generates Metrics)    ↓
                                                     ↓                    ↓
                                                 Prometheus →    MCP    → State Observer
```

### Event Flow Steps:
1. The Load Generator produces events. It can send HTTP traffic directly to the Consumer Workload App to create resource pressure, or it can send a scaling trigger event via REST or Kafka.
2. The Consumer Workload App receives the trigger event on its /events/trigger endpoint or via its integrated Kafka consumer.
3. Upon receiving an event, the app notifies the LangGraph Operator to initiate a scaling analysis.
4. The AI agents within the operator observe cluster metrics via Prometheus, make a decision, and execute a scaling action.
5. Metrics from the action's outcome are fed back into the system, allowing the AI to learn continuously.

## Technology Stack

### ML Platform Choice: MLflow vs Kubeflow

#### Primary Choice: MLflow (Phase 1)
✅ Lightweight: Minimal infrastructure overhead, perfect for research projects.
✅ Easy Setup: Simple deployment and management.
✅ Research Focused: Excellent for experimentation and model development.
✅ Local Development: Works well with Docker Desktop and Kind clusters.

#### Optional Upgrade: Kubeflow (Phase 3)
🔮 Enterprise Scale: Complex ML workflows for production environments.
🔮 Distributed Training: Multi-node training for large models.
🔮 Advanced Pipelines: Sophisticated ML orchestration.

**Recommendation:** Start with MLflow for development and research, optionally migrate to Kubeflow if enterprise-scale features are needed.

### Core Technologies
- Python 3.11
- FastAPI
- LangGraph
- Kubernetes 1.28+
- Docker

### Event Streaming & Messaging
- Apache Kafka
- kafka-python

### AI/ML Libraries
- NumPy & Pandas
- Scikit-learn
- TensorFlow 2.x (Phase 2)
- Gymnasium

### ML Platform & Lifecycle Management
- MLflow (Primary Choice)
- Jupyter
- Kubeflow (Optional Upgrade)

### GitOps & Deployment
- ArgoCD (Phase 2)
- Helm
- Kustomize

### Observability
- Prometheus, Grafana, Tempo, Loki, OpenTelemetry

### Infrastructure
- KEDA, Kubernetes HPA, MCP

## Configuration

### Q-Learning Parameters
```
learning_rate: 0.1
discount_factor: 0.95
epsilon_start: 1.0
epsilon_end: 0.01
epsilon_decay: 0.995
```

### LSTM Model Configuration
```
sequence_length: 60
hidden_units: 128
num_layers: 3
dropout_rate: 0.2
```

### Scaling Parameters
```
min_replicas: 1
max_replicas: 50
cooldown_period: 300
```

## License

This project is licensed under the MIT License.

## Related Projects

- [KEDA](https://keda.sh/)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Model Context Protocol](https://github.com/modelcontextprotocol/mcp)
- [Prometheus](https://prometheus.io/)
- [MLflow](https://mlflow.org/)
- [ArgoCD](https://argoproj.github.io/cd/)
- [Kubeflow](https://www.kubeflow.org/)