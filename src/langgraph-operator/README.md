# NimbusGuard LangGraph Operator

AI-powered Kubernetes operator for intelligent autoscaling using LangGraph workflows and Q-learning reinforcement learning.

## Overview

The NimbusGuard operator is designed to run **inside a Kubernetes cluster**, not in docker-compose. It monitors deployments and makes intelligent scaling decisions using AI agents and Q-learning.

## Features

- 🤖 AI-powered scaling decisions using LangGraph workflows
- 🧠 Q-learning reinforcement learning for optimization
- 📊 Prometheus metrics endpoint at `/metrics`
- 🔍 Comprehensive observability and logging
- ⚕️ Health checks at `/healthz` and `/readyz`

## Prerequisites

- Kubernetes cluster (>= v1.20)
- kubectl configured to access your cluster
- OpenAI API key for AI agents

## Building the Image

```bash
# Build the operator image
cd src/langgraph-operator
docker build -t nimbusguard-operator:latest .

# Tag for your registry (replace with your registry)
docker tag nimbusguard-operator:latest your-registry/nimbusguard-operator:latest
docker push your-registry/nimbusguard-operator:latest
```

## Deployment

### 1. Create Namespace

```bash
kubectl create namespace nimbusguard
```

### 2. Configure OpenAI API Key

```bash
kubectl create secret generic openai-secret \
  --from-literal=OPENAI_API_KEY=your-openai-api-key-here \
  -n nimbusguard
```

### 3. Apply RBAC and Deployment

```bash
# Apply RBAC permissions
kubectl apply -f config/rbac.yaml

# Apply Custom Resource Definitions
kubectl apply -f config/crd.yaml

# Apply the operator deployment
kubectl apply -f config/deployment.yaml
```

## Configuration

The operator can be configured via environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `LOG_LEVEL`: Logging level (default: INFO)
- `NIMBUSGUARD_NAMESPACE`: Target namespace to monitor (default: default)
- `PROMETHEUS_URL`: Prometheus endpoint for metrics collection

## Monitoring

The operator exposes metrics at port 8080:

- `/metrics` - Prometheus metrics
- `/healthz` - Liveness probe
- `/readyz` - Readiness probe

Example metrics:
- `nimbusguard_workflows_total` - Total workflows created
- `nimbusguard_active_workflows` - Current active workflows
- `nimbusguard_scaling_actions_total` - Total scaling actions executed
- `nimbusguard_q_learning_epsilon` - Current Q-learning exploration rate

## Development

For local testing against a Kubernetes cluster:

```bash
# Set up environment
export OPENAI_API_KEY=your-key-here
export KUBECONFIG=~/.kube/config

# Install dependencies
pip install -r requirements.txt

# Run operator locally
python operator.py
```

## Architecture

The operator consists of:

1. **Supervisor Agent** - Orchestrates the workflow
2. **State Observer Agent** - Collects metrics and system state
3. **Decision Agent** - Makes scaling decisions using AI
4. **Action Executor** - Executes scaling actions on Kubernetes
5. **Reward Calculator** - Provides feedback for Q-learning

## Custom Resource Definitions

The operator watches for `ScalingPolicy` custom resources:

```yaml
apiVersion: nimbusguard.io/v1
kind: ScalingPolicy
metadata:
  name: my-app-scaling
  namespace: default
spec:
  targetDeployment: my-app
  minReplicas: 1
  maxReplicas: 10
  metrics:
    - type: cpu
      target: 70
    - type: memory
      target: 80
```

## Troubleshooting

Check operator logs:
```bash
kubectl logs -f deployment/nimbusguard-operator -n nimbusguard
```

Check operator status:
```bash
kubectl get pods -n nimbusguard
kubectl describe deployment nimbusguard-operator -n nimbusguard
```

## License

This project is part of the NimbusGuard platform. 