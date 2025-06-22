# NimbusGuard LangGraph Migration Guide

## 🚀 Overview

This guide explains the migration from the traditional DQN-based operator to the new **LangGraph-powered** NimbusGuard operator, which combines ML models with LLM reasoning for more intelligent autoscaling decisions.

## 🔄 What Changed

### Before (Traditional)
- Simple DQN agent for scaling decisions
- Basic observability collection
- Rule-based decision logic
- Limited reasoning capabilities

### After (LangGraph)
- **Graph-based workflow** orchestration
- **LLM analysis** for intelligent reasoning
- **Enhanced observability** with multi-source integration
- **State management** with workflow history
- **Risk assessment** and confidence scoring
- **Tool-based architecture** for modularity

## 📋 Architecture Comparison

### Traditional Architecture
```
Kopf Timer → Handler → ObservabilityCollector → DQN Agent → Kubernetes API
```

### LangGraph Architecture
```
Kopf Timer → LangGraphHandler → LangGraph Workflow:
├── 1. Collect Observability (Prometheus + Loki + Tempo)
├── 2. LLM Analysis & Risk Assessment
├── 3. ML Model Decision (KServe)
├── 4. Execute Scaling Action
└── 5. Store Workflow History
```

## 🆕 New Features

### 1. **LLM-Powered Analysis**
- GPT-4o-mini analyzes observability data
- Provides human-readable reasoning
- Risk assessment for scaling decisions
- Context-aware recommendations

### 2. **Enhanced State Management**
- Workflow history tracking
- Performance metrics collection
- Reasoning step documentation
- Execution error tracking

### 3. **Advanced Observability**
- Multi-source data collection (Prometheus, Loki, Tempo)
- Health monitoring of all systems
- Confidence scoring for data quality
- Business KPI integration

### 4. **Tool-Based Architecture**
- Modular LangGraph tools
- Reusable components
- Easy testing and debugging
- Extensible design

## 🔧 Migration Steps

### 1. **Update Dependencies**
The new `requirements.txt` includes LangGraph and LangChain dependencies:
```bash
# Rebuild the container with new dependencies
make build-images
```

### 2. **Environment Variables**
Add required environment variables:
```bash
# Required for LLM analysis
export OPENAI_API_KEY="your-openai-api-key"

# Required for ML model inference
export KSERVE_ENDPOINT="http://your-kserve-endpoint/v1/models/dqn-model:predict"

# Optional: Custom observability endpoints
export PROMETHEUS_URL="http://prometheus.monitoring.svc.cluster.local:9090"
export LOKI_URL="http://loki.monitoring.svc.cluster.local:3100"
export TEMPO_URL="http://tempo.monitoring.svc.cluster.local:3100"
```

### 3. **Choose Implementation**
You can run either implementation:

**Option A: LangGraph (Recommended)**
```bash
# Use the new LangGraph-based main
python main_langgraph.py
```

**Option B: Traditional (Legacy)**
```bash
# Use the original implementation
python main.py
```

### 4. **Update Deployment**
Update your operator deployment to use the new main file:
```yaml
# In your deployment manifest
containers:
- name: operator
  image: nimbusguard-operator:latest
  command: ["python", "main_langgraph.py"]  # Changed from main.py
  env:
  - name: OPENAI_API_KEY
    valueFrom:
      secretKeyRef:
        name: langgraph-secrets
        key: openai-api-key
  - name: KSERVE_ENDPOINT
    value: "http://dqn-model.nimbusguard-serving.svc.cluster.local/v1/models/dqn-model:predict"
```

## 📊 Monitoring & Observability

### New Metrics Available
The LangGraph implementation provides enhanced metrics:

```bash
# Check operator health
kubectl get pod -l app=nimbusguard-operator -o jsonpath='{.items[0].status.phase}'

# View LangGraph-specific logs
kubectl logs -l app=nimbusguard-operator | grep "LangGraph"

# Check workflow metrics via operator status
kubectl get intelligentscaling -o yaml
```

### Health Endpoints
The operator now provides comprehensive health information:
- **Workflow Status**: Success/failure rates
- **System Health**: Individual component status
- **AI Analysis**: LLM and ML model availability
- **Performance Metrics**: Workflow duration and confidence scores

## 🔍 Debugging & Troubleshooting

### Common Issues

**1. Missing OpenAI API Key**
```
Error: LLM analysis failed: OpenAI API key not found
Solution: Set OPENAI_API_KEY environment variable
```

**2. KServe Endpoint Not Accessible**
```
Error: KServe inference failed: connection refused
Solution: Check KSERVE_ENDPOINT and ensure model is deployed
```

**3. Workflow Timeout**
```
Error: Workflow execution timeout
Solution: Check observability systems (Prometheus, Loki, Tempo) connectivity
```

### Debug Mode
Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
python main_langgraph.py
```

## 🎯 Benefits of Migration

### 1. **Intelligent Decision Making**
- LLM provides contextual analysis
- Risk assessment prevents harmful scaling
- Human-readable explanations for decisions

### 2. **Enhanced Reliability**
- Multi-system health monitoring
- Graceful degradation when systems fail
- Comprehensive error handling

### 3. **Better Observability**
- Workflow history tracking
- Performance metrics collection
- Detailed reasoning documentation

### 4. **Extensibility**
- Easy to add new tools
- Modular architecture
- Simple to test individual components

## 🔄 Rollback Plan

If you need to rollback to the traditional implementation:

1. **Update deployment** to use `main.py` instead of `main_langgraph.py`
2. **Remove new environment variables** (OPENAI_API_KEY, etc.)
3. **Optionally downgrade** dependencies to remove LangGraph packages

The traditional implementation remains fully functional and can be used as a fallback.

## 📈 Performance Considerations

### Resource Usage
- **Memory**: +50-100MB for LangGraph components
- **CPU**: +10-20% during LLM analysis
- **Network**: Additional API calls to OpenAI (optional)

### Latency
- **Traditional**: ~1-3 seconds per evaluation
- **LangGraph**: ~3-8 seconds per evaluation (including LLM analysis)

The increased latency is offset by significantly better decision quality and reasoning.

## 🎉 Next Steps

1. **Deploy** the LangGraph version in a test environment
2. **Monitor** workflow metrics and performance
3. **Compare** decision quality with traditional version
4. **Gradually migrate** production workloads
5. **Extend** with custom tools and analysis as needed

The LangGraph implementation provides a solid foundation for more advanced AI-powered autoscaling capabilities! 