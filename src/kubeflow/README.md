# 🚀 NimbusGuard Kubeflow Integration

This directory contains the complete Kubeflow migration implementation for NimbusGuard, transforming your ML-powered Kubernetes autoscaling from a research prototype into a production-ready MLOps platform.

## 📁 Directory Structure

```
kubeflow/
├── Dockerfile                     # Unified Kubeflow container
├── requirements.txt               # Python dependencies  
├── test_e2e.py                    # End-to-end validation
├── pipelines/
│   └── dqn_training_pipeline.py   # Complete training pipeline
├── experiments/
│   └── katib_training.py          # Training script for Katib
└── serving/
    └── transformer.py             # Model preprocessing/postprocessing

# Kubernetes manifests are in:
kubernetes-manifests/components/kubeflow/
├── namespaces.yaml                # Kubeflow namespace setup  
├── katib-experiment.yaml          # Katib experiment config
├── kserve-inference.yaml          # KServe model serving
└── kustomization.yaml             # Kustomize configuration
```

## 🎯 Quick Start

### Interactive Setup (Recommended)
```bash
# Main development environment with ML options
make k8s-dev
```
Then choose your setup:
- **1)** Traditional ML (local models, fast setup)
- **2)** Kubeflow ML Pipeline (distributed training & serving)
- **3)** Hybrid (traditional + Kubeflow integration)

### Advanced: Component-by-Component
```bash
# Install infrastructure only
make kubeflow-install

# Deploy specific components
make kubeflow-pipelines     # Training pipelines
make kubeflow-experiments   # Hyperparameter tuning
make kubeflow-serving       # Model serving
```

## 🧪 Validation

### Quick Health Check
```bash
make kubeflow-status
```

### Comprehensive Testing
```bash
# Run complete E2E test suite
cd kubeflow
python test_e2e.py
```

### Manual Tests
```bash
# Test model serving
make kubeflow-serving-test

# Check experiment progress
make kubeflow-experiments-status

# Monitor logs
make kubeflow-logs
```

## 🔧 Components Overview

### 1. Training Pipelines (`pipelines/`)

**File**: `dqn_training_pipeline.py`
- Complete Kubeflow Pipeline for DQN training
- Automated data collection from Prometheus
- Model training with your 5-action space (SCALE_DOWN_2, SCALE_DOWN_1, NO_ACTION, SCALE_UP_1, SCALE_UP_2)
- Model validation and artifact storage

**Usage**:
```bash
make kubeflow-pipelines
```

### 2. Hyperparameter Tuning (`experiments/`)

**Files**: 
- `dqn-hyperparameter-tuning.yaml`: Katib experiment configuration
- `katib_training.py`: Training script optimized for your action space

**Features**:
- Bayesian optimization for hyperparameter search
- Parallel trial execution (3 concurrent trials)
- Automatic best parameter selection

**Usage**:
```bash
make kubeflow-experiments
make kubeflow-experiments-best  # Get optimal parameters
```

### 3. Model Serving (`serving/`)

**Files**:
- `dqn-inference-service.yaml`: KServe configuration
- `transformer.py`: Request/response preprocessing

**Features**:
- High-availability model serving with KServe
- Automatic scaling based on load
- Health monitoring and failover
- Integration with your existing operator

**Usage**:
```bash
make kubeflow-serving
make kubeflow-serving-test
```

### 4. Enhanced Operator Integration

**Location**: `src/nimbusguard-operator/`

**New Features**:
- Async KServe client integration
- Fallback to local models when serving unavailable
- Automatic model health validation
- Enhanced experience collection for distributed training

**Key Files Modified**:
- `handler.py`: Enhanced with Kubeflow integration
- `ml/kubeflow_integration.py`: New integration layer

## ⚙️ Configuration

### Environment Setup
```bash
# Copy and customize configuration
cp kubeflow/.env.example kubeflow/.env
# Edit configuration values as needed
```

### Key Configuration Options

#### Model Serving
```bash
KSERVE_ENDPOINT=http://nimbusguard-dqn-model.nimbusguard-serving.svc.cluster.local/v1/models/nimbusguard-dqn:predict
FALLBACK_TO_LOCAL=true
MODEL_VALIDATION_INTERVAL=300
```

#### Training Pipeline
```bash
DQN_STATE_DIM=11
DQN_ACTION_DIM=5
TRAINING_EPOCHS=100
DATA_COLLECTION_HOURS=24
```

#### Hyperparameter Tuning
```bash
KATIB_MAX_TRIAL_COUNT=20
KATIB_PARALLEL_TRIAL_COUNT=3
LEARNING_RATE_MIN=1e-5
LEARNING_RATE_MAX=1e-2
```

## 📊 Monitoring Integration

### New Metrics Available
- `nimbusguard_pipeline_runs_total`: Pipeline execution count
- `nimbusguard_model_training_duration_seconds`: Training time
- `nimbusguard_model_accuracy`: Model performance
- `nimbusguard_model_serving_requests_total`: Inference requests
- `nimbusguard_hyperparameter_experiments_total`: Katib experiments

### Grafana Dashboards
The system provides enhanced metrics that you can use to build dashboards showing:
- Pipeline success rates
- Model performance trends
- Serving request patterns
- Hyperparameter optimization progress

## 🔄 Workflow Integration

### Your Existing Scaling Actions
The integration maintains compatibility with your 5-action scaling system:
- `0: SCALE_DOWN_2` (-2 replicas)
- `1: SCALE_DOWN_1` (-1 replica)
- `2: NO_ACTION` (0 replicas)
- `3: SCALE_UP_1` (+1 replica)
- `4: SCALE_UP_2` (+2 replicas)

### Enhanced Decision Making
```
Local Training (Before) → Kubeflow Pipelines (After)
In-Memory Models → KServe Production Serving
Manual Tuning → Automated Katib Optimization
Basic Persistence → Complete Model Lifecycle
```

## 🚨 Troubleshooting

### Common Issues

1. **KServe Endpoint Not Available**
   ```bash
   kubectl get inferenceservices -n nimbusguard-serving
   kubectl describe inferenceservice nimbusguard-dqn-model -n nimbusguard-serving
   ```

2. **Pipeline Failures**
   ```bash
   kubectl get workflow -n kubeflow
   kubectl logs -n kubeflow <workflow-pod-name>
   ```

3. **Katib Experiments Not Starting**
   ```bash
   kubectl describe experiment nimbusguard-dqn-hyperparameter-tuning -n nimbusguard-experiments
   ```

### Debug Commands
```bash
# Overall status
make kubeflow-status

# Component logs  
make kubeflow-logs

# Test specific features
make kubeflow-serving-test
make kubeflow-experiments-status

# Operator integration
kubectl logs -f deployment/nimbusguard-operator -n nimbusguard | grep -i kubeflow
```

## 🎯 Migration Benefits

### Before (Current Implementation)
- ❌ Local DQN training within operator
- ❌ Manual hyperparameter tuning
- ❌ In-memory model serving
- ❌ Basic model persistence
- ❌ Limited scalability

### After (Kubeflow Integration)
- ✅ **Distributed Training**: Scalable across multiple nodes
- ✅ **Automated Tuning**: Katib finds optimal hyperparameters
- ✅ **Production Serving**: KServe with autoscaling and health checks
- ✅ **MLOps Workflow**: Complete lifecycle automation
- ✅ **Enhanced Monitoring**: Comprehensive ML pipeline observability

## 🚀 Next Steps

### Immediate
1. Deploy and test the system:
   ```bash
   make k8s-dev  # Choose option 2 for full Kubeflow
   python kubeflow/test_e2e.py
   ```

2. Generate load and observe ML decisions:
   ```bash
   curl -X POST "http://localhost:8080/api/v1/workload/cpu/start" \
     -H "Content-Type: application/json" \
     -d '{"intensity": 80, "duration": 300}'
   ```

### Short Term (1-2 weeks)
- Customize hyperparameter ranges in Katib experiments
- Build Grafana dashboards for ML pipeline monitoring
- Configure automated retraining schedules

### Medium Term (1-2 months)
- Set up persistent model storage with backup
- Implement A/B testing for model updates
- Configure multi-environment deployment (dev/staging/prod)

### Long Term (3+ months)
- Explore distributed training for larger datasets
- Implement federated learning across clusters
- Add neural architecture search capabilities

## 📚 Additional Resources

- **Main Migration Guide**: `../KUBEFLOW_MIGRATION.md`
- **Implementation Summary**: `../KUBEFLOW_IMPLEMENTATION.md`
- **Kubeflow Documentation**: https://www.kubeflow.org/docs/
- **KServe Documentation**: https://kserve.github.io/website/
- **Katib Documentation**: https://www.kubeflow.org/docs/components/katib/

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Run the E2E test suite for diagnostics
3. Review component logs using `make kubeflow-logs`
4. Consult the main documentation files

---

**Ready to transform your ML workflow?**
```bash
make k8s-dev
```

Choose option 2 for the complete production-ready MLOps platform! 🎉
