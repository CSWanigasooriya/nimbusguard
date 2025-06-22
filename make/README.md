# 📁 Modular Makefiles

This directory contains modular makefiles that organize NimbusGuard's build and deployment functionality into focused, maintainable components.

## 🏗️ Structure

```
make/
├── Makefile.infrastructure    # Docker builds, namespaces, secrets
├── Makefile.components        # Workload deployment, operator, KEDA
├── Makefile.monitoring        # Alloy, Prometheus, Grafana
└── Makefile.dev               # Port forwarding, status, cleanup
```

## 📋 Makefile Overview

### `Makefile.infrastructure`
**Purpose**: Core infrastructure setup
- Docker image building for all components
- Kubernetes namespace and CRD deployment
- Secret management (OpenAI API keys)
- Local model directory preparation

**Key Targets**:
- `setup-infrastructure` - Complete infrastructure setup
- `build-images` - Build all Docker images
- `create-operator-secret` - Setup API keys
- `rebuild-operator` - Quick operator rebuild

### `Makefile.components`
**Purpose**: Application component deployment
- Workload deployment (Kafka, Consumer, Load Generator)
- NimbusGuard operator deployment (local and Kubeflow modes)
- KEDA installation and scaling configuration
- Component health checks and management

**Key Targets**:
- `deploy-workloads` - Deploy core workloads
- `deploy-operator-only` - Deploy local operator
- `deploy-kubeflow-operator` - Deploy with KServe integration
- `install-keda` - Install KEDA auto-scaling
- `deploy-traditional-scaling` - Interactive scaling setup

### `Makefile.monitoring`
**Purpose**: Observability and monitoring
- Alloy observability pipeline installation
- Prometheus and Grafana deployment
- Monitoring configuration management
- Health checks and troubleshooting

**Key Targets**:
- `deploy-monitoring` - Deploy complete monitoring stack
- `install-alloy` - Install Grafana Alloy
- `check-monitoring` - Health check monitoring components
- `port-forward-monitoring` - Access monitoring UIs

### `Makefile.dev`
**Purpose**: Development utilities
- Port forwarding management
- System status and health checks
- Quick testing and validation
- Cleanup and troubleshooting tools

**Key Targets**:
- `forward` - Setup port forwarding
- `status` - Complete system status
- `health-check` - Endpoint health validation
- `quick-test` - Integration test
- `clean` / `clean-all` - Resource cleanup

## 🎯 Usage Patterns

### Main Workflow
```bash
# Use the main entry point
make k8s-dev

# Or run individual modules
make setup-infrastructure
make deploy-traditional-scaling
make deploy-monitoring
make forward
```

### Development
```bash
# Quick status check
make status

# Health validation
make health-check

# Component restart
make restart-operator
make restart-monitoring
```

### Troubleshooting
```bash
# View logs
make logs-operator
make logs-monitoring

# Debug info
make debug-info
make describe-failed-pods
```

### Component Management
```bash
# Infrastructure
make build-images
make rebuild-operator

# Components
make check-workloads
make check-keda

# Monitoring
make check-monitoring
make validate-monitoring-config
```

## 🔧 Benefits of Modular Structure

### 📦 **Organization**
- Related functionality grouped together
- Easy to find and modify specific features
- Clear separation of concerns

### 🔄 **Maintainability**
- Changes to one area don't affect others
- Easier to debug and test individual components
- Simpler to add new functionality

### 👥 **Team Development**
- Different team members can work on different modules
- Reduced merge conflicts
- Specialized expertise areas

### 🧪 **Testing**
- Test individual components in isolation
- Granular CI/CD pipeline stages
- Faster development cycles

## 🚀 Adding New Functionality

### 1. Choose the Right Makefile
- **Infrastructure**: New build processes, base setup
- **Components**: New services, deployment logic
- **Monitoring**: New dashboards, alerts, metrics
- **Dev**: New developer tools, utilities

### 2. Follow Naming Conventions
```bash
# Target naming
component-action:           # deploy-workloads
check-component:           # check-monitoring
clean-component:           # clean-keda

# Variable naming
COMPONENT_CONFIG := value  # KEDA_VERSION := 2.17.1
```

### 3. Add Documentation
- Update this README
- Add comments to complex targets
- Document new environment variables

### 4. Test Integration
```bash
# Test individual makefile
make -f make/Makefile.components deploy-workloads

# Test main workflow
make k8s-dev
```

## 📚 Examples

### Adding a New Service
1. Add build target to `Makefile.infrastructure`
2. Add deployment target to `Makefile.components`
3. Add health check to `Makefile.dev`
4. Update monitoring in `Makefile.monitoring`

### Adding New Development Tool
1. Add to `Makefile.dev`
2. Update help text in main `Makefile`
3. Test with `make your-new-tool`

### Adding Monitoring Component
1. Add installation to `Makefile.monitoring`
2. Add health checks
3. Add port forwarding if needed
4. Update `status` target in `Makefile.dev`

This modular approach makes NimbusGuard's build system much more maintainable and developer-friendly! 🎉
