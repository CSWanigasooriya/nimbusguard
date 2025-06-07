#!/bin/bash

# NimbusGuard Deployment Script
# Deploys the complete NimbusGuard AI-powered Kubernetes autoscaling platform

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="nimbusguard"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MANIFESTS_DIR="$PROJECT_ROOT/kubernetes-manifests"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your context."
        exit 1
    fi
    
    # Check if kustomize is available
    if ! command -v kustomize &> /dev/null; then
        log_warning "kustomize not found. Using kubectl apply with -k flag."
    fi
    
    log_success "Prerequisites check passed"
}

# Wait for deployment to be ready
wait_for_deployment() {
    local deployment=$1
    local namespace=${2:-$NAMESPACE}
    local timeout=${3:-300}
    
    log_info "Waiting for deployment $deployment to be ready..."
    
    if kubectl rollout status deployment/$deployment -n $namespace --timeout=${timeout}s; then
        log_success "Deployment $deployment is ready"
        return 0
    else
        log_error "Deployment $deployment failed to become ready"
        return 1
    fi
}

# Wait for pods to be ready
wait_for_pods() {
    local label_selector=$1
    local namespace=${2:-$NAMESPACE}
    local timeout=${3:-300}
    
    log_info "Waiting for pods with selector '$label_selector' to be ready..."
    
    if kubectl wait --for=condition=ready pod -l "$label_selector" -n $namespace --timeout=${timeout}s; then
        log_success "Pods with selector '$label_selector' are ready"
        return 0
    else
        log_error "Pods with selector '$label_selector' failed to become ready"
        return 1
    fi
}

# Deploy base components
deploy_base() {
    log_info "Deploying base components..."
    
    cd "$MANIFESTS_DIR/base"
    
    # Apply base manifests using kustomization
    if kubectl apply -k . ; then
        log_success "Base components deployed successfully"
    else
        log_error "Failed to deploy base components"
        exit 1
    fi
    
    # Wait for critical services
    log_info "Waiting for base services to be ready..."
    
    # Wait for Kafka (KRaft mode - no Zookeeper needed)
    wait_for_deployment "kafka" "$NAMESPACE" 180
    
    # Wait for core applications
    wait_for_deployment "consumer-workload" "$NAMESPACE" 120
    wait_for_deployment "load-generator" "$NAMESPACE" 120
    wait_for_deployment "langgraph-operator" "$NAMESPACE" 120
}

# Deploy observability stack
deploy_observability() {
    log_info "Deploying observability stack..."
    
    cd "$MANIFESTS_DIR/components/observability"
    
    if kubectl apply -k . ; then
        log_success "Observability stack deployed successfully"
    else
        log_error "Failed to deploy observability stack"
        exit 1
    fi
    
    # Wait for observability services
    log_info "Waiting for observability services to be ready..."
    
    wait_for_deployment "prometheus-server" "$NAMESPACE" 120
    wait_for_deployment "grafana" "$NAMESPACE" 120
    wait_for_deployment "loki" "$NAMESPACE" 120
    wait_for_deployment "tempo" "$NAMESPACE" 120
    wait_for_deployment "otel-collector" "$NAMESPACE" 120
    
    # Wait for Promtail DaemonSet
    log_info "Waiting for Promtail DaemonSet..."
    kubectl rollout status daemonset/promtail -n $NAMESPACE --timeout=120s || log_warning "Promtail DaemonSet may not be fully ready"
}

# Deploy KEDA components
deploy_keda() {
    log_info "Deploying KEDA components..."
    
    cd "$MANIFESTS_DIR/components/keda"
    
    if kubectl apply -k . ; then
        log_success "KEDA components deployed successfully"
    else
        log_error "Failed to deploy KEDA components"
        exit 1
    fi
    
    # Wait for KEDA operator
    log_info "Waiting for KEDA operator to be ready..."
    wait_for_deployment "keda-operator" "keda-system" 120
    wait_for_deployment "keda-metrics-apiserver" "keda-system" 120
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check namespace
    if kubectl get namespace $NAMESPACE &>/dev/null; then
        log_success "Namespace $NAMESPACE exists"
    else
        log_error "Namespace $NAMESPACE not found"
        return 1
    fi
    
    # Check core services
    local services=("consumer-workload" "load-generator" "langgraph-operator" "kafka")
    for service in "${services[@]}"; do
        if kubectl get service $service -n $NAMESPACE &>/dev/null; then
            log_success "Service $service is available"
        else
            log_error "Service $service not found"
        fi
    done
    
    # Check observability services
    local obs_services=("prometheus-server" "grafana" "loki" "tempo" "otel-collector")
    for service in "${obs_services[@]}"; do
        if kubectl get service $service -n $NAMESPACE &>/dev/null; then
            log_success "Observability service $service is available"
        else
            log_error "Observability service $service not found"
        fi
    done
    
    # Check CRDs
    if kubectl get crd scalingpolicies.nimbusguard.io &>/dev/null; then
        log_success "ScalingPolicy CRD is installed"
    else
        log_warning "ScalingPolicy CRD not found"
    fi
    
    if kubectl get crd aimodels.nimbusguard.io &>/dev/null; then
        log_success "AIModel CRD is installed"
    else
        log_warning "AIModel CRD not found"
    fi
    
    # Show pod status
    log_info "Current pod status:"
    kubectl get pods -n $NAMESPACE
    
    log_info "Current service status:"
    kubectl get services -n $NAMESPACE
}

# Show access information
show_access_info() {
    echo ""
    log_info "=== NimbusGuard Deployment Complete ==="
    echo ""
    log_success "🎯 All services are deployed and ready!"
    echo ""
    log_info "To access the services, run:"
    echo "  ./scripts/utilities/port-forward.sh"
    echo ""
    log_info "📊 Service URLs (after port forwarding):"
    echo "  Grafana Dashboard:    http://localhost:3000 (admin/nimbusguard)"
    echo "  Prometheus:           http://localhost:9090"
    echo "  Loki (Logs):          http://localhost:3100"
    echo "  Tempo (Tracing):      http://localhost:3200"
    echo "  Consumer Workload:    http://localhost:8080"
    echo "  Load Generator:       http://localhost:8081"
    echo "  LangGraph Operator:   http://localhost:8082"
    echo ""
    log_info "🧪 Test the system:"
    echo "  # Generate CPU load"
    echo "  curl -X POST 'http://localhost:8080/workload/cpu?intensity=80&duration=300'"
    echo ""
    echo "  # Generate load pattern"
    echo "  curl -X POST http://localhost:8081/load/generate \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"pattern\":\"spike\",\"duration\":300,\"target\":\"http\"}'"
    echo ""
    log_info "🔍 Monitor the system:"
    echo "  kubectl get pods -n $NAMESPACE -w"
    echo "  kubectl get hpa -n $NAMESPACE"
    echo "  kubectl get scaledobjects -n $NAMESPACE"
    echo ""
}

# Main deployment function
main() {
    echo ""
    log_info "🚀 Starting NimbusGuard deployment..."
    echo ""
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Check prerequisites
    check_prerequisites
    
    # Deploy components in order
    deploy_base
    deploy_observability
    deploy_keda
    
    # Verify deployment
    verify_deployment
    
    # Show access information
    show_access_info
    
    log_success "🎉 NimbusGuard deployment completed successfully!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Deploy NimbusGuard AI-powered Kubernetes autoscaling platform"
            echo ""
            echo "Options:"
            echo "  --help     Show this help message"
            echo ""
            echo "Components deployed:"
            echo "  - Base infrastructure (namespace, RBAC, ConfigMaps)"
            echo "  - Kafka event streaming"
            echo "  - Consumer workload application"
            echo "  - Load generator service"
            echo "  - LangGraph AI operator"
            echo "  - Complete observability stack (Prometheus, Grafana, Loki, Tempo)"
            echo "  - KEDA autoscaling components"
            echo ""
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main 