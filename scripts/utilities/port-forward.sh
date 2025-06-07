#!/bin/bash

# NimbusGuard Port Forwarding Script
# Forwards ports for all NimbusGuard services for local development

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="nimbusguard"
KEDA_NAMESPACE="keda"

# Functions
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

# Array to store background process PIDs
PIDS=()

cleanup() {
    log_info "Cleaning up port forwards..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            log_info "Stopping port forward (PID: $pid)"
            kill "$pid" 2>/dev/null || true
        fi
    done
    log_success "Cleanup completed"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

start_port_forward() {
    local service=$1
    local local_port=$2
    local service_port=$3
    local namespace=${4:-$NAMESPACE}
    
    log_info "Starting port forward for $service ($local_port -> $service_port)"
    kubectl port-forward -n "$namespace" "svc/$service" "$local_port:$service_port" > /dev/null 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    
    # Give it a moment to start
    sleep 2
    
    # Check if the port forward is still running
    if kill -0 "$pid" 2>/dev/null; then
        log_success "Port forward for $service started successfully (PID: $pid)"
        return 0
    else
        log_error "Failed to start port forward for $service"
        return 1
    fi
}

check_service_exists() {
    local service=$1
    local namespace=${2:-$NAMESPACE}
    
    if kubectl get service "$service" -n "$namespace" &>/dev/null; then
        return 0
    else
        log_warning "Service $service not found in namespace $namespace"
        return 1
    fi
}

show_access_info() {
    echo ""
    log_info "=== NimbusGuard Services Access ==="
    echo ""
    log_info "🎯 Main Services:"
    echo "  Consumer Workload:    http://localhost:8080"
    echo "  Load Generator:       http://localhost:8081"
    echo "  LangGraph Operator:   http://localhost:8082"
    echo ""
    log_info "📊 Monitoring & Observability:"
    echo "  Grafana Dashboard:    http://localhost:3000 (admin/nimbusguard)"
    echo "  Prometheus:           http://localhost:9090"
    echo "  Loki (Logs):          http://localhost:3100"
    echo "  Tempo (Tracing):      http://localhost:3200"
    echo "  OpenTelemetry:        gRPC http://localhost:4317, HTTP http://localhost:4318"
    echo ""
    log_info "📋 API Endpoints:"
    echo "  Health Checks:"
    echo "    curl http://localhost:8080/health"
    echo "    curl http://localhost:8081/health"
    echo "    curl http://localhost:8082/healthz"
    echo ""
    echo "  Generate CPU Load:"
    echo "    curl -X POST 'http://localhost:8080/workload/cpu?intensity=80&duration=300'"
    echo ""
    echo "  Generate Memory Load:"
    echo "    curl -X POST 'http://localhost:8080/workload/memory?intensity=70&duration=300'"
    echo ""
    echo "  Trigger Scaling Event:"
    echo "    curl -X POST http://localhost:8080/events/trigger \\"
    echo "      -H 'Content-Type: application/json' \\"
    echo "      -d '{\"event_type\":\"high_cpu_usage\",\"service\":\"consumer-workload\",\"value\":90}'"
    echo ""
    echo "  Generate Load Pattern:"
    echo "    curl -X POST http://localhost:8081/load/generate \\"
    echo "      -H 'Content-Type: application/json' \\"
    echo "      -d '{\"pattern\":\"spike\",\"duration\":300,\"target\":\"http\"}'"
    echo ""
    log_info "🔧 Management:"
    echo "  View Scaling Policies: kubectl get scalingpolicies -n $NAMESPACE"
    echo "  View AI Models:        kubectl get aimodels -n $NAMESPACE"
    echo "  View Pods:             kubectl get pods -n $NAMESPACE"
    echo "  View HPA:              kubectl get hpa -n $NAMESPACE"
    echo "  View KEDA Objects:     kubectl get scaledobjects -n $NAMESPACE"
    echo ""
    log_warning "Press Ctrl+C to stop all port forwards"
}

main() {
    log_info "Starting NimbusGuard port forwarding..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your context."
        exit 1
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
        log_error "Namespace $NAMESPACE not found. Please deploy NimbusGuard first."
        exit 1
    fi
    
    log_info "Setting up port forwards for NimbusGuard services..."
    
    # Core application services
    if check_service_exists "consumer-workload"; then
        start_port_forward "consumer-workload" "8080" "8080"
    fi
    
    if check_service_exists "load-generator"; then
        start_port_forward "load-generator" "8081" "8081"
    fi
    
    if check_service_exists "langgraph-operator"; then
        start_port_forward "langgraph-operator" "8082" "8082"
    fi
    
    # Monitoring services
    if check_service_exists "grafana"; then
        start_port_forward "grafana" "3000" "3000"
    fi
    
    if check_service_exists "prometheus-server"; then
        start_port_forward "prometheus-server" "9090" "9090"
    fi
    
    # Observability services
    if check_service_exists "loki"; then
        start_port_forward "loki" "3100" "3100"
    fi
    
    if check_service_exists "tempo"; then
        start_port_forward "tempo" "3200" "3200"
    fi
    
    if check_service_exists "otel-collector"; then
        start_port_forward "otel-collector" "4317" "4317"
        start_port_forward "otel-collector" "4318" "4318"
    fi
    
    # Kafka (optional)
    if check_service_exists "kafka"; then
        start_port_forward "kafka" "9092" "9092"
    fi
    
    # Show access information
    show_access_info
    
    # Wait for signals
    log_info "Port forwarding active. Waiting for termination signal..."
    wait
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Sets up port forwarding for all NimbusGuard services"
            echo ""
            echo "Port mappings:"
            echo "  8080 -> consumer-workload:8080"
            echo "  8081 -> load-generator:8081"
            echo "  8082 -> langgraph-operator:8082"
            echo "  3000 -> grafana:3000"
            echo "  9090 -> prometheus-server:9090"
            echo "  3100 -> loki:3100"
            echo "  3200 -> tempo:3200"
            echo "  4317 -> otel-collector:4317 (OTLP gRPC)"
            echo "  4318 -> otel-collector:4318 (OTLP HTTP)"
            echo "  9092 -> kafka:9092"
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