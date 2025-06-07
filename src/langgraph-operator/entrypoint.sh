#!/bin/bash

# NimbusGuard LangGraph Operator Entrypoint Script
set -e  # Exit on any error

echo "🚀 Starting NimbusGuard LangGraph Operator..."

# Function to check if environment variable is set
check_env_var() {
    local var_name=$1
    local var_value=$(eval echo \$$var_name)
    
    if [ -z "$var_value" ]; then
        echo "❌ Error: Environment variable $var_name is not set"
        return 1
    else
        echo "✅ $var_name is configured"
        return 0
    fi
}

# Function to validate OpenAI API key format
validate_openai_key() {
    if [[ $OPENAI_API_KEY =~ ^sk-[a-zA-Z0-9]{48,}$ ]]; then
        echo "✅ OpenAI API key format is valid"
        return 0
    else
        echo "❌ Error: OpenAI API key format is invalid (should start with 'sk-' and be at least 51 characters)"
        return 1
    fi
}

# Environment validation
echo "🔍 Validating environment configuration..."

# Check required environment variables
VALIDATION_FAILED=false

if ! check_env_var "OPENAI_API_KEY"; then
    VALIDATION_FAILED=true
fi

# Validate OpenAI key format if it exists
if [ ! -z "$OPENAI_API_KEY" ]; then
    if ! validate_openai_key; then
        VALIDATION_FAILED=true
    fi
fi

# Check optional but important variables
echo "📋 Optional configuration:"
echo "   LOG_LEVEL: ${LOG_LEVEL:-INFO}"
echo "   DEBUG: ${DEBUG:-false}"
echo "   DEV_MODE: ${DEV_MODE:-true}"
echo "   PROMETHEUS_URL: ${PROMETHEUS_URL:-http://prometheus:9090}"

# Exit if validation failed
if [ "$VALIDATION_FAILED" = true ]; then
    echo ""
    echo "💡 To fix these issues:"
    echo "   1. Set your OpenAI API key: export OPENAI_API_KEY='sk-your-key-here'"
    echo "   2. Or use the setup script: source ./setup-env.sh"
    echo ""
    exit 1
fi

echo "✅ Environment validation passed!"

# Kubernetes configuration check
echo "🔧 Checking Kubernetes configuration..."
if [ "$KUBERNETES_IN_CLUSTER" = "true" ]; then
    echo "✅ Running in Kubernetes cluster mode"
else
    echo "🏠 Running in local development mode"
    if [ -f "/home/nimbusguard/.kube/config" ]; then
        echo "✅ Kubernetes config found at /home/nimbusguard/.kube/config"
    else
        echo "⚠️  Warning: No Kubernetes config found - operator may not be able to connect to cluster"
    fi
fi

# Wait for dependencies
echo "⏳ Waiting for dependencies..."

# Wait for Prometheus
echo "   Waiting for Prometheus..."
until curl -sf "$PROMETHEUS_URL/api/v1/status/buildinfo" > /dev/null 2>&1; do
    echo "   Prometheus not ready, waiting..."
    sleep 2
done
echo "✅ Prometheus is ready"

# Wait for consumer workload
echo "   Waiting for Consumer Workload..."
until curl -sf "${CONSUMER_WORKLOAD_URL}/health" > /dev/null 2>&1; do
    echo "   Consumer workload not ready, waiting..."
    sleep 2
done
echo "✅ Consumer Workload is ready"

echo ""
echo "🎯 All dependencies are ready!"
echo "🚀 Starting LangGraph Operator..."
echo ""

# Start the operator
exec python -m kopf run operator.py \
    --liveness=http://0.0.0.0:8080/healthz \
    --verbose \
    --log-format=json 