#!/bin/bash

# NimbusGuard LangGraph Operator Entrypoint Script
set -e  # Exit on any error

echo "🚀 Starting NimbusGuard LangGraph Operator..."

# Function to read value from .env file
read_env_file() {
    local var_name=$1
    local env_file="/app/.env"
    
    # First try to read from mounted .env file in the app directory
    if [ -f "$env_file" ]; then
        local value=$(grep "^${var_name}=" "$env_file" | cut -d'=' -f2- | tr -d '"' | tr -d "'")
        echo "$value"
        return 0
    fi
    
    # Fallback to environment variable
    local var_value=$(eval echo \$$var_name)
    echo "$var_value"
    return 0
}

# Function to check if OpenAI API key is configured
check_openai_key() {
    local api_key=$(read_env_file "OPENAI_API_KEY")
    
    if [ -z "$api_key" ]; then
        echo "❌ Error: OPENAI_API_KEY not found in .env file or environment"
        return 1
    else
        echo "✅ OPENAI_API_KEY is configured"
        # Set the environment variable for the Python application
        export OPENAI_API_KEY="$api_key"
        return 0
    fi
}

# Function to validate OpenAI API key format
validate_openai_key() {
    local api_key=$(read_env_file "OPENAI_API_KEY")
    
    # Allow demo key for development
    if [[ $api_key == "sk-demo-key-for-development"* ]]; then
        echo "⚠️  Using demo OpenAI API key for development"
        return 0
    fi
    
    # Support both legacy (sk-...) and new project-based (sk-proj-...) API key formats
    if [[ $api_key =~ ^sk-[a-zA-Z0-9]{48,}$ ]] || [[ $api_key =~ ^sk-proj-[a-zA-Z0-9_-]{90,}$ ]]; then
        echo "✅ OpenAI API key format is valid"
        return 0
    else
        echo "❌ Error: OpenAI API key format is invalid"
        echo "   Expected: 'sk-...' (legacy) or 'sk-proj-...' (project-based)"
        echo "   Found key: ${api_key:0:20}..."
        return 1
    fi
}

# Environment validation
echo "🔍 Validating environment configuration..."

# Check required configuration
VALIDATION_FAILED=false

if ! check_openai_key; then
    VALIDATION_FAILED=true
fi

# Validate OpenAI key format
if ! validate_openai_key; then
    VALIDATION_FAILED=true
fi

# Note: Running in Kubernetes cluster, no special configuration needed
echo "🔧 Kubernetes operator will connect using in-cluster config or mounted kubeconfig"

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

# Wait for dependencies (temporarily disabled for controller pattern testing)
echo "⏳ Skipping dependency checks for controller pattern testing..."

# # Wait for Prometheus
# echo "   Waiting for Prometheus..."
# until curl -sf "$PROMETHEUS_URL/api/v1/status/buildinfo" > /dev/null 2>&1; do
#     echo "   Prometheus not ready, waiting..."
#     sleep 2
# done
# echo "✅ Prometheus is ready"

# # Wait for consumer workload
# echo "   Waiting for Consumer Workload..."
# until curl -sf "${CONSUMER_WORKLOAD_URL}/health" > /dev/null 2>&1; do
#     echo "   Consumer workload not ready, waiting..."
#     sleep 2
# done
# echo "✅ Consumer Workload is ready"

echo ""
echo "🎯 Starting without dependency checks for testing!"
echo "🚀 Starting LangGraph Operator..."
echo ""

# Start the operator
export PYTHONPATH="/app:$PYTHONPATH"
cd /app
exec python -m kopf run nimbusguard_operator.py \
    --liveness=http://0.0.0.0:8080/healthz \
    --verbose \
    --log-format=json 