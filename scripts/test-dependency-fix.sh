#!/bin/bash

# Test script to verify dependency compatibility fixes
# This script tests if the Docker build will work with fixed dependencies

set -e

echo "🔍 Testing dependency compatibility fixes..."

# Test 1: Check if requirements files have compatible prometheus-client versions
echo "📋 Checking prometheus-client versions in requirements files..."

echo "Consumer workload:"
grep "prometheus-client" src/consumer-workload/requirements.txt || echo "Not found"

echo "Operator:"
grep "prometheus-client" src/nimbusguard-operator/requirements.txt || echo "Not found"

echo "Kubeflow:"
grep "prometheus-client" src/kubeflow/requirements.txt || echo "Not found"

echo "KServe version:"
grep "kserve" src/kubeflow/requirements.txt || echo "Not found"

# Test 2: Try to build the base image to verify compatibility
echo ""
echo "🏗️  Testing base image build with fixed dependencies..."

# Create a minimal test Dockerfile
cat > /tmp/test-deps.Dockerfile << 'EOF'
FROM python:3.11-slim

# Copy requirements files
COPY src/consumer-workload/requirements.txt /tmp/consumer-requirements.txt
COPY src/load-generator/requirements.txt /tmp/load-generator-requirements.txt  
COPY src/nimbusguard-operator/requirements.txt /tmp/langgraph-requirements.txt
COPY src/kubeflow/requirements.txt /tmp/kubeflow-requirements.txt

# Test dependency resolution without actually installing
RUN pip install --dry-run -r /tmp/consumer-requirements.txt && \
    pip install --dry-run -r /tmp/load-generator-requirements.txt && \
    pip install --dry-run -r /tmp/langgraph-requirements.txt && \
    pip install --dry-run -r /tmp/kubeflow-requirements.txt

EOF

# Test build
docker build -f /tmp/test-deps.Dockerfile -t test-deps . && \
echo "✅ Dependencies are compatible!" || \
echo "❌ Dependencies still have conflicts"

# Cleanup
rm -f /tmp/test-deps.Dockerfile
docker rmi test-deps 2>/dev/null || true

echo ""
echo "🧪 Dependency test completed!" 