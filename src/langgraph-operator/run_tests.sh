#!/bin/bash

# NimbusGuard Test Runner Script
# This script sets up the environment and runs all tests

set -e  # Exit on any error

echo "🧪 NimbusGuard Test Runner"
echo "=========================="

# Check if we're in the right directory
if [ ! -f "nimbusguard_operator.py" ]; then
    echo "❌ Error: Please run this script from the langgraph-operator directory"
    exit 1
fi

# Install test dependencies if needed
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

echo "🔧 Activating virtual environment..."
source .venv/bin/activate

echo "📥 Installing test dependencies..."
pip install -r test-requirements.txt > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1

# Set Python path to include current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "🚀 Running tests..."
echo ""

# Run different test suites
echo "Running unit tests..."
pytest tests/test_scaling_state.py tests/test_q_learning.py -m "not integration" -v

echo ""
echo "Running integration tests..."
pytest tests/test_integration.py -m "not slow" -v

echo ""
echo "Running controller tests..."
pytest tests/test_nimbusguard_operator.py -v

echo ""
echo "✅ All tests completed!"
echo ""
echo "📊 Test Coverage Report:"
pytest --cov=. --cov-report=term-missing tests/ --tb=no -q

echo ""
echo "🎉 Test run finished successfully!" 