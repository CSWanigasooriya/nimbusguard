.PHONY: all help dev k8s-dev clean k8s-clean status forward keda-setup keda-clean stop-forward build-base build-images kopf-dev create-sample-scaling create-operator-secret

all: help

help:
	@echo "NimbusGuard - Quick Development Commands"
	@echo "========================================"
	@echo ""
	@echo "🚀 Development:"
	@echo "  dev          - Docker development environment"
	@echo "  k8s-dev      - Kubernetes development environment"
	@echo "  kopf-dev     - Run operator locally with kopf"
	@echo ""
	@echo "🏗️  Build:"
	@echo "  build-base   - Build base Docker image"
	@echo "  build-images - Build all service images"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  clean        - Clean Docker environment"
	@echo "  k8s-clean    - Clean Kubernetes environment"
	@echo "  stop-forward - Stop all port forwarding"
	@echo ""
	@echo "📊 Status:"
	@echo "  status       - Show service status"
	@echo "  forward      - Forward ports for local access"
	@echo "  create-sample-scaling - Create sample scaling config"

# Run operator locally with kopf
kopf-dev:
	@echo "🚀 Starting operator locally with kopf..."
	@echo "   Checking for OpenAI API key..."
	@read -p "Enter your OpenAI API key (or press Enter to skip): " api_key; \
	export OPENAI_API_KEY="$$api_key"; \
	export PROMETHEUS_URL="http://localhost:9090"; \
	cd src/nimbusguard-operator && \
	kopf run --standalone --namespace nimbusguard nimbusguard_operator.py

# Build base image
build-base:
	@echo "🏗️  Building base image..."
	@DOCKER_BUILDKIT=1 docker build \
		-t nimbusguard/base:latest \
		-f docker/base.Dockerfile .
	@echo "✅ Base image built!"

# Build service images
build-images: build-base
	@echo "🏗️  Building workload images..."
	@DOCKER_BUILDKIT=1 docker build \
		-t nimbusguard/consumer-workload:latest \
		-f src/consumer-workload/Dockerfile \
		src/consumer-workload
	@DOCKER_BUILDKIT=1 docker build \
		-t nimbusguard/load-generator:latest \
		-f src/load-generator/Dockerfile \
		src/load-generator
	@DOCKER_BUILDKIT=1 docker build \
		-t nimbusguard/nimbusguard-operator:latest \
		-f src/nimbusguard-operator/Dockerfile \
		src/nimbusguard-operator
	@echo "✅ Workload images built!"

# Docker Development
dev: build-images
	@echo "🐳 Starting Docker development environment..."
	@docker-compose up -d
	@echo ""
	@echo "✅ Services available at:"
	@echo "   Consumer Workload: http://localhost:8080"
	@echo "   Load Generator:    http://localhost:8081"
	@echo "   Prometheus:        http://localhost:9090"

# Stop port forwarding
stop-forward:
	@echo "🛑 Stopping port forwarding..."
	@pkill -f "kubectl port-forward" || true
	@echo "✅ Port forwarding stopped"

# Create operator secret
create-operator-secret:
	@echo "🔑 Setting up operator secrets..."
	@read -p "Enter your OpenAI API key (or press Enter to skip): " api_key; \
	if [ -n "$$api_key" ]; then \
		kubectl create secret generic operator-secrets \
			--namespace nimbusguard \
			--from-literal=openai_api_key="$$api_key" \
			--dry-run=client -o yaml | kubectl apply -f -; \
		echo "✅ OpenAI API key configured!"; \
	else \
		kubectl create secret generic operator-secrets \
			--namespace nimbusguard \
			--from-literal=openai_api_key="" \
			--dry-run=client -o yaml | kubectl apply -f -; \
		echo "⚠️  No OpenAI API key provided - operator will use basic decision making"; \
	fi

# Kubernetes Development
k8s-dev: create-operator-secret
	@echo "🚀 Starting Kubernetes development environment..."
	@echo "🏗️  Building base image..."
	@DOCKER_BUILDKIT=1 docker build \
		-t nimbusguard/base:latest \
		-f docker/base.Dockerfile .
	@echo "🏗️  Building workload images..."
	@DOCKER_BUILDKIT=1 docker build \
		-t nimbusguard/consumer-workload:latest \
		-f src/consumer-workload/Dockerfile \
		src/consumer-workload
	@DOCKER_BUILDKIT=1 docker build \
		-t nimbusguard/load-generator:latest \
		-f src/load-generator/Dockerfile \
		src/load-generator
	@DOCKER_BUILDKIT=1 docker build \
		-t nimbusguard/nimbusguard-operator:latest \
		-f src/nimbusguard-operator/Dockerfile \
		src/nimbusguard-operator
	@echo "✅ All images built!"
	@echo "   Creating namespace..."
	@kubectl create namespace nimbusguard --dry-run=client -o yaml | kubectl apply -f -
	@echo "   Applying Kubernetes manifests..."
	@kubectl apply -k kubernetes-manifests/base
	@echo "   Waiting for pods to be ready..."
	@echo "   (This may take a few minutes for first-time setup...)"
	@kubectl wait --for=condition=ready pod -l app=kafka -n nimbusguard --timeout=300s || echo "⚠️  Kafka not ready yet"
	@kubectl wait --for=condition=ready pod -l app=consumer-workload -n nimbusguard --timeout=300s || echo "⚠️  Consumer not ready yet"
	@kubectl wait --for=condition=ready pod -l app=load-generator -n nimbusguard --timeout=300s || echo "⚠️  Load generator not ready yet"
	@kubectl wait --for=condition=ready pod -l app=nimbusguard-operator -n nimbusguard --timeout=300s || echo "⚠️  Operator not ready yet"
	@echo "   Setting up port forwarding..."
	@make stop-forward
	@kubectl port-forward -n nimbusguard svc/consumer-workload 8080:8080 > /dev/null 2>&1 & \
	kubectl port-forward -n nimbusguard svc/load-generator 8081:8081 > /dev/null 2>&1 & \
	kubectl port-forward -n nimbusguard svc/nimbusguard-operator 8082:8080 > /dev/null 2>&1 & \
	sleep 2
	@echo "   ✅ Environment ready!"
	@echo ""
	@echo "   Services available at:"
	@echo "   Consumer Workload: http://localhost:8080"
	@echo "   Load Generator:    http://localhost:8081"
	@echo "   Operator:          http://localhost:8082"

# Cleanup Commands
clean:
	@echo "🧹 Cleaning Docker environment..."
	@docker-compose down -v --remove-orphans 2>/dev/null || true
	@docker system prune -f
	@echo "✅ Docker cleanup completed"

k8s-clean: stop-forward
	@echo "🧹 Cleaning up Kubernetes environment..."
	@kubectl delete -k kubernetes-manifests/base || true
	@kubectl delete namespace nimbusguard || true
	@echo "✅ Environment cleaned up!"

# Status Commands
status:
	@echo "📊 Checking service status..."
	@echo ""
	@echo "Namespace:"
	@kubectl get namespace nimbusguard 2>/dev/null || echo "❌ nimbusguard namespace not found"
	@echo ""
	@echo "Pods:"
	@kubectl get pods -n nimbusguard 2>/dev/null || echo "❌ No pods found in nimbusguard namespace"
	@echo ""
	@echo "Services:"
	@kubectl get svc -n nimbusguard 2>/dev/null || echo "❌ No services found in nimbusguard namespace"
	@echo ""
	@echo "StatefulSets:"
	@kubectl get statefulsets -n nimbusguard 2>/dev/null || echo "❌ No statefulsets found"
	@echo ""
	@echo "IntelligentScaling CRs:"
	@kubectl get intelligentscaling -n nimbusguard 2>/dev/null || echo "❌ No IntelligentScaling resources found"
	@echo ""
	@echo "Port forwarding status:"
	@pgrep -f "kubectl port-forward" > /dev/null && echo "✅ Port forwarding active" || echo "❌ No port forwarding active"

# Forward ports
forward: stop-forward
	@echo "🔌 Setting up port forwarding..."
	@for port in 8080 8081 8082 9090; do \
		if lsof -i :$$port > /dev/null 2>&1; then \
			echo "Port $$port is already in use. Stopping existing port forward..."; \
			pkill -f "kubectl port-forward.*:$$port" || true; \
			sleep 2; \
		fi; \
	done
	@echo "   Waiting for pods to be ready..."
	@kubectl wait --for=condition=ready pod -l app=consumer-workload -n nimbusguard --timeout=60s 2>/dev/null || echo "⚠️  Consumer workload not ready"
	@kubectl wait --for=condition=ready pod -l app=load-generator -n nimbusguard --timeout=60s 2>/dev/null || echo "⚠️  Load generator not ready"
	@kubectl wait --for=condition=ready pod -l app=kafka -n nimbusguard --timeout=60s 2>/dev/null || echo "⚠️  Kafka not ready"
	@kubectl wait --for=condition=ready pod -l app=nimbusguard-operator -n nimbusguard --timeout=60s 2>/dev/null || echo "⚠️  Operator not ready"
	@echo "   Starting port forwarding..."
	@kubectl port-forward -n nimbusguard svc/consumer-workload 8080:8080 > /dev/null 2>&1 & \
	kubectl port-forward -n nimbusguard svc/load-generator 8081:8081 > /dev/null 2>&1 & \
	kubectl port-forward -n nimbusguard svc/nimbusguard-operator 8082:8080 > /dev/null 2>&1 & \
	kubectl port-forward -n monitoring svc/prometheus-server 9090:9090 > /dev/null 2>&1 & \
	sleep 2
	@echo "✅ Port forwarding started!"
	@echo ""
	@echo "Services available at:"
	@echo "   Consumer Workload: http://localhost:8080"
	@echo "   Load Generator:    http://localhost:8081"
	@echo "   Operator:          http://localhost:8082"
	@echo "   Prometheus:        http://localhost:9090 (if monitoring stack deployed)"
