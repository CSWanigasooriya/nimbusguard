.PHONY: help dev build test clean deploy k8s-dev k8s-deploy k8s-clean logs health port-forward status observability observability-status test-load test-ai test-kafka metrics

help:
	@echo "NimbusGuard Development & Operations Commands"
	@echo "============================================="
	@echo ""
	@echo "🐳 Docker Development:"
	@echo "  dev              - Start development environment with Docker Compose"
	@echo "  build            - Build all Docker images"
	@echo "  clean            - Clean up Docker containers and images"
	@echo ""
	@echo "☸️  Kubernetes Operations:"
	@echo "  k8s-dev          - 🚀 One-command K8s setup: build + deploy + port-forward"
	@echo "  k8s-deploy       - Deploy complete NimbusGuard system to Kubernetes"
	@echo "  k8s-clean        - Clean up Kubernetes deployment"
	@echo "  observability    - Deploy only observability stack"
	@echo "  observability-status - Check observability stack status"
	@echo "  port-forward     - Set up port forwarding for all services"
	@echo ""
	@echo "🧪 Testing & Monitoring:"
	@echo "  test             - Run system health checks"
	@echo "  test-load        - Generate test load and scaling events"
	@echo "  test-ai          - Test AI-powered scaling workflow"
	@echo "  test-kafka       - Test Kafka connectivity and topics"
	@echo "  status           - Show system status and metrics"
	@echo "  logs             - Follow application logs"
	@echo ""
	@echo "🔧 Utilities:"
	@echo "  health           - Check all service health"
	@echo "  metrics          - Show current metrics"

# Docker Development Environment
dev:
	@echo "🐳 Starting NimbusGuard development environment..."
	docker-compose up --build -d
	@echo "✅ Services starting..."
	@echo "   Consumer Workload: http://localhost:8080"
	@echo "   Load Generator:    http://localhost:8081"
	@echo "   LangGraph Operator:http://localhost:8082"
	@echo "   Prometheus:        http://localhost:9090"
	@echo ""
	@echo "💡 Run 'make test' to verify services are ready"

build:
	@echo "🔨 Building Docker images..."
	docker build -t nimbusguard/consumer-workload:latest src/consumer-workload/
	docker build -t nimbusguard/load-generator:latest src/load-generator/
	docker build -t nimbusguard/langgraph-operator:latest src/langgraph-operator/
	@echo "✅ All images built successfully"

clean:
	@echo "🧹 Cleaning up Docker environment..."
	docker-compose down -v --remove-orphans
	docker system prune -f
	@echo "✅ Docker cleanup completed"

# Kubernetes Operations
k8s-dev:
	@echo "🚀 Starting complete NimbusGuard Kubernetes development environment..."
	@echo "   This will: Build images → Deploy system → Setup port forwarding"
	@echo ""
	@echo "📦 Step 1/3: Building Docker images..."
	@make build
	@echo ""
	@echo "🚀 Step 2/3: Deploying to Kubernetes..."
	@make k8s-deploy
	@echo ""
	@echo "⏳ Step 3/3: Waiting for services to be ready..."
	@echo "   Waiting for core services to start..."
	@sleep 10
	@kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=consumer-workload -n nimbusguard --timeout=120s || echo "⚠️  Consumer workload taking longer than expected"
	@kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=kafka -n nimbusguard --timeout=60s || echo "⚠️  Kafka taking longer than expected"
	@echo ""
	@echo "🔗 Setting up port forwarding..."
	@echo "   🌟 NimbusGuard will be available at:"
	@echo "   📊 Grafana Dashboard:  http://localhost:3000 (admin/nimbusguard)"
	@echo "   🎯 Consumer Workload:  http://localhost:8080"
	@echo "   🚀 Load Generator:     http://localhost:8081"
	@echo "   🧠 LangGraph Operator: http://localhost:8082"
	@echo "   📈 Prometheus:         http://localhost:9090"
	@echo ""
	@echo "   💡 Press Ctrl+C to stop port forwarding when done"
	@echo ""
	@make port-forward

k8s-deploy:
	@echo "🚀 Deploying complete NimbusGuard system to Kubernetes..."
	@echo "   This includes: Base infrastructure, Observability stack, KEDA"
	@chmod +x scripts/operations/deploy-nimbusguard.sh
	@./scripts/operations/deploy-nimbusguard.sh
	@echo ""
	@echo "✅ Deployment completed! Run 'make port-forward' to access services"

k8s-clean:
	@echo "🧹 Cleaning up Kubernetes deployment..."
	@echo "⚠️  This will delete the entire nimbusguard namespace!"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	kubectl delete namespace nimbusguard --ignore-not-found=true
	kubectl delete crd scalingpolicies.nimbusguard.io aimodels.nimbusguard.io --ignore-not-found=true
	@echo "✅ Kubernetes cleanup completed"

observability:
	@echo "📊 Deploying observability stack (Prometheus, Grafana, Loki, Tempo)..."
	kubectl apply -k kubernetes-manifests/components/observability/
	@echo "✅ Observability stack deployed"
	@echo "   Run 'make port-forward' to access all services"

observability-status:
	@echo "📊 Observability Stack Status"
	@echo "============================="
	@echo ""
	@echo "🔍 Services:"
	@kubectl get services -n nimbusguard -l component=observability 2>/dev/null || echo "   ❌ No observability services found"
	@echo ""
	@echo "📋 Pods:"
	@kubectl get pods -n nimbusguard -l component=observability 2>/dev/null || echo "   ❌ No observability pods found"
	@echo ""
	@echo "📊 Service Health (requires port-forward):"
	@echo "Grafana:"
	@curl -f http://localhost:3000/api/health 2>/dev/null && echo "  ✅ Healthy" || echo "  ❌ Not responding (run 'make port-forward')"
	@echo "Prometheus:"
	@curl -f http://localhost:9090/-/healthy 2>/dev/null && echo "  ✅ Healthy" || echo "  ❌ Not responding (run 'make port-forward')"
	@echo "Loki:"
	@curl -f http://localhost:3100/ready 2>/dev/null && echo "  ✅ Healthy" || echo "  ❌ Not responding (run 'make port-forward')"
	@echo "Tempo:"
	@curl -f http://localhost:3200/ready 2>/dev/null && echo "  ✅ Healthy" || echo "  ❌ Not responding (run 'make port-forward')"

port-forward:
	@echo "🔗 Setting up port forwarding for all NimbusGuard services..."
	@chmod +x scripts/utilities/port-forward.sh
	@./scripts/utilities/port-forward.sh

# Testing & Validation
test:
	@echo "🧪 Running system health checks..."
	@echo "=== Core Services ==="
	@echo "Consumer Workload:"
	@curl -f http://localhost:8080/health 2>/dev/null && echo "  ✅ Healthy" || echo "  ❌ Not responding"
	@echo "Load Generator:"
	@curl -f http://localhost:8081/health 2>/dev/null && echo "  ✅ Healthy" || echo "  ❌ Not responding"
	@echo "LangGraph Operator:"
	@curl -f http://localhost:8082/healthz 2>/dev/null && echo "  ✅ Healthy" || echo "  ❌ Not responding"
	@echo ""
	@echo "=== Observability Stack ==="
	@echo "Prometheus:"
	@curl -f http://localhost:9090/-/healthy 2>/dev/null && echo "  ✅ Healthy" || echo "  ❌ Not responding"
	@echo "Grafana:"
	@curl -f http://localhost:3000/api/health 2>/dev/null && echo "  ✅ Healthy" || echo "  ❌ Not responding"
	@echo "Loki:"
	@curl -f http://localhost:3100/ready 2>/dev/null && echo "  ✅ Healthy" || echo "  ❌ Not responding"
	@echo "Tempo:"
	@curl -f http://localhost:3200/ready 2>/dev/null && echo "  ✅ Healthy" || echo "  ❌ Not responding"
	@echo "OpenTelemetry Collector:"
	@curl -f http://localhost:13133/ 2>/dev/null && echo "  ✅ Healthy" || echo "  ❌ Not responding"

test-load:
	@echo "🚀 Generating test load and scaling events..."
	@echo "1. Generating CPU load (80% for 5 minutes)..."
	@curl -X POST 'http://localhost:8080/workload/cpu?intensity=80&duration=300' \
		-H 'Content-Type: application/json' && echo " ✅ CPU load started" || echo " ❌ Failed"
	@echo ""
	@echo "2. Generating memory load (70% for 5 minutes)..."
	@curl -X POST 'http://localhost:8080/workload/memory?intensity=70&duration=300' \
		-H 'Content-Type: application/json' && echo " ✅ Memory load started" || echo " ❌ Failed"
	@echo ""
	@echo "3. Generating HTTP load pattern (spike for 5 minutes)..."
	@curl -X POST http://localhost:8081/load/generate \
		-H 'Content-Type: application/json' \
		-d '{"pattern":"spike","duration":300,"target":"http"}' && echo " ✅ HTTP load started" || echo " ❌ Failed"
	@echo ""
	@echo "4. Triggering scaling event..."
	@curl -X POST http://localhost:8080/events/trigger \
		-H 'Content-Type: application/json' \
		-d '{"event_type":"high_cpu_usage","service":"consumer-workload","value":85}' && echo " ✅ Scaling event triggered" || echo " ❌ Failed"
	@echo ""
	@echo "🔍 Monitor scaling with: kubectl get pods -n nimbusguard -w"

test-ai:
	@echo "🤖 Testing AI-powered scaling workflow..."
	@echo "1. Checking current pod count..."
	@kubectl get pods -n nimbusguard -l app.kubernetes.io/component=consumer-workload --no-headers | wc -l | xargs echo "   Current pods:"
	@echo ""
	@echo "2. Generating high load to trigger AI scaling..."
	@curl -X POST 'http://localhost:8080/workload/cpu?intensity=95&duration=180' >/dev/null 2>&1 &
	@curl -X POST 'http://localhost:8080/workload/memory?intensity=85&duration=180' >/dev/null 2>&1 &
	@echo "   ✅ High load generated"
	@echo ""
	@echo "3. Waiting for AI decision (30 seconds)..."
	@sleep 30
	@echo ""
	@echo "4. Checking for scaling decision..."
	@kubectl get pods -n nimbusguard -l app.kubernetes.io/component=consumer-workload --no-headers | wc -l | xargs echo "   New pod count:"
	@echo "   📊 Check Grafana for metrics: http://localhost:3000"

status:
	@echo "📈 NimbusGuard System Status"
	@echo "============================"
	@echo ""
	@echo "☸️  Kubernetes Pods:"
	@kubectl get pods -n nimbusguard 2>/dev/null || echo "   ❌ Namespace not found - run 'make k8s-deploy'"
	@echo ""
	@echo "🔄 HPA Status:"
	@kubectl get hpa -n nimbusguard 2>/dev/null || echo "   ❌ No HPA found"
	@echo ""
	@echo "📊 KEDA ScaledObjects:"
	@kubectl get scaledobjects -n nimbusguard 2>/dev/null || echo "   ❌ No ScaledObjects found"
	@echo ""
	@echo "🎯 Custom Resources:"
	@kubectl get scalingpolicies -n nimbusguard 2>/dev/null || echo "   ❌ No ScalingPolicies found"

health:
	@echo "💊 Comprehensive Health Check"
	@echo "============================="
	@echo ""
	@echo "🐳 Docker Services (if running):"
	@docker-compose ps 2>/dev/null || echo "   Docker Compose not running"
	@echo ""
	@echo "☸️  Kubernetes Services:"
	@kubectl get services -n nimbusguard 2>/dev/null || echo "   ❌ Namespace not found"
	@echo ""
	@echo "🌐 Service Endpoints (if port-forwarded):"
	@make test

metrics:
	@echo "📊 Current System Metrics"
	@echo "========================"
	@echo ""
	@echo "📈 Consumer Workload Metrics:"
	@curl -s http://localhost:8080/metrics 2>/dev/null | grep -E "(cpu_usage|memory_usage|requests_total)" | head -5 || echo "   ❌ Metrics not available"
	@echo ""
	@echo "🚀 Load Generator Metrics:"
	@curl -s http://localhost:8081/metrics 2>/dev/null | grep -E "(requests_generated|load_pattern)" | head -3 || echo "   ❌ Metrics not available"
	@echo ""
	@echo "🧠 AI Operator Status:"
	@curl -s http://localhost:8082/metrics 2>/dev/null | grep -E "(scaling_decisions|ai_confidence)" | head -3 || echo "   ❌ Metrics not available"
	@echo ""
	@echo "📊 Prometheus Targets:"
	@curl -s http://localhost:9090/api/v1/targets 2>/dev/null | jq -r '.data.activeTargets | length' 2>/dev/null | xargs -I {} echo "  {} active targets" || echo "   ❌ Prometheus not available"
	@echo ""
	@echo "📋 Loki Status:"
	@curl -s http://localhost:3100/api/v1/label 2>/dev/null | jq -r '.data | length' 2>/dev/null | xargs -I {} echo "  {} log labels" || echo "   ❌ Loki not available"

logs:
	@echo "📋 Following application logs..."
	@echo "Choose log source:"
	@echo "1. Docker Compose logs"
	@echo "2. Kubernetes logs (consumer-workload)"
	@echo "3. Kubernetes logs (load-generator)" 
	@echo "4. Kubernetes logs (langgraph-operator)"
	@read -p "Select option (1-4): " choice; \
	case $$choice in \
		1) docker-compose logs -f ;; \
		2) kubectl logs -f -n nimbusguard -l app.kubernetes.io/component=consumer-workload ;; \
		3) kubectl logs -f -n nimbusguard -l app.kubernetes.io/component=load-generator ;; \
		4) kubectl logs -f -n nimbusguard -l app.kubernetes.io/component=langgraph-operator ;; \
		*) echo "Invalid option" ;; \
	esac

test-kafka:
	@echo "🔗 Testing Kafka connectivity and topics..."
	@echo "=== Kafka Service ==="
	@kubectl get service kafka -n nimbusguard 2>/dev/null && echo "  ✅ Service exists" || echo "  ❌ Service not found"
	@echo "=== Kafka Pod ==="
	@kubectl get pods -n nimbusguard -l app.kubernetes.io/component=kafka 2>/dev/null && echo "  ✅ Pod running" || echo "  ❌ Pod not found"
	@echo "=== Kafka Topics ==="
	@echo "Available topics:"
	@kubectl exec -n nimbusguard $$(kubectl get pods -n nimbusguard -l app.kubernetes.io/component=kafka -o jsonpath='{.items[0].metadata.name}') -- kafka-topics --bootstrap-server localhost:9092 --list 2>/dev/null | grep nimbusguard | sed 's/^/  - /' || echo "  ❌ Cannot list topics"
	@echo "=== Kafka Log Configuration ==="
	@kubectl exec -n nimbusguard $$(kubectl get pods -n nimbusguard -l app.kubernetes.io/component=kafka -o jsonpath='{.items[0].metadata.name}') -- ls -la /opt/kafka/config/custom-log4j.properties 2>/dev/null && echo "  ✅ Custom log4j.properties mounted" || echo "  ❌ Custom log4j.properties missing"

# Legacy deploy command for compatibility
deploy: k8s-deploy 