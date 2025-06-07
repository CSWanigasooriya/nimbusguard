.PHONY: help dev k8s-dev clean k8s-clean status forward

help:
	@echo "NimbusGuard - Quick Development Commands"
	@echo "========================================"
	@echo ""
	@echo "🚀 Development:"
	@echo "  dev          - Docker development environment"
	@echo "  k8s-dev      - Kubernetes development (auto setup + port-forward)"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  clean        - Clean Docker environment"
	@echo "  k8s-clean    - Clean Kubernetes environment"
	@echo ""
	@echo "📊 Monitoring:"
	@echo "  status       - Show system status (with port-forward)"
	@echo "  forward      - Setup persistent port forwarding only"

# Docker Development
dev:
	@echo "🐳 Starting Docker development environment..."
	@echo "   Checking for base image..."
	@if ! docker image inspect nimbusguard/base:latest >/dev/null 2>&1; then \
		echo "   Building base image (first time setup)..."; \
		DOCKER_BUILDKIT=1 docker build \
			-t nimbusguard/base:latest \
			-f docker/base.Dockerfile .; \
		echo "   ✅ Base image built!"; \
	else \
		echo "   ✅ Base image found"; \
	fi
	@echo "   Building services with cache optimization..."
	@DOCKER_BUILDKIT=1 docker-compose build --parallel
	@docker-compose up -d
	@echo ""
	@echo "✅ Services available at:"
	@echo "   Consumer Workload: http://localhost:8080"
	@echo "   Load Generator:    http://localhost:8081"
	@echo "   LangGraph Operator:http://localhost:8082"
	@echo "   Prometheus:        http://localhost:9090"

# Kubernetes Development - All-in-one command
k8s-dev:
	@echo "🚀 Starting Kubernetes development environment..."
	@echo ""
	@echo "🔨 Building images with caching..."
	@echo "   Checking for base image..."
	@if ! docker image inspect nimbusguard/base:latest >/dev/null 2>&1; then \
		echo "   Building base image (first time setup)..."; \
		DOCKER_BUILDKIT=1 docker build \
			-t nimbusguard/base:latest \
			-f docker/base.Dockerfile .; \
		echo "   ✅ Base image built!"; \
	else \
		echo "   ✅ Base image found"; \
	fi
	@echo "   Building consumer workload..."
	@DOCKER_BUILDKIT=1 docker build \
		-t nimbusguard/consumer-workload:latest \
		src/consumer-workload/
	@echo "   Building load generator..."
	@DOCKER_BUILDKIT=1 docker build \
		-t nimbusguard/load-generator:latest \
		src/load-generator/
	@echo "   Building LangGraph operator..."
	@DOCKER_BUILDKIT=1 docker build \
		-t nimbusguard/langgraph-operator:latest \
		src/langgraph-operator/
	@echo ""
	@echo "☸️  Deploying to Kubernetes..."
	@kubectl apply -f kubernetes-manifests/base/namespace.yaml || true
	@kubectl apply -f kubernetes-manifests/base/crds.yaml || true
	@sleep 3
	@kubectl apply -k kubernetes-manifests/base/ || true
	@kubectl apply -k kubernetes-manifests/components/observability/ || true
	@kubectl apply -k kubernetes-manifests/components/keda/ || true
	@echo ""
	@echo "🔑 Setting up OpenAI API key..."
	@echo "   Enter your OpenAI API key (required for AI operator):"; \
	read -p "   API Key (sk-...): " OPENAI_KEY; \
	if [ -n "$$OPENAI_KEY" ] && [ "$${OPENAI_KEY#sk-}" != "$$OPENAI_KEY" ]; then \
		echo "   Applying OpenAI secret..."; \
		kubectl create secret generic openai-api-key -n nimbusguard \
			--from-literal=api-key="$$OPENAI_KEY" \
			--dry-run=client -o yaml | kubectl apply -f -; \
		echo "   Restarting operator to pick up new key..."; \
		kubectl rollout restart deployment/langgraph-operator -n nimbusguard 2>/dev/null || true; \
		echo "   ✅ OpenAI key configured and operator restarted"; \
	else \
		echo "   ❌ Invalid OpenAI key format - must start with 'sk-'"; \
		echo "   ⚠️  Operator will fail without valid OpenAI key"; \
	fi
	@echo ""
	@echo "⏳ Waiting for pods to start..."
	@sleep 20
	@echo ""
	@echo "🔗 Setting up port forwarding..."
	@echo "   Starting port forwards..."
	@kubectl port-forward -n nimbusguard $$(kubectl get pods -n nimbusguard -l app.kubernetes.io/component=observability | grep grafana | head -1 | awk '{print $$1}') 3000:3000 >/dev/null 2>&1 &
	@kubectl port-forward -n nimbusguard $$(kubectl get pods -n nimbusguard -l app.kubernetes.io/component=base | grep consumer-workload | head -1 | awk '{print $$1}') 8080:8080 >/dev/null 2>&1 &
	@kubectl port-forward -n nimbusguard $$(kubectl get pods -n nimbusguard -l app.kubernetes.io/component=base | grep load-generator | head -1 | awk '{print $$1}') 8081:8081 >/dev/null 2>&1 &
	@kubectl port-forward -n nimbusguard $$(kubectl get pods -n nimbusguard -l app.kubernetes.io/component=operator | tail -1 | awk '{print $$1}') 8082:8080 >/dev/null 2>&1 &
	@kubectl port-forward -n nimbusguard $$(kubectl get pods -n nimbusguard -l app.kubernetes.io/component=observability | grep prometheus | head -1 | awk '{print $$1}') 9090:9090 >/dev/null 2>&1 &
	@sleep 5
	@echo ""
	@echo "🌟 NimbusGuard is ready!"
	@echo "========================"
	@echo "📊 Grafana:       http://localhost:3000 (admin/nimbusguard)"
	@echo "🎯 Workload:      http://localhost:8080"
	@echo "🚀 Load Gen:      http://localhost:8081"
	@echo "🧠 AI Operator:   http://localhost:8082"
	@echo "📈 Prometheus:    http://localhost:9090"
	@echo ""
	@echo "💡 Run 'make status' to check system health"
	@echo "💡 Use Ctrl+C to stop, then 'make k8s-clean' to cleanup"

# Cleanup Commands
clean:
	@echo "🧹 Cleaning Docker environment..."
	@docker-compose down -v --remove-orphans 2>/dev/null || true
	@docker system prune -f
	@echo "✅ Docker cleanup completed"

k8s-clean:
	@echo "🧹 Cleaning Kubernetes environment..."
	@echo "⚠️  This will delete the entire nimbusguard namespace!"
	@read -p "Continue? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	@pkill -f "kubectl port-forward" 2>/dev/null || true
	@kubectl delete namespace nimbusguard --ignore-not-found=true
	@kubectl delete crd scalingpolicies.nimbusguard.io aimodels.nimbusguard.io --ignore-not-found=true
	@echo "✅ Kubernetes cleanup completed"

# Status and Health
status:
	@echo "📊 NimbusGuard System Status"
	@echo "============================="
	@echo ""
	@echo "☸️  Kubernetes Pods:"
	@kubectl get pods -n nimbusguard 2>/dev/null || echo "   ❌ Namespace not found - run 'make k8s-dev'"
	@echo ""
	@echo "🌐 Service Health (via port-forward):"
	@echo "   Setting up port forwards..."
	@pkill -f "kubectl port-forward.*nimbusguard" 2>/dev/null || true
	@sleep 2
	@kubectl port-forward -n nimbusguard $$(kubectl get pods -n nimbusguard -l app.kubernetes.io/component=observability | grep grafana | head -1 | awk '{print $$1}') 3000:3000 >/dev/null 2>&1 & \
	kubectl port-forward -n nimbusguard $$(kubectl get pods -n nimbusguard -l app.kubernetes.io/component=base | grep consumer-workload | head -1 | awk '{print $$1}') 8080:8080 >/dev/null 2>&1 & \
	kubectl port-forward -n nimbusguard $$(kubectl get pods -n nimbusguard -l app.kubernetes.io/component=base | grep load-generator | head -1 | awk '{print $$1}') 8081:8081 >/dev/null 2>&1 & \
	kubectl port-forward -n nimbusguard $$(kubectl get pods -n nimbusguard -l app.kubernetes.io/component=operator | tail -1 | awk '{print $$1}') 8082:8080 >/dev/null 2>&1 & \
	kubectl port-forward -n nimbusguard $$(kubectl get pods -n nimbusguard -l app.kubernetes.io/component=observability | grep prometheus | head -1 | awk '{print $$1}') 9090:9090 >/dev/null 2>&1 & \
	sleep 10
	@printf "Grafana:      " && curl -f -s http://localhost:3000/api/health 2>/dev/null >/dev/null && echo "✅ Ready" || echo "❌ Not ready"
	@printf "Workload:     " && curl -f -s http://localhost:8080/health 2>/dev/null >/dev/null && echo "✅ Ready" || echo "❌ Not ready"  
	@printf "Load Gen:     " && curl -f -s http://localhost:8081/health 2>/dev/null >/dev/null && echo "✅ Ready" || echo "❌ Not ready"
	@printf "AI Operator:  " && curl -f -s http://localhost:8082/healthz 2>/dev/null | grep -q "nimbusguard" && echo "✅ Ready" || echo "❌ Not ready"
	@printf "Prometheus:   " && curl -f -s http://localhost:9090/-/ready 2>/dev/null >/dev/null && echo "✅ Ready" || echo "❌ Not ready"
	@echo ""
	@echo "🔗 Active Port Forwards:"
	@echo "   📊 Grafana:       http://localhost:3000 (admin/nimbusguard)"
	@echo "   🎯 Workload:      http://localhost:8080"
	@echo "   🚀 Load Gen:      http://localhost:8081"
	@echo "   🧠 AI Operator:   http://localhost:8082"
	@echo "   📈 Prometheus:    http://localhost:9090"
	@echo ""
	@echo "💡 Port forwards will remain active until stopped with Ctrl+C"
	@echo "💡 Run 'make k8s-clean' to cleanup everything"

# Port Forwarding Only
forward:
	@echo "🔗 Setting up persistent port forwarding..."
	@pkill -f "kubectl port-forward.*nimbusguard" 2>/dev/null || true
	@sleep 2
	@echo "   Starting port forwards..."
	@kubectl port-forward -n nimbusguard $$(kubectl get pods -n nimbusguard -l app.kubernetes.io/component=observability | grep grafana | head -1 | awk '{print $$1}') 3000:3000 >/dev/null 2>&1 & \
	kubectl port-forward -n nimbusguard $$(kubectl get pods -n nimbusguard -l app.kubernetes.io/component=base | grep consumer-workload | head -1 | awk '{print $$1}') 8080:8080 >/dev/null 2>&1 & \
	kubectl port-forward -n nimbusguard $$(kubectl get pods -n nimbusguard -l app.kubernetes.io/component=base | grep load-generator | head -1 | awk '{print $$1}') 8081:8081 >/dev/null 2>&1 & \
	kubectl port-forward -n nimbusguard $$(kubectl get pods -n nimbusguard -l app.kubernetes.io/component=operator | tail -1 | awk '{print $$1}') 8082:8080 >/dev/null 2>&1 & \
	kubectl port-forward -n nimbusguard $$(kubectl get pods -n nimbusguard -l app.kubernetes.io/component=observability | grep prometheus | head -1 | awk '{print $$1}') 9090:9090 >/dev/null 2>&1 &
	@sleep 3
	@echo ""
	@echo "🌟 Port forwards active!"
	@echo "======================="
	@echo "📊 Grafana:       http://localhost:3000 (admin/nimbusguard)"
	@echo "🎯 Workload:      http://localhost:8080"
	@echo "🚀 Load Gen:      http://localhost:8081" 
	@echo "🧠 AI Operator:   http://localhost:8082"
	@echo "📈 Prometheus:    http://localhost:9090"
	@echo ""
	@echo "💡 Port forwards will remain active in background"
	@echo "💡 Use Ctrl+C to stop this command, forwards will continue"
	@echo "💡 Run 'pkill -f \"kubectl port-forward.*nimbusguard\"' to stop all forwards" 