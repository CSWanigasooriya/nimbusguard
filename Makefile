# NimbusGuard Makefile

.PHONY: help build build-base build-consumer build-generator build-all dev prod run forward stop-forward status logs restart clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-10s %s\n", $$1, $$2}'

setup: ## Setup development environment (install latest tools)
	@echo "🔧 Setting up NimbusGuard development environment..."
	@echo "📅 Installing LATEST versions of all tools..."
	@echo ""
	
	# Check if running on macOS
	@if [ "$$(uname)" = "Darwin" ]; then \
		echo "🍎 Detected macOS - using Homebrew for installations"; \
		if ! command -v brew >/dev/null 2>&1; then \
			echo "❌ Homebrew not found. Please install it first: https://brew.sh"; \
			exit 1; \
		fi; \
		echo "🔄 Updating Homebrew..."; \
		brew update; \
	else \
		echo "🐧 Detected Linux - using latest release downloads"; \
	fi
	@echo ""
	
	# Install/Update kubectl to latest stable
	@echo "📦 Installing/Updating kubectl to latest stable..."
	@if [ "$$(uname)" = "Darwin" ]; then \
		if command -v kubectl >/dev/null 2>&1; then \
			brew upgrade kubernetes-cli 2>/dev/null || brew install kubernetes-cli; \
		else \
			brew install kubernetes-cli; \
		fi; \
	else \
		KUBECTL_VERSION=$$(curl -L -s https://dl.k8s.io/release/stable.txt); \
		echo "📥 Downloading kubectl $$KUBECTL_VERSION..."; \
		curl -LO "https://dl.k8s.io/release/$$KUBECTL_VERSION/bin/linux/amd64/kubectl" && \
		chmod +x kubectl && sudo mv kubectl /usr/local/bin/; \
	fi
	@echo "✅ kubectl installed: $$(kubectl version --client --short 2>/dev/null || kubectl version --client)"
	
	# Install/Update Helm to latest
	@echo "📦 Installing/Updating Helm to latest..."
	@if [ "$$(uname)" = "Darwin" ]; then \
		if command -v helm >/dev/null 2>&1; then \
			brew upgrade helm 2>/dev/null || brew install helm; \
		else \
			brew install helm; \
		fi; \
	else \
		curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 && \
		chmod 700 get_helm.sh && ./get_helm.sh && rm get_helm.sh; \
	fi
	@echo "✅ Helm installed: $$(helm version --short)"
	

	
	# Install/Update Docker (macOS only)
	@if [ "$$(uname)" = "Darwin" ]; then \
		echo "📦 Checking Docker Desktop..."; \
		if ! command -v docker >/dev/null 2>&1; then \
			echo "📥 Installing Docker Desktop..."; \
			brew install --cask docker; \
			echo "⚠️  Please start Docker Desktop manually after installation"; \
		else \
			echo "🔄 Updating Docker Desktop..."; \
			brew upgrade --cask docker 2>/dev/null || echo "✅ Docker already up to date"; \
		fi; \
	fi
	@if command -v docker >/dev/null 2>&1; then \
		echo "✅ Docker installed: $$(docker --version)"; \
	fi
	
	# Install/Update additional useful tools
	@echo "📦 Installing additional development tools..."
	@if [ "$$(uname)" = "Darwin" ]; then \
		if ! command -v jq >/dev/null 2>&1; then brew install jq; fi; \
		if ! command -v yq >/dev/null 2>&1; then brew install yq; fi; \
		if ! command -v k9s >/dev/null 2>&1; then brew install k9s; fi; \
	else \
		if ! command -v jq >/dev/null 2>&1; then \
			JQ_VERSION=$$(curl -s "https://api.github.com/repos/jqlang/jq/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/'); \
			curl -L "https://github.com/jqlang/jq/releases/download/$$JQ_VERSION/jq-linux-amd64" -o jq && \
			chmod +x jq && sudo mv jq /usr/local/bin/; \
		fi; \
		if ! command -v yq >/dev/null 2>&1; then \
			YQ_VERSION=$$(curl -s "https://api.github.com/repos/mikefarah/yq/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/'); \
			curl -L "https://github.com/mikefarah/yq/releases/download/$$YQ_VERSION/yq_linux_amd64" -o yq && \
			chmod +x yq && sudo mv yq /usr/local/bin/; \
		fi; \
		if ! command -v k9s >/dev/null 2>&1; then \
			K9S_VERSION=$$(curl -s "https://api.github.com/repos/derailed/k9s/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/'); \
			curl -L "https://github.com/derailed/k9s/releases/download/$$K9S_VERSION/k9s_Linux_amd64.tar.gz" | tar xz && \
			chmod +x k9s && sudo mv k9s /usr/local/bin/; \
		fi; \
	fi
	
	# Setup and update Helm repositories
	@echo ""
	@echo "🔄 Setting up and updating Helm repositories..."
	@helm repo add kedacore https://kedacore.github.io/charts 2>/dev/null || true
	@helm repo add grafana https://grafana.github.io/helm-charts 2>/dev/null || true
	@helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
	@helm repo add jetstack https://charts.jetstack.io 2>/dev/null || true
	@helm repo update
	@echo "✅ Helm repositories configured and updated"
	
	@echo ""
	@echo "🎉 Environment setup complete with LATEST versions!"
	@echo "📋 Installed tools:"
	@echo "   • kubectl: $$(kubectl version --client --short 2>/dev/null || kubectl version --client | head -1)"
	@echo "   • helm: $$(helm version --short)"

	@echo "   • docker: $$(docker --version 2>/dev/null || echo 'not installed')"
	@echo "   • jq: $$(jq --version 2>/dev/null || echo 'not installed')"
	@echo "   • yq: $$(yq --version 2>/dev/null || echo 'not installed')"
	@echo "   • k9s: $$(k9s version -s 2>/dev/null || echo 'not installed')"
	@echo ""
	@echo "🚀 Ready to deploy! Try: make keda-install && make dev"

setup-update: ## Update all existing tools to latest versions
	@echo "🔄 Updating all tools to latest versions..."
	@if [ "$$(uname)" = "Darwin" ]; then \
		brew update && brew upgrade kubernetes-cli helm jq yq k9s 2>/dev/null; \
		brew upgrade --cask docker 2>/dev/null || true; \
	else \
		echo "🐧 For Linux, please run 'make setup' to get latest versions"; \
	fi
	@helm repo update
	@echo "✅ All tools updated!"

keda-install: ## Install KEDA using Helm
	@echo "🎯 Installing KEDA..."
	@if ! helm list -n keda | grep -q keda 2>/dev/null; then \
		helm install keda kedacore/keda --namespace keda --create-namespace \
			--set operator.replicaCount=1 \
			--set webhooks.enabled=true \
			--set prometheus.metricServer.enabled=false; \
		echo "✅ KEDA installed successfully"; \
	else \
		echo "✅ KEDA already installed"; \
	fi
	@kubectl get pods -n keda

keda-uninstall: ## Uninstall KEDA
	@echo "🗑️  Uninstalling KEDA..."
	@helm uninstall keda -n keda 2>/dev/null || true
	@kubectl delete namespace keda --ignore-not-found=true
	@echo "✅ KEDA uninstalled"

build-base: ## Build base Docker image
	@echo "🔨 Building nimbusguard-base image..."
	@docker buildx bake -f docker-bake.hcl --set *.output=type=docker nimbusguard-base
	@echo "✅ Base image built"

build-consumer: build-base ## Build consumer Docker image
	@echo "🔨 Building nimbusguard-consumer image..."
	@docker buildx bake -f docker-bake.hcl --set *.output=type=docker consumer
	@echo "✅ Consumer image built"

build-generator: build-base ## Build load generator Docker image
	@echo "🔨 Building nimbusguard-generator image..."
	@docker buildx bake -f docker-bake.hcl --set *.output=type=docker generator
	@echo "✅ Generator image built"

build-all: ## Build all Docker images
	@echo "🔨 Building all nimbusguard images..."
	@docker buildx bake -f docker-bake.hcl --set *.output=type=docker all
	@echo "✅ All images built"

build: build-all ## Alias for build-all

dev: build-all ## Build images and deploy to development
	@echo "🚀 Building images and deploying to development..."
	kubectl apply -k kubernetes-manifests/overlays/development
	@echo "✅ Development deployment complete!"

prod: ## Deploy to production  
	kubectl apply -k kubernetes-manifests/overlays/production

run: ## Dry run deployment
	kubectl apply -k kubernetes-manifests/overlays/development --dry-run=client

forward: stop-forward ## Port forward ALL services at once
	@echo "🚀 Starting all port forwarding in background..."
	@kubectl port-forward -n nimbusguard svc/consumer 8000:8000 > /dev/null 2>&1 &
	@kubectl port-forward -n nimbusguard svc/prometheus 9090:9090 > /dev/null 2>&1 &
	@kubectl port-forward -n nimbusguard svc/grafana 3000:3000 > /dev/null 2>&1 &
	@kubectl port-forward -n nimbusguard svc/loki 3100:3100 > /dev/null 2>&1 &
	@kubectl port-forward -n nimbusguard svc/tempo 3200:3200 > /dev/null 2>&1 &
	@kubectl port-forward -n nimbusguard svc/alloy 8080:8080 > /dev/null 2>&1 &
	@sleep 2
	@echo "✅ All services forwarded!"
	@echo "📊 Consumer:    http://localhost:8000"
	@echo "📈 Prometheus:  http://localhost:9090"
	@echo "📋 Grafana:     http://localhost:3000  (admin/admin)"
	@echo "📜 Loki:        http://localhost:3100"
	@echo "🔍 Tempo:       http://localhost:3200"
	@echo "🤖 Alloy:       http://localhost:8080"
	@echo ""
	@echo "Use 'make stop-forward' to stop all forwarding"

stop-forward: ## Stop ALL port forwarding
	@echo "🛑 Stopping all port forwarding..."
	@pkill -f "kubectl port-forward.*nimbusguard" || true
	@echo "✅ All port forwarding stopped"

status: ## Check deployment status
	kubectl get pods,svc -n nimbusguard -l app=nimbusguard

logs: ## Follow consumer logs
	kubectl logs -n nimbusguard -l app.kubernetes.io/name=consumer -f

restart: ## Restart consumer deployment
	kubectl rollout restart deployment/consumer -n nimbusguard

clean: ## Delete all resources
	kubectl delete namespace nimbusguard --ignore-not-found=true