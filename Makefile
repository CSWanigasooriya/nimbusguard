# NimbusGuard Makefile

.PHONY: help build build-base build-consumer build-generator build-all dev prod run forward stop-forward status logs restart clean
.PHONY: load-test-light load-test-medium load-test-heavy load-test-sustained load-test-burst load-test-memory load-test-cpu
.PHONY: load-clean load-status
.PHONY: helm-install helm-upgrade helm-uninstall helm-dev helm-prod helm-test helm-lint helm-template

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-20s %s\n", $$1, $$2}'

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
	else \
		echo "🐧 Detected Linux - using latest release downloads"; \
	fi
	@echo ""
	
	# Install/Update kubectl to latest stable
	@echo "📦 Checking kubectl..."
	@if ! command -v kubectl >/dev/null 2>&1; then \
		echo "📥 Installing kubectl..."; \
		if [ "$$(uname)" = "Darwin" ]; then \
			brew install kubernetes-cli; \
		else \
			KUBECTL_VERSION=$$(curl -L -s https://dl.k8s.io/release/stable.txt); \
			curl -LO "https://dl.k8s.io/release/$$KUBECTL_VERSION/bin/linux/amd64/kubectl" && \
			chmod +x kubectl && sudo mv kubectl /usr/local/bin/; \
		fi; \
	else \
		echo "✅ kubectl already installed"; \
	fi
	@echo "✅ kubectl: $$(kubectl version --client --short 2>/dev/null || kubectl version --client | head -1)"
	
	# Install/Update Helm to latest
	@echo "📦 Checking Helm..."
	@if ! command -v helm >/dev/null 2>&1; then \
		echo "📥 Installing Helm..."; \
		if [ "$$(uname)" = "Darwin" ]; then \
			brew install helm; \
		else \
			curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 && \
			chmod 700 get_helm.sh && ./get_helm.sh && rm get_helm.sh; \
		fi; \
	else \
		echo "✅ Helm already installed"; \
	fi
	@echo "✅ Helm: $$(helm version --short)"
	

	
	# Install/Check Docker (macOS only)
	@if [ "$$(uname)" = "Darwin" ]; then \
		echo "📦 Checking Docker Desktop..."; \
		if ! command -v docker >/dev/null 2>&1; then \
			echo "📥 Installing Docker Desktop..."; \
			brew install --cask docker; \
			echo "⚠️  Please start Docker Desktop manually after installation"; \
		else \
			echo "✅ Docker Desktop already installed"; \
		fi; \
	fi
	@if command -v docker >/dev/null 2>&1; then \
		echo "✅ Docker: $$(docker --version)"; \
	fi
	
	# Install/Check additional useful tools
	@echo "📦 Checking additional development tools..."
	@if [ "$$(uname)" = "Darwin" ]; then \
		if ! command -v jq >/dev/null 2>&1; then echo "📥 Installing jq..." && brew install jq; else echo "✅ jq already installed"; fi; \
		if ! command -v yq >/dev/null 2>&1; then echo "📥 Installing yq..." && brew install yq; else echo "✅ yq already installed"; fi; \
		if ! command -v k9s >/dev/null 2>&1; then echo "📥 Installing k9s..." && brew install k9s; else echo "✅ k9s already installed"; fi; \
	else \
		if ! command -v jq >/dev/null 2>&1; then \
			echo "📥 Installing jq..."; \
			JQ_VERSION=$$(curl -s "https://api.github.com/repos/jqlang/jq/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/'); \
			curl -L "https://github.com/jqlang/jq/releases/download/$$JQ_VERSION/jq-linux-amd64" -o jq && \
			chmod +x jq && sudo mv jq /usr/local/bin/; \
		else \
			echo "✅ jq already installed"; \
		fi; \
		if ! command -v yq >/dev/null 2>&1; then \
			echo "📥 Installing yq..."; \
			YQ_VERSION=$$(curl -s "https://api.github.com/repos/mikefarah/yq/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/'); \
			curl -L "https://github.com/mikefarah/yq/releases/download/$$YQ_VERSION/yq_linux_amd64" -o yq && \
			chmod +x yq && sudo mv yq /usr/local/bin/; \
		else \
			echo "✅ yq already installed"; \
		fi; \
		if ! command -v k9s >/dev/null 2>&1; then \
			echo "📥 Installing k9s..."; \
			K9S_VERSION=$$(curl -s "https://api.github.com/repos/derailed/k9s/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/'); \
			curl -L "https://github.com/derailed/k9s/releases/download/$$K9S_VERSION/k9s_Linux_amd64.tar.gz" | tar xz && \
			chmod +x k9s && sudo mv k9s /usr/local/bin/; \
		else \
			echo "✅ k9s already installed"; \
		fi; \
	fi
	
	# Setup Helm repositories (only update if needed)
	@echo ""
	@echo "📦 Checking Helm repositories..."
	@if ! helm repo list | grep -q kedacore 2>/dev/null; then \
		echo "📥 Adding Helm repositories..."; \
		helm repo add kedacore https://kedacore.github.io/charts 2>/dev/null || true; \
		helm repo add grafana https://grafana.github.io/helm-charts 2>/dev/null || true; \
		helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true; \
		helm repo add jetstack https://charts.jetstack.io 2>/dev/null || true; \
		helm repo update; \
		echo "✅ Helm repositories configured"; \
	else \
		echo "✅ Helm repositories already configured"; \
	fi
	
	# Install metrics-server for CPU/Memory monitoring
	@echo ""
	@echo "📊 Installing metrics-server for k9s and KEDA monitoring..."
	@if ! kubectl get deployment metrics-server -n kube-system >/dev/null 2>&1; then \
		kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml; \
		echo "⏳ Patching metrics-server for Docker Desktop compatibility..."; \
		kubectl patch deployment metrics-server -n kube-system --type='json' -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-insecure-tls"}]' 2>/dev/null || true; \
		kubectl patch deployment metrics-server -n kube-system --type='json' -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname"}]' 2>/dev/null || true; \
		echo "✅ Metrics-server installed and configured"; \
	else \
		echo "✅ Metrics-server already installed"; \
	fi
	
	@echo ""
	@echo "🎉 Environment setup complete!"
	@echo "📋 Available tools:"
	@echo "   • kubectl: $$(kubectl version --client --short 2>/dev/null || kubectl version --client | head -1)"
	@echo "   • helm: $$(helm version --short)"
	@echo "   • docker: $$(docker --version 2>/dev/null || echo 'not installed')"
	@echo "   • jq: $$(jq --version 2>/dev/null || echo 'not installed')"
	@echo "   • yq: $$(yq --version 2>/dev/null || echo 'not installed')"
	@echo "   • k9s: $$(k9s version -s 2>/dev/null || echo 'not installed')"
	@echo ""
	@echo "🚀 Ready to deploy! Try: make helm-dev (recommended) or make dev (legacy)"

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

## 🎯 Helm Chart Commands (Recommended)

helm-lint: ## Lint the Helm chart
	@echo "🔍 Linting Helm chart..."
	@helm lint helm-chart
	@echo "✅ Helm chart linting complete"

helm-template: ## Generate Kubernetes manifests from Helm chart (dry-run)
	@echo "📋 Generating Kubernetes manifests from Helm chart..."
	@helm template nimbusguard helm-chart --debug

helm-install: build-all ## Install NimbusGuard using Helm chart
	@echo "🚀 Installing NimbusGuard with Helm..."
	@echo "🔧 Adding required Helm repositories..."
	@helm repo add kedacore https://kedacore.github.io/charts 2>/dev/null || true
	@helm repo update
	@echo "📦 Installing NimbusGuard..."
	@helm install nimbusguard helm-chart --create-namespace --wait --timeout=600s
	@echo "✅ NimbusGuard installed successfully!"

helm-upgrade: build-all ## Upgrade NimbusGuard installation
	@echo "🔄 Upgrading NimbusGuard with Helm..."
	@helm upgrade nimbusguard helm-chart --wait --timeout=600s
	@echo "✅ NimbusGuard upgraded successfully!"

helm-dev: build-all ## Install/upgrade NimbusGuard for development
	@echo "🚀 Deploying NimbusGuard for development..."
	@helm repo add kedacore https://kedacore.github.io/charts 2>/dev/null || true
	@helm repo update
	@if helm list | grep -q nimbusguard 2>/dev/null; then \
		echo "🔄 Upgrading existing installation..."; \
		helm upgrade nimbusguard helm-chart \
			--set monitoring.grafana.adminPassword=admin \
			--set consumer.image.tag=latest \
			--set keda.scaledObject.minReplicaCount=1 \
			--wait --timeout=600s; \
	else \
		echo "📦 Installing fresh deployment..."; \
		helm install nimbusguard helm-chart \
			--set monitoring.grafana.adminPassword=admin \
			--set consumer.image.tag=latest \
			--set keda.scaledObject.minReplicaCount=1 \
			--create-namespace --wait --timeout=600s; \
	fi
	@echo "✅ Development deployment complete!"

helm-prod: build-all ## Install/upgrade NimbusGuard for production
	@echo "🚀 Deploying NimbusGuard for production..."
	@helm repo add kedacore https://kedacore.github.io/charts 2>/dev/null || true
	@helm repo update
	@if [ -z "$$GRAFANA_PASSWORD" ]; then \
		echo "❌ GRAFANA_PASSWORD environment variable is required for production"; \
		echo "   Set it with: export GRAFANA_PASSWORD=your-secure-password"; \
		exit 1; \
	fi
	@if helm list | grep -q nimbusguard 2>/dev/null; then \
		echo "🔄 Upgrading existing installation..."; \
		helm upgrade nimbusguard helm-chart \
			--set monitoring.grafana.adminPassword=$$GRAFANA_PASSWORD \
			--set consumer.image.tag=latest \
			--set keda.scaledObject.minReplicaCount=2 \
			--set keda.scaledObject.maxReplicaCount=20 \
			--wait --timeout=600s; \
	else \
		echo "📦 Installing fresh deployment..."; \
		helm install nimbusguard helm-chart \
			--set monitoring.grafana.adminPassword=$$GRAFANA_PASSWORD \
			--set consumer.image.tag=latest \
			--set keda.scaledObject.minReplicaCount=2 \
			--set keda.scaledObject.maxReplicaCount=20 \
			--create-namespace --wait --timeout=600s; \
	fi
	@echo "✅ Production deployment complete!"

helm-test: ## Run Helm chart tests
	@echo "🧪 Running Helm chart tests..."
	@helm test nimbusguard
	@echo "✅ Helm tests completed!"

helm-uninstall: ## Uninstall NimbusGuard Helm release
	@echo "🗑️  Uninstalling NimbusGuard..."
	@helm uninstall nimbusguard 2>/dev/null || echo "NimbusGuard not found"
	@echo "🗑️  Cleaning up KEDA..."
	@helm uninstall keda -n keda 2>/dev/null || echo "KEDA not found"
	@kubectl delete namespace keda --ignore-not-found=true
	@kubectl delete namespace nimbusguard --ignore-not-found=true
	@echo "✅ Cleanup complete!"

## 🛠️ Legacy Kubernetes Commands (for reference)

keda-install: ## Install KEDA using Helm
	@echo "🎯 Installing KEDA..."
	@if ! helm list -n keda | grep -q keda 2>/dev/null; then \
		helm install keda kedacore/keda --namespace keda --create-namespace \
			--set operator.replicaCount=1 \
			--set webhooks.enabled=true \
			--set prometheus.metricServer.enabled=false; \
		echo "⏳ Waiting for KEDA operator to be ready..."; \
		sleep 30; \
		echo "✅ KEDA installed successfully"; \
	else \
		echo "✅ KEDA already installed"; \
	fi
	@kubectl get pods -n keda 2>/dev/null || echo "KEDA pods starting..."

# Remove KEDA-managed objects that are no longer present in the manifests (eg, after commenting out the KEDA component)
keda-prune: ## Prune KEDA ScaledObjects/HPA that no longer exist in manifests
	@echo "🧹 Deleting ScaledObjects and HPAs managed by KEDA in nimbusguard namespace..."
	@kubectl delete scaledobject -n nimbusguard --ignore-not-found=true --all
	@kubectl delete hpa -n nimbusguard -l app.kubernetes.io/managed-by=keda-operator --ignore-not-found=true
	@echo "✅ Autoscaling disabled – KEDA resources cleaned up"

# Pause and resume autoscaling via annotations
keda-pause: ## Pause KEDA autoscaling (REPLICAS=<n> to keep n replicas)
	@echo "⏸️  Pausing KEDA autoscaling..."
	@if [ -z "$(REPLICAS)" ]; then \
		kubectl annotate scaledobject -n nimbusguard consumer-scaler autoscaling.keda.sh/paused="true" --overwrite; \
	else \
		kubectl annotate scaledobject -n nimbusguard consumer-scaler \
			autoscaling.keda.sh/paused="true" autoscaling.keda.sh/paused-replicas="$(REPLICAS)" --overwrite; \
	fi
	@echo "✅ Autoscaling paused"

keda-resume: ## Resume KEDA autoscaling (remove pause annotations)
	@echo "▶️  Resuming KEDA autoscaling..."
	@kubectl annotate scaledobject -n nimbusguard consumer-scaler \
		autoscaling.keda.sh/paused- autoscaling.keda.sh/paused-replicas- --overwrite
	@echo "✅ Autoscaling resumed"

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

dev: build-all ## Build images and deploy to development (legacy)
	@echo "🚀 Building images and deploying to development..."
	@echo "🔍 Checking KEDA installation..."
	@if ! helm list -n keda | grep -q keda 2>/dev/null; then \
		echo "⚠️  KEDA not found, installing it first..."; \
		$(MAKE) keda-install; \
	else \
		echo "✅ KEDA already installed"; \
	fi
	@echo "🚀 Deploying to development..."
	kubectl apply -k kubernetes-manifests/overlays/development
	@echo "✅ Development deployment complete!"
	@echo "💡 Consider using 'make helm-dev' for better deployment management!"

prod: ## Deploy to production (legacy)
	kubectl apply -k kubernetes-manifests/overlays/production

run: ## Dry run deployment (legacy)
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

clean: ## Delete all resources (legacy)
	kubectl delete namespace nimbusguard --ignore-not-found=true

## 🧪 Load Testing Commands

load-test-light: build-generator ## Run light load test (quick validation)
	@echo "🟢 Starting LIGHT load test..."
	@kubectl apply -f kubernetes-manifests/components/load-generator/job-light.yaml
	@echo "✅ Light load test job applied. Monitor with k9s in nimbusguard namespace."

load-test-medium: build-generator ## Run medium load test (moderate scaling)
	@echo "🟡 Starting MEDIUM load test..."
	@kubectl apply -f kubernetes-manifests/components/load-generator/job-medium.yaml
	@echo "✅ Medium load test job applied. Monitor with k9s in nimbusguard namespace."

load-test-heavy: build-generator ## Run heavy load test (trigger immediate KEDA scaling)
	@echo "🔴 Starting HEAVY load test..."
	@kubectl apply -f kubernetes-manifests/components/load-generator/job-heavy.yaml
	@echo "✅ Heavy load test job applied. Monitor with k9s in nimbusguard namespace."

load-test-sustained: build-generator ## Run sustained load test (long-term scaling cycle)
	@echo "🔵 Starting SUSTAINED load test..."
	@kubectl apply -f kubernetes-manifests/components/load-generator/job-sustained.yaml
	@echo "✅ Sustained load test job applied. Monitor with k9s in nimbusguard namespace."

load-test-burst: build-generator ## Run burst load test (sudden spikes)
	@echo "⚡ Starting BURST load test..."
	@kubectl apply -f kubernetes-manifests/components/load-generator/job-burst.yaml
	@echo "✅ Burst load test job applied. Monitor with k9s in nimbusguard namespace."

load-test-memory: build-generator ## Run memory stress test (test memory-based scaling)
	@echo "🧠 Starting MEMORY stress test..."
	@kubectl apply -f kubernetes-manifests/components/load-generator/job-memory-stress.yaml
	@echo "✅ Memory stress test job applied. Monitor with k9s in nimbusguard namespace."

load-test-cpu: build-generator ## Run CPU stress test (test CPU-based scaling)
	@echo "⚙️  Starting CPU stress test..."
	@kubectl apply -f kubernetes-manifests/components/load-generator/job-cpu-stress.yaml
	@echo "✅ CPU stress test job applied. Monitor with k9s in nimbusguard namespace."



load-status: ## Show status of all load test jobs and consumer pods
	@echo "📊 Load Test Status Report"
	@echo "=========================="
	@echo ""
	@echo "🧪 Load Test Jobs:"
	@kubectl get jobs -n nimbusguard --sort-by=.metadata.creationTimestamp | grep -E "(load-test|quick-test)" || echo "   No load test jobs found"
	@echo ""
	@echo "🚀 Consumer Pods:"
	@kubectl get pods -n nimbusguard -l app.kubernetes.io/name=consumer
	@echo ""
	@echo "🎯 KEDA ScaledObjects:"
	@kubectl get scaledobjects -n nimbusguard
	@echo ""
	@echo "📈 HPA Status:"
	@kubectl get hpa -n nimbusguard || echo "   No HPA found (KEDA creates HPA automatically)"

load-clean: ## Clean up completed load test jobs
	@echo "🧹 Cleaning up load test jobs..."
	@kubectl delete jobs -n nimbusguard -l job-name --field-selector=status.successful=1 2>/dev/null || true
	@kubectl delete jobs -n nimbusguard --field-selector=status.failed=1 2>/dev/null || true
	@echo "✅ Cleanup complete"

load-clean-all: ## Clean up ALL load test jobs (including running ones)
	@echo "🗑️  WARNING: Deleting ALL load test jobs..."
	@kubectl delete jobs -n nimbusguard -l job-name 2>/dev/null || true
	@kubectl delete jobs -n nimbusguard --selector='job-name' 2>/dev/null || true
	@for job in $$(kubectl get jobs -n nimbusguard -o name | grep -E "load-test|quick-test"); do \
		kubectl delete -n nimbusguard $$job 2>/dev/null || true; \
	done
	@echo "✅ All load test jobs deleted"