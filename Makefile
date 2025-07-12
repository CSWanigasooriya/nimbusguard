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
	
	@echo ""
	@echo "📦 Checking Helm repositories..."
	@if ! helm repo list | grep -q kedacore 2>/dev/null; then \
		echo "📥 Adding Helm repositories..."; \
		helm repo add kedacore https://kedacore.github.io/charts >/dev/null 2>&1; \
		helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1; \
		echo "✅ Helm repositories configured"; \
	else \
		echo "✅ Helm repositories already configured"; \
	fi
	@helm repo update >/dev/null 2>&1
	
	@echo ""
	@echo "🔧 Creating 'nimbusguard' namespace if it doesn't exist..."
	@kubectl create namespace nimbusguard >/dev/null 2>&1 || echo "✅ Namespace 'nimbusguard' already exists."
	@echo ""
	@echo "🔑 Configuring OpenAI API Key..."
	@echo "The dqn-adapter requires an OpenAI API key to function."
	@echo "You can get one from https://platform.openai.com/api-keys"
	@read -s -p "Enter your OpenAI API Key (leave blank to skip): " OPENAI_API_KEY; \
	if [ -n "$$OPENAI_API_KEY" ]; then \
		kubectl create secret generic openai-api-key --namespace=nimbusguard \
			--from-literal=key=$$OPENAI_API_KEY \
			--dry-run=client -o yaml | kubectl apply -f - >/dev/null; \
		echo "\n✅ OpenAI API Key secret configured in 'nimbusguard' namespace."; \
	else \
		echo "\n⚠️  Skipped OpenAI API Key configuration. The dqn-adapter may not work."; \
		echo "   You can create the secret manually later with: "; \
		echo "   kubectl create secret generic openai-api-key -n nimbusguard --from-literal=key=YOUR_API_KEY"; \
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
	@echo "📊 Installed cluster components:"
	@echo "   • metrics-server (CPU/Memory monitoring)"
	@echo ""
	@echo "🚀 Ready to deploy!"

# KServe installation removed - no longer needed with combined DQN architecture
# The DQN model is now loaded locally in the adapter for optimal performance

# -----------------------------------------------------------------------------
# KEDA Installation
# -----------------------------------------------------------------------------

keda-install: ## Install the KEDA operator in the 'nimbusguard' namespace
	@echo "📦 Installing KEDA operator into nimbusguard namespace..."
	@if ! helm list -n nimbusguard | grep -q keda 2>/dev/null; then \
		helm repo add kedacore https://kedacore.github.io/charts >/dev/null 2>&1; \
		helm repo update kedacore >/dev/null 2>&1; \
		helm install keda kedacore/keda --namespace nimbusguard --wait; \
		echo "✅ KEDA installed successfully"; \
	else \
		echo "✅ KEDA already installed in nimbusguard namespace"; \
	fi

keda-uninstall: ## Uninstall the KEDA operator
	@echo "🗑️  Uninstalling KEDA operator from nimbusguard namespace..."
	@helm uninstall keda -n nimbusguard 2>/dev/null || true
	@echo "✅ KEDA uninstalled"

prometheus-adapter-uninstall: ## Uninstall the Prometheus Adapter
	@echo "🗑️  Uninstalling Prometheus Adapter from nimbusguard namespace..."
	@helm uninstall prometheus-adapter -n nimbusguard 2>/dev/null || true
	@echo "✅ Prometheus Adapter uninstalled"

# -----------------------------------------------------------------------------
# KEDA Management Commands
# -----------------------------------------------------------------------------

keda-pause: ## Pause KEDA autoscaling (REPLICAS=<n> to set replica count)
	@echo "⏸️  Pausing KEDA autoscaling..."
	@if [ -z "$(REPLICAS)" ]; then \
		kubectl annotate scaledobject -n nimbusguard nimbusguard-scaler autoscaling.keda.sh/paused="true" --overwrite; \
	else \
		kubectl annotate scaledobject -n nimbusguard nimbusguard-scaler \
			autoscaling.keda.sh/paused="true" autoscaling.keda.sh/paused-replicas="$(REPLICAS)" --overwrite; \
	fi
	@echo "✅ Autoscaling paused."

keda-resume: ## Resume KEDA autoscaling
	@echo "▶️  Resuming KEDA autoscaling..."
	@kubectl annotate scaledobject -n nimbusguard nimbusguard-scaler \
		autoscaling.keda.sh/paused- autoscaling.keda.sh/paused-replicas- --overwrite
	@echo "✅ Autoscaling resumed"

# -----------------------------------------------------------------------------
# Load Testing Commands
# -----------------------------------------------------------------------------

load-test-light: docker-build ## Run light load test (quick validation)
	@echo "🟢 Starting LIGHT load test in nimbusguard namespace..."
	@kubectl apply -f kubernetes-manifests/components/load-generator/job-light.yaml -n nimbusguard
	@echo "✅ Light load test job applied. Monitor with k9s in nimbusguard namespace."

load-test-medium: docker-build ## Run medium load test (moderate scaling)
	@echo "🟡 Starting MEDIUM load test in nimbusguard namespace..."
	@kubectl apply -f kubernetes-manifests/components/load-generator/job-medium.yaml -n nimbusguard
	@echo "✅ Medium load test job applied. Monitor with k9s in nimbusguard namespace."

load-test-heavy: docker-build ## Run heavy load test (trigger immediate KEDA scaling)
	@echo "🔴 Starting HEAVY load test in nimbusguard namespace..."
	@kubectl apply -f kubernetes-manifests/components/load-generator/job-heavy.yaml -n nimbusguard
	@echo "✅ Heavy load test job applied. Monitor with k9s in nimbusguard namespace."

load-test-sustained: docker-build ## Run sustained load test (long-term scaling cycle)
	@echo "🔵 Starting SUSTAINED load test in nimbusguard namespace..."
	@kubectl apply -f kubernetes-manifests/components/load-generator/job-sustained.yaml -n nimbusguard
	@echo "✅ Sustained load test job applied. Monitor with k9s in nimbusguard namespace."

load-test-burst: docker-build ## Run burst load test (sudden spikes)
	@echo "⚡ Starting BURST load test in nimbusguard namespace..."
	@kubectl apply -f kubernetes-manifests/components/load-generator/job-burst.yaml -n nimbusguard
	@echo "✅ Burst load test job applied. Monitor with k9s in nimbusguard namespace."

load-test-memory: docker-build ## Run memory stress test (test memory-based scaling)
	@echo "🧠 Starting MEMORY stress test in nimbusguard namespace..."
	@kubectl apply -f kubernetes-manifests/components/load-generator/job-memory-stress.yaml -n nimbusguard
	@echo "✅ Memory stress test job applied. Monitor with k9s in nimbusguard namespace."

load-test-cpu: docker-build ## Run CPU stress test (test CPU-based scaling)
	@echo "⚙️  Starting CPU stress test in nimbusguard namespace..."
	@kubectl apply -f kubernetes-manifests/components/load-generator/job-cpu-stress.yaml -n nimbusguard
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

docker-build-base: ## Build the base Docker image
	@echo "🔨 Building nimbusguard-base image..."
	@docker build -t nimbusguard-base:latest -f docker/base.Dockerfile docker/

docker-build: docker-build-base ## Build all necessary Docker images
	@echo "🔨 Building application images..."
	@docker build -t nimbusguard-consumer:latest src/consumer/
	@docker build -t nimbusguard-generator:latest src/generator/
	@docker build -t nimbusguard-dqn-adapter:latest src/dqn-adapter/



docker-build-clean: ## Build images without cache (clean build)
	@echo "🧹 Clean build without cache..."
	@docker build --no-cache -t nimbusguard-base:latest docker/
	@docker build --no-cache -t nimbusguard-consumer:latest src/consumer/
	@docker build --no-cache -t nimbusguard-generator:latest src/generator/
	@docker build --no-cache -t nimbusguard-dqn-adapter:latest src/dqn-adapter/

# Push Docker images
docker-push: ## Push all necessary Docker images to a registry
	$(eval REPO_URL := $(shell echo $(DOCKER_REPO_URL)))
	docker tag nimbusguard-consumer:latest $(REPO_URL)/nimbusguard-consumer:latest
	docker tag nimbusguard-dqn-adapter:latest $(REPO_URL)/nimbusguard-dqn-adapter:latest
	docker tag nimbusguard-generator:latest $(REPO_URL)/nimbusguard-generator:latest
	docker push $(REPO_URL)/nimbusguard-consumer:latest
	docker push $(REPO_URL)/nimbusguard-dqn-adapter:latest
	docker push $(REPO_URL)/nimbusguard-generator:latest

# -----------------------------------------------------------------------------
# Deployment Commands
# -----------------------------------------------------------------------------

deploy: docker-build ## Build images and deploy all components to the cluster
	@echo "🔍 Checking KEDA installation..."
	@if ! helm list -n nimbusguard | grep -q keda 2>/dev/null; then \
		echo "⚠️  KEDA operator not found in nimbusguard namespace, installing it first..."; \
		$(MAKE) keda-install; \
	else \
		echo "✅ KEDA operator already installed"; \
	fi
	@echo "🚀 Deploying all components to the cluster..."
	@kubectl apply -k kubernetes-manifests/overlays/development
	@echo "✅ Deployment complete! Use 'make ports' to access services."

dev: deploy ## Alias for 'deploy' - builds and deploys all components for development

clean: ## NUCLEAR cleanup - immediate brutal force deletion of everything
	@echo "💥 NUCLEAR OPTION: Immediately destroying all NimbusGuard resources..."
	@echo "⚠️  WARNING: This will BRUTALLY FORCE DELETE everything!"
	@echo ""
	
	# Step 1: Kill all port forwards immediately
	@echo "🔪 Killing all port forwards..."
	@pkill -f "kubectl port-forward.*nimbusguard" 2>/dev/null || true
	@pkill -f "kubectl port-forward.*9090" 2>/dev/null || true
	@pkill -f "kubectl port-forward.*3000" 2>/dev/null || true
	
	# Step 2: Nuclear namespace deletion - no mercy, no waiting
	@echo "💥 NUCLEAR namespace deletion..."
	@for ns in nimbusguard kubeflow keda-system monitoring; do \
		if kubectl get namespace $$ns >/dev/null 2>&1; then \
			echo "   💀 Destroying namespace: $$ns"; \
			kubectl patch namespace $$ns -p '{"metadata":{"finalizers":[]}}' --type=merge 2>/dev/null || true; \
			kubectl delete namespace $$ns --force --grace-period=0 2>/dev/null & \
		fi; \
	done
	
	# Step 3: Delete KEDA and Prometheus Adapter CRDs immediately (no waiting for namespace cleanup)
	@echo "💥 Nuclear KEDA and Prometheus Adapter destruction..."
	@kubectl get crd 2>/dev/null | grep -E "(keda|metrics\.k8s\.io)" | awk '{print $$1}' | xargs -r kubectl delete crd --force --grace-period=0 2>/dev/null &
	@helm uninstall prometheus-adapter -n nimbusguard 2>/dev/null &
	@helm uninstall keda -n nimbusguard 2>/dev/null &
	
	# Step 4: Destroy all RBAC resources matching our patterns
	@echo "💥 Nuclear RBAC destruction..."
	@kubectl get clusterrole,clusterrolebinding --no-headers 2>/dev/null | grep -E "(nimbusguard|dqn-adapter|mcp-server|alloy|beyla|prometheus|kube-state-metrics|prometheus-adapter)" | awk '{print $$1}' | xargs -r kubectl delete --force --grace-period=0 2>/dev/null &
	
	# Step 5: Kill webhook configurations
	@echo "💥 Nuclear webhook destruction..."
	@kubectl delete validatingwebhookconfiguration,mutatingwebhookconfiguration --all --force --grace-period=0 2>/dev/null &
	
	# Step 6: Clean up project resources from default namespace (but don't delete the namespace)
	@echo "💥 Cleaning project resources from default namespace..."
	@kubectl delete pods,services,deployments,configmaps,secrets,jobs,cronjobs -n default -l app=nimbusguard --force --grace-period=0 2>/dev/null || true
	@kubectl delete pods,services,deployments,configmaps,secrets,jobs,cronjobs -n default -l component=keda-dqn --force --grace-period=0 2>/dev/null || true
	@kubectl delete scaledobjects,hpa -n default --all --force --grace-period=0 2>/dev/null || true
	
	# Step 7: Wait briefly for background deletions then force-finalize stuck namespaces
	@echo "⏳ Waiting 5 seconds for background deletions..."
	@sleep 5
	@for ns in nimbusguard kubeflow keda-system monitoring; do \
		if kubectl get namespace $$ns 2>/dev/null | grep -q "Terminating"; then \
			echo "   🔨 Force-finalizing stuck namespace: $$ns"; \
			kubectl get namespace $$ns -o json | jq '.spec.finalizers = []' | kubectl replace --raw /api/v1/namespaces/$$ns/finalize -f - 2>/dev/null || true; \
		fi; \
	done
	
	# Step 8: Final brute force check
	@echo "🔍 Final nuclear verification..."
	@sleep 3
	@REMAINING=$$(kubectl get namespaces --no-headers 2>/dev/null | grep -E "(nimbusguard|kubeflow|keda-system|monitoring)" | awk '{print $$1}' || true); \
	if [ -n "$$REMAINING" ]; then \
		echo "💀 Still found stubborn namespaces: $$REMAINING"; \
		echo "   Applying final nuclear option..."; \
		for ns in $$REMAINING; do \
			kubectl delete namespace $$ns --force --grace-period=0 2>/dev/null || true; \
		done; \
	else \
		echo "✅ Nuclear cleanup successful - all targets eliminated!"; \
	fi
	
	@echo ""
	@echo "💥 NUCLEAR cleanup complete! Everything should be destroyed."
	@echo "🔄 Ready for fresh deployment with 'make deploy'"

# Port forwards
ports: ## Port forward all relevant services in the background
	@echo "🚀 Starting all port forwarding in the background..."
	@nohup kubectl port-forward -n nimbusguard svc/prometheus 9090:9090 > .ports.log 2>&1 &
	@nohup kubectl port-forward -n nimbusguard svc/grafana 3000:3000 > .ports.log 2>&1 &
	@nohup kubectl port-forward -n nimbusguard svc/dqn-adapter 8080:8080 > .ports.log 2>&1 &
	@nohup kubectl port-forward -n nimbusguard svc/redis 6379:6379 > .ports.log 2>&1 &
	@nohup kubectl port-forward -n nimbusguard svc/minio 9000:9000 > .ports.log 2>&1 &
	@nohup kubectl port-forward -n nimbusguard svc/minio 9001:9001 > .ports.log 2>&1 &
	@sleep 2
	@echo "✅ All services are being forwarded in the background."
	@echo "   Use 'make ports-stop' to terminate them."
	@echo "-----------------------------------------"
	@echo "📈 Prometheus:         http://localhost:9090"
	@echo "📋 Grafana:            http://localhost:3000  (admin/admin)"
	@echo "🧠 DQN Adapter HTTP:   http://localhost:8080"
	@echo "    ├── /healthz      Health check"
	@echo "    ├── /metrics      Prometheus metrics (includes nimbusguard_dqn_desired_replicas)"
	@echo "    └── /evaluate     Manual evaluation trigger"
	@echo "💾 Redis (CLI):        redis-cli -p 6379"
	@echo "🗄️ MinIO API:          http://localhost:9000"
	@echo "🖥️ MinIO Console:      http://localhost:9001  (minioadmin/minioadmin)"
	@echo "-----------------------------------------"

# Stop port forwards
ports-stop:
	@echo "🛑 Stopping all port forwarding..."
	@pkill -f "kubectl port-forward.*nimbusguard" || true
	@pkill -f "kubectl port-forward.*kubeflow" || true
	@echo "✅ All port forwarding stopped"

# Experimental comparison targets
.PHONY: test-hpa-baseline test-dqn-baseline

test-hpa-baseline: ## Run HPA-only test with deterministic load
	@echo "🧪 Starting HPA baseline test with deterministic load..."
	kubectl delete jobs --all -n nimbusguard --ignore-not-found=true
	kubectl scale deployment consumer --replicas=1 -n nimbusguard
	kubectl apply -f kubernetes-manifests/components/consumer/hpa.yaml
	sleep 10
	kubectl apply -f kubernetes-manifests/components/load-generator/job-comparison-baseline.yaml -n nimbusguard
	@echo "✅ HPA test started. Monitor with: kubectl logs -f job/load-test-comparison-baseline"
	@echo "⏱️  Test duration: 30 minutes"

test-dqn-baseline: ## Run DQN test with identical deterministic load  
	@echo "🧠 Starting DQN test with identical deterministic load..."
	kubectl delete jobs --all -n nimbusguard --ignore-not-found=true
	kubectl delete hpa --all -n nimbusguard --ignore-not-found=true
	kubectl scale deployment consumer --replicas=1 -n nimbusguard
	kubectl apply -k kubernetes-manifests/overlays/development/
	sleep 30  # Wait for DQN to initialize
	kubectl apply -f kubernetes-manifests/components/load-generator/job-comparison-baseline.yaml -n nimbusguard
	@echo "✅ DQN test started. Monitor with: kubectl logs -f job/load-test-comparison-baseline"
	@echo "⏱️  Test duration: 30 minutes"