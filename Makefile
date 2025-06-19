.PHONY: k8s-dev create-operator-secret stop-forward install-keda uninstall-keda reinstall-keda forward prepare-local-models

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

# Uninstall KEDA
uninstall-keda:
	@echo "🗑️  Uninstalling old KEDA..."
	@helm uninstall keda -n keda 2>/dev/null || true
	@kubectl delete namespace keda --ignore-not-found=true
	@kubectl delete apiservice v1beta1.external.metrics.k8s.io --ignore-not-found=true
	@kubectl delete crd scaledobjects.keda.sh --ignore-not-found=true
	@kubectl delete crd scaledjobs.keda.sh --ignore-not-found=true
	@kubectl delete crd triggerauthentications.keda.sh --ignore-not-found=true
	@kubectl delete crd clustertriggerauthentications.keda.sh --ignore-not-found=true
	@echo "⏳ Waiting for cleanup..."
	@sleep 10
	@echo "✅ KEDA uninstalled!"

# Install KEDA (compatible version for Kubernetes v1.33+)
install-keda:
	@echo "🚀 Installing KEDA..."
	@helm repo add kedacore https://kedacore.github.io/charts 2>/dev/null || true
	@helm repo update
	@kubectl create namespace keda --dry-run=client -o yaml | kubectl apply -f -
	@helm upgrade --install keda kedacore/keda \
		--version 2.17.1 \
		--namespace keda \
		--wait \
		--timeout 300s \
		--set installCRDs=true \
		--set operator.replicaCount=1 \
		--set metricsServer.replicaCount=1 \
		--set metricsServer.useHostNetwork=false \
		--set metricsServer.port=6443 \
		--set operator.grpcPort=9666 \
		--set certs.autoGenerate=true \
		--set certs.certDir=/certs \
		--set certs.certSecretName=kedaorg-certs \
		--set certs.caSecretName=kedaorg-ca \
		--force
	@echo "⏳ Waiting for KEDA to be ready..."
	@kubectl wait --for=condition=ready pod -l app=keda-operator -n keda --timeout=300s || echo "⚠️  KEDA operator not ready yet"
	@kubectl wait --for=condition=ready pod -l app=keda-operator-metrics-apiserver -n keda --timeout=300s || echo "⚠️  KEDA metrics server not ready yet"
	@kubectl wait --for=condition=ready pod -l app=keda-admission-webhooks -n keda --timeout=300s || echo "⚠️  KEDA webhooks not ready yet"
	@echo "✅ KEDA installed successfully!"

# Reinstall KEDA with correct version
reinstall-keda: uninstall-keda install-keda

# Uninstall Alloy
uninstall-alloy:
	@echo "🗑️  Uninstalling old Alloy..."
	@helm uninstall alloy -n monitoring 2>/dev/null || true
	@kubectl delete namespace monitoring --ignore-not-found=true
	@echo "⏳ Waiting for cleanup..."
	@sleep 10
	@echo "✅ Alloy uninstalled!"

# Install Alloy (using Helm)
install-alloy:
	@echo "🚀 Installing Grafana Alloy..."
	@helm repo add grafana https://grafana.github.io/helm-charts 2>/dev/null || true
	@helm repo update
	@helm upgrade --install alloy grafana/alloy \
		--namespace monitoring \
		--create-namespace \
		--wait \
		--timeout 300s \
		-f helm/values-alloy.yaml
	@echo "⏳ Waiting for Alloy to be ready..."
	@kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=alloy -n monitoring --timeout=300s
	@echo "✅ Alloy installed successfully!"

# Reinstall Alloy with correct version
reinstall-alloy: uninstall-alloy install-alloy

# Reset Kubernetes resources
reset-k8s-resources:
	@echo "🗑️  Deleting all resources from base and monitoring kustomize..."
	@kubectl delete -k kubernetes-manifests/base || true
	@kubectl delete -k kubernetes-manifests/monitoring || true
	@echo "✅ All resources deleted!"

# Kubernetes Development - Infrastructure + Component Selection
k8s-dev:
	@echo "🚀 Starting Kubernetes development environment..."
	@echo "📋 Setting up common infrastructure..."
	$(MAKE) setup-infrastructure
	@echo ""
	@echo "🎯 Choose scaling components to deploy (workloads always included):"
	@echo "  1) Workloads only (no auto-scaling)"
	@echo "  2) Workloads + KEDA (traditional auto-scaling)"
	@echo "  3) Workloads + Operator (ML-based scaling)"
	@echo "  4) Workloads + KEDA + Operator (both scaling methods - default)"
	@read -p "Enter choice [1-4, default 4]: " choice; \
	echo "[COMPONENTS] Deploying workloads (always required)..."; \
	$(MAKE) deploy-workloads; \
	if [ "$$choice" = "1" ]; then \
	  echo "[SCALING] No additional scaling components selected"; \
	elif [ "$$choice" = "2" ]; then \
	  echo "[SCALING] Adding KEDA auto-scaling..."; \
	  $(MAKE) install-keda; \
	  $(MAKE) deploy-keda-scaling; \
	elif [ "$$choice" = "3" ]; then \
	  echo "[SCALING] Adding ML-based operator scaling..."; \
	  $(MAKE) deploy-operator-only; \
	elif [ "$$choice" = "4" ] || [ -z "$$choice" ]; then \
	  echo "[SCALING] Adding both KEDA and Operator scaling..."; \
	  $(MAKE) install-keda; \
	  $(MAKE) deploy-operator-only; \
	  $(MAKE) deploy-keda-scaling; \
	else \
	  echo "❌ Invalid choice. Exiting."; \
	  exit 1; \
	fi; \
	echo ""; \
	$(MAKE) deploy-monitoring; \
	$(MAKE) wait-pods; \
	$(MAKE) setup-port-forwarding

# Common infrastructure setup (always required)
setup-infrastructure:
	@echo "🏗️  Building Docker images..."
	@DOCKER_BUILDKIT=1 docker build \
		-t nimbusguard/base:latest \
		-f docker/base.Dockerfile .
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
	@echo "📦 Applying base manifests (namespaces + CRDs)..."
	@kubectl apply -k kubernetes-manifests/base
	@echo "🔑 Creating operator secrets..."
	@$(MAKE) create-operator-secret
	@echo "📊 Installing Alloy (observability pipeline)..."
	@if ! helm status alloy -n monitoring > /dev/null 2>&1; then \
		$(MAKE) install-alloy; \
	else \
		echo "✅ Alloy already installed."; \
	fi
	@echo "📁 Preparing local models directory..."
	@$(MAKE) prepare-local-models
	@echo "✅ Infrastructure setup complete!"

# Deploy just the workloads (Kafka + Consumer + Load Generator)
deploy-workloads:
	@echo "🚀 Deploying workload components..."
	@kubectl apply -k kubernetes-manifests/components/base
	@echo "✅ Workloads deployed!"

# Deploy KEDA scaling components
deploy-keda-scaling:
	@echo "📈 Deploying KEDA scaling components..."
	@kubectl apply -k kubernetes-manifests/components/keda
	@echo "✅ KEDA scaling deployed!"

# Deploy only the operator (without workloads)
deploy-operator-only:
	@echo "🤖 Deploying NimbusGuard operator..."
	@PROJECT_DIR=$$(pwd); \
	sed "s|\$${NIMBUSGUARD_PROJECT_PATH}|$$PROJECT_DIR|g" \
		kubernetes-manifests/components/operator/operator.yaml > \
		/tmp/nimbusguard-operator.yaml; \
	kubectl apply -f /tmp/nimbusguard-operator.yaml; \
	rm -f /tmp/nimbusguard-operator.yaml
	@echo "✅ Operator deployed!"

# Deploy monitoring stack
deploy-monitoring:
	@echo "📊 Deploying monitoring stack..."
	@kubectl apply -k kubernetes-manifests/monitoring
	@echo "✅ Monitoring deployed!"

# Legacy deploy-operator target (calls new structure)
deploy-operator:
	@echo "🚀 Using new infrastructure + operator deployment..."
	$(MAKE) setup-infrastructure
	$(MAKE) deploy-operator-only

wait-pods:
	@echo "   Waiting for pods to be ready..."
	@echo "   (This may take a few minutes for first-time setup...)"
	@kubectl wait --for=condition=ready pod -l app=kafka -n nimbusguard --timeout=300s || echo "⚠️  Kafka not ready yet"
	@kubectl wait --for=condition=ready pod -l app=consumer-workload -n nimbusguard --timeout=300s || echo "⚠️  Consumer not ready yet"
	@kubectl wait --for=condition=ready pod -l app=load-generator -n nimbusguard --timeout=300s || echo "⚠️  Load generator not ready yet"
	@kubectl wait --for=condition=ready pod -l app=nimbusguard-operator -n nimbusguard --timeout=300s || echo "⚠️  Operator not ready yet"
	@kubectl wait --for=condition=ready pod -l app=prometheus -n monitoring --timeout=300s || echo "⚠️  Prometheus not ready yet"
	@kubectl wait --for=condition=ready pod -l app=grafana -n monitoring --timeout=300s || echo "⚠️  Grafana not ready yet"

setup-port-forwarding:
	@echo "   Setting up port forwarding..."
	@make stop-forward
	@kubectl port-forward -n nimbusguard svc/consumer-workload 8080:8080 > /dev/null 2>&1 & \
	kubectl port-forward -n nimbusguard svc/load-generator 8081:8081 > /dev/null 2>&1 & \
	kubectl port-forward -n nimbusguard svc/nimbusguard-operator 9080:9080 > /dev/null 2>&1 & \
	kubectl port-forward -n monitoring svc/prometheus 9090:9090 > /dev/null 2>&1 & \
	kubectl port-forward -n monitoring svc/grafana 3000:3000 > /dev/null 2>&1 & \
	sleep 2
	@echo "   ✅ Environment ready!"
	@echo ""
	@echo "   Services available at:"
	@echo "   Consumer Workload: http://localhost:8080"
	@echo "   Load Generator:    http://localhost:8081"
	@echo "   Operator:          http://localhost:9080"
	@echo "   Prometheus:        http://localhost:9090"
	@echo "   Grafana:           http://localhost:3000 (admin/admin)"

# Prepare local models directory for development
prepare-local-models:
	@echo "📁 Preparing local models directory..."
	@PROJECT_DIR=$$(pwd); \
	mkdir -p "$$PROJECT_DIR/models"; \
	chmod 777 "$$PROJECT_DIR/models"; \
	echo "✅ Models directory ready at: $$PROJECT_DIR/models"

# Forward all relevant ports for local development
forward:
	@echo "🔀 Setting up port forwarding for all services..."
	@kubectl port-forward -n nimbusguard svc/consumer-workload 8080:8080 > /dev/null 2>&1 & \
	kubectl port-forward -n nimbusguard svc/load-generator 8081:8081 > /dev/null 2>&1 & \
	kubectl port-forward -n nimbusguard svc/nimbusguard-operator 9080:9080 > /dev/null 2>&1 & \
	kubectl port-forward -n monitoring svc/prometheus 9090:9090 > /dev/null 2>&1 & \
	kubectl port-forward -n monitoring svc/grafana 3000:3000 > /dev/null 2>&1 & \
	sleep 2
	@echo "   ✅ Port forwarding ready!"
	@echo "   Consumer Workload: http://localhost:8080"
	@echo "   Load Generator:    http://localhost:8081"
	@echo "   Operator:          http://localhost:9080"
	@echo "   Prometheus:        http://localhost:9090"
	@echo "   Grafana:           http://localhost:3000 (admin/admin)"
