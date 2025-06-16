.PHONY: k8s-dev create-operator-secret stop-forward install-keda uninstall-keda reinstall-keda forward

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
	@helm upgrade --install keda kedacore/keda \
		--version 2.17.1 \
		--namespace keda \
		--create-namespace \
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
		--set certs.caSecretName=kedaorg-ca
	@echo "⏳ Waiting for KEDA to be ready..."
	@kubectl wait --for=condition=ready pod -l app=keda-operator -n keda --timeout=300s
	@kubectl wait --for=condition=ready pod -l app=keda-operator-metrics-apiserver -n keda --timeout=300s
	@kubectl wait --for=condition=ready pod -l app=keda-admission-webhooks -n keda --timeout=300s
	@echo "✅ KEDA installed successfully!"

# Reinstall KEDA with correct version
reinstall-keda: uninstall-keda install-keda

# Kubernetes Development
k8s-dev: install-keda create-operator-secret
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
	@echo "   Creating namespaces..."
	@kubectl create namespace nimbusguard --dry-run=client -o yaml | kubectl apply -f -
	@kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
	@kubectl apply -k kubernetes-manifests/base
	@echo "   Deploying monitoring stack..."
	@kubectl apply -k kubernetes-manifests/monitoring
	@echo "   Waiting for pods to be ready..."
	@echo "   (This may take a few minutes for first-time setup...)"
	@kubectl wait --for=condition=ready pod -l app=kafka -n nimbusguard --timeout=300s || echo "⚠️  Kafka not ready yet"
	@kubectl wait --for=condition=ready pod -l app=consumer-workload -n nimbusguard --timeout=300s || echo "⚠️  Consumer not ready yet"
	@kubectl wait --for=condition=ready pod -l app=load-generator -n nimbusguard --timeout=300s || echo "⚠️  Load generator not ready yet"
	@kubectl wait --for=condition=ready pod -l app=nimbusguard-operator -n nimbusguard --timeout=300s || echo "⚠️  Operator not ready yet"
	@kubectl wait --for=condition=ready pod -l app=prometheus -n monitoring --timeout=300s || echo "⚠️  Prometheus not ready yet"
	@kubectl wait --for=condition=ready pod -l app=grafana -n monitoring --timeout=300s || echo "⚠️  Grafana not ready yet"
	@echo "   Setting up port forwarding..."
	@make stop-forward
	@kubectl port-forward -n nimbusguard svc/consumer-workload 8080:8080 > /dev/null 2>&1 & \
	kubectl port-forward -n nimbusguard svc/load-generator 8081:8081 > /dev/null 2>&1 & \
	kubectl port-forward -n nimbusguard svc/nimbusguard-operator 8090:8090 > /dev/null 2>&1 & \
	kubectl port-forward -n monitoring svc/prometheus 9090:9090 > /dev/null 2>&1 & \
	kubectl port-forward -n monitoring svc/grafana 3000:3000 > /dev/null 2>&1 & \
	sleep 2
	@echo "   ✅ Environment ready!"
	@echo ""
	@echo "   Services available at:"
	@echo "   Consumer Workload: http://localhost:8080"
	@echo "   Load Generator:    http://localhost:8081"
	@echo "   Operator:          http://localhost:8090"
	@echo "   Prometheus:        http://localhost:9090"
	@echo "   Grafana:           http://localhost:3000 (admin/admin)"

# Forward all relevant ports for local development
forward:
	@echo "🔀 Setting up port forwarding for all services..."
	@kubectl port-forward -n nimbusguard svc/consumer-workload 8080:8080 > /dev/null 2>&1 & \
	kubectl port-forward -n nimbusguard svc/load-generator 8081:8081 > /dev/null 2>&1 & \
	kubectl port-forward -n nimbusguard svc/nimbusguard-operator 8090:8090 > /dev/null 2>&1 & \
	kubectl port-forward -n monitoring svc/prometheus 9090:9090 > /dev/null 2>&1 & \
	kubectl port-forward -n monitoring svc/grafana 3000:3000 > /dev/null 2>&1 & \
	sleep 2
	@echo "   ✅ Port forwarding ready!"
	@echo "   Consumer Workload: http://localhost:8080"
	@echo "   Load Generator:    http://localhost:8081"
	@echo "   Operator:          http://localhost:8090"
	@echo "   Prometheus:        http://localhost:9090"
	@echo "   Grafana:           http://localhost:3000 (admin/admin)"
