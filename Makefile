# NimbusGuard Makefile

.PHONY: help dev prod run forward stop-forward status logs restart clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-10s %s\n", $$1, $$2}'

dev: ## Deploy to development
	kubectl apply -k kubernetes-manifests/overlays/development

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