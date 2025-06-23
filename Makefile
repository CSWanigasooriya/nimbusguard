# NimbusGuard Makefile

.PHONY: help dev prod run forward status logs restart clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-10s %s\n", $$1, $$2}'

dev: ## Deploy to development
	kubectl apply -k kubernetes-manifests/overlays/development

prod: ## Deploy to production  
	kubectl apply -k kubernetes-manifests/overlays/production

run: ## Dry run deployment
	kubectl apply -k kubernetes-manifests/overlays/development --dry-run=client

forward: ## Port forward to access locally
	kubectl port-forward -n nimbusguard svc/consumer 8000:8000

status: ## Check deployment status
	kubectl get pods,svc -n nimbusguard -l app=nimbusguard

logs: ## Follow consumer logs
	kubectl logs -n nimbusguard -l app.kubernetes.io/name=consumer -f

restart: ## Restart consumer deployment
	kubectl rollout restart deployment/consumer -n nimbusguard

clean: ## Delete all resources
	kubectl delete namespace nimbusguard --ignore-not-found=true