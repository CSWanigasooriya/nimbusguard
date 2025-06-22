# NimbusGuard Main Makefile
# Orchestrates high-level workflows using modular makefiles

.PHONY: k8s-dev help setup-kubeflow-environment clean clean-all

# Include all modular makefiles
include make/Makefile.infrastructure
include make/Makefile.components
include make/Makefile.monitoring
include make/Makefile.dev

# Colors for better output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# =============================================================================
# Main Help
# =============================================================================

help:
	@echo "$(BLUE)🚀 NimbusGuard Development Environment$(NC)"
	@echo ""
	@echo "$(GREEN)🎯 Quick Start:$(NC)"
	@echo "  $(YELLOW)k8s-dev$(NC)              Interactive setup with scaling options"
	@echo ""
	@echo "$(GREEN)📋 Development:$(NC)"
	@echo "  $(YELLOW)forward$(NC)              Setup port forwarding"
	@echo "  $(YELLOW)status$(NC)               Show system status"
	@echo "  $(YELLOW)health-check$(NC)         Check endpoint health"
	@echo "  $(YELLOW)quick-test$(NC)           Run integration test"
	@echo ""
	@echo "$(GREEN)🧹 Cleanup:$(NC)"
	@echo "  $(YELLOW)clean$(NC)                Clean up NimbusGuard resources"
	@echo "  $(YELLOW)clean-all$(NC)            Clean everything including Kubeflow"
	@echo ""
	@echo "$(GREEN)🔧 Kubeflow ML Operations:$(NC)"
	@echo "  $(YELLOW)kubeflow-install$(NC)     Install Kubeflow components"
	@echo "  $(YELLOW)kubeflow-pipelines$(NC)   Deploy training pipelines"
	@echo "  $(YELLOW)kubeflow-experiments$(NC) Run hyperparameter tuning"
	@echo "  $(YELLOW)kubeflow-serving$(NC)     Deploy model serving"
	@echo "  $(YELLOW)kubeflow-status$(NC)      Check Kubeflow status"
	@echo ""
	@echo "$(GREEN)🗄️ Storage & Data:$(NC)"
	@echo "  $(YELLOW)deploy-minio$(NC)         Deploy MinIO object storage"
	@echo "  $(YELLOW)minio-status$(NC)         Check MinIO status"
	@echo "  $(YELLOW)upload-dataset$(NC)       Upload training dataset to MinIO"
	@echo ""
	@echo "$(GREEN)🎛️ Dashboards & UI:$(NC)"
	@echo "  $(YELLOW)kubeflow-dashboard$(NC)   Access Kubeflow Pipelines & Katib UIs"
	@echo "  $(YELLOW)minio-console$(NC)        Access MinIO web console"
	@echo "  $(YELLOW)dashboards$(NC)           Access all dashboards (Kubeflow, MinIO, Grafana)"
	@echo ""
	@echo "$(GREEN)🆘 Help & Info:$(NC)"
	@echo "  $(YELLOW)info$(NC)                 Show detailed command info"
	@echo "  $(YELLOW)debug-info$(NC)           System debug information"
	@echo "  $(YELLOW)kubeflow-help$(NC)        Show Kubeflow-specific commands"
	@echo ""
	@echo "$(BLUE)💡 Getting Started:$(NC)"
	@echo "  $(GREEN)1.$(NC) Run: $(YELLOW)make k8s-dev$(NC)"
	@echo "  $(GREEN)2.$(NC) Choose your scaling setup (Traditional KEDA or Kubeflow ML)"
	@echo "  $(GREEN)3.$(NC) Access services via port forwarding"

# =============================================================================
# Main Development Entry Point
# =============================================================================

k8s-dev:
	@echo "$(BLUE)🚀 Welcome to NimbusGuard Development Environment$(NC)"
	@echo ""
	@echo "$(BLUE)📋 Choose your scaling approach:$(NC)"
	@echo "  $(GREEN)1)$(NC) $(BLUE)Traditional Scaling$(NC) (workloads + KEDA)"
	@echo "     • Workload components (Kafka, Consumer, Load Generator)"
	@echo "     • KEDA-based auto-scaling with metrics"
	@echo "     • Rule-based scaling policies"
	@echo "     • ⚡ Fast setup, proven scaling approach"
	@echo ""
	@echo "  $(GREEN)2)$(NC) $(BLUE)Kubeflow ML Pipeline$(NC) (distributed training & serving)"
	@echo "     • Workload components + Kubeflow operator"
	@echo "     • Automated training pipelines with Kubeflow"
	@echo "     • Hyperparameter optimization with Katib"
	@echo "     • Production model serving with KServe"
	@echo "     • Complete MLOps workflow with intelligent scaling"
	@echo "     • 🚀 Production-ready, enterprise ML features"
	@echo ""
	@read -p "Enter choice [1-2, default 2]: " choice; \
	echo "$(BLUE)[SETUP] Setting up common infrastructure...$(NC)"; \
	$(MAKE) setup-infrastructure; \
	echo ""; \
	if [ "$$choice" = "1" ]; then \
	  echo "$(BLUE)[SCALING] ⚡ Setting up traditional KEDA scaling...$(NC)"; \
	  $(MAKE) deploy-traditional-scaling; \
	else \
	  echo "$(BLUE)[ML] 🚀 Setting up Kubeflow ML Pipeline environment...$(NC)"; \
	  $(MAKE) setup-kubeflow-environment; \
	fi; \
	echo ""; \
	$(MAKE) deploy-monitoring; \
	$(MAKE) wait-pods; \
	$(MAKE) dashboards; \
	echo ""; \
	$(MAKE) show-completion-summary

# =============================================================================
# High-Level Workflow Orchestration
# =============================================================================

# Kubeflow environment setup (called from k8s-dev)
setup-kubeflow-environment:
	@echo "$(BLUE)[KUBEFLOW] Installing Kubeflow components...$(NC)"
	@$(MAKE) kubeflow-install
	@echo "$(BLUE)[KUBEFLOW] Building ML pipeline images...$(NC)"
	@$(MAKE) kubeflow-build-images
	@echo "$(BLUE)[KUBEFLOW] Deploying workloads with Kubeflow integration...$(NC)"
	@$(MAKE) deploy-workloads
	@$(MAKE) deploy-kubeflow-operator
	@echo "$(BLUE)[KUBEFLOW] Deploying Kubeflow ML components (including model storage)...$(NC)"
	@$(MAKE) deploy-kubeflow
	@echo "$(BLUE)[STORAGE] Setting up MinIO object storage...$(NC)"
	@$(MAKE) deploy-minio
	@echo "$(BLUE)[KUBEFLOW] Setting up model serving...$(NC)"
	@$(MAKE) kubeflow-serving
	@echo "$(GREEN)✅ Kubeflow environment ready!$(NC)"

# Traditional scaling setup (workloads + KEDA only)
deploy-traditional-scaling:
	@echo "$(BLUE)[SCALING] Setting up traditional KEDA-based scaling...$(NC)"
	@$(MAKE) deploy-workloads
	@$(MAKE) install-keda
	@$(MAKE) deploy-keda-scaling
	@echo "$(GREEN)✅ Traditional scaling environment ready!$(NC)"

# =============================================================================
# Legacy Targets (for backward compatibility)
# =============================================================================

# Legacy deploy-operator target (now redirects to Kubeflow)
deploy-operator:
	@echo "$(BLUE)🚀 NimbusGuard now uses Kubeflow-only deployment...$(NC)"
	@echo "$(YELLOW)Redirecting to Kubeflow setup...$(NC)"
	$(MAKE) setup-infrastructure
	$(MAKE) setup-kubeflow-environment

# Reset Kubernetes resources (legacy)
reset-k8s-resources:
	@echo "$(RED)🗑️  Deleting all resources from base and monitoring kustomize...$(NC)"
	@kubectl delete -k kubernetes-manifests/base || true
	@kubectl delete -k kubernetes-manifests/monitoring || true
	@echo "$(GREEN)✅ All resources deleted!$(NC)"

# =============================================================================
# Cleanup Targets  
# =============================================================================

clean: ## Clean up NimbusGuard resources only
	@echo "$(RED)🧹 Cleaning NimbusGuard resources...$(NC)"
	@$(MAKE) clean-all-components
	@$(MAKE) clean-monitoring
	@echo "$(GREEN)✅ NimbusGuard resources cleaned!$(NC)"

clean-all: ## Clean everything including Kubeflow
	@echo "$(RED)🧹 Cleaning everything including KEDA and Kubeflow...$(NC)"
	@read -p "Are you sure you want to clean everything? (y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
	  $(MAKE) clean-all-components; \
	  $(MAKE) clean-monitoring; \
	  $(MAKE) clean-base-manifests; \
	  $(MAKE) clean-images; \
	  $(MAKE) clean-kubeflow-installation; \
	  echo "$(RED)Cleaning Kubeflow namespaces...$(NC)"; \
	  kubectl delete namespace nimbusguard-ml nimbusguard-serving nimbusguard-experiments --ignore-not-found=true; \
	  echo "$(GREEN)✅ Everything cleaned!$(NC)"; \
	else \
	  echo "$(YELLOW)Cleanup cancelled.$(NC)"; \
	fi

# =============================================================================
# Completion Summary
# =============================================================================

show-completion-summary:
	@echo "$(GREEN)🎉 NimbusGuard Development Environment Ready!$(NC)"
	@echo ""
	@echo "$(BLUE)🌐 Available Dashboards:$(NC)"
	@echo "  $(GREEN)• Kubeflow Pipelines:$(NC) http://localhost:8082 (ML workflows)"
	@echo "  $(GREEN)• Kubeflow Katib:$(NC)     http://localhost:8083/katib/ (Hyperparameter tuning)"
	@echo "  $(GREEN)• MinIO Console:$(NC)      http://localhost:30901 (Object storage - nimbusguard/nimbusguard123)"
	@echo "  $(GREEN)• Grafana:$(NC)            http://localhost:3000 (Monitoring - admin/admin)"
	@echo "  $(GREEN)• Prometheus:$(NC)         http://localhost:9090 (Metrics)"
	@echo "  $(GREEN)• Consumer Workload:$(NC)  http://localhost:8080 (API endpoints)"
	@echo "  $(GREEN)• Load Generator:$(NC)     http://localhost:8081 (Load controls)"
	@echo "  $(GREEN)• NimbusGuard Operator:$(NC) http://localhost:9080 (Operator API)"
	@echo ""
	@echo "$(BLUE)🚀 Quick Commands:$(NC)"
	@echo "  $(YELLOW)make dashboards$(NC)       - Access all dashboards at once"
	@echo "  $(YELLOW)make status$(NC)           - Check system health"
	@echo "  $(YELLOW)make health-check$(NC)     - Validate endpoints"
	@echo "  $(YELLOW)make quick-test$(NC)       - Run integration test"
	@echo ""
	@echo "$(BLUE)📊 ML Operations:$(NC)"
	@echo "  $(YELLOW)make kubeflow-status$(NC)  - Check ML pipeline status"
	@echo "  $(YELLOW)make minio-status$(NC)     - Check object storage"
	@echo "  $(YELLOW)make upload-dataset$(NC)   - Prepare datasets for training"
	@echo ""
	@echo "$(YELLOW)💡 Notes:$(NC)"
	@echo "  • Kubeflow Pipelines: Ready immediately"
	@echo "  • Kubeflow Katib: Access via /katib/ path"
	@echo "  • MinIO Console: NodePort service (may need a moment to be accessible)"
	@echo ""
	@echo "$(GREEN)✨ Your intelligent Kubernetes scaling platform is ready for action!$(NC)"

# Reinstall helpers (legacy)
reinstall-keda: uninstall-keda install-keda
reinstall-alloy: uninstall-alloy install-alloy
