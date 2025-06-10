.PHONY: help dev k8s-dev clean k8s-clean status forward keda-setup keda-clean

help:
	@echo "NimbusGuard - Quick Development Commands"
	@echo "========================================"
	@echo ""
	@echo "🚀 Development:"
	@echo "  dev          - Docker development environment"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  clean        - Clean Docker environment"

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

# Cleanup Commands
clean:
	@echo "🧹 Cleaning Docker environment..."
	@docker-compose down -v --remove-orphans 2>/dev/null || true
	@docker system prune -f
	@echo "✅ Docker cleanup completed"