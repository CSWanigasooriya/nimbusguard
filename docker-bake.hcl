# Docker Bake configuration for NimbusGuard
# This file optimizes builds with advanced caching strategies

variable "REGISTRY" {
  default = "localhost"
}

variable "TAG" {
  default = "latest"
}

# Global cache configuration
variable "CACHE_DIR" {
  default = "/tmp/buildkit-cache"
}

group "default" {
  targets = ["consumer-workload", "load-generator", "langgraph-operator"]
}

target "consumer-workload" {
  context = "./src/consumer-workload"
  dockerfile = "Dockerfile"
  tags = ["${REGISTRY}/consumer-workload:${TAG}"]
  cache-from = [
    "type=local,src=${CACHE_DIR}/consumer-workload"
  ]
  cache-to = [
    "type=local,dest=${CACHE_DIR}/consumer-workload,mode=max"
  ]
  platforms = ["linux/amd64"]
}

target "load-generator" {
  context = "./src/load-generator"
  dockerfile = "Dockerfile"
  tags = ["${REGISTRY}/load-generator:${TAG}"]
  cache-from = [
    "type=local,src=${CACHE_DIR}/load-generator"
  ]
  cache-to = [
    "type=local,dest=${CACHE_DIR}/load-generator,mode=max"
  ]
  platforms = ["linux/amd64"]
}

target "langgraph-operator" {
  context = "./src/langgraph-operator"
  dockerfile = "Dockerfile"
  tags = ["${REGISTRY}/langgraph-operator:${TAG}"]
  cache-from = [
    "type=local,src=${CACHE_DIR}/langgraph-operator"
  ]
  cache-to = [
    "type=local,dest=${CACHE_DIR}/langgraph-operator,mode=max"
  ]
  platforms = ["linux/amd64"]
}

# Development target - builds all services
target "dev" {
  name = "dev"
  inherits = ["consumer-workload", "load-generator", "langgraph-operator"]
} 