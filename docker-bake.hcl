variable "REGISTRY" {
  default = "ghcr.io/nimbusguard"
}

variable "TAG" {
  default = "latest"
}

group "default" {
  targets = ["consumer"]
}

group "base" {
  targets = ["nimbusguard-base"]
}

group "all" {
  targets = ["nimbusguard-base", "consumer"]
}

target "_common" {
  platforms = ["linux/amd64", "linux/arm64"]
  annotations = [
    "org.opencontainers.image.authors=NimbusGuard Team",
    "org.opencontainers.image.source=https://github.com/nimbusguard/nimbusguard",
    "org.opencontainers.image.description=NimbusGuard - Intelligent Kubernetes Resource Management"
  ]
  attest = [
    "type=provenance,mode=max",
    "type=sbom"
  ]
}

target "nimbusguard-base" {
  inherits = ["_common"]
  context = "docker"
  dockerfile = "base.Dockerfile"
  tags = [
    "${REGISTRY}/nimbusguard-base:${TAG}",
    "${REGISTRY}/nimbusguard-base:latest"
  ]
}

target "consumer" {
  inherits = ["_common"]
  context = "src/consumer"
  dockerfile = "Dockerfile"
  tags = [
    "${REGISTRY}/nimbusguard-consumer:${TAG}",
    "${REGISTRY}/nimbusguard-consumer:latest"
  ]
  contexts = {
    nimbusguard-base = "target:nimbusguard-base"
  }
} 