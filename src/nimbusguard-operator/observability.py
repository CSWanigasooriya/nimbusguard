# engine/observability.py - Optimized for a 7-state DQN model
# ============================================================================
# Focused Observability Data Collector
# ============================================================================

import asyncio
import logging
from typing import Dict, Any, List, Tuple, Optional

import aiohttp
import numpy as np

from config import health_status
from prometheus import PrometheusClient

# Note: Tempo and Loki clients are no longer needed for building the core feature vector.
# They can be used for status reporting, but are removed here for clarity.

LOG = logging.getLogger(__name__)


def _calculate_confidence_score(prometheus_available: bool) -> float:
    """Simplified confidence score based only on Prometheus availability."""
    return 1.0 if prometheus_available else 0.0


def _calculate_health_score(feature_vector: List[float]) -> float:
    """Calculate a unified health score from the feature vector."""
    if not feature_vector:
        return 0.0
    # Simple average, assuming all features are normalized where lower is better (or neutral)
    # This might need more sophisticated weighting based on your goals.
    return 1.0 - np.mean(feature_vector)


def _normalize_feature(raw_value: float, scale_factor: float) -> float:
    """Normalize a feature value to be between 0 and 1."""
    if scale_factor <= 0:
        LOG.warning(f"Invalid scale factor {scale_factor}, using 1.0 instead")
        scale_factor = 1.0
    normalized = raw_value / scale_factor
    return max(0.0, min(1.0, normalized))


async def _try_prometheus_query(client: PrometheusClient, query: str) -> Optional[float]:
    """Try a single Prometheus query."""
    try:
        value = await client.query(query)
        if value is not None:
            LOG.debug(f"Successfully queried: {query[:80]}... = {value}")
            return value
    except Exception as e:
        LOG.debug(f"Query failed: {query[:80]}... - {e}")
    return None


def _initialize_feature_config() -> Dict[str, Dict[str, Any]]:
    """
    Defines the 7 core features for the DQN agent.
    The PromQL queries are designed to use metrics from your application
    (nimbusguard_*) and standard Kubernetes metrics (kube_*).
    """
    return {
        "core_metrics": {
            "cpu_usage": {
                "query": "avg(nimbusguard_cpu_usage_percent)",
                "default": 50.0,  # Default to 50%
                "scale": 100.0,  # Scale against 100%
            },
            "memory_usage": {
                "query": "avg(nimbusguard_memory_usage_percent)",
                "default": 50.0,
                "scale": 100.0,
            },
            "request_rate": {
                "query": "sum(rate(nimbusguard_http_requests_total[2m]))",
                "default": 10.0,
                "scale": 1000.0,  # Assumes a max of 1000 rps
            },
            "response_time_p95": {
                "query": "histogram_quantile(0.95, sum(rate(nimbusguard_http_request_duration_seconds_bucket[5m])) by (le))",
                "default": 0.2,  # Default to 200ms
                "scale": 1.0,  # Scale against a 1s target
            },
            "error_rate": {
                "query": '(sum(rate(nimbusguard_http_requests_total{status=~"5.."}[2m])) / sum(rate(nimbusguard_http_requests_total[2m])))',
                "default": 0.0,
                "scale": 0.05,  # Scale against a 5% error rate
            },
            "pod_restart_rate": {
                "query": "sum(rate(kube_pod_container_status_restarts_total{pod=~'consumer-workload.*'}[10m])) > 0",
                "default": 0.0,
                "scale": 1.0,  # A rate > 0 is bad
            },
            "queue_size": {
                "query": "sum(nimbusguard_queue_size)",
                "default": 0.0,
                "scale": 100.0,  # Scale against a max queue of 100
            },
        }
    }


class ObservabilityCollector:
    """
    Optimized collector that focuses on gathering the 7 core features
    from Prometheus for the DQN agent's state.
    """

    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url
        self.session = None
        self.feature_config = _initialize_feature_config()

    async def collect_unified_state(self, metrics_config: Dict[str, Any], target_labels: Dict[str, str],
                                    service_name: str) -> Dict[str, Any]:
        """Collects the 7 core metrics to form the DQN state vector."""
        if not self.session:
            self.session = aiohttp.ClientSession()

        # Collect raw data from Prometheus
        raw_metrics, prometheus_available = await self._collect_prometheus_metrics()

        # Create the feature vector
        feature_vector = self._create_feature_vector(raw_metrics)
        feature_names = list(self.feature_config["core_metrics"].keys())

        # Calculate scores
        health_score = _calculate_health_score(feature_vector)
        confidence_score = _calculate_confidence_score(prometheus_available)

        return {
            "raw_metrics": raw_metrics,
            "feature_vector": feature_vector,
            "feature_names": feature_names,
            "health_score": health_score,
            "confidence_score": confidence_score,
            "data_sources_available": {"prometheus": prometheus_available}
        }

    async def _collect_prometheus_metrics(self) -> Tuple[Dict[str, float], bool]:
        """Collects only the core metrics defined in the feature config."""
        client = PrometheusClient(self.prometheus_url)
        metrics = {}

        feature_definitions = self.feature_config.get("core_metrics", {})
        tasks = []

        # Create a task for each metric query
        for name, config in feature_definitions.items():
            tasks.append(asyncio.create_task(_try_prometheus_query(client, config["query"])))

        # Run all queries in parallel
        results = await asyncio.gather(*tasks)

        # Process results
        for (name, config), value in zip(feature_definitions.items(), results):
            if value is not None:
                metrics[name] = value
            else:
                metrics[name] = config["default"]
                LOG.warning(f"Using default value for '{name}': {config['default']}")

        prometheus_available = health_status.get("prometheus", False)
        return metrics, prometheus_available

    def _create_feature_vector(self, raw_data: Dict[str, float]) -> List[float]:
        """Creates and normalizes the 7-element feature vector."""
        feature_vector = []
        feature_definitions = self.feature_config.get("core_metrics", {})

        for name, config in feature_definitions.items():
            raw_value = raw_data.get(name, config["default"])
            scale_factor = config.get("scale", 1.0)
            normalized_value = _normalize_feature(raw_value, scale_factor)
            feature_vector.append(normalized_value)

        return feature_vector