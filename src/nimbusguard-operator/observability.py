# engine/observability.py - Enhanced with better error handling and fallbacks
# ============================================================================
# Robust Observability Data Collector
# ============================================================================

import asyncio
import logging
from typing import Dict, Any, List, Tuple, Optional

import aiohttp
import numpy as np

from config import health_status
from prometheus import PrometheusClient

LOG = logging.getLogger(__name__)


def _calculate_confidence_score(prometheus_available: bool, query_success_rate: float = 0.0) -> float:
    """
    Enhanced confidence score based on Prometheus availability and query success rate.
    Returns a lower confidence when data sources are unreliable.
    """
    if not prometheus_available:
        LOG.warning("Prometheus unavailable - returning low confidence score")
        return 0.0

    # If Prometheus is available but queries are failing, reduce confidence
    if query_success_rate < 0.5:  # Less than 50% of queries successful
        confidence = query_success_rate * 0.8  # Max 0.4 confidence
        LOG.warning(f"Low query success rate ({query_success_rate:.2f}) - confidence: {confidence:.2f}")
        return confidence

    return min(1.0, query_success_rate)


def _calculate_health_score(feature_vector: List[float]) -> float:
    """Calculate a unified health score from the feature vector."""
    if not feature_vector:
        return 0.0
    # Simple average, assuming all features are normalized where lower is better (or neutral)
    return 1.0 - float(np.mean(feature_vector))


def _normalize_feature(raw_value: float, scale_factor: float) -> float:
    """Normalize a feature value to be between 0 and 1."""
    if scale_factor <= 0:
        LOG.warning(f"Invalid scale factor {scale_factor}, using 1.0 instead")
        scale_factor = 1.0
    normalized = raw_value / scale_factor
    return max(0.0, min(1.0, normalized))


async def _try_prometheus_query(client: PrometheusClient, query: str, timeout: float = 5.0) -> Optional[float]:
    """Try a single Prometheus query with timeout and better error handling."""
    try:
        # Add timeout to prevent hanging queries
        value = await asyncio.wait_for(client.query(query), timeout=timeout)
        if value is not None:
            LOG.debug(f"Successfully queried: {query[:80]}... = {value}")
            return value
        else:
            LOG.debug(f"Query returned no data: {query[:80]}...")
            return None
    except asyncio.TimeoutError:
        LOG.warning(f"Query timed out ({timeout}s): {query[:80]}...")
        return None
    except Exception as e:
        LOG.debug(f"Query failed: {query[:80]}... - {e}")
        return None


def _initialize_feature_config() -> Dict[str, Dict[str, Any]]:
    """
    Defines the 7 core features for the DQN agent with fallback configurations.
    """
    return {
        "core_metrics": {
            "cpu_usage": {
                "query": "avg(nimbusguard_cpu_usage_percent)",
                "fallback_queries": [
                    "avg(rate(container_cpu_usage_seconds_total{pod=~'consumer-workload.*'}[5m])) * 100",
                    "avg(node_cpu_seconds_total{mode='idle'}) * 100"
                ],
                "default": 30.0,  # Conservative default
                "scale": 100.0,
            },
            "memory_usage": {
                "query": "avg(nimbusguard_memory_usage_percent)",
                "fallback_queries": [
                    "avg(container_memory_usage_bytes{pod=~'consumer-workload.*'} / container_spec_memory_limit_bytes{pod=~'consumer-workload.*'}) * 100"
                ],
                "default": 40.0,
                "scale": 100.0,
            },
            "request_rate": {
                "query": "sum(rate(nimbusguard_http_requests_total[2m]))",
                "fallback_queries": [
                    "sum(rate(http_requests_total{job='consumer-workload'}[2m]))",
                    "sum(rate(nginx_http_requests_total[2m]))"
                ],
                "default": 5.0,  # Low default request rate
                "scale": 100.0,
            },
            "response_time_p95": {
                "query": "histogram_quantile(0.95, sum(rate(nimbusguard_http_request_duration_seconds_bucket[5m])) by (le))",
                "fallback_queries": [
                    "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job='consumer-workload'}[5m])) by (le))"
                ],
                "default": 0.1,  # 100ms default
                "scale": 1.0,
            },
            "error_rate": {
                "query": '(sum(rate(nimbusguard_http_requests_total{status=~"5.."}[2m])) / sum(rate(nimbusguard_http_requests_total[2m])))',
                "fallback_queries": [
                    '(sum(rate(http_requests_total{status=~"5..",job="consumer-workload"}[2m])) / sum(rate(http_requests_total{job="consumer-workload"}[2m])))'
                ],
                "default": 0.01,  # 1% default error rate
                "scale": 0.05,
            },
            "pod_restart_rate": {
                "query": "sum(rate(kube_pod_container_status_restarts_total{pod=~'consumer-workload.*'}[10m]))",
                "fallback_queries": [
                    "sum(increase(kube_pod_container_status_restarts_total{pod=~'consumer-workload.*'}[10m]))"
                ],
                "default": 0.0,
                "scale": 1.0,
            },
            "queue_size": {
                "query": "sum(nimbusguard_queue_size)",
                "fallback_queries": [
                    "sum(kafka_consumer_lag_sum{consumer_group='background-consumer'})",
                    "sum(rabbitmq_queue_messages_ready)"
                ],
                "default": 10.0,  # Small default queue
                "scale": 100.0,
            },
        }
    }


class ObservabilityCollector:
    """
    Enhanced collector with better error handling, fallbacks, and connection management.
    """

    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url
        self.session = None
        self.feature_config = _initialize_feature_config()
        self.connection_retry_count = 0
        self.max_retries = 3

    async def collect_unified_state(self, metrics_config: Dict[str, Any], target_labels: Dict[str, str],
                                    service_name: str) -> Dict[str, Any]:
        """Collects the 7 core metrics to form the DQN state vector with enhanced error handling."""
        if not self.session:
            self.session = aiohttp.ClientSession()

        # Test Prometheus connectivity first
        prometheus_available = await self._test_prometheus_connectivity()

        if not prometheus_available:
            LOG.warning("Prometheus not available - using all default values")
            raw_metrics, query_success_rate = self._get_default_metrics(), 0.0
        else:
            # Collect raw data from Prometheus with fallbacks
            raw_metrics, query_success_rate = await self._collect_prometheus_metrics_with_fallbacks()

        # Create the feature vector
        feature_vector = self._create_feature_vector(raw_metrics)
        feature_names = list(self.feature_config["core_metrics"].keys())

        # Calculate scores
        health_score = _calculate_health_score(feature_vector)
        confidence_score = _calculate_confidence_score(prometheus_available, query_success_rate)

        result = {
            "raw_metrics": raw_metrics,
            "feature_vector": feature_vector,
            "feature_names": feature_names,
            "health_score": health_score,
            "confidence_score": confidence_score,
            "data_sources_available": {
                "prometheus": prometheus_available,
                "query_success_rate": query_success_rate
            }
        }

        LOG.info(f"Observability collection complete - Confidence: {confidence_score:.2f}, Health: {health_score:.2f}")
        return result

    async def _test_prometheus_connectivity(self) -> bool:
        """Test if Prometheus is reachable and responding."""
        try:
            client = PrometheusClient(self.prometheus_url)
            # Simple query to test connectivity
            result = await asyncio.wait_for(client.query("up"), timeout=3.0)
            if result is not None:
                LOG.info("Prometheus connectivity test passed")
                health_status["prometheus"] = True
                self.connection_retry_count = 0
                return True
        except Exception as e:
            self.connection_retry_count += 1
            LOG.warning(f"Prometheus connectivity test failed (attempt {self.connection_retry_count}): {e}")
            health_status["prometheus"] = False

        return False

    async def _collect_prometheus_metrics_with_fallbacks(self) -> Tuple[Dict[str, float], float]:
        """Collects metrics with fallback queries and tracks success rate."""
        client = PrometheusClient(self.prometheus_url)
        metrics = {}
        successful_queries = 0
        total_queries = 0

        feature_definitions = self.feature_config.get("core_metrics", {})

        for name, config in feature_definitions.items():
            value = None
            queries_to_try = [config["query"]] + config.get("fallback_queries", [])

            # Try primary query first, then fallbacks
            for query in queries_to_try:
                total_queries += 1
                value = await _try_prometheus_query(client, query)
                if value is not None:
                    successful_queries += 1
                    break

            # Use value if found, otherwise use default
            if value is not None:
                metrics[name] = value
            else:
                metrics[name] = config["default"]
                LOG.warning(f"Using default value for '{name}': {config['default']} (all queries failed)")

        success_rate = successful_queries / total_queries if total_queries > 0 else 0.0
        LOG.info(f"Prometheus query success rate: {successful_queries}/{total_queries} ({success_rate:.2f})")

        return metrics, success_rate

    def _get_default_metrics(self) -> Dict[str, float]:
        """Returns default values for all metrics when Prometheus is unavailable."""
        feature_definitions = self.feature_config.get("core_metrics", {})
        return {name: config["default"] for name, config in feature_definitions.items()}

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

    async def close(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
