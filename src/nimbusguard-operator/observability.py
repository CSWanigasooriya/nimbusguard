# engine/observability.py - Enhanced version with availability checks
# ============================================================================
# Unified Multi-Source Observability Data Collector with Fallback Handling
# ============================================================================

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

from config import health_status
from prometheus import PrometheusClient
from tempo import TempoClient
from loki import LokiClient

LOG = logging.getLogger(__name__)


class ObservabilityCollector:
    """
    Enhanced unified collector that gracefully handles missing metrics and
    provides comprehensive fallback mechanisms for DQN feature vector creation.
    """

    def __init__(self, prometheus_url: str, tempo_url: str, loki_url: str):
        """
        Initializes all observability clients.

        Args:
            prometheus_url: Prometheus server URL
            tempo_url: Tempo server URL
            loki_url: Loki server URL
        """
        self.prometheus = PrometheusClient(prometheus_url)
        self.tempo = TempoClient(tempo_url)
        self.loki = LokiClient(loki_url)

        # Enhanced feature configuration with availability tracking
        self.feature_config = self._initialize_feature_config()
        self.metric_availability = {}
        self.last_availability_check = 0

    def _initialize_feature_config(self) -> Dict[str, Dict[str, Any]]:
        """Enhanced feature vector structure with fallback queries."""
        return {
            # System Metrics (10 features) - Enhanced with fallbacks
            "system_metrics": {
                "cpu_usage": {
                    "source": "prometheus",
                    "default": 0.5,
                    "scale": 1.0,
                    "queries": [
                        'avg(rate(container_cpu_usage_seconds_total[5m]))',
                        'avg(node_cpu_seconds_total[5m])',
                        'avg(cpu_usage_percent)'  # Fallback for simple metrics
                    ]
                },
                "memory_usage": {
                    "source": "prometheus",
                    "default": 0.5,
                    "scale": 1.0,
                    "queries": [
                        'avg(container_memory_usage_bytes / container_spec_memory_limit_bytes)',
                        'avg(node_memory_usage_percent)',
                        'avg(memory_usage_percent)'
                    ]
                },
                "network_io_rate": {
                    "source": "prometheus",
                    "default": 0.0,
                    "scale": 0.001,
                    "queries": [
                        'sum(rate(container_network_receive_bytes_total[5m]) + rate(container_network_transmit_bytes_total[5m]))',
                        'sum(rate(node_network_receive_bytes_total[5m]) + rate(node_network_transmit_bytes_total[5m]))'
                    ]
                },
                "disk_io_rate": {
                    "source": "prometheus",
                    "default": 0.0,
                    "scale": 0.001,
                    "queries": [
                        'sum(rate(container_fs_reads_bytes_total[5m]) + rate(container_fs_writes_bytes_total[5m]))',
                        'sum(rate(node_disk_read_bytes_total[5m]) + rate(node_disk_written_bytes_total[5m]))'
                    ]
                },
                "pod_readiness_ratio": {
                    "source": "prometheus",
                    "default": 1.0,
                    "scale": 1.0,
                    "queries": [
                        'sum(kube_pod_status_ready{condition="true"}) / sum(kube_pod_status_ready)',
                        'sum(up) / count(up)'  # Fallback: general service health
                    ]
                },
                "pod_restart_rate": {
                    "source": "prometheus",
                    "default": 0.0,
                    "scale": 1.0,
                    "queries": [
                        'sum(rate(kube_pod_container_status_restarts_total[10m]))',
                        'sum(increase(kube_pod_container_status_restarts_total[10m]))'
                    ]
                },
                "node_resource_pressure": {
                    "source": "prometheus",
                    "default": 0.0,
                    "scale": 1.0,
                    "queries": [
                        'avg(kube_node_status_condition{condition="MemoryPressure", status="true"}) + avg(kube_node_status_condition{condition="DiskPressure", status="true"})',
                        'avg(node_memory_utilization) + avg(node_disk_utilization)'
                    ]
                },
                "container_throttling": {
                    "source": "prometheus",
                    "default": 0.0,
                    "scale": 1.0,
                    "queries": [
                        'sum(rate(container_cpu_cfs_throttled_seconds_total[5m]))',
                        'sum(rate(container_cpu_throttled_time_total[5m]))'
                    ]
                },
                "filesystem_usage": {
                    "source": "prometheus",
                    "default": 0.0,
                    "scale": 1.0,
                    "queries": [
                        'avg(container_fs_usage_bytes / container_fs_limit_bytes)',
                        'avg(node_filesystem_avail_bytes / node_filesystem_size_bytes)'
                    ]
                },
                "load_average": {
                    "source": "prometheus",
                    "default": 0.0,
                    "scale": 0.1,
                    "queries": [
                        'avg(node_load1)',
                        'avg(system_load1)',
                        'avg(load_average_1m)'
                    ]
                },
            },

            # Application Metrics (8 features) - With fallbacks
            "application_metrics": {
                "request_rate": {
                    "source": "prometheus",
                    "default": 0.0,
                    "scale": 0.01,
                    "queries": [
                        'sum(rate(http_requests_total[5m]))',
                        'sum(rate(requests_total[5m]))',
                        'sum(rate(api_requests_total[5m]))'
                    ]
                },
                "response_time_p50": {
                    "source": "prometheus",
                    "default": 0.1,
                    "scale": 0.001,
                    "queries": [
                        'histogram_quantile(0.5, rate(http_request_duration_seconds_bucket[5m]))',
                        'histogram_quantile(0.5, rate(request_duration_seconds_bucket[5m]))',
                        'avg(response_time_p50_seconds)'
                    ]
                },
                "response_time_p95": {
                    "source": "prometheus",
                    "default": 0.2,
                    "scale": 0.001,
                    "queries": [
                        'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))',
                        'histogram_quantile(0.95, rate(request_duration_seconds_bucket[5m]))',
                        'avg(response_time_p95_seconds)'
                    ]
                },
                "response_time_p99": {
                    "source": "prometheus",
                    "default": 0.5,
                    "scale": 0.001,
                    "queries": [
                        'histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))',
                        'histogram_quantile(0.99, rate(request_duration_seconds_bucket[5m]))',
                        'avg(response_time_p99_seconds)'
                    ]
                },
                "error_rate": {
                    "source": "prometheus",
                    "default": 0.0,
                    "scale": 1.0,
                    "queries": [
                        'sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))',
                        'sum(rate(requests_total{status="error"}[5m])) / sum(rate(requests_total[5m]))',
                        'avg(error_rate_percent)'
                    ]
                },
                "throughput": {
                    "source": "prometheus",
                    "default": 0.0,
                    "scale": 0.01,
                    "queries": [
                        'sum(rate(http_requests_total[5m]))',
                        'sum(rate(throughput_total[5m]))'
                    ]
                },
                "active_connections": {
                    "source": "prometheus",
                    "default": 0.0,
                    "scale": 0.01,
                    "queries": [
                        'sum(node_netstat_Tcp_CurrEstab)',
                        'sum(active_connections)',
                        'sum(current_connections)'
                    ]
                },
                "queue_size": {
                    "source": "prometheus",
                    "default": 0.0,
                    "scale": 0.01,
                    "queries": [
                        'sum(queue_size)',
                        'sum(pending_requests)',
                        'sum(backlog_size)'
                    ]
                },
            },

            # Trace Insights (7 features) - from Tempo
            "trace_insights": {
                "service_graph_latency_p50": {"source": "tempo", "default": 0.1, "scale": 0.001},
                "service_graph_latency_p95": {"source": "tempo", "default": 0.2, "scale": 0.001},
                "service_graph_latency_p99": {"source": "tempo", "default": 0.5, "scale": 0.001},
                "dependency_error_rate": {"source": "tempo", "default": 0.0, "scale": 1.0},
                "dependency_health_score": {"source": "tempo", "default": 1.0, "scale": 1.0},
                "service_bottleneck_score": {"source": "tempo", "default": 0.0, "scale": 1.0},
                "trace_anomaly_score": {"source": "tempo", "default": 0.0, "scale": 1.0},
            },

            # Log Insights (5 features) - from Loki
            "log_insights": {
                "error_log_frequency": {"source": "loki", "default": 0.0, "scale": 0.1},
                "critical_error_rate": {"source": "loki", "default": 0.0, "scale": 1.0},
                "log_anomaly_score": {"source": "loki", "default": 0.0, "scale": 1.0},
                "business_kpi_trend": {"source": "loki", "default": 0.5, "scale": 1.0},
                "transaction_error_rate": {"source": "loki", "default": 0.0, "scale": 1.0},
            }
        }

    async def collect_unified_state(self,
                                    metrics_config: Dict[str, Any],
                                    target_labels: Dict[str, str],
                                    service_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhanced collection with availability checks and graceful degradation.
        """
        LOG.info("Collecting unified observability state with enhanced fallback handling")

        # Determine service name for trace analysis
        if not service_name:
            service_name = target_labels.get("app", "unknown-service")

        # Collect data with timeout and error handling
        collection_results = await self._collect_all_sources_with_timeout(
            metrics_config, target_labels, service_name
        )

        prometheus_data = collection_results.get("prometheus", {})
        tempo_data = collection_results.get("tempo", {})
        loki_data = collection_results.get("loki", {})

        # Create enhanced feature vector with availability tracking
        feature_vector, availability_stats = self._create_enhanced_feature_vector(
            prometheus_data, tempo_data, loki_data
        )

        # Calculate health and confidence scores
        health_score = self._calculate_enhanced_health_score(availability_stats)
        confidence_score = self._calculate_enhanced_confidence_score(
            prometheus_data, tempo_data, loki_data, availability_stats
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "raw_metrics": {
                "prometheus": prometheus_data,
                "tempo": tempo_data,
                "loki": loki_data
            },
            "feature_vector": feature_vector,
            "feature_names": self._get_feature_names(),
            "health_score": health_score,
            "confidence_score": confidence_score,
            "data_sources_available": {
                "prometheus": bool(prometheus_data),
                "tempo": bool(tempo_data),
                "loki": bool(loki_data)
            },
            "availability_stats": availability_stats,
            "collection_metadata": {
                "total_metrics_attempted": sum(len(cat["features"]) for cat in availability_stats.values()),
                "total_metrics_collected": sum(sum(cat["features"].values()) for cat in availability_stats.values()),
                "source_health": collection_results.get("source_health", {})
            }
        }

    async def _collect_all_sources_with_timeout(self,
                                                metrics_config: Dict[str, Any],
                                                target_labels: Dict[str, str],
                                                service_name: str) -> Dict[str, Any]:
        """Collect from all sources with individual timeouts and error handling."""

        async def safe_prometheus_collection():
            try:
                return await asyncio.wait_for(
                    self._collect_enhanced_prometheus_data(metrics_config, target_labels),
                    timeout=10.0
                )
            except Exception as e:
                LOG.error(f"Prometheus collection failed: {e}")
                return {}

        async def safe_tempo_collection():
            try:
                return await asyncio.wait_for(
                    self._collect_tempo_data(service_name),
                    timeout=8.0
                )
            except Exception as e:
                LOG.error(f"Tempo collection failed: {e}")
                return {}

        async def safe_loki_collection():
            try:
                return await asyncio.wait_for(
                    self._collect_loki_data(target_labels),
                    timeout=8.0
                )
            except Exception as e:
                LOG.error(f"Loki collection failed: {e}")
                return {}

        # Collect all sources concurrently
        prometheus_data, tempo_data, loki_data = await asyncio.gather(
            safe_prometheus_collection(),
            safe_tempo_collection(),
            safe_loki_collection(),
            return_exceptions=True
        )

        # Handle exceptions
        if isinstance(prometheus_data, Exception):
            prometheus_data = {}
        if isinstance(tempo_data, Exception):
            tempo_data = {}
        if isinstance(loki_data, Exception):
            loki_data = {}

        return {
            "prometheus": prometheus_data,
            "tempo": tempo_data,
            "loki": loki_data,
            "source_health": {
                "prometheus": health_status.get("prometheus", False),
                "tempo": health_status.get("tempo", False),
                "loki": health_status.get("loki", False)
            }
        }

    async def _collect_enhanced_prometheus_data(self,
                                                metrics_config: Dict[str, Any],
                                                target_labels: Dict[str, str]) -> Dict[str, float]:
        """Enhanced Prometheus collection with fallback queries."""
        prometheus_url = metrics_config.get("prometheus_url", "http://prometheus.monitoring.svc.cluster.local:9090")
        client = PrometheusClient(prometheus_url)

        metrics = {}

        # Try to collect each metric from feature config
        for category, features in self.feature_config.items():
            if category in ["system_metrics", "application_metrics"]:
                for feature_name, config in features.items():
                    if "queries" in config:
                        # Try each query until one succeeds
                        value = await self._try_prometheus_queries(client, config["queries"])
                        if value is not None:
                            metrics[feature_name] = value
                        else:
                            # Use configured default if no query succeeds
                            metrics[feature_name] = config["default"]
                            LOG.debug(f"Using default value for {feature_name}: {config['default']}")

        # Also collect any custom metrics from CRD config
        for metric_conf in metrics_config.get("metrics", []):
            query = metric_conf.get("query")
            if query:
                value = await client.query(query)
                if value is not None:
                    name = metric_conf.get("name", self._extract_metric_name(query))
                    metrics[name] = value

        return metrics

    async def _try_prometheus_queries(self, client: PrometheusClient, queries: List[str]) -> Optional[float]:
        """Try multiple Prometheus queries until one succeeds."""
        for query in queries:
            try:
                value = await client.query(query)
                if value is not None:
                    LOG.debug(f"Successfully queried: {query} = {value}")
                    return value
            except Exception as e:
                LOG.debug(f"Query failed: {query} - {e}")
                continue
        return None

    def _create_enhanced_feature_vector(self,
                                        prometheus_data: Dict[str, float],
                                        tempo_data: Dict[str, float],
                                        loki_data: Dict[str, float]) -> tuple[List[float], Dict[str, Any]]:
        """Creates feature vector with availability tracking."""
        feature_vector = []
        availability_stats = {}

        # Process each feature category
        for category, features in self.feature_config.items():
            category_stats = {"attempted": 0, "collected": 0, "features": {}}

            for feature_name, config in features.items():
                category_stats["attempted"] += 1
                source = config["source"]
                default_value = config["default"]
                scale_factor = config["scale"]

                # Get value from appropriate data source
                if source == "prometheus":
                    raw_value = prometheus_data.get(feature_name, default_value)
                elif source == "tempo":
                    raw_value = tempo_data.get(feature_name, default_value)
                elif source == "loki":
                    raw_value = loki_data.get(feature_name, default_value)
                else:
                    raw_value = default_value

                # Track if we got real data vs default
                got_real_data = (
                        (source == "prometheus" and feature_name in prometheus_data) or
                        (source == "tempo" and feature_name in tempo_data) or
                        (source == "loki" and feature_name in loki_data)
                )

                if got_real_data:
                    category_stats["collected"] += 1
                    category_stats["features"][feature_name] = 1
                else:
                    category_stats["features"][feature_name] = 0

                # Normalize the value
                normalized_value = self._normalize_feature(raw_value, scale_factor)
                feature_vector.append(normalized_value)

            availability_stats[category] = category_stats

        return feature_vector, availability_stats

    def _calculate_enhanced_health_score(self, availability_stats: Dict[str, Any]) -> float:
        """Calculate health score based on metric availability and source health."""

        # Base health from source availability
        source_health = sum(1 for component in ["prometheus", "tempo", "loki"]
                            if health_status.get(component, False)) / 3

        # Metric availability score
        total_attempted = sum(stats["attempted"] for stats in availability_stats.values())
        total_collected = sum(stats["collected"] for stats in availability_stats.values())

        metric_availability = total_collected / total_attempted if total_attempted > 0 else 0

        # Weighted score (source health is more important than individual metric availability)
        health_score = (source_health * 0.7) + (metric_availability * 0.3)

        return min(max(health_score, 0.0), 1.0)

    def _calculate_enhanced_confidence_score(self,
                                             prometheus_data: Dict,
                                             tempo_data: Dict,
                                             loki_data: Dict,
                                             availability_stats: Dict[str, Any]) -> float:
        """Enhanced confidence calculation based on data quality and coverage."""

        # Base confidence on data source availability
        data_sources = [prometheus_data, tempo_data, loki_data]
        non_empty_sources = sum(1 for data in data_sources if data)
        base_confidence = non_empty_sources / len(data_sources)

        # Adjust based on critical metrics availability (Prometheus is most important)
        prometheus_coverage = availability_stats.get("system_metrics", {}).get("collected", 0) / max(
            availability_stats.get("system_metrics", {}).get("attempted", 1), 1
        )

        app_metrics_coverage = availability_stats.get("application_metrics", {}).get("collected", 0) / max(
            availability_stats.get("application_metrics", {}).get("attempted", 1), 1
        )

        # Weight different metric categories
        weighted_confidence = (
                base_confidence * 0.4 +  # Source availability
                prometheus_coverage * 0.4 +  # System metrics (most critical)
                app_metrics_coverage * 0.2  # Application metrics
        )

        return min(max(weighted_confidence, 0.0), 1.0)

    # ... (keep existing methods for tempo and loki collection, normalize_feature, etc.)

    def get_availability_report(self) -> Dict[str, Any]:
        """Get detailed availability report for debugging."""
        return {
            "metric_availability": self.metric_availability,
            "source_health": {
                "prometheus": health_status.get("prometheus", False),
                "tempo": health_status.get("tempo", False),
                "loki": health_status.get("loki", False)
            },
            "feature_config_summary": {
                category: {
                    "total_features": len(features),
                    "features_with_fallbacks": sum(1 for f in features.values() if "queries" in f)
                }
                for category, features in self.feature_config.items()
            }
        }