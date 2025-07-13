"""Predefined Prometheus queries for the 9 selected features."""
from typing import Dict


class PrometheusQueries:
    """Prometheus query definitions for selected features."""
    
    @staticmethod
    def get_feature_queries() -> Dict[str, str]:
        """Get consumer-focused feature queries based on HPA baseline analysis."""
        return {
            # PRIMARY INDICATOR: Request latency (35-542s spikes indicate scaling need)
            "http_request_duration_seconds_sum_rate": 
                'sum(rate(http_request_duration_seconds_sum{job="prometheus.scrape.nimbusguard_consumer"}[1m])) or vector(0)',
            
            # Request rate - actual workload indicator
            "http_request_duration_seconds_count_rate":
                'sum(rate(http_request_duration_seconds_count{job="prometheus.scrape.nimbusguard_consumer"}[1m])) or vector(0)',
            
            # CPU per pod - shows resource pressure (0.48-15.51s spikes)
            "process_cpu_seconds_total_rate": 
                'sum(rate(process_cpu_seconds_total{job="prometheus.scrape.nimbusguard_consumer"}[1m])) or vector(0)',
            
            # Memory per pod - shows memory pressure (61-197MB spikes)
            "process_resident_memory_bytes":
                'sum(process_resident_memory_bytes{job="prometheus.scrape.nimbusguard_consumer"}) or vector(0)',
            
            # Actual workload requests to /process endpoint (fallback to all HTTP requests)
            "http_requests_total_process_rate":
                'sum(rate(http_requests_total{job="prometheus.scrape.nimbusguard_consumer", handler="/process"}[1m])) or sum(rate(http_requests_total{job="prometheus.scrape.nimbusguard_consumer"}[1m])) or vector(0)',
            
            # Response throughput - indicates processing capacity
            "http_response_size_bytes_sum_rate":
                'sum(rate(http_response_size_bytes_sum{job="prometheus.scrape.nimbusguard_consumer"}[1m])) or vector(0)',
            
            # Connection pressure - file descriptors
            "process_open_fds":
                'sum(process_open_fds{job="prometheus.scrape.nimbusguard_consumer"}) or vector(0)',
            
            # Resource constraints - CPU limits from kube-state-metrics
            "kube_pod_container_resource_limits_cpu":
                'sum(kube_pod_container_resource_limits{resource="cpu", namespace="nimbusguard", pod=~"consumer-.*"}) or vector(0)',
            
            # Active connections (fallback to request count if not available)
            "http_server_active_connections":
                'sum(rate(http_requests_total{job="prometheus.scrape.nimbusguard_consumer"}[1m])) or vector(0)'
        }
    
    @staticmethod
    def get_range_query(feature_name: str, duration_minutes: int) -> str:
        """Get a range query for historical data collection."""
        base_query = PrometheusQueries.get_feature_queries().get(feature_name)
        if not base_query:
            raise ValueError(f"Unknown feature: {feature_name}")
        
        # For range queries, we need to adjust rate functions
        if "[30s]" in base_query:
            # Replace 30s rate with 1m rate for more granular historical data
            range_query = base_query.replace("[30s]", "[1m]")
        else:
            range_query = base_query
        
        return f"{range_query}[{duration_minutes}m:1m]"
    
    @staticmethod
    def get_current_replicas_query(deployment: str, namespace: str) -> str:
        """Get query for current replica count."""
        return f'kube_deployment_status_replicas{{deployment="{deployment}",namespace="{namespace}"}}'
    
    @staticmethod
    def get_aggregated_query(feature_name: str) -> str:
        """Get aggregated query across all consumer pods."""
        # All queries are already aggregated with sum() in get_feature_queries()
        return PrometheusQueries.get_feature_queries().get(feature_name) 