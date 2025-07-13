"""Predefined Prometheus queries for the 9 selected features."""
from typing import Dict


class PrometheusQueries:
    """Prometheus query definitions for selected features."""
    
    @staticmethod
    def get_feature_queries() -> Dict[str, str]:
        """Get all feature queries mapped by feature name."""
        return {
            # CPU metrics - aggregated across consumer pods
            "process_cpu_seconds_total_rate": 
                'sum(rate(process_cpu_seconds_total{job="prometheus.scrape.annotated_pods", instance=~".*8000"}[30s])) or vector(0)',
            
            # Memory metrics - aggregated across consumer pods
            "process_resident_memory_bytes":
                'sum(process_resident_memory_bytes{job="prometheus.scrape.annotated_pods", instance=~".*8000"}) or vector(0)',
            
            "process_virtual_memory_bytes":
                'sum(process_virtual_memory_bytes{job="prometheus.scrape.annotated_pods", instance=~".*8000"}) or vector(0)',
            
            # Request latency metrics - ONLY for /process endpoint (actual workload)
            "http_request_duration_seconds_sum_rate":
                'sum(rate(http_request_duration_seconds_sum{job="prometheus.scrape.annotated_pods", instance=~".*8000", handler="/process"}[30s])) or vector(0)',
            
            # Request rate metrics - ONLY for /process endpoint (actual workload)
            "http_requests_total_rate":
                'sum(rate(http_requests_total{job="prometheus.scrape.annotated_pods", instance=~".*8000", handler="/process"}[30s])) or vector(0)',
            
            "http_request_duration_seconds_count_rate":
                'sum(rate(http_request_duration_seconds_count{job="prometheus.scrape.annotated_pods", instance=~".*8000", handler="/process"}[30s])) or vector(0)',
            
            # File descriptor metrics - aggregated across consumer pods
            "process_open_fds":
                'sum(process_open_fds{job="prometheus.scrape.annotated_pods", instance=~".*8000"}) or vector(0)',
            
            # Response size metrics - ONLY for /process endpoint (actual workload)
            "http_response_size_bytes_sum_rate":
                'sum(rate(http_response_size_bytes_sum{job="prometheus.scrape.annotated_pods", instance=~".*8000", handler="/process"}[30s])) or vector(0)',
            
            # Request size metrics - ONLY for /process endpoint (actual workload)
            "http_request_size_bytes_count_rate":
                'sum(rate(http_request_size_bytes_count{job="prometheus.scrape.annotated_pods", instance=~".*8000", handler="/process"}[30s])) or vector(0)'
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