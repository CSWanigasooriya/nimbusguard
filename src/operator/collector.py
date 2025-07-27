import logging
import requests
import pandas as pd
from typing import Dict, Any, Optional, List
from collections import deque
from state_manager import state
from config import system_config

class MetricsCollector:
    """Handles collection and processing of Kubernetes and Prometheus metrics."""
    
    def __init__(self, prometheus_url: Optional[str] = None):
        """
        Initialize the metrics collector.
        
        Args:
            prometheus_url: Prometheus URL (optional, uses config default if None)
        """
        self.prometheus_url = prometheus_url or system_config.prometheus_url
        logging.info(f"MetricsCollector initialized with Prometheus URL: {self.prometheus_url}")
    
    def fetch_per_instance_metrics(self, query: str) -> Dict[str, tuple]:
        """
        Fetches metrics from Prometheus, returning a dictionary of pod/instance-level data.
        Handles both instant queries (single value) and range queries (array of values).
        For range queries, returns the latest value from the 15s window.
        Each value is a tuple of (timestamp, value).
        """
        api_endpoint = "/api/v1/query"
        full_url = f"{self.prometheus_url}{api_endpoint}"
        metrics = {}
        
        # Determine if this is a range query
        is_range_query = '[' in query and ']' in query
        
        try:
            response = requests.get(full_url, params={'query': query}, timeout=5)
            response.raise_for_status()
            result = response.json()
            
            if result.get('status') == 'success':
                results = result.get('data', {}).get('result', [])
                logging.debug(f"[Prometheus Fetch] Query: {query} returned {len(results)} results (Range: {is_range_query})")
                
                for item in results:
                    metric_labels = item.get('metric', {})
                    
                    # For cAdvisor metrics, prefer 'pod' label, fallback to 'instance'
                    identifier = metric_labels.get('pod') or metric_labels.get('instance')
                    
                    if identifier:
                        container = metric_labels.get('container', '')
                        job = metric_labels.get('job', '')
                        
                        if is_range_query and item.get('values'):
                            # For range queries, use the latest value from the 15s window
                            values_array = item['values']
                            if values_array:
                                # Get the latest (most recent) value
                                latest_timestamp, latest_value = values_array[-1]
                                timestamp = pd.to_datetime(float(latest_timestamp), unit='s', utc=True)
                                value = float(latest_value)
                                
                                logging.debug(f"  [Range Metric] Pod: {identifier}, Container: {container}, Job: {job}, Latest Value: {value} ({len(values_array)} samples)")
                                metrics[identifier] = (timestamp, value)
                            else:
                                logging.warning(f"  [Skip] Empty values array for {identifier}")
                        
                        elif not is_range_query and item.get('value'):
                            # For instant queries, use single value
                            timestamp = pd.to_datetime(item['value'][0], unit='s', utc=True)
                            value = float(item['value'][1])
                            
                            logging.debug(f"  [Instant Metric] Pod: {identifier}, Container: {container}, Job: {job}, Value: {value}")
                            metrics[identifier] = (timestamp, value)
                        
                        else:
                            logging.warning(f"  [Skip] Missing data for {identifier}: range={is_range_query}, has_values={bool(item.get('values'))}, has_value={bool(item.get('value'))}")
                    else:
                        logging.warning(f"  [Skip] Missing identifier in metric: {metric_labels}")
            else:
                logging.error(f"Prometheus query failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            logging.error(f"Could not fetch Prometheus metrics for query '{query}': {e}")
        return metrics
    
    def collect_cpu_metrics(self) -> Dict[str, tuple]:
        """Collect CPU metrics for consumer containers."""
        cpu_query = 'container_cpu_usage_seconds_total{container="consumer",job="cadvisor"}[15s]'
        return self.fetch_per_instance_metrics(cpu_query)
    
    def collect_memory_metrics(self) -> Dict[str, tuple]:
        """Collect memory metrics for consumer containers."""
        mem_query = 'container_memory_working_set_bytes{container="consumer",job="cadvisor"}[15s]'
        return self.fetch_per_instance_metrics(mem_query)
    
    def collect_and_process_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Main method to collect all metrics, process them, and store structured memory data.
        
        Returns:
            Dictionary containing processed metrics, or None if collection fails
        """
        try:
            # Collect raw metrics
            current_cpu_metrics = self.collect_cpu_metrics()
            current_mem_metrics = self.collect_memory_metrics()
            
            if not current_cpu_metrics and not current_mem_metrics:
                logging.warning("No metrics collected from Prometheus")
                return None
            
            # Get all unique instances
            all_instances = set(current_cpu_metrics.keys()) | set(current_mem_metrics.keys())
            
            if not all_instances:
                logging.warning("No pod instances found in metrics")
                return None
            
            total_current_cpu_rate = 0
            total_current_mem = 0
            current_memory_values = []
            
            logging.debug(f"Processing metrics for {len(all_instances)} instances")
            
            # Process each instance
            for instance in all_instances:
                # Initialize pod history if needed
                if instance not in state.pod_history:
                    state.pod_history[instance] = {
                        'cpu_raw': deque(maxlen=state.sequence_length + 1)
                    }
                
                logging.debug(f"Processing instance: {instance}")
                
                # Process CPU metrics
                if instance in current_cpu_metrics:
                    timestamp, value = current_cpu_metrics[instance]
                    state.pod_history[instance]['cpu_raw'].append((timestamp, value))
                    logging.debug(f"  [CPU] Added cumulative value: {value:.2f} seconds")
                    
                    # Calculate CPU rate if we have enough history
                    cpu_history_raw = list(state.pod_history[instance]['cpu_raw'])
                    if len(cpu_history_raw) > 1:
                        prev_time, prev_val = cpu_history_raw[-2]
                        curr_time, curr_val = cpu_history_raw[-1]
                        time_delta = (curr_time - prev_time).total_seconds()
                        
                        if curr_val >= prev_val and time_delta > 0:
                            cpu_delta = curr_val - prev_val
                            instance_cpu_rate = cpu_delta / time_delta
                            total_current_cpu_rate += instance_cpu_rate
                            logging.debug(f"  [CPU Rate] {instance}: {instance_cpu_rate:.6f} cores/second")
                        else:
                            logging.warning(f"  [CPU] Skipping calculation for {instance}: invalid delta")
                    else:
                        logging.debug(f"  [CPU] Not enough history for {instance}")
                
                # Process Memory metrics
                if instance in current_mem_metrics:
                    timestamp, value = current_mem_metrics[instance]
                    current_memory_values.append(value)
                    
                    # Store structured data for LSTM prediction
                    memory_entry = {
                        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        'pod_name': instance,
                        'memory_bytes': int(value)
                    }
                    state.global_memory_history.append(memory_entry)
                    
                    logging.debug(f"  [Memory] {instance}: {value / (1024*1024):.2f} MB")
                    total_current_mem += value
            
            # Calculate aggregated metrics
            if current_memory_values:
                max_memory_value = max(current_memory_values)
                avg_memory_value = sum(current_memory_values) / len(current_memory_values)
                
                logging.info(f"Metrics collected: {len(all_instances)} pods, "
                            f"Total CPU: {total_current_cpu_rate:.3f} cores/sec, "
                            f"Total Memory: {total_current_mem/(1024*1024):.1f} MB")
            else:
                max_memory_value = 0
                avg_memory_value = 0
                logging.warning("No memory metrics collected")
            
            return {
                'total_cpu_rate': total_current_cpu_rate,
                'total_memory_bytes': total_current_mem,
                'max_memory_bytes': max_memory_value,
                'avg_memory_bytes': avg_memory_value,
                'pod_count': len(all_instances),
                'instances': list(all_instances),
                'memory_values': current_memory_values
            }
            
        except Exception as e:
            logging.error(f"Error in collect_and_process_metrics: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None
    
    def get_historical_data(self) -> List[Dict[str, Any]]:
        """Returns the global memory history."""
        return list(state.global_memory_history)
    
    def has_sufficient_data(self) -> bool:
        """Check if we have sufficient historical data."""
        return len(state.global_memory_history) >= state.sequence_length
    
    def has_sufficient_aggregated_data(self) -> bool:
        """Check if we have sufficient aggregated historical data."""
        try:
            if len(state.global_memory_history) < state.sequence_length:
                return False
            
            # Convert to DataFrame and check aggregated intervals
            history_df = pd.DataFrame(list(state.global_memory_history))
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            history_df['timestamp_rounded'] = history_df['timestamp'].dt.round('15s')
            aggregated_intervals = len(history_df.groupby('timestamp_rounded'))
            
            return aggregated_intervals >= state.sequence_length
        except Exception as e:
            logging.warning(f"Could not check aggregated data: {e}")
            return False
    
    def get_data_status(self) -> Dict[str, Any]:
        """Get status of data collection."""
        history_count = len(state.global_memory_history)
        sufficient_data = self.has_sufficient_data()
        sufficient_aggregated = self.has_sufficient_aggregated_data()
        
        return {
            'history_count': history_count,
            'required_count': state.sequence_length,
            'sufficient_data': sufficient_data,
            'sufficient_aggregated': sufficient_aggregated,
            'pod_history_count': len(state.pod_history)
        }
    
    def clear_history(self):
        """Clear all historical data."""
        state.global_memory_history.clear()
        state.pod_history.clear()
        logging.info("Historical data cleared")
    
    def get_prometheus_status(self) -> Dict[str, Any]:
        """Check Prometheus connectivity."""
        try:
            response = requests.get(f"{self.prometheus_url}/api/v1/status/config", timeout=5)
            return {
                'connected': response.status_code == 200,
                'url': self.prometheus_url,
                'status_code': response.status_code
            }
        except Exception as e:
            return {
                'connected': False,
                'url': self.prometheus_url,
                'error': str(e)
            } 