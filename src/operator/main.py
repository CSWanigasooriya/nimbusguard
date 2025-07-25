import kopf
import logging
import requests
import pandas as pd
import threading
import time
from collections import deque

# --- Global State Management ---
# For storing pod metric history for both CPU and Memory
# Format: { 'instance_ip': {'cpu': deque(...), 'memory': deque(...)} }
POD_HISTORY = {}
SEQUENCE_LENGTH = 10

# For managing the background polling threads
# Format: { 'deployment_uid': (thread_object, stop_event) }
POLLING_THREADS = {}


# --- Helper Functions for Prometheus ---

def fetch_cpu_metrics(query):
    """Queries Prometheus for CPU metrics and returns a dictionary of the results."""
    # This function remains unchanged
    prometheus_url = "http://prometheus.nimbusguard.svc:9090"
    api_endpoint = "/api/v1/query"
    full_url = f"{prometheus_url}{api_endpoint}"
    
    metrics = {}
    try:
        response = requests.get(full_url, params={'query': query}, timeout=5)
        response.raise_for_status()
        result = response.json()

        if result.get('status') == 'success':
            for item in result.get('data', {}).get('result', []):
                instance = item.get('metric', {}).get('instance')
                timestamp = pd.to_datetime(item['value'][0], unit='s', utc=True)
                value = float(item['value'][1])
                if instance:
                    metrics[instance] = (timestamp, value)
    except Exception as e:
        logging.error(f"Could not fetch or parse Prometheus CPU metrics: {e}")
    return metrics

def fetch_memory_metrics(query):
    """
    ✨ NEW: Queries Prometheus for Memory metrics and returns a dictionary of the results.
    This function specifically fetches memory usage, which is a GAUGE, not a COUNTER.
    """
    prometheus_url = "http://prometheus.nimbusguard.svc:9090"
    api_endpoint = "/api/v1/query"
    full_url = f"{prometheus_url}{api_endpoint}"
    
    metrics = {}
    try:
        response = requests.get(full_url, params={'query': query}, timeout=5)
        response.raise_for_status()
        result = response.json()

        if result.get('status') == 'success':
            for item in result.get('data', {}).get('result', []):
                instance = item.get('metric', {}).get('instance')
                # For gauges, the timestamp is still the first element.
                timestamp = pd.to_datetime(item['value'][0], unit='s', utc=True)
                # The value is the direct memory usage in bytes.
                value = float(item['value'][1])
                if instance:
                    metrics[instance] = (timestamp, value)
    except Exception as e:
        logging.error(f"Could not fetch or parse Prometheus Memory metrics: {e}")
    return metrics


# --- Background Polling Function ---
def poll_consumer_metrics(stop_event, name, namespace, **kwargs):
    """
    MODIFIED: This function now polls, processes, and logs both CPU and Memory metrics.
    """
    logging.info(f"Polling thread started for Deployment '{namespace}/{name}'.")
    spec = kwargs.get('spec', {})
    
    while not stop_event.is_set():
        # --- Print Deployment Info ---
        replicas = spec.get('replicas', 'N/A')
        cpu_requests, mem_requests = 'N/A', 'N/A'
        cpu_limits, mem_limits = 'N/A', 'N/A'
        try:
            resources = spec['template']['spec']['containers'][0]['resources']
            cpu_requests = resources.get('requests', {}).get('cpu', 'N/A')
            mem_requests = resources.get('requests', {}).get('memory', 'N/A')
            cpu_limits = resources.get('limits', {}).get('cpu', 'N/A')
            mem_limits = resources.get('limits', {}).get('memory', 'N/A')
        except (KeyError, IndexError):
            logging.warning(f"Could not find resource specifications for '{name}'.")

        print(f"\n--- App: {name} (Polled) ---")
        print(f"  Replicas: {replicas}")
        print(f"  Requests -> CPU: {cpu_requests}, Memory: {mem_requests}")
        print(f"  Limits   -> CPU: {cpu_limits}, Memory: {mem_limits}")
        print("-" * 25)

        # --- Prometheus Metric Polling ---
        # Query for CPU (a counter that increases over time)
        cpu_query = 'process_cpu_seconds_total{job=~"prometheus.scrape.annotated_pods", instance=~".*:8000"}'
        # Query for Memory (a gauge showing current usage in bytes)
        mem_query = 'process_resident_memory_bytes{job=~"prometheus.scrape.annotated_pods", instance=~".*:8000"}'
        
        current_cpu_metrics = fetch_cpu_metrics(cpu_query)
        current_mem_metrics = fetch_memory_metrics(mem_query)

        # Get a unique set of all instances that reported either CPU or memory
        all_instances = set(current_cpu_metrics.keys()) | set(current_mem_metrics.keys())

        for instance in all_instances:
            # Initialize history for a new instance
            if instance not in POD_HISTORY:
                POD_HISTORY[instance] = {
                    'cpu': deque(maxlen=SEQUENCE_LENGTH + 1),
                    'memory': deque(maxlen=SEQUENCE_LENGTH) # No need for +1 as we don't calculate rate
                }
            
            print(f"--- Instance: {instance} ---")
            
            # --- Process CPU Metrics ---
            if instance in current_cpu_metrics:
                timestamp, value = current_cpu_metrics[instance]
                
                # Avoid adding duplicate data points if the process is idle
                if not POD_HISTORY[instance]['cpu'] or POD_HISTORY[instance]['cpu'][-1][1] != value:
                    POD_HISTORY[instance]['cpu'].append((timestamp, value))
                
                cpu_history = list(POD_HISTORY[instance]['cpu'])
                
                if len(cpu_history) < 2:
                    print(f"  CPU: Collecting initial data points... ({len(cpu_history)} collected)")
                else:
                    cpu_rates = []
                    for i in range(1, len(cpu_history)):
                        prev_time, prev_val = cpu_history[i-1]
                        curr_time, curr_val = cpu_history[i]
                        time_delta = (curr_time - prev_time).total_seconds()
                        if curr_val < prev_val or time_delta <= 0: continue
                        rate = (curr_val - prev_val) / time_delta
                        cpu_rates.append(rate)
                    print(f"  CPU Rates: {[float(f'{r:.6f}') for r in cpu_rates]}")

            # --- Process Memory Metrics ---
            if instance in current_mem_metrics:
                timestamp, value = current_mem_metrics[instance]
                
                # Memory is a gauge, so we just append the value
                POD_HISTORY[instance]['memory'].append(value)
                
                # Convert bytes to megabytes for readability
                mem_history_mb = [v / (1024 * 1024) for v in POD_HISTORY[instance]['memory']]
                print(f"  Memory Usage (MB): {[float(f'{m:.2f}') for m in mem_history_mb]}")
        
        stop_event.wait(5)
        
    logging.info(f"Polling thread stopped for Deployment '{namespace}/{name}'.")

# --- Kopf Handlers for Lifecycle Management ---
# (These handlers remain unchanged as they correctly manage the thread lifecycle)

@kopf.on.resume('apps', 'v1', 'deployments', 
                when=lambda name, **_: name == 'consumer')
@kopf.on.create('apps', 'v1', 'deployments', 
                when=lambda name, **_: name == 'consumer')
def start_polling_for_consumer(uid, **kwargs):
    """Create and start a background polling thread for the consumer deployment."""
    if uid not in POLLING_THREADS or not POLLING_THREADS[uid][0].is_alive():
        logging.info(f"Starting poller for {uid}")
        stop_event = threading.Event()
        thread = threading.Thread(target=poll_consumer_metrics, args=(stop_event,), kwargs=kwargs)
        POLLING_THREADS[uid] = (thread, stop_event)
        thread.start()
    else:
        logging.info(f"Poller for {uid} is already running.")


@kopf.on.delete('apps', 'v1', 'deployments', 
               when=lambda name, **_: name == 'consumer')
def stop_polling_for_consumer(uid, **kwargs):
    """Signal the polling thread to stop and wait for it to exit."""
    if uid in POLLING_THREADS:
        logging.info(f"Stopping poller for {uid}")
        thread, stop_event = POLLING_THREADS[uid]
        stop_event.set()
        thread.join(timeout=10) # Wait for thread to finish
        del POLLING_THREADS[uid]
        logging.info(f"Poller for {uid} stopped and cleaned up.")