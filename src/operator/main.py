import kopf
import logging
import requests
import pandas as pd
import threading
import time
from collections import deque
import joblib
import numpy as np
import tensorflow as tf
import sklearn # Required for joblib to load the scaler object

# --- Global State Management ---

# For storing pod metric history
POD_HISTORY = {}
SEQUENCE_LENGTH = 10 # This MUST match the sequence_length used for training

# For managing background polling threads
POLLING_THREADS = {}

# For holding loaded models and scalers
CPU_MODEL = None
CPU_SCALER = None
MEMORY_MODEL = None
MEMORY_SCALER = None
MODELS_LOADED = threading.Event() # Use an event to track loading status


# --- Model Loading and Prediction Functions ---

def load_models_and_scalers():
    """
    Loads the pre-trained models and scalers from the /tmp/ directory.
    This function is called once when the operator starts.
    """
    global CPU_MODEL, CPU_SCALER, MEMORY_MODEL, MEMORY_SCALER
    
    model_dir = "/tmp"
    cpu_model_path = f"{model_dir}/cpu.keras"
    cpu_scaler_path = f"{model_dir}/cpu.pkl"
    mem_model_path = f"{model_dir}/memory.keras"
    mem_scaler_path = f"{model_dir}/memory.pkl"

    logging.info("--- Attempting to load forecasting models and scalers ---")
    try:
        logging.info(f"Loading CPU model from: {cpu_model_path}")
        CPU_MODEL = tf.keras.models.load_model(cpu_model_path)
        
        logging.info(f"Loading CPU scaler from: {cpu_scaler_path}")
        CPU_SCALER = joblib.load(cpu_scaler_path)
        
        logging.info(f"Loading Memory model from: {mem_model_path}")
        MEMORY_MODEL = tf.keras.models.load_model(mem_model_path)
        
        logging.info(f"Loading Memory scaler from: {mem_scaler_path}")
        MEMORY_SCALER = joblib.load(mem_scaler_path)
        
        MODELS_LOADED.set() # Signal that models are loaded
        logging.info("All forecasting models and scalers loaded successfully.")
        
    except Exception as e:
        logging.error(f"CRITICAL: Failed to load models. Forecasting will be disabled. Error: {e}")
        logging.error("Please ensure cpu.keras, cpu.pkl, memory.keras, and memory.pkl are in the /tmp/ directory.")
    
    logging.info("----------------------------------------------------------")

def make_prediction(model, scaler, history_sequence, feature_name):
    """
    Takes a model, a scaler, a sequence of historical data, and the feature name
    the scaler was trained on, and returns a single predicted value.
    """
    if not model or not scaler:
        logging.warning("Prediction skipped: Model or scaler not loaded.")
        return None

    try:
        # 1. Convert history to a pandas DataFrame with the correct feature name.
        input_df = pd.DataFrame(history_sequence, columns=[feature_name])

        # 2. Scale the data using the DataFrame
        scaled_data = scaler.transform(input_df)

        # 3. Reshape the scaled data for the LSTM model
        reshaped_data = scaled_data.reshape(1, SEQUENCE_LENGTH, 1)

        # 4. Make the prediction
        prediction_scaled = model.predict(reshaped_data, verbose=0)

        # 5. Inverse transform the prediction to get the actual value
        prediction = scaler.inverse_transform(prediction_scaled)

        return prediction[0][0]
    except Exception as e:
        logging.error(f"Error during prediction for feature '{feature_name}': {e}")
        return None


# --- Helper Functions for Prometheus ---

def fetch_cpu_metrics(query):
    """Queries Prometheus for CPU metrics."""
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
    """Queries Prometheus for Memory metrics."""
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
        logging.error(f"Could not fetch or parse Prometheus Memory metrics: {e}")
    return metrics


# --- Background Polling Function ---
def poll_consumer_metrics(stop_event, name, namespace, **kwargs):
    """Polls metrics and, if models are loaded, makes predictions."""
    logging.info(f"Polling thread started for Deployment '{namespace}/{name}'.")
    spec = kwargs.get('spec', {})
    
    while not stop_event.is_set():
        # Deployment Info... (no changes here)
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

        # Metric Polling... (no changes here)
        cpu_query = 'process_cpu_seconds_total{job=~"prometheus.scrape.annotated_pods", instance=~".*:8000"}'
        mem_query = 'process_resident_memory_bytes{job=~"prometheus.scrape.annotated_pods", instance=~".*:8000"}'
        current_cpu_metrics = fetch_cpu_metrics(cpu_query)
        current_mem_metrics = fetch_memory_metrics(mem_query)
        all_instances = set(current_cpu_metrics.keys()) | set(current_mem_metrics.keys())

        for instance in all_instances:
            if instance not in POD_HISTORY:
                POD_HISTORY[instance] = {
                    'cpu_raw': deque(maxlen=SEQUENCE_LENGTH + 1), 
                    'cpu_rates': deque(maxlen=SEQUENCE_LENGTH),
                    'memory': deque(maxlen=SEQUENCE_LENGTH)
                }
            
            print(f"--- Instance: {instance} ---")
            
            # --- Process CPU Metrics ---
            if instance in current_cpu_metrics:
                timestamp, value = current_cpu_metrics[instance]
                if not POD_HISTORY[instance]['cpu_raw'] or POD_HISTORY[instance]['cpu_raw'][-1][1] != value:
                    POD_HISTORY[instance]['cpu_raw'].append((timestamp, value))
                
                cpu_history_raw = list(POD_HISTORY[instance]['cpu_raw'])
                if len(cpu_history_raw) > 1:
                    prev_time, prev_val = cpu_history_raw[-2]
                    curr_time, curr_val = cpu_history_raw[-1]
                    time_delta = (curr_time - prev_time).total_seconds()
                    if curr_val >= prev_val and time_delta > 0:
                        rate = (curr_val - prev_val) / time_delta
                        POD_HISTORY[instance]['cpu_rates'].append(rate)

                cpu_rates_history = list(POD_HISTORY[instance]['cpu_rates'])
                if len(cpu_rates_history) < SEQUENCE_LENGTH:
                    print(f"  CPU: Collecting history... ({len(cpu_rates_history)}/{SEQUENCE_LENGTH} points collected)")
                else:
                    print(f"  CPU History (last {SEQUENCE_LENGTH}): {[float(f'{r:.6f}') for r in cpu_rates_history]}")
                    if MODELS_LOADED.is_set():
                        predicted_cpu = make_prediction(CPU_MODEL, CPU_SCALER, cpu_rates_history, 'cpu_rate')
                        if predicted_cpu is not None:
                            print(f"  Predicted Next CPU Rate: {predicted_cpu:.6f}")
            
            # --- Process Memory Metrics ---
            if instance in current_mem_metrics:
                timestamp, value = current_mem_metrics[instance] # value is in bytes
                
                # ✨ FIX: Store the raw byte value for the model
                POD_HISTORY[instance]['memory'].append(value)
                
                mem_history_bytes = list(POD_HISTORY[instance]['memory'])
                
                if len(mem_history_bytes) < SEQUENCE_LENGTH:
                    print(f"  Memory: Collecting history... ({len(mem_history_bytes)}/{SEQUENCE_LENGTH} points collected)")
                else:
                    # For logging, convert the byte history to MB for readability
                    mem_history_mb = [b / (1024*1024) for b in mem_history_bytes]
                    print(f"  Memory History (MB, last {SEQUENCE_LENGTH}): {[float(f'{m:.2f}') for m in mem_history_mb]}")
                    
                    if MODELS_LOADED.is_set():
                        # ✨ FIX: Use the correct feature name ('memory_bytes') and the byte history for prediction
                        predicted_mem_bytes = make_prediction(MEMORY_MODEL, MEMORY_SCALER, mem_history_bytes, 'memory_bytes')
                        
                        if predicted_mem_bytes is not None:
                            # ✨ FIX: Convert the predicted byte value to MB for logging
                            predicted_mem_mb = predicted_mem_bytes / (1024 * 1024)
                            print(f"  🔮 Predicted Next Memory (MB): {predicted_mem_mb:.2f}")
        
        stop_event.wait(5)
        
    logging.info(f"Polling thread stopped for Deployment '{namespace}/{name}'.")

# --- Kopf Handlers for Lifecycle Management ---

@kopf.on.resume('apps', 'v1', 'deployments', 
                when=lambda name, **_: name == 'consumer')
@kopf.on.create('apps', 'v1', 'deployments', 
                when=lambda name, **_: name == 'consumer')
def start_polling_for_consumer(uid, **kwargs):
    """Loads models on first run and starts the background polling thread."""
    if not MODELS_LOADED.is_set():
        load_models_and_scalers()

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
        thread.join(timeout=10)
        del POLLING_THREADS[uid]
        logging.info(f"Poller for {uid} stopped and cleaned up.")
