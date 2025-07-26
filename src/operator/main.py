import kopf
import logging
import requests
import pandas as pd
import threading
import time
import joblib
import numpy as np
import tensorflow as tf
import sklearn
from kubernetes import client, config
from collections import deque
from prometheus_client import start_http_server
import metrics

# --- Application State & Models ---
from state_manager import state
# This now imports the 'state' object which includes our dqn_agent, pod_history, etc.
# IMPORTANT: Ensure the DQN agent's state_size is set to 4 in state_manager.py

# --- Kubernetes Configuration ---
try:
    config.load_incluster_config()
except config.ConfigException:
    config.load_kube_config()
k8s_apps_v1 = client.AppsV1Api()

# --- Model Loading & Prediction ---
def load_models_and_scalers():
    """Loads only the memory forecasting model and scaler."""
    model_dir = "/tmp"
    mem_model_path = f"{model_dir}/memory.keras"
    mem_scaler_path = f"{model_dir}/memory.pkl"
    logging.info("--- Attempting to load memory forecasting model and scaler ---")
    try:
        # REMOVED CPU model loading
        state.memory_model = tf.keras.models.load_model(mem_model_path)
        state.memory_scaler = joblib.load(mem_scaler_path)
        state.models_loaded.set()
        logging.info("Memory forecasting model and scaler loaded successfully.")
    except Exception as e:
        logging.error(f"CRITICAL: Failed to load memory model. Forecasting will be disabled. Error: {e}")

def make_prediction(model, scaler, history_sequence, feature_name):
    if not model or not scaler:
        return None
    try:
        input_df = pd.DataFrame(history_sequence, columns=[feature_name])
        scaled_data = scaler.transform(input_df)
        reshaped_data = scaled_data.reshape(1, state.sequence_length, 1)
        prediction_scaled = model.predict(reshaped_data, verbose=0)
        prediction = scaler.inverse_transform(prediction_scaled)
        return prediction[0][0]
    except Exception as e:
        logging.error(f"Error during prediction for feature '{feature_name}': {e}")
        return None

def fetch_per_instance_metrics(query):
    """
    Fetches metrics from Prometheus, returning a dictionary of instance-level data.
    Each value is a tuple of (timestamp, value).
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
                # Use pandas for robust timestamp parsing
                timestamp = pd.to_datetime(item['value'][0], unit='s', utc=True)
                value = float(item['value'][1])
                if instance:
                    # --- ADDED LOGGING ---
                    logging.info(f"  [Prometheus Fetch] Instance: {instance}, Value: {value}, Timestamp: {timestamp}")
                    metrics[instance] = (timestamp, value)
    except Exception as e:
        logging.error(f"Could not fetch Prometheus metrics for query '{query}': {e}")
    return metrics

# --- Kubernetes Resource Helper ---
def parse_resource_value(resource_str):
    if not resource_str: return 0.0
    resource_str = resource_str.lower()
    if resource_str.endswith('m'): return float(resource_str[:-1]) / 1000.0
    if resource_str.endswith('gi'): return float(resource_str[:-2]) * (1024**3)
    if resource_str.endswith('mi'): return float(resource_str[:-2]) * (1024**2)
    if resource_str.endswith('ki'): return float(resource_str[:-2]) * 1024
    return float(resource_str)

def calculate_reward(current_cpu_util, current_mem_util, predicted_mem_util, action, replicas, min_replicas, max_replicas):
    """
    Calculates a reward that is tolerant of prediction uncertainty by using
    a target band instead of a single point.
    """
    # --- 1. Define Targets, Thresholds, and Weights ---
    cpu_target_util = 70.0
    
    # --- CHANGE 1: Define a target band for memory to absorb prediction variance ---
    mem_target_band_low = 70.0
    mem_target_band_high = 85.0
    
    low_util_threshold = 30.0
    
    # --- CHANGE 2: Reduce weight of the prediction to be more conservative ---
    current_weight = 0.5
    predicted_weight = 0.5
    
    weighted_mem_util = (current_mem_util * current_weight) + (predicted_mem_util * predicted_weight)

    # --- 2. Initialize Reward ---
    total_reward = 0.0

    # --- 3. Check for Low Utilization Zone ---
    if current_cpu_util < low_util_threshold and weighted_mem_util < low_util_threshold:
        logging.info("--> Operating in LOW UTILIZATION ZONE")
        if action == 2: # SCALE DOWN
            total_reward += 15.0
            logging.info("      [+15.0] Rewarding correct action (Scale Down).")
        elif action == 0: # DO NOTHING
            total_reward -= 5.0
            logging.info("      [-5.0] Penalizing inaction.")
        elif action == 1: # SCALE UP
            total_reward -= 20.0
            logging.info("      [-20.0] Penalizing incorrect action (Scale Up).")
        
        if replicas == min_replicas:
            total_reward += 5.0
            logging.info("      [+5.0] Efficiency bonus for being at min_replicas.")

    # --- 4. Normal/High Utilization Zone ---
    else:
        logging.info("--> Operating in NORMAL/HIGH UTILIZATION ZONE")
        total_reward = 10.0
        
        # --- CHANGE 3: Calculate error based on the distance from the target band ---
        mem_error = 0.0
        if weighted_mem_util > mem_target_band_high:
            mem_error = (weighted_mem_util - mem_target_band_high) / 100.0
            logging.info(f"      Memory is ABOVE target band by {mem_error*100:.2f}%.")
        elif weighted_mem_util < mem_target_band_low:
            mem_error = (mem_target_band_low - weighted_mem_util) / 100.0
            logging.info(f"      Memory is BELOW target band by {mem_error*100:.2f}%.")
        else:
            logging.info("      Memory is WITHIN target band. No penalty.")

        cpu_error = (current_cpu_util - cpu_target_util) / 100.0
        
        cpu_penalty = 20 * (cpu_error ** 2)
        mem_penalty = 15 * (mem_error ** 2) # Penalize deviation from the band
        
        total_reward -= cpu_penalty
        total_reward -= mem_penalty
        
        logging.info(f"      [-{cpu_penalty:.2f}] CPU penalty. [-{mem_penalty:.2f}] Memory penalty.")

        if action in [1, 2]:
            total_reward -= 1.0
            logging.info("      [-1.0] Penalty for scaling action.")

    # --- 5. Add Boundary Penalties ---
    if action == 1 and replicas >= max_replicas:
        total_reward -= 20.0
        logging.info("      [-20.0] Penalty for attempting to scale up at max_replicas.")
    if action == 2 and replicas <= min_replicas:
        total_reward -= 10.0
        logging.info("      [-10.0] Penalty for attempting to scale down at min_replicas.")

    metrics.DQN_REWARD_TOTAL.set(total_reward)
    logging.info(f"Final Calculated Reward: {total_reward:.2f} (CPU: {current_cpu_util:.2f}%, WeightedMem: {weighted_mem_util:.2f}%)")
    return total_reward

# --- Main Logic ---
def run_scaling_cycle(name, namespace):
    logging.info(f"{'='*22}  Starting Scaling Cycle   {'='*22}")
    # --- 1. OBSERVE & FORECAST (Per-Instance) ---
    try:
        deployment = k8s_apps_v1.read_namespaced_deployment(name=name, namespace=namespace)
        current_replicas = deployment.spec.replicas
        metrics.NIMBUSGUARD_CURRENT_REPLICAS.set(current_replicas)
        resources = deployment.spec.template.spec.containers[0].resources
        cpu_limit = parse_resource_value(resources.limits.get('cpu', resources.requests.get('cpu')))
        mem_limit = parse_resource_value(resources.limits.get('memory', resources.requests.get('memory')))
        annotations = deployment.metadata.annotations or {}
        min_replicas = int(annotations.get('nimbusguard.io/min-replicas', '1'))
        max_replicas = int(annotations.get('nimbusguard.io/max-replicas', '10'))

        cpu_query = 'process_cpu_seconds_total{job=~"prometheus.scrape.annotated_pods", instance=~".*:8000"}'
        mem_query = 'process_resident_memory_bytes{job=~"prometheus.scrape.annotated_pods", instance=~".*:8000"}'
        current_cpu_metrics = fetch_per_instance_metrics(cpu_query)
        current_mem_metrics = fetch_per_instance_metrics(mem_query)
        all_instances = set(current_cpu_metrics.keys()) | set(current_mem_metrics.keys())

        total_current_cpu_rate = 0
        total_current_mem = 0
        total_predicted_mem = 0 # REMOVED: total_predicted_cpu

        logging.info(f"--- App: {name} | Replicas: {current_replicas} | Constraints: {min_replicas}-{max_replicas} ---")

        for instance in all_instances:
            if instance not in state.pod_history:
                state.pod_history[instance] = {
                    'cpu_raw': deque(maxlen=state.sequence_length + 1),
                    'memory': deque(maxlen=state.sequence_length)
                }
            
            logging.info(f"--- Processing Instance: {instance} ---")

            # --- Process CPU Metrics (for current value only) ---
            if instance in current_cpu_metrics:
                timestamp, value = current_cpu_metrics[instance]
                if not state.pod_history[instance]['cpu_raw'] or state.pod_history[instance]['cpu_raw'][-1][1] != value:
                    state.pod_history[instance]['cpu_raw'].append((timestamp, value))
                
                cpu_history_raw = list(state.pod_history[instance]['cpu_raw'])
                if len(cpu_history_raw) > 1:
                    prev_time, prev_val = cpu_history_raw[-2]
                    curr_time, curr_val = cpu_history_raw[-1]
                    time_delta = (curr_time - prev_time).total_seconds()
                    if curr_val >= prev_val and time_delta > 0:
                        instance_cpu_rate = (curr_val - prev_val) / time_delta
                        total_current_cpu_rate += instance_cpu_rate

            # --- Process Memory Metrics and Prediction ---
            if instance in current_mem_metrics:
                timestamp, value = current_mem_metrics[instance]
                state.pod_history[instance]['memory'].append(value)
                total_current_mem += value
            
            mem_history_bytes = list(state.pod_history[instance]['memory'])

            if len(mem_history_bytes) < state.sequence_length:
                logging.info(f"  Collecting memory history... ({len(mem_history_bytes)}/{state.sequence_length} points)")
            else:
                logging.info(f"  Mem History (MB, last {state.sequence_length}): {[float(f'{m/(1024*1024):.2f}') for m in mem_history_bytes]}")
                if state.models_loaded.is_set():
                    # REMOVED: CPU prediction logic
                    predicted_mem = make_prediction(state.memory_model, state.memory_scaler, mem_history_bytes, 'memory_bytes')
                    if predicted_mem is not None:
                        logging.info(f"  Predicted Next Memory (MB): {predicted_mem / (1024*1024):.2f}")
                        total_predicted_mem += predicted_mem
        
        if total_predicted_mem == 0:
            logging.info("Memory predictions not yet available for any instance. Skipping DQN cycle.")
            return
        
        metrics.LSTM_FORECAST_MEMORY_BYTES.set(total_predicted_mem)

    except Exception as e:
        logging.error(f"Failed during OBSERVE stage: {e}", exc_info=True)
        return

    # --- 2. CONSTRUCT DQN STATE (using aggregated values) ---
    current_cpu_util = (total_current_cpu_rate / cpu_limit) * 100 if cpu_limit else 0
    current_mem_util = (total_current_mem / mem_limit) * 100 if mem_limit else 0
    predicted_mem_util = (total_predicted_mem / mem_limit) * 100 if mem_limit else 0
    # REMOVED: predicted_cpu_util

    # MODIFIED: State vector no longer includes predicted_cpu_util. State size is now 4.
    current_state_list = [predicted_mem_util, current_cpu_util, current_mem_util, current_replicas]
    current_state = np.reshape(current_state_list, [1, state.dqn_agent.state_size])
    logging.info(f"DQN State Vector: [PredMem%={predicted_mem_util:.2f}, CurrCPU%={current_cpu_util:.2f}, CurrMem%={current_mem_util:.2f}, Replicas={current_replicas}]")

    # --- 3. LEARN (from previous action) ---
    if state.last_state is not None and state.last_action is not None:
        last_replicas = int(state.last_state[0][-1])
        # MODIFIED: call to calculate_reward no longer passes predicted_cpu_util
        reward = calculate_reward(current_cpu_util, current_mem_util, predicted_mem_util, state.last_action, last_replicas, min_replicas, max_replicas)
        logging.info(f"Calculated Reward: {reward:.2f} for previous action.")
        state.dqn_agent.remember(state.last_state, state.last_action, reward, current_state, False)
        metrics.DQN_REPLAY_BUFFER_SIZE.set(len(state.dqn_agent.memory))
        metrics.DQN_EXPERIENCES_ADDED_TOTAL.inc()
        logging.info("Training agent with new experience...")
        state.dqn_agent.replay()

    # --- 4. ACT ---
    action = state.dqn_agent.act(current_state)
    logging.info(f"DQN Agent chose Action: {action} (0:None, 1:Up, 2:Down) with Epsilon: {state.dqn_agent.epsilon:.3f}")

    # --- 5. EXECUTE ---
    new_replicas = current_replicas
    if action == 1: new_replicas = min(current_replicas + 1, max_replicas)
    elif action == 2: new_replicas = max(current_replicas - 1, min_replicas)

    if new_replicas != current_replicas:
        try:
            logging.info(f"SCALING: Changing replicas from {current_replicas} to {new_replicas}.")
            patch = {'spec': {'replicas': new_replicas}}
            k8s_apps_v1.patch_namespaced_deployment_scale(name=name, namespace=namespace, body=patch)
        except Exception as e:
            logging.error(f"Failed to scale deployment: {e}")
    else:
        logging.info("No scaling action required.")

    # --- 6. PREPARE FOR NEXT CYCLE ---
    metrics.NIMBUSGUARD_DESIRED_REPLICAS.set(new_replicas)
    state.last_state = current_state
    state.last_action = action
    logging.info(f"{'='*22} End of Scaling Cycle {'='*22}")


# --- Background Polling Function ---
def poll_consumer_metrics(stop_event, name, namespace, **kwargs):
    logging.info(f"Polling thread started for Deployment '{namespace}/{name}'.")
    while not stop_event.is_set():
        run_scaling_cycle(name, namespace)
        stop_event.wait(15)
    logging.info(f"Polling thread stopped for Deployment '{namespace}/{name}'.")


# --- Kopf Handlers ---
@kopf.on.resume('apps', 'v1', 'deployments', when=lambda name, **_: name == 'consumer')
@kopf.on.create('apps', 'v1', 'deployments', when=lambda name, **_: name == 'consumer')
def start_polling_for_consumer(uid, name, namespace, **kwargs):
    if not hasattr(state, 'prometheus_server_started'):
        logging.info("Starting Prometheus client server on port 8080.")
        start_http_server(8080)
        state.prometheus_server_started = True

    if not state.models_loaded.is_set():
        load_models_and_scalers()
    if uid not in state.polling_threads or not state.polling_threads[uid][0].is_alive():
        logging.info(f"Starting poller for {uid}")
        stop_event = threading.Event()
        thread = threading.Thread(target=poll_consumer_metrics, args=(stop_event, name, namespace), kwargs=kwargs)
        state.polling_threads[uid] = (thread, stop_event)
        thread.start()

@kopf.on.delete('apps', 'v1', 'deployments', when=lambda name, **_: name == 'consumer')
def stop_polling_for_consumer(uid, **kwargs):
    if uid in state.polling_threads:
        logging.info(f"Stopping poller for {uid}")
        thread, stop_event = state.polling_threads[uid]
        stop_event.set()
        thread.join(timeout=10)
        del state.polling_threads[uid]
        logging.info(f"Poller for {uid} stopped and cleaned up.")