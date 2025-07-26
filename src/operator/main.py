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

# --- Application State & Models ---
from state_manager import state
# This now imports the 'state' object which includes our dqn_agent, pod_history, etc.

# --- Kubernetes Configuration ---
try:
    config.load_incluster_config()
except config.ConfigException:
    config.load_kube_config()
k8s_apps_v1 = client.AppsV1Api()

# --- Model Loading & Prediction ---
def load_models_and_scalers():
    model_dir = "/tmp"
    cpu_model_path = f"{model_dir}/cpu.keras"
    cpu_scaler_path = f"{model_dir}/cpu.pkl"
    mem_model_path = f"{model_dir}/memory.keras"
    mem_scaler_path = f"{model_dir}/memory.pkl"
    logging.info("--- Attempting to load forecasting models and scalers ---")
    try:
        state.cpu_model = tf.keras.models.load_model(cpu_model_path)
        state.cpu_scaler = joblib.load(cpu_scaler_path)
        state.memory_model = tf.keras.models.load_model(mem_model_path)
        state.memory_scaler = joblib.load(mem_scaler_path)
        state.models_loaded.set()
        logging.info("All forecasting models and scalers loaded successfully.")
    except Exception as e:
        logging.error(f"CRITICAL: Failed to load models. Forecasting will be disabled. Error: {e}")

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

# --- Prometheus Helpers ---
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

# --- REWARD FUNCTION ---
def calculate_reward(current_cpu_util, current_mem_util, predicted_cpu_util, predicted_mem_util, action, replicas, min_replicas, max_replicas):
    """
    Calculates a reward based on a weighted average of current and predicted utilization.
    """
    # --- 1. Define Targets and Weights ---
    cpu_target_util = 70.0
    mem_target_util = 80.0
    current_weight = 0.4
    predicted_weight = 0.6

    # --- 2. Calculate Weighted Utilization ---
    # This creates a single "effective" utilization that the agent will be rewarded on.
    weighted_cpu_util = (current_cpu_util * current_weight) + (predicted_cpu_util * predicted_weight)
    weighted_mem_util = (current_mem_util * current_weight) + (predicted_mem_util * predicted_weight)

    # --- 3. Calculate Reward based on distance from target ---
    # Reward is higher when closer to the target. Penalize deviation quadratically.
    cpu_error = (weighted_cpu_util - cpu_target_util) / 100.0 # Normalize error
    mem_error = (weighted_mem_util - mem_target_util) / 100.0

    # Start with a max possible reward and subtract penalties
    total_reward = 10.0
    total_reward -= 20 * (cpu_error ** 2) # Quadratic penalty for CPU deviation
    total_reward -= 15 * (mem_error ** 2) # Quadratic penalty for Memory deviation

    # --- 4. Add penalties for actions and boundary conditions ---
    if action in [1, 2]: # If action was Scale Up or Scale Down
        total_reward -= 1.0

    # Penalize trying to scale up when already at max replicas
    if action == 1 and replicas >= max_replicas:
        total_reward -= 20.0 

    # Penalize trying to scale down when already at min replicas
    if action == 2 and replicas <= min_replicas:
        total_reward -= 10.0

    logging.info(f"Calculated Reward: {total_reward:.2f} (WeightedCPU: {weighted_cpu_util:.2f}%, WeightedMem: {weighted_mem_util:.2f}%)")
    return total_reward

# --- Main Logic ---
def run_scaling_cycle(name, namespace):
    logging.info(f"{'='*22}  Starting Scaling Cycle   {'='*22}")
    # --- 1. OBSERVE & FORECAST (Per-Instance) ---
    try:
        deployment = k8s_apps_v1.read_namespaced_deployment(name=name, namespace=namespace)
        current_replicas = deployment.spec.replicas
        resources = deployment.spec.template.spec.containers[0].resources
        cpu_limit = parse_resource_value(resources.limits.get('cpu', resources.requests.get('cpu')))
        mem_limit = parse_resource_value(resources.limits.get('memory', resources.requests.get('memory')))
        annotations = deployment.metadata.annotations or {}
        min_replicas = int(annotations.get('nimbusguard.io/min-replicas', '1'))
        max_replicas = int(annotations.get('nimbusguard.io/max-replicas', '10'))

        # ✨ Use per-instance queries
        cpu_query = 'process_cpu_seconds_total{job=~"prometheus.scrape.annotated_pods", instance=~".*:8000"}'
        mem_query = 'process_resident_memory_bytes{job=~"prometheus.scrape.annotated_pods", instance=~".*:8000"}'
        current_cpu_metrics = fetch_per_instance_metrics(cpu_query)
        current_mem_metrics = fetch_per_instance_metrics(mem_query)
        all_instances = set(current_cpu_metrics.keys()) | set(current_mem_metrics.keys())

        # Initialize totals for aggregation
        total_current_cpu_rate = 0
        total_current_mem = 0
        total_predicted_cpu = 0
        total_predicted_mem = 0
        
        logging.info(f"--- App: {name} | Replicas: {current_replicas} | Constraints: {min_replicas}-{max_replicas} ---")

        # ✨ Loop through each pod instance
        for instance in all_instances:
            if instance not in state.pod_history:
                state.pod_history[instance] = {
                    'cpu_raw': deque(maxlen=state.sequence_length + 1),
                    'cpu_rates': deque(maxlen=state.sequence_length),
                    'memory': deque(maxlen=state.sequence_length)
                }
            
            logging.info(f"--- Processing Instance: {instance} ---")
            
            # --- Process CPU Metrics ---
            instance_cpu_rate = 0
            if instance in current_cpu_metrics:
                timestamp, value = current_cpu_metrics[instance]
                # Add new raw value if it has changed
                if not state.pod_history[instance]['cpu_raw'] or state.pod_history[instance]['cpu_raw'][-1][1] != value:
                    state.pod_history[instance]['cpu_raw'].append((timestamp, value))
                
                # Calculate rate if we have at least two points
                cpu_history_raw = list(state.pod_history[instance]['cpu_raw'])
                if len(cpu_history_raw) > 1:
                    prev_time, prev_val = cpu_history_raw[-2]
                    curr_time, curr_val = cpu_history_raw[-1]
                    time_delta = (curr_time - prev_time).total_seconds()
                    if curr_val >= prev_val and time_delta > 0:
                        instance_cpu_rate = (curr_val - prev_val) / time_delta
                        state.pod_history[instance]['cpu_rates'].append(instance_cpu_rate)
                total_current_cpu_rate += instance_cpu_rate

            # --- Process Memory Metrics ---
            instance_mem = 0
            if instance in current_mem_metrics:
                timestamp, value = current_mem_metrics[instance]
                state.pod_history[instance]['memory'].append(value)
                instance_mem = value
                total_current_mem += instance_mem

            # --- Make Predictions for this instance ---
            cpu_rates_history = list(state.pod_history[instance]['cpu_rates'])
            mem_history_bytes = list(state.pod_history[instance]['memory'])

            if len(cpu_rates_history) < state.sequence_length:
                logging.info(f"  Collecting history... ({len(cpu_rates_history)}/{state.sequence_length} points)")
            else:
                logging.info(f"  CPU History (last {state.sequence_length}): {[float(f'{r:.4f}') for r in cpu_rates_history]}")
                logging.info(f"  Mem History (MB, last {state.sequence_length}): {[float(f'{m/(1024*1024):.2f}') for m in mem_history_bytes]}")
                if state.models_loaded.is_set():
                    predicted_cpu = make_prediction(state.cpu_model, state.cpu_scaler, cpu_rates_history, 'cpu_rate')
                    if predicted_cpu is not None:
                        logging.info(f"  Predicted Next CPU Rate: {predicted_cpu:.4f}")
                        total_predicted_cpu += predicted_cpu
                    
                    predicted_mem = make_prediction(state.memory_model, state.memory_scaler, mem_history_bytes, 'memory_bytes')
                    if predicted_mem is not None:
                        logging.info(f"  Predicted Next Memory (MB): {predicted_mem / (1024*1024):.2f}")
                        total_predicted_mem += predicted_mem
        
        # Check if we have predictions to act on
        if total_predicted_cpu == 0 and total_predicted_mem == 0:
            logging.info("Predictions not yet available for any instance. Skipping DQN cycle.")
            return

    except Exception as e:
        logging.error(f"Failed during OBSERVE stage: {e}", exc_info=True)
        return

    # --- 2. CONSTRUCT DQN STATE (using aggregated values) ---
    current_cpu_util = (total_current_cpu_rate / cpu_limit) * 100 if cpu_limit else 0
    current_mem_util = (total_current_mem / mem_limit) * 100 if mem_limit else 0
    predicted_cpu_util = (total_predicted_cpu / cpu_limit) * 100 if cpu_limit else 0
    predicted_mem_util = (total_predicted_mem / mem_limit) * 100 if mem_limit else 0

    current_state_list = [predicted_cpu_util, predicted_mem_util, current_cpu_util, current_mem_util, current_replicas]
    current_state = np.reshape(current_state_list, [1, state.dqn_agent.state_size])
    logging.info(f"DQN State Vector: [PredCPU%={predicted_cpu_util:.2f}, PredMem%={predicted_mem_util:.2f}, CurrCPU%={current_cpu_util:.2f}, CurrMem%={current_mem_util:.2f}, Replicas={current_replicas}]")

    # --- 3. LEARN (from previous action) ---
    if state.last_state is not None and state.last_action is not None:
        last_replicas = int(state.last_state[0][-1])
        reward = calculate_reward(current_cpu_util, current_mem_util, predicted_cpu_util, predicted_mem_util, state.last_action, last_replicas, min_replicas, max_replicas)
        logging.info(f"Calculated Reward: {reward:.2f} for previous action.")
        state.dqn_agent.remember(state.last_state, state.last_action, reward, current_state, False)
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
    state.last_state = current_state
    state.last_action = action
    logging.info(f"{'='*22} End of Scaling Cycle {'='*22}")


# --- Background Polling Function ---
def poll_consumer_metrics(stop_event, name, namespace, **kwargs):
    logging.info(f"Polling thread started for Deployment '{namespace}/{name}'.")
    while not stop_event.is_set():
        run_scaling_cycle(name, namespace)
        stop_event.wait(5)
    logging.info(f"Polling thread stopped for Deployment '{namespace}/{name}'.")


# --- Kopf Handlers ---
@kopf.on.resume('apps', 'v1', 'deployments', when=lambda name, **_: name == 'consumer')
@kopf.on.create('apps', 'v1', 'deployments', when=lambda name, **_: name == 'consumer')
def start_polling_for_consumer(uid, name, namespace, **kwargs):
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
