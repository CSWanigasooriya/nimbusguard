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
    """Loads the LSTM model and scalers separately."""
    model_dir = "/tmp"
    mem_model_path = f"{model_dir}/memory.keras"
    mem_scaler_path = f"{model_dir}/memory.pkl"
    logging.info("--- Attempting to load LSTM model and scalers ---")
    try:
        # Load the Keras model
        state.memory_model = tf.keras.models.load_model(mem_model_path)
        logging.info("LSTM model loaded successfully.")
        
        # Load the scalers
        scalers_data = joblib.load(mem_scaler_path)
        state.memory_scaler = scalers_data['feature_scaler']
        state.target_scaler = scalers_data['target_scaler']
        logging.info("Scalers loaded successfully.")
        
        # Verify prediction mode
        predict_both = scalers_data.get('predict_both', True)
        lookback_window = scalers_data.get('lookback_window', 20)
        logging.info(f"Model configuration: predict_both={predict_both}, lookback_window={lookback_window}")
        
        state.models_loaded.set()
        logging.info("LSTM model and scalers loaded successfully.")
    except Exception as e:
        logging.error(f"CRITICAL: Failed to load LSTM model and scalers. Forecasting will be disabled. Error: {e}")
        import traceback
        logging.error(traceback.format_exc())

def make_lstm_prediction(model, pod_data):
    """
    LSTM prediction function using raw Keras model directly.
    pod_data format: [{'timestamp': str, 'pod_name': str, 'memory_bytes': int}, ...]
    
    Args:
        model: Raw Keras model
        pod_data: List of dictionaries with timestamp, pod_name, and memory_bytes
    
    Returns:
        Dictionary with prediction results, or None if prediction fails
    """
    if not model or not pod_data:
        return None
    
    try:
        # Check if we have enough data points (need 20+ intervals)
        if len(pod_data) < state.sequence_length:
            logging.info(f"Not enough data for LSTM prediction. Have {len(pod_data)}, need {state.sequence_length}")
            return None
            
        # Convert raw pod data to the format the model expects
        df = pd.DataFrame(pod_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Round timestamps to nearest 15 seconds to group pods with slightly different timestamps
        df['timestamp_rounded'] = df['timestamp'].dt.round('15s')
        
        # Aggregate by rounded timestamp (same as training)
        aggregated = df.groupby('timestamp_rounded').agg({
            'memory_bytes': 'sum',    # Total memory across all consumer pods
            'pod_name': 'count'       # Number of consumer pods
        }).reset_index()
        
        # Rename column back to timestamp
        aggregated = aggregated.rename(columns={'timestamp_rounded': 'timestamp'})
        
        # Rename columns to match training format
        aggregated.columns = ['timestamp', 'total_memory_bytes', 'pod_count']
        
        # Convert to MB (same as training)
        aggregated['total_memory_mb'] = aggregated['total_memory_bytes'] / (1024 * 1024)
        
        # Sort by timestamp
        aggregated = aggregated.sort_values('timestamp').reset_index(drop=True)
        
        # Debug: Show recent aggregated data
        logging.info(f"--- LSTM Aggregation: {len(aggregated)} time intervals from {len(pod_data)} pod entries ---")
        if len(aggregated) > 0:
            latest_agg = aggregated.iloc[-1]
            logging.info(f"    Latest aggregated: {latest_agg['timestamp']} | {latest_agg['total_memory_mb']:.1f} MB | {latest_agg['pod_count']} pods")
        
        # Check if we have enough history after aggregation
        if len(aggregated) < state.sequence_length:
            logging.info(f"Not enough aggregated data for prediction. Need {state.sequence_length}, got {len(aggregated)}")
            logging.info(f"LSTM needs {state.sequence_length * 15} seconds ({state.sequence_length * 15 / 60:.1f} minutes) of data for prediction")
            return None
        
        # Extract features for last N intervals
        features = ['total_memory_mb', 'pod_count']
        recent_data = aggregated[features].tail(state.sequence_length).values
        
        # Filter out periods with pod count changes to get stable prediction data
        current_pod_count = aggregated['pod_count'].iloc[-1]
        stable_data = aggregated[aggregated['pod_count'] == current_pod_count]
        
        if len(stable_data) < state.sequence_length:
            logging.info(f"Not enough stable data (same pod count={current_pod_count}). Have {len(stable_data)}, need {state.sequence_length}")
            logging.info("LSTM requires stable pod count period for accurate prediction")
            return None
        
        # Use stable data for prediction
        recent_data = stable_data[features].tail(state.sequence_length).values
        
        # Debug: Show the actual sequence we're feeding to LSTM
        recent_timestamps = stable_data['timestamp'].tail(state.sequence_length)
        recent_memory = stable_data['total_memory_mb'].tail(state.sequence_length)
        recent_pods = stable_data['pod_count'].tail(state.sequence_length)
        
        logging.info(f"--- LSTM Input Sequence (last {state.sequence_length} stable intervals with {current_pod_count} pods) ---")
        logging.info(f"Memory range: {recent_memory.min():.1f} - {recent_memory.max():.1f} MB")
        logging.info(f"Pod count: {current_pod_count} (constant)")
        logging.info(f"First 3 intervals:")
        for i in range(min(3, len(recent_timestamps))):
            idx = recent_timestamps.index[i]
            logging.info(f"  [{i}] {recent_timestamps.iloc[i]} | {recent_memory.iloc[i]:.1f} MB | {recent_pods.iloc[i]} pods")
        logging.info(f"Last 3 intervals:")
        for i in range(max(0, len(recent_timestamps)-3), len(recent_timestamps)):
            idx = recent_timestamps.index[i]
            logging.info(f"  [{i}] {recent_timestamps.iloc[i]} | {recent_memory.iloc[i]:.1f} MB | {recent_pods.iloc[i]} pods")
        
        # Scale using the scaler
        if not hasattr(state, 'memory_scaler') or state.memory_scaler is None:
            logging.error("Memory scaler not loaded")
            return None
            
        scaled_data = state.memory_scaler.transform(recent_data)
        
        # Reshape for LSTM (batch_size=1, sequence_length=N, features=2)
        model_input = scaled_data.reshape(1, state.sequence_length, len(features))
        
        # Make prediction using the raw Keras model
        prediction_scaled = model.predict(model_input, verbose=0)
        
        # Debug: Show raw model output
        logging.info(f"--- LSTM Raw Output (scaled): {prediction_scaled[0]} ---")
        
        # Inverse transform prediction using target scaler
        if not hasattr(state, 'target_scaler') or state.target_scaler is None:
            logging.error("Target scaler not loaded")
            return None
            
        prediction = state.target_scaler.inverse_transform(prediction_scaled)[0]
        
        # Debug: Show final prediction
        logging.info(f"--- LSTM Final Prediction: Memory={prediction[0]:.1f} MB, Pods={prediction[1]:.1f} ---")
        
        # Get current values for comparison (use stable data, not all aggregated data)
        latest_stable = stable_data.iloc[-1]
        latest_all = aggregated.iloc[-1]
        
        # Debug logging for LSTM input vs current totals
        logging.info(f"--- LSTM Debug: Latest stable data: {latest_stable['total_memory_mb']:.1f} MB ---")
        logging.info(f"--- LSTM Debug: Latest all data: {latest_all['total_memory_mb']:.1f} MB ---")
        logging.info(f"--- LSTM Raw Prediction vs Latest Stable: {prediction[0]:.1f} MB vs {latest_stable['total_memory_mb']:.1f} MB ---")
        
        # Return structured result (use latest_all for current comparison since DQN uses current total)
        result = {
            'predicted_memory_mb': round(prediction[0], 1),
            'predicted_pod_count': round(prediction[1], 0),
            'predicted_memory_bytes': int(prediction[0] * 1024 * 1024),
            'current_memory_bytes': int(latest_all['total_memory_bytes']),
            'memory_change_mb': round(prediction[0] - latest_all['total_memory_mb'], 1),
            'pod_change': round(prediction[1] - latest_all['pod_count'], 0)
        }
        
        # Additional debug: Check if prediction equals current
        if abs(result['memory_change_mb']) < 0.1:
            logging.warning(f"LSTM predicting essentially no change! Raw prediction: {prediction[0]:.6f} MB, Current: {latest_all['total_memory_mb']:.6f} MB")
            logging.warning(f"Raw model output was: {prediction_scaled[0]}")
            
        logging.info(f"LSTM prediction successful: Memory {result['predicted_memory_mb']:.0f}MB (+{result['memory_change_mb']:.0f}), Pods {result['predicted_pod_count']:.0f} (+{result['pod_change']:.0f})")
        return result
        
    except Exception as e:
        logging.error(f"Error during LSTM prediction: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def fetch_per_instance_metrics(query):
    """
    Fetches metrics from Prometheus, returning a dictionary of pod/instance-level data.
    Handles both instant queries (single value) and range queries (array of values).
    For range queries, returns the latest value from the 15s window.
    Each value is a tuple of (timestamp, value).
    """
    prometheus_url = "http://prometheus.nimbusguard.svc:9090"
    api_endpoint = "/api/v1/query"
    full_url = f"{prometheus_url}{api_endpoint}"
    metrics = {}
    
    # Determine if this is a range query
    is_range_query = '[' in query and ']' in query
    
    try:
        response = requests.get(full_url, params={'query': query}, timeout=5)
        response.raise_for_status()
        result = response.json()
        
        if result.get('status') == 'success':
            results = result.get('data', {}).get('result', [])
            logging.info(f"[Prometheus Fetch] Query: {query} returned {len(results)} results (Range: {is_range_query})")
            
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
                            
                            logging.info(f"  [Range Metric] Pod: {identifier}, Container: {container}, Job: {job}, Latest Value: {value} ({len(values_array)} samples)")
                            metrics[identifier] = (timestamp, value)
                        else:
                            logging.warning(f"  [Skip] Empty values array for {identifier}")
                    
                    elif not is_range_query and item.get('value'):
                        # For instant queries, use single value
                        timestamp = pd.to_datetime(item['value'][0], unit='s', utc=True)
                        value = float(item['value'][1])
                        
                        logging.info(f"  [Instant Metric] Pod: {identifier}, Container: {container}, Job: {job}, Value: {value}")
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

        # Use cAdvisor container metrics with 15s range to match DQN decision interval
        # Get 15-second range data and filter by container name and job to get actual consumer containers
        cpu_query = 'container_cpu_usage_seconds_total{container="consumer",job="cadvisor"}[15s]'
        mem_query = 'container_memory_working_set_bytes{container="consumer",job="cadvisor"}[15s]'
        current_cpu_metrics = fetch_per_instance_metrics(cpu_query)
        current_mem_metrics = fetch_per_instance_metrics(mem_query)
        all_instances = set(current_cpu_metrics.keys()) | set(current_mem_metrics.keys())

        total_current_cpu_rate = 0
        total_current_mem = 0
        current_memory_values = []  # Collect all current memory values to find max

        logging.info(f"--- App: {name} | Replicas: {current_replicas} | Constraints: {min_replicas}-{max_replicas} ---")

        for instance in all_instances:
            if instance not in state.pod_history:
                state.pod_history[instance] = {
                    'cpu_raw': deque(maxlen=state.sequence_length + 1)
                    # No longer storing per-pod memory history
                }
            
            logging.info(f"--- Processing Instance: {instance} ---")

            # --- Process CPU Metrics (cAdvisor cumulative CPU seconds from 15s range) ---
            if instance in current_cpu_metrics:
                timestamp, value = current_cpu_metrics[instance]
                # Always append - deque with maxlen automatically removes oldest values
                state.pod_history[instance]['cpu_raw'].append((timestamp, value))
                logging.info(f"  [CPU] Added cumulative value: {value:.2f} seconds (from 15s range, queue size: {len(state.pod_history[instance]['cpu_raw'])})")
                
                cpu_history_raw = list(state.pod_history[instance]['cpu_raw'])
                if len(cpu_history_raw) > 1:
                    prev_time, prev_val = cpu_history_raw[-2]
                    curr_time, curr_val = cpu_history_raw[-1]
                    time_delta = (curr_time - prev_time).total_seconds()
                    # For cAdvisor metrics: calculate CPU rate from cumulative seconds over ~15s intervals
                    if curr_val >= prev_val and time_delta > 0:
                        cpu_delta = curr_val - prev_val
                        instance_cpu_rate = cpu_delta / time_delta
                        cpu_percentage = (instance_cpu_rate / cpu_limit) * 100 if cpu_limit else 0
                        logging.info(f"  [CPU Rate] Instance {instance}: {instance_cpu_rate:.6f} cores/second (Δ{cpu_delta:.6f}s over {time_delta:.2f}s = {cpu_percentage:.2f}%)")
                        total_current_cpu_rate += instance_cpu_rate
                        
                        # Additional debugging for very small rates
                        if instance_cpu_rate < 0.0001:
                            logging.info(f"    [DEBUG] Very low CPU rate detected - this is normal for idle/low-usage pods")
                    else:
                        logging.warning(f"  [CPU] Skipping calculation for {instance}: curr_val={curr_val}, prev_val={prev_val}, time_delta={time_delta}")
                else:
                    logging.info(f"  [CPU] Not enough history for rate calculation on {instance} (need 2+ data points)")
            else:
                logging.warning(f"  [CPU] No metrics found for instance {instance}")

            # --- Collect Memory Metrics (for structured data storage) ---
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
                
                logging.info(f"  [Memory] Pod {instance}: {value / (1024*1024):.2f} MB")
                total_current_mem += value

        # --- CPU Summary Logging ---
        active_pods = len(all_instances)
        pods_with_cpu_activity = len([i for i in all_instances if i in current_cpu_metrics])
        logging.info(f"--- CPU Summary: Total rate {total_current_cpu_rate:.6f} cores/second across {pods_with_cpu_activity}/{active_pods} pods ---")
        
        # --- Process Structured Memory Data and Prediction ---
        if current_memory_values:
            max_memory_value = max(current_memory_values)
            
            # Debug logging for memory values
            memory_values_mb = [val / (1024*1024) for val in current_memory_values]
            logging.info(f"--- Memory Values (MB): {[f'{val:.3f}' for val in memory_values_mb]} ---")
            logging.info(f"--- Max Memory: {max_memory_value} bytes = {max_memory_value / (1024*1024):.6f} MB ---")
            logging.info(f"--- Total Current Memory: {total_current_mem} bytes = {total_current_mem / (1024*1024):.1f} MB ---")
            logging.info(f"--- Sum of current_memory_values: {sum(current_memory_values)} bytes = {sum(current_memory_values) / (1024*1024):.1f} MB ---")
            
            # Log structured data info
            history_data = list(state.global_memory_history)
            logging.info(f"--- Structured Memory History: {len(history_data)} entries stored ---")
            if history_data:
                latest_entries = history_data[-min(3, len(history_data)):]
                for entry in latest_entries:
                    logging.info(f"    {entry['timestamp']} | {entry['pod_name']} | {entry['memory_bytes']/(1024*1024):.2f} MB")
            
            # Check if we have enough structured data for prediction
            if len(history_data) < state.sequence_length:
                logging.info(f"Collecting structured memory history... ({len(history_data)}/{state.sequence_length} points) - DQN decisions pending")
                total_predicted_mem = total_current_mem  # Use current total instead of max
            else:
                logging.info(f"--- Structured Memory History: {len(history_data)} entries available for LSTM prediction ---")
                
                if state.models_loaded.is_set():
                    # Use new LSTM prediction with structured data
                    prediction_result = make_lstm_prediction(state.memory_model, history_data)
                    if prediction_result is not None:
                        # Extract predicted memory in bytes from the result dictionary
                        total_predicted_mem = prediction_result.get('predicted_memory_bytes', total_current_mem)
                        
                        # Log detailed prediction information
                        pred_mb = prediction_result.get('predicted_memory_mb', 0)
                        change_mb = prediction_result.get('memory_change_mb', 0)
                        
                        logging.info(f"LSTM Predicted Next Max Memory: {pred_mb:.1f} MB ({total_predicted_mem} bytes)")
                        logging.info(f"Memory Change Prediction: {change_mb:+.1f} MB")
                        
                        # If the model predicts both memory and pods, log pod prediction too
                        if 'predicted_pod_count' in prediction_result:
                            pred_pods = prediction_result.get('predicted_pod_count', 0)
                            pod_change = prediction_result.get('pod_change', 0)
                            logging.info(f"LSTM Predicted Pod Count: {pred_pods:.0f} (+{pod_change:.0f})")
                    else:
                        total_predicted_mem = total_current_mem
                        logging.warning("LSTM prediction failed, using current total memory")
                else:
                    total_predicted_mem = total_current_mem
                    logging.info("Models not loaded, using current total memory")
        else:
            total_predicted_mem = 0
        
        if total_predicted_mem == 0:
            logging.info("No memory data available from any pod. Skipping DQN cycle.")
            return

        # Check if structured memory history has enough data before making DQN decisions
        if len(state.global_memory_history) < state.sequence_length:
            logging.info(f"Global memory history not yet full ({len(state.global_memory_history)}/{state.sequence_length}). DQN will wait for more data.")
            return
        
        # Additional check: Ensure we have enough aggregated time intervals for LSTM prediction
        # Convert structured history to DataFrame and check aggregated intervals
        try:
            history_df = pd.DataFrame(list(state.global_memory_history))
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            history_df['timestamp_rounded'] = history_df['timestamp'].dt.round('15s')
            aggregated_intervals = len(history_df.groupby('timestamp_rounded'))
            
            if aggregated_intervals < state.sequence_length:
                logging.info(f"Not enough aggregated time intervals for LSTM prediction. Have {aggregated_intervals}, need {state.sequence_length}")
                logging.info(f"DQN waiting for LSTM to be ready. Need {state.sequence_length * 15} seconds ({state.sequence_length * 15 / 60:.1f} minutes) of aggregated data.")
                return
            else:
                logging.info(f"LSTM data requirements met: {aggregated_intervals} time intervals available")
        except Exception as e:
            logging.warning(f"Could not check LSTM data requirements: {e}. Proceeding with DQN.")
            # Continue with DQN if we can't check LSTM requirements
        
        metrics.LSTM_FORECAST_MEMORY_BYTES.set(total_predicted_mem)

    except Exception as e:
        logging.error(f"Failed during OBSERVE stage: {e}", exc_info=True)
        return

    # --- 2. CONSTRUCT DQN STATE (using aggregated values) ---
    logging.info("Proceeding with DQN decision making...")
    current_cpu_util = (total_current_cpu_rate / cpu_limit) * 100 if cpu_limit else 0
    current_mem_util = (total_current_mem / mem_limit) * 100 if mem_limit else 0
    predicted_mem_util = (total_predicted_mem / mem_limit) * 100 if mem_limit else 0
    
    # Enhanced logging for CPU utilization
    logging.info(f"--- Resource Utilization ---")
    logging.info(f"  CPU: {total_current_cpu_rate:.6f} cores used / {cpu_limit:.2f} limit = {current_cpu_util:.3f}%")
    logging.info(f"  Memory: {total_current_mem/(1024*1024):.1f} MB used / {mem_limit/(1024*1024):.1f} MB limit = {current_mem_util:.3f}%")
    logging.info(f"  Predicted Memory: {total_predicted_mem/(1024*1024):.1f} MB = {predicted_mem_util:.3f}%")
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
    if action == 0:
        metrics.DQN_ACTION_KEEP_SAME_TOTAL.inc()
    elif action == 1:
        metrics.DQN_ACTION_SCALE_UP_TOTAL.inc()
    elif action == 2:
        metrics.DQN_ACTION_SCALE_DOWN_TOTAL.inc()
        
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
@kopf.on.startup()
def startup_handler(**kwargs):
    """Start Prometheus metrics server on startup."""
    logging.info("Starting Prometheus client server on port 8080.")
    start_http_server(8080)
    state.prometheus_server_started = True

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