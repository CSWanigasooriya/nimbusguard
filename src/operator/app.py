import kopf
import logging
import threading
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from kubernetes import client, config
from prometheus_client import start_http_server
import metrics

# Import our modular components
from state_manager import state
from collector import MetricsCollector
from forecaster import MemoryForecaster
from decision_engine import DecisionEngine
from validator import ScalingValidator
from executor import ClusterExecutor
from reward_calculator import RewardCalculator
from config import system_config, dqn_config

# --- Kubernetes Configuration ---
try:
    config.load_incluster_config()
except config.ConfigException:
    config.load_kube_config()

k8s_apps_v1 = client.AppsV1Api()

# --- LangGraph State Definition ---

class AutoscalerState(TypedDict):
    """State dictionary for the LangGraph autoscaler workflow."""
    # Input data
    deployment_name: str
    deployment_namespace: str
    
    # Collected metrics
    metrics_data: Optional[Dict]
    historical_data: list
    current_cpu_util: float
    current_mem_util: float
    total_current_mem: int
    
    # Deployment info
    deployment_info: Optional[Dict]
    current_replicas: int
    min_replicas: int
    max_replicas: int
    cpu_limit: float
    mem_limit: int
    
    # Forecasting results
    prediction_result: Optional[Dict]
    predicted_mem_util: float
    prediction_ready: bool
    
    # Decision making
    state_vector: Optional[Any]  # np.ndarray serialized as list
    dqn_action: int
    target_replicas: int
    
    # Validation
    validation_result: Optional[Dict]
    is_valid: bool
    validation_reason: str
    adjusted_target: int
    
    # Execution
    execution_result: Optional[Dict]
    scaling_successful: bool
    
    # Reward calculation
    reward: float
    reward_components: Optional[Dict]
    
    # Control flow
    should_continue: bool
    error_message: str
    cycle_complete: bool

# --- Initialize Components ---
collector = MetricsCollector()
forecaster = MemoryForecaster()
decision_engine = DecisionEngine()
validator = ScalingValidator()
executor = ClusterExecutor()
reward_calculator = RewardCalculator()

# --- LangGraph Node Functions ---

def collect_metrics_node(state: AutoscalerState) -> AutoscalerState:
    """Node 1: Collect metrics and deployment information."""
    logging.info("[COLLECTOR] Starting metrics collection...")
    
    try:
        # Collect metrics using the MetricsCollector
        metrics_data = collector.collect_and_process_metrics()
        if not metrics_data:
            state["error_message"] = "Failed to collect metrics"
            state["should_continue"] = False
            return state
        
        # Get deployment information
        deployment_info = executor.get_deployment_info(state["deployment_name"], state["deployment_namespace"])
        if not deployment_info:
            state["error_message"] = "Failed to get deployment info"
            state["should_continue"] = False
            return state
        
        # Update state with collected data
        state["metrics_data"] = metrics_data
        state["deployment_info"] = deployment_info
        state["historical_data"] = collector.get_historical_data()
        
        # Extract key metrics
        state["total_current_mem"] = metrics_data['total_memory_bytes']
        state["current_replicas"] = deployment_info['current_replicas']
        state["min_replicas"] = deployment_info['min_replicas']
        state["max_replicas"] = deployment_info['max_replicas']
        state["cpu_limit"] = deployment_info['cpu_limit']
        state["mem_limit"] = deployment_info['memory_limit']
        
        # Calculate utilization percentages
        state["current_cpu_util"] = (metrics_data['total_cpu_rate'] / state["cpu_limit"]) * 100 if state["cpu_limit"] else 0
        state["current_mem_util"] = (state["total_current_mem"] / state["mem_limit"]) * 100 if state["mem_limit"] else 0
        
        logging.info(f"[COLLECTOR] Metrics collected - CPU: {state['current_cpu_util']:.1f}%, Memory: {state['current_mem_util']:.1f}%")
        logging.info(f"[COLLECTOR] Replicas: {state['current_replicas']}, Historical entries: {len(state['historical_data'])}")
        
    except Exception as e:
        logging.error(f"[COLLECTOR] Error: {e}")
        state["error_message"] = f"Collection error: {e}"
        state["should_continue"] = False
    
    return state

def forecast_memory_node(state: AutoscalerState) -> AutoscalerState:
    """Node 2: Generate memory predictions using LSTM."""
    logging.info("[FORECASTER] Starting memory prediction...")
    
    try:
        # Check if forecaster is ready
        if not forecaster.is_ready():
            logging.warning("[FORECASTER] LSTM models not loaded yet")
            state["predicted_mem_util"] = state["current_mem_util"]  # Fallback to current
            state["prediction_ready"] = False
            return state
        
        # Check if we have sufficient data
        if not collector.has_sufficient_aggregated_data():
            logging.info(f"[FORECASTER] Insufficient data for prediction")
            state["predicted_mem_util"] = state["current_mem_util"]  # Fallback to current
            state["prediction_ready"] = False
            return state
        
        # Make prediction
        prediction_result = forecaster.predict_next_interval(state["historical_data"])
        
        if prediction_result:
            state["prediction_result"] = prediction_result
            state["predicted_mem_util"] = (prediction_result['predicted_memory_bytes'] / state["mem_limit"]) * 100
            state["prediction_ready"] = True
            
            logging.info(f"[FORECASTER] Prediction successful:")
            logging.info(f"    Memory: {prediction_result['predicted_memory_mb']:.1f} MB ({state['predicted_mem_util']:.1f}%)")
            logging.info(f"    Change: {prediction_result['memory_change_mb']:+.1f} MB")
            logging.info(f"    Pods: {prediction_result['predicted_pod_count']:.0f} (+{prediction_result['pod_change']:.0f})")
        else:
            logging.warning("[FORECASTER] Prediction failed, using current memory")
            state["predicted_mem_util"] = state["current_mem_util"]
            state["prediction_ready"] = False
        
    except Exception as e:
        logging.error(f"[FORECASTER] Error: {e}")
        state["predicted_mem_util"] = state["current_mem_util"]
        state["prediction_ready"] = False
    
    return state

def make_decision_node(state: AutoscalerState) -> AutoscalerState:
    """Node 3: Make scaling decision using DQN."""
    logging.info("[DECISION] Making scaling decision...")
    
    try:
        # Check if decision engine is ready
        is_ready, ready_message = decision_engine.is_ready()
        if not is_ready:
            logging.error(f"[DECISION] Decision engine not ready: {ready_message}")
            state["error_message"] = f"Decision engine error: {ready_message}"
            state["should_continue"] = False
            return state
        
        # Construct state vector
        state_vector = decision_engine.construct_state_vector(
            state["current_cpu_util"],
            state["current_mem_util"],
            state["predicted_mem_util"],
            state["current_replicas"]
        )
        
        # Validate state vector
        is_valid_vector, vector_error = decision_engine.validate_state_vector(state_vector)
        if not is_valid_vector:
            logging.error(f"[DECISION] Invalid state vector: {vector_error}")
            state["error_message"] = f"State vector error: {vector_error}"
            state["should_continue"] = False
            return state
        
        # Store state vector as list for serialization
        state["state_vector"] = state_vector.tolist() if state_vector is not None else None
        
        # Make decision
        dqn_action = decision_engine.make_decision(state_vector)
        state["dqn_action"] = dqn_action
        
        # Calculate target replicas
        target_replicas = executor.calculate_new_replicas(
            dqn_action, 
            state["current_replicas"], 
            state["min_replicas"], 
            state["max_replicas"]
        )
        state["target_replicas"] = target_replicas
        
        action_name = decision_engine.get_action_name(dqn_action)
        logging.info(f"[DECISION] Action: {dqn_action} ({action_name})")
        logging.info(f"[DECISION] Target replicas: {state['current_replicas']} → {target_replicas}")
        
    except Exception as e:
        logging.error(f"[DECISION] Error: {e}")
        state["error_message"] = f"Decision error: {e}"
        state["should_continue"] = False
    
    return state

def validate_action_node(state: AutoscalerState) -> AutoscalerState:
    """Node 4: Validate the scaling action."""
    logging.info("[VALIDATOR] Validating scaling action...")
    
    try:
        # Prepare validation info
        validation_deployment_info = {
            **state["deployment_info"],
            'current_cpu_util': state["current_cpu_util"],
            'current_mem_util': state["current_mem_util"],
            'predicted_mem_util': state["predicted_mem_util"]
        }
        
        # Check for forced actions first
        forced_action, force_reason = validator.should_force_action(
            state["current_cpu_util"],
            state["current_mem_util"],
            state["predicted_mem_util"],
            state["current_replicas"],
            state["min_replicas"],
            state["max_replicas"]
        )
        
        if forced_action is not None:
            logging.warning(f"[VALIDATOR] {force_reason}")
            state["dqn_action"] = forced_action
            state["target_replicas"] = executor.calculate_new_replicas(
                forced_action, state["current_replicas"], state["min_replicas"], state["max_replicas"]
            )
        
        # Validate the action
        is_valid, validation_reason, adjusted_target = validator.validate_scaling_action(
            state["dqn_action"],
            state["current_replicas"],
            state["target_replicas"],
            state["min_replicas"],
            state["max_replicas"],
            validation_deployment_info
        )
        
        state["is_valid"] = is_valid
        state["validation_reason"] = validation_reason
        state["adjusted_target"] = adjusted_target
        
        # Get comprehensive validation summary
        validation_result = validator.get_validation_summary(
            state["dqn_action"],
            state["current_replicas"],
            state["target_replicas"],
            state["min_replicas"],
            state["max_replicas"],
            validation_deployment_info
        )
        state["validation_result"] = validation_result
        
        # Use adjusted target if validation modified it
        if adjusted_target != state["target_replicas"]:
            logging.info(f"[VALIDATOR] Target adjusted: {state['target_replicas']} → {adjusted_target}")
            state["target_replicas"] = adjusted_target
        
        if is_valid:
            logging.info(f"[VALIDATOR] Action validated: {validation_reason}")
        else:
            logging.warning(f"[VALIDATOR] Action rejected: {validation_reason}")
        
    except Exception as e:
        logging.error(f"[VALIDATOR] Error: {e}")
        state["error_message"] = f"Validation error: {e}"
        state["should_continue"] = False
    
    return state

def execute_scaling_node(state: AutoscalerState) -> AutoscalerState:
    """Node 5: Execute the scaling action."""
    logging.info("[EXECUTOR] Executing scaling action...")
    
    try:
        if not state["is_valid"]:
            logging.info("[EXECUTOR] Skipping execution - action not valid")
            state["scaling_successful"] = False
            return state
        
        if state["target_replicas"] == state["current_replicas"]:
            logging.info("[EXECUTOR] No scaling needed - target equals current")
            state["scaling_successful"] = True
            state["execution_result"] = {
                'success': True,
                'action': 'none',
                'reason': 'No change needed'
            }
            return state
        
        # Execute the scaling with validation reason
        execution_result = executor.scale_deployment(
            state["deployment_name"],
            state["deployment_namespace"],
            state["target_replicas"],
            reason=state["validation_reason"]
        )
        
        state["execution_result"] = execution_result
        state["scaling_successful"] = execution_result.get('success', False)
        
        if state["scaling_successful"]:
            action_name = execution_result.get('action', 'unknown')
            logging.info(f"[EXECUTOR] Scaling successful: {action_name}")
            logging.info(f"[EXECUTOR] Replicas: {state['current_replicas']} → {state['target_replicas']}")
        else:
            error = execution_result.get('error', 'Unknown error')
            logging.error(f"[EXECUTOR] Scaling failed: {error}")
        
    except Exception as e:
        logging.error(f"[EXECUTOR] Error: {e}")
        state["execution_result"] = {'success': False, 'error': str(e)}
        state["scaling_successful"] = False
    
    return state

def calculate_reward_node(state: AutoscalerState) -> AutoscalerState:
    """Node 6: Calculate reward for the DQN agent."""
    logging.info("[REWARD] Calculating reward...")
    
    try:
        # Only calculate reward if we have a previous action to learn from
        if state["state_vector"] is not None:
            # Calculate reward
            reward = reward_calculator.calculate_reward(
                state["current_cpu_util"],
                state["current_mem_util"],
                state["predicted_mem_util"],
                state["dqn_action"],
                state["current_replicas"],
                state["min_replicas"],
                state["max_replicas"]
            )
            state["reward"] = reward
            
            # Get reward components for analysis
            reward_components = reward_calculator.get_reward_components(
                state["current_cpu_util"],
                state["current_mem_util"],
                state["predicted_mem_util"],
                state["dqn_action"],
                state["current_replicas"],
                state["min_replicas"],
                state["max_replicas"]
            )
            state["reward_components"] = reward_components
            
            # Convert state vector back to numpy array for DQN
            state_vector_np = np.array(state["state_vector"]).reshape(1, -1) if state["state_vector"] else None
            
            # Update DQN agent with experience
            decision_engine.learn_from_experience(reward, state_vector_np)
            
            # Prepare for next cycle
            decision_engine.prepare_for_next_cycle(state_vector_np, state["dqn_action"])
            
            logging.info(f"[REWARD] Reward calculated: {reward:.2f}")
            
            # Update metrics
            metrics.DQN_REWARD_TOTAL.set(reward)
            metrics.LSTM_FORECAST_MEMORY_BYTES.set(state["total_current_mem"])
            metrics.NIMBUSGUARD_CURRENT_REPLICAS.set(state["current_replicas"])
            metrics.NIMBUSGUARD_DESIRED_REPLICAS.set(state["target_replicas"])
        else:
            logging.info("[REWARD] No previous state to learn from (first cycle)")
        
        state["cycle_complete"] = True
        
    except Exception as e:
        logging.error(f"[REWARD] Error: {e}")
        state["error_message"] = f"Reward calculation error: {e}"
    
    return state

# --- LangGraph Workflow Setup ---

def create_autoscaler_workflow() -> CompiledStateGraph:
    """Create and compile the LangGraph autoscaler workflow."""
    
    # Create the state graph
    workflow = StateGraph(AutoscalerState)
    
    # Add nodes
    workflow.add_node("collector", collect_metrics_node)
    workflow.add_node("forecaster", forecast_memory_node)
    workflow.add_node("decision", make_decision_node)
    workflow.add_node("validator", validate_action_node)
    workflow.add_node("executor", execute_scaling_node)
    workflow.add_node("reward", calculate_reward_node)
    
    # Define the flow: collector -> forecaster -> decision -> validator -> executor -> reward
    workflow.add_edge(START, "collector")
    workflow.add_edge("collector", "forecaster")
    workflow.add_edge("forecaster", "decision")
    workflow.add_edge("decision", "validator")
    workflow.add_edge("validator", "executor")
    workflow.add_edge("executor", "reward")
    workflow.add_edge("reward", END)
    
    # Compile the workflow
    return workflow.compile()

# --- Global workflow instance ---
autoscaler_workflow = create_autoscaler_workflow()

# --- Scaling Cycle Function ---
def run_scaling_cycle(name: str, namespace: str):
    """Run a complete scaling cycle using the LangGraph workflow."""
    logging.info(f"{'='*20} Starting LangGraph Scaling Cycle {'='*20}")
    
    try:
        # Initialize state as dictionary
        initial_state: AutoscalerState = {
            # Input data
            "deployment_name": name,
            "deployment_namespace": namespace,
            
            # Collected metrics
            "metrics_data": None,
            "historical_data": [],
            "current_cpu_util": 0.0,
            "current_mem_util": 0.0,
            "total_current_mem": 0,
            
            # Deployment info
            "deployment_info": None,
            "current_replicas": 0,
            "min_replicas": 1,
            "max_replicas": 10,
            "cpu_limit": 1.0,
            "mem_limit": 1024 * 1024 * 1024,  # 1GB
            
            # Forecasting results
            "prediction_result": None,
            "predicted_mem_util": 0.0,
            "prediction_ready": False,
            
            # Decision making
            "state_vector": None,
            "dqn_action": 0,
            "target_replicas": 0,
            
            # Validation
            "validation_result": None,
            "is_valid": False,
            "validation_reason": "",
            "adjusted_target": 0,
            
            # Execution
            "execution_result": None,
            "scaling_successful": False,
            
            # Reward calculation
            "reward": 0.0,
            "reward_components": None,
            
            # Control flow
            "should_continue": True,
            "error_message": "",
            "cycle_complete": False
        }
        
        # Execute the workflow
        result = autoscaler_workflow.invoke(initial_state)
        
        # Log cycle completion
        if result["cycle_complete"]:
            logging.info(f"{'='*20} Scaling Cycle Complete {'='*20}")
            if result["scaling_successful"]:
                logging.info(f"Summary: {result['current_replicas']} → {result['target_replicas']} replicas, Reward: {result['reward']:.2f}")
            else:
                logging.info(f"Summary: No scaling performed, Reward: {result['reward']:.2f}")
        else:
            logging.error(f"Scaling cycle incomplete: {result['error_message']}")
            
    except Exception as e:
        logging.error(f"Scaling cycle failed: {e}")
        import traceback
        logging.error(traceback.format_exc())

# --- Background Polling Function ---
def poll_consumer_metrics(stop_event, name, namespace, **kwargs):
    """Background polling function for continuous autoscaling."""
    logging.info(f"Polling thread started for Deployment '{namespace}/{name}' using LangGraph workflow.")
    logging.info(f"Scaling interval: {system_config.scaling_interval} seconds")
    
    while not stop_event.is_set():
        try:
            run_scaling_cycle(name, namespace)
        except Exception as e:
            logging.error(f"Error in scaling cycle: {e}")
        
        # Wait for next cycle (configurable interval)
        stop_event.wait(system_config.scaling_interval)
    
    logging.info(f"Polling thread stopped for Deployment '{namespace}/{name}'.")

# --- Model Loading Function ---
def load_models_and_scalers():
    """Load LSTM models and scalers using configured paths."""
    return forecaster.load_models(
        system_config.lstm_model_path, 
        system_config.lstm_scaler_path
    )

# --- Kopf Handlers ---
@kopf.on.startup()
def startup_handler(**kwargs):
    """Start Prometheus metrics server and initialize components on startup."""
    logging.info("Starting NimbusGuard Autoscaler with LangGraph workflow...")
    
    # Log configuration
    logging.info("Configuration loaded:")
    logging.info(f"  DQN: gamma={dqn_config.gamma}, epsilon={dqn_config.epsilon}, batch_size={dqn_config.batch_size}")
    logging.info(f"  System: interval={system_config.scaling_interval}s, sequence_length={system_config.lstm_sequence_length}")
    logging.info(f"  Target: {system_config.target_deployment} in {system_config.target_namespace}")
    
    # Start Prometheus metrics server
    if not state.prometheus_server_started:
        logging.info(f"Starting Prometheus client server on port {system_config.prometheus_port}.")
        start_http_server(system_config.prometheus_port)
        state.prometheus_server_started = True
    
    # Load LSTM models
    if not state.models_loaded.is_set():
        logging.info("Loading LSTM models...")
        if load_models_and_scalers():
            logging.info("LSTM models loaded successfully")
        else:
            logging.warning("Failed to load LSTM models - will use fallback predictions")
    
    logging.info("NimbusGuard startup complete - LangGraph workflow ready!")

@kopf.on.resume('apps', 'v1', 'deployments', when=lambda name, **_: name == system_config.target_deployment)
@kopf.on.create('apps', 'v1', 'deployments', when=lambda name, **_: name == system_config.target_deployment)
def start_polling_for_consumer(uid, name, namespace, **kwargs):
    """Start polling for the configured target deployment."""
    
    # Load models if not already loaded
    if not state.models_loaded.is_set():
        logging.info("Loading LSTM models...")
        load_models_and_scalers()
    
    # Start polling thread if not already running
    if uid not in state.polling_threads or not state.polling_threads[uid][0].is_alive():
        logging.info(f"Starting LangGraph autoscaler for {uid}")
        stop_event = threading.Event()
        thread = threading.Thread(
            target=poll_consumer_metrics, 
            args=(stop_event, name, namespace), 
            kwargs=kwargs
        )
        state.polling_threads[uid] = (thread, stop_event)
        thread.start()
        logging.info(f"LangGraph workflow polling started for {namespace}/{name}")

@kopf.on.delete('apps', 'v1', 'deployments', when=lambda name, **_: name == system_config.target_deployment)
def stop_polling_for_consumer(uid, **kwargs):
    """Stop polling for the configured target deployment."""
    if uid in state.polling_threads:
        logging.info(f"Stopping LangGraph autoscaler for {uid}")
        thread, stop_event = state.polling_threads[uid]
        stop_event.set()
        thread.join(timeout=10)
        del state.polling_threads[uid]
        logging.info(f"LangGraph workflow polling stopped for {uid}")

# --- Main Entry Point ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.info("NimbusGuard LangGraph Autoscaler starting...")
