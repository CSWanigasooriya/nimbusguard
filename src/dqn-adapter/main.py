import os
import logging
import asyncio
import time
from typing import TypedDict, List, Dict, Any
import json

import requests
import numpy as np
import joblib
import redis
import kopf
from dotenv import load_dotenv
from prometheus_client import Gauge, generate_latest
from aiohttp import web

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langchain_mcp_adapters.client import MultiServerMCPClient

# --- Basic Setup ---
load_dotenv()
# Note: kopf handles its own logging setup, we'll configure it in the startup handler
logger = logging.getLogger("DQN_Adapter")

# --- Kopf Health Check Probes ---
@kopf.on.probe(id='status')
def health_status(**kwargs):
    """Health status probe for Kopf liveness check"""
    return {"status": "healthy", "service": "dqn-adapter"}

@kopf.on.probe(id='redis_connection')
def redis_health(**kwargs):
    """Check Redis connection health"""
    try:
        if redis_client and redis_client.ping():
            return {"redis": "connected"}
        else:
            return {"redis": "disconnected"}
    except Exception as e:
        return {"redis": f"error: {str(e)}"}

@kopf.on.probe(id='scaler_loaded')
def scaler_health(**kwargs):
    """Check if the feature scaler is loaded"""
    return {"scaler": "loaded" if scaler else "not_loaded"}

# --- KEDA Metrics Server with HTTP/2 Error Handling ---
async def metrics_handler(request):
    """Provide Prometheus metrics for KEDA"""
    metrics_data = generate_latest()
    return web.Response(text=metrics_data.decode('utf-8'), content_type='text/plain; version=0.0.4; charset=utf-8')

async def metrics_api_handler(request):
    """KEDA Metrics API endpoint - returns DQN desired replicas in simple JSON format"""
    current_value = float(DESIRED_REPLICAS_GAUGE._value._value)
    
    response = {
        "dqn": {
            "desired_replicas": current_value,
            "status": "active",
            "timestamp": int(time.time())
        }
    }
    return web.json_response(response)

class HTTP2ErrorFilter(logging.Filter):
    """Filter to suppress HTTP/2 protocol error logs"""
    def filter(self, record):
        if record.name == "aiohttp.server":
            message = record.getMessage()
            if "PRI/Upgrade" in message or "BadHttpMessage" in message:
                return False  # Suppress this log
        return True  # Allow other logs

# --- Environment & Configuration ---
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus.nimbusguard.svc:9090")
KSERVE_URL = os.getenv("KSERVE_URL")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")
TARGET_DEPLOYMENT = os.getenv("TARGET_DEPLOYMENT", "consumer")
TARGET_NAMESPACE = os.getenv("TARGET_NAMESPACE", "nimbusguard")
SCALER_PATH = os.getenv("SCALER_PATH", "/app/feature_scaler.gz")
STABILIZATION_PERIOD_SECONDS = int(os.getenv("STABILIZATION_PERIOD_SECONDS", 60)) # Time to wait after action
REWARD_LATENCY_WEIGHT = float(os.getenv("REWARD_LATENCY_WEIGHT", 10.0)) # Higher = more penalty for latency
REWARD_REPLICA_COST = float(os.getenv("REWARD_REPLICA_COST", 0.1)) # Cost per replica
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
REPLAY_BUFFER_KEY = "replay_buffer"

# Enhanced feature set - matching the learner
FEATURE_ORDER = [
    'request_rate', 'request_latency_p95', 'error_rate', 'current_replicas',
    'cpu_usage_avg', 'memory_usage_avg', 'memory_usage_max', 'gc_pressure',
    'request_rate_trend', 'hour_of_day', 'day_of_week'
]

# --- Prometheus Metrics ---
# These are defined globally and will be exposed by kopf's built-in metrics server.
DESIRED_REPLICAS_GAUGE = Gauge('nimbusguard_dqn_desired_replicas', 'Desired replicas calculated by the DQN adapter')
CURRENT_REPLICAS_GAUGE = Gauge('nimbusguard_current_replicas', 'Current replicas of the target deployment as seen by the adapter')

# --- Global Clients and Models ---
# These will be initialized in the kopf startup handler
prometheus_client = None
llm = None
scaler = None
validator_agent = None
redis_client = None
metrics_server = None

# --- LangGraph State ---
class Experience(TypedDict):
    state: Dict[str, Any]
    action: str
    reward: float
    next_state: Dict[str, Any]
    
class ScalingState(TypedDict):
    current_metrics: Dict[str, float]
    current_replicas: int
    dqn_prediction: Dict[str, Any]
    llm_validation_response: Dict[str, Any]
    final_decision: int
    experience: Experience
    error: str

# --- LangGraph Nodes ---
async def get_live_metrics(state: ScalingState, is_next_state: bool = False) -> Dict[str, Any]:
    node_name = "observe_next_state" if is_next_state else "get_live_metrics"
    logger.info(f"==> Node: {node_name}")
    
    from datetime import datetime
    current_time = datetime.now()
    
    # Enhanced metrics matching the learner's feature extraction
    queries = {
        # Workload features
        "request_rate": 'sum(rate(http_requests_total{job="prometheus.scrape.annotated_pods"}[1m]))',
        "request_latency_p95": 'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job="prometheus.scrape.annotated_pods"}[2m])) by (le))',
        "error_rate": 'sum(rate(http_requests_total{job="prometheus.scrape.annotated_pods",code!~"2.."}[1m])) / sum(rate(http_requests_total{job="prometheus.scrape.annotated_pods"}[1m]))',
        
        # Resource features
        "current_replicas": f'kube_deployment_status_replicas{{deployment="{TARGET_DEPLOYMENT}",namespace="{TARGET_NAMESPACE}"}}',
        "cpu_usage_avg": f'avg(rate(container_cpu_usage_seconds_total{{container="{TARGET_DEPLOYMENT}",namespace="{TARGET_NAMESPACE}"}}[2m]))',
        "memory_usage_avg": f'avg(container_memory_usage_bytes{{container="{TARGET_DEPLOYMENT}",namespace="{TARGET_NAMESPACE}"}}) / 1024 / 1024',
        "memory_usage_max": f'max(container_memory_usage_bytes{{container="{TARGET_DEPLOYMENT}",namespace="{TARGET_NAMESPACE}"}}) / 1024 / 1024',
        
        # System health
        "gc_pressure": 'sum(rate(go_gc_duration_seconds_sum{job="prometheus.scrape.annotated_pods"}[2m])) / sum(rate(go_gc_duration_seconds_count{job="prometheus.scrape.annotated_pods"}[2m]))',
        
        # Trend features
        "request_rate_trend": 'rate(sum(rate(http_requests_total{job="prometheus.scrape.annotated_pods"}[1m]))[5m:])'
    }
    
    tasks = {name: prometheus_client.query(query) for name, query in queries.items()}
    results = await asyncio.gather(*tasks.values())
    
    metrics = dict(zip(tasks.keys(), results))
    current_replicas = int(metrics.pop('current_replicas', 1))
    
    # Add temporal features
    metrics['hour_of_day'] = current_time.hour
    metrics['day_of_week'] = current_time.weekday()
    
    CURRENT_REPLICAS_GAUGE.set(current_replicas)
    logger.info(f"  - Enhanced Metrics: request_rate={metrics.get('request_rate', 0):.2f}, "
                f"latency_p95={metrics.get('request_latency_p95', 0):.3f}, "
                f"replicas={current_replicas}")
    return {"current_metrics": metrics, "current_replicas": current_replicas}

async def get_dqn_recommendation(state: ScalingState) -> Dict[str, Any]:
    logger.info("==> Node: get_dqn_recommendation")
    if not KSERVE_URL or not scaler:
        logger.error("KSERVE_URL or scaler not configured. Skipping DQN prediction.")
        return {"error": "DQN_NOT_CONFIGURED"}

    metrics = state['current_metrics'].copy()
    metrics['current_replicas'] = state['current_replicas']
    
    # Enhanced feature vector
    feature_vector = [metrics.get(feat, 0.0) for feat in FEATURE_ORDER]
    scaled_features = scaler.transform([feature_vector])
    logger.info(f"  - Using enhanced features ({len(feature_vector)} dimensions)")
    
    payload = {"inputs": [{"name": "dqn_input", "shape": list(scaled_features.shape), "datatype": "FP32", "data": scaled_features.tolist()}]}
    
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: requests.post(KSERVE_URL, json=payload))
        response.raise_for_status()
        prediction = response.json()['outputs'][0]['data']
        action_index = np.argmax(prediction)
        action_map = {0: "Scale Down", 1: "Keep Same", 2: "Scale Up"}
        action_name = action_map.get(action_index, "Unknown")
        
        logger.info(f"  - DQN recommends: '{action_name}' (Q-values: {[f'{q:.3f}' for q in prediction]})")
        experience_update = {"state": metrics, "action": action_name}
        return {"dqn_prediction": {"action_name": action_name, "q_values": prediction}, "experience": experience_update}
    except Exception as e:
        logger.error(f"DQN inference failed: {e}", exc_info=True)
        return {"error": f"DQN_INFERENCE_FAILED: {e}"}

async def validate_with_llm(state: ScalingState) -> Dict[str, Any]:
    logger.info("==> Node: validate_with_llm")
    if not validator_agent:
        logger.warning("  - Validator agent not initialized, skipping.")
        return {"llm_validation_response": {"approved": True, "reason": "Agent not available."}}

    prompt = f"""An automated DQN model suggests the scaling action: '{state['dqn_prediction']['action_name']}'.
    The current state of the deployment '{TARGET_DEPLOYMENT}' is:
    - Current Replicas: {state['current_replicas']}
    - Current Metrics: {state['current_metrics']}

    Validate if this is a safe action. Use your tools to check the live cluster state if necessary.
    Respond with a JSON object: {{"approved": boolean, "reason": "your reasoning"}}.
    """
    try:
        response = await validator_agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
        last_message = response['messages'][-1].content
        # TODO: Robust JSON parsing from last_message
        return {"llm_validation_response": {"approved": True, "reason": last_message}}
    except Exception as e:
        logger.error(f"LLM validation failed: {e}")
        return {"llm_validation_response": {"approved": True, "reason": f"Validation failed: {e}"}}

def plan_final_action(state: ScalingState) -> Dict[str, Any]:
    logger.info("==> Node: plan_final_action")
    current_replicas = state['current_replicas']
    action_name = state['dqn_prediction']['action_name']
    
    new_replicas = current_replicas
    if action_name == 'Scale Up': new_replicas += 1
    elif action_name == 'Scale Down': new_replicas -= 1
    
    final_decision = max(1, new_replicas)
    DESIRED_REPLICAS_GAUGE.set(final_decision)
    logger.info(f"  - Final Decision: Scale from {current_replicas} to {final_decision}. Gauge updated.")
    return {"final_decision": final_decision}

async def wait_for_system_to_stabilize(state: ScalingState) -> None:
    logger.info(f"==> Node: wait_for_system_to_stabilize (waiting {STABILIZATION_PERIOD_SECONDS}s)")
    await asyncio.sleep(STABILIZATION_PERIOD_SECONDS)
    return {}

async def observe_next_state_and_calculate_reward(state: ScalingState) -> Dict[str, Any]:
    logger.info("==> Node: observe_next_state_and_calculate_reward")
    next_state_data = await get_live_metrics(state, is_next_state=True)
    latency = next_state_data.get("current_metrics", {}).get("avg_latency", 1.0)
    if latency == 0: latency = 0.001
    
    num_replicas = next_state_data.get("current_replicas", 1)
    latency_reward = REWARD_LATENCY_WEIGHT / latency
    cost = REWARD_REPLICA_COST * num_replicas
    reward = latency_reward - cost
    
    logger.info(f"  - Reward calculated: {reward:.4f} (Latency Reward: {latency_reward:.4f}, Cost: {cost:.4f})")
    
    experience = state['experience']
    experience['reward'] = reward
    experience['next_state'] = {**next_state_data['current_metrics'], 'current_replicas': num_replicas}
    return {"experience": experience}
    
def log_experience(state: ScalingState) -> Dict[str, Any]:
    logger.info("==> Node: log_experience")
    exp = state['experience']
    if redis_client:
        experience_json = json.dumps(exp)
        redis_client.lpush(REPLAY_BUFFER_KEY, experience_json)
        logger.info(f"--- Experience Tuple Sent to Redis ---")
    else:
        logger.warning("Redis client not available. Could not log experience.")
    return {}

def create_graph():
    workflow = StateGraph(ScalingState)
    workflow.add_node("get_live_metrics", get_live_metrics)
    workflow.add_node("get_dqn_recommendation", get_dqn_recommendation)
    workflow.add_node("validate_with_llm", validate_with_llm)
    workflow.add_node("plan_final_action", plan_final_action)
    workflow.add_node("wait_for_system_to_stabilize", wait_for_system_to_stabilize)
    workflow.add_node("observe_next_state_and_calculate_reward", observe_next_state_and_calculate_reward)
    workflow.add_node("log_experience", log_experience)
    
    workflow.set_entry_point("get_live_metrics")
    workflow.add_edge("get_live_metrics", "get_dqn_recommendation")
    workflow.add_edge("get_dqn_recommendation", "validate_with_llm")
    workflow.add_edge("validate_with_llm", "plan_final_action")
    workflow.add_edge("plan_final_action", "wait_for_system_to_stabilize")
    workflow.add_edge("wait_for_system_to_stabilize", "observe_next_state_and_calculate_reward")
    workflow.add_edge("observe_next_state_and_calculate_reward", "log_experience")
    workflow.add_edge("log_experience", END)
    
    return workflow.compile()

# --- kopf Operator Setup ---
@kopf.on.startup()
async def configure(settings: kopf.OperatorSettings, **kwargs):
    # kopf's default logging format is slightly different, so we align it
    logging.getLogger().handlers.clear()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("🚀 NimbusGuard DQN Operator starting up...")
    
    # Initialize all global clients
    global prometheus_client, llm, scaler, validator_agent, redis_client, metrics_server
    
    prometheus_client = PrometheusClient() # Use our custom PrometheusClient
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    try:
        scaler = joblib.load(SCALER_PATH)
        logger.info(f"Successfully loaded feature scaler from {SCALER_PATH}")
    except Exception as e:
        logger.error(f"FATAL: Could not load scaler from {SCALER_PATH}. Inference will fail.")
        raise kopf.PermanentError(f"Scaler not found at {SCALER_PATH}")

    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        logger.info("Successfully connected to Redis for experience logging.")
    except Exception as e:
        logger.error(f"Failed to connect to Redis during startup: {e}")
        # Not raising a permanent error, as the operator might still function for decision-making
        
    # Temporarily disable MCP validation to eliminate connection errors
    # TODO: Fix MCP client configuration later
    logger.info("MCP validation temporarily disabled to eliminate connection errors.")
    validator_agent = create_react_agent(llm, tools=[])
    
    # if MCP_SERVER_URL:
    #     try:
    #         mcp_client = MultiServerMCPClient({
    #             "kubernetes": {
    #                 "url": f"{MCP_SERVER_URL}/sse/",
    #                 "transport": "streamable_http"
    #             }
    #         })
    #         tools = await mcp_client.get_tools()
    #         validator_agent = create_react_agent(llm, tools=tools)
    #         logger.info(f"LLM validator agent initialized with {len(tools)} tools from MCP.")
    #     except Exception as e:
    #         logger.error(f"Failed to initialize validator agent from MCP: {e}", exc_info=True)
    # else:
    #     logger.warning("MCP_SERVER_URL not set. Validator agent will have no tools.")
    #     validator_agent = create_react_agent(llm, tools=[])

    # Initialize gauges to a default value
    CURRENT_REPLICAS_GAUGE.set(1)
    DESIRED_REPLICAS_GAUGE.set(1)
    
    # Add HTTP/2 error log suppression
    aiohttp_server_logger = logging.getLogger("aiohttp.server")
    aiohttp_server_logger.addFilter(HTTP2ErrorFilter())
    
    # Start metrics server for KEDA
    app = web.Application()
    app.router.add_get('/metrics', metrics_handler)
    app.router.add_get('/api/v1/dqn-metrics', metrics_api_handler)
    
    # Add a root handler
    async def root_handler(request):
        return web.json_response({"message": "NimbusGuard DQN Adapter Metrics Server", "status": "running"})
    
    app.router.add_get('/', root_handler)
    
    metrics_server = web.AppRunner(app)
    await metrics_server.setup()
    site = web.TCPSite(metrics_server, '0.0.0.0', 8080)
    await site.start()
    logger.info("🌐 KEDA Metrics API server with HTTP/2 error suppression started on port 8080")
        
    logger.info(f"✅ Startup complete. Watching for ScaledObject events.")

@kopf.on.cleanup()
async def cleanup(**kwargs):
    logger.info("Operator shutting down.")
    if redis_client:
        try:
            redis_client.close()
        except Exception as e:
            logger.error(f"Error closing Redis client: {e}")
    
    if metrics_server:
        try:
            await metrics_server.cleanup()
            logger.info("Metrics server stopped.")
        except Exception as e:
            logger.error(f"Error stopping metrics server: {e}")

# --- ScaledObject Event Handlers ---
@kopf.on.create('keda.sh', 'v1alpha1', 'scaledobjects', labels={'component': 'keda-dqn'})
@kopf.on.update('keda.sh', 'v1alpha1', 'scaledobjects', labels={'component': 'keda-dqn'})
async def on_scaledobject_event(spec, status, meta, **kwargs):
    """React to ScaledObject events - this is where the intelligent scaling happens"""
    logger.info(f"🎯 ScaledObject event detected: {meta['name']}")
    
    # Check if this is our DQN ScaledObject
    if meta['name'] != 'consumer-scaler-dqn':
        return
    
    # Get current scaling status from KEDA
    current_replicas = status.get('originalReplicaCount', 1)
    is_active = any(condition.get('type') == 'Active' and condition.get('status') == 'True' 
                   for condition in status.get('conditions', []))
    
    logger.info(f"📊 Current state: {current_replicas} replicas, Active: {is_active}")
    
    # Run DQN decision making process
    await run_intelligent_scaling_decision()

async def run_intelligent_scaling_decision():
    """Execute the intelligent scaling decision using DQN"""
    logger.info("🧠 Starting intelligent scaling decision process...")
    graph = create_graph()
    try:
        final_state = await graph.ainvoke({}, {"recursion_limit": 15})
        logger.info(f"✅ Intelligent scaling decision complete. Final decision: {final_state.get('final_decision', 'N/A')} replicas")
    except Exception as e:
        logger.error(f"❌ Critical error in intelligent scaling process: {e}", exc_info=True)

class PrometheusClient:
    """ A simple async-wrapper for the requests library for Prometheus. """
    async def query(self, promql: str) -> float:
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': promql}))
            response.raise_for_status()
            result = response.json()['data']['result']
            if result: return float(result[0]['value'][1])
            return 0.0
        except Exception as e:
            logger.error(f"Prometheus query failed for '{promql}': {e}")
            return 0.0