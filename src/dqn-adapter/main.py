import asyncio
import logging
from io import BytesIO
import os
from urllib.parse import urlparse
from models.network import EnhancedQNetwork
import joblib
import kopf
import redis
import torch
from aiohttp import web
from data.prometheus import PrometheusClient
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from minio import Minio
from config import get_config
from monitoring.metrics import *
# Import evaluator
from monitoring.evaluator import DQNEvaluator
from ai.graph import create_graph
from core.services import ServiceContainer

# Global services container (initialized in configure function)
services: ServiceContainer = None
from ai.reasoning import DecisionReasoning
from models.trainer import DQNTrainer
from core.controller import setup_probes, setup_http_server, setup_logging_filters, set_decision_function, set_services
# --- Basic Setup ---
load_dotenv()
# Note: kopf handles its own logging setup, we'll configure it in the startup handler
logger = logging.getLogger("DQN_Adapter")

# Load configuration at module level for decorator access
config = get_config()

# Setup probes from controller
setup_probes()


# OLD HTTP-based KEDA endpoints removed - now using gRPC External Scaler
# These fake HTTP endpoints never worked with KEDA external scaler anyway!
# gRPC External Scaler provides truly dynamic DQN control without artificial baselines.

# --- kopf Operator Setup ---
@kopf.on.startup()
async def configure(settings: kopf.OperatorSettings, **kwargs):
    # Configure Kopf to use a different port to avoid conflict with our metrics server
    settings.networking.health_listening_port = config.server.kopf_health_port  # Use config value
    
    # Use annotations storage to avoid K8s 1.16+ schema pruning issues
    # This prevents "Patching failed with inconsistencies" errors
    settings.persistence.progress_storage = kopf.AnnotationsProgressStorage()
    settings.persistence.diffbase_storage = kopf.AnnotationsDiffBaseStorage()
    
    # Production logging format - structured and regex-extractable
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=getattr(logging, config.log_level.value), 
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger.info("STARTUP: NimbusGuard DQN Operator initializing components")
    logger.info("KOPF: using_annotations_storage to_avoid_schema_pruning")
    logger.info(f"IMPROVEMENTS: stabilization_period={config.kubernetes.stabilization_period_seconds}s improved_rewards={config.reward.enable_improved_rewards}")
    
    # Log all configurable hyperparameters for research transparency
    logger.info("=== MODULAR CONFIGURATION ===")
    logger.info(f"DQN_EXPLORATION: epsilon_start={config.dqn.epsilon_start} epsilon_end={config.dqn.epsilon_end} epsilon_decay={config.dqn.epsilon_decay}")
    logger.info(f"DQN_TRAINING: gamma={config.dqn.gamma} lr={config.dqn.learning_rate} memory_capacity={config.dqn.memory_capacity} batch_size={config.dqn.batch_size}")
    logger.info(f"DQN_ARCHITECTURE: hidden_dims={config.dqn.hidden_dims}")
    logger.info(f"REWARD_WEIGHTS: performance={config.reward.performance_weight:.0%} resource={config.reward.resource_weight:.0%} health={config.reward.health_weight:.0%} cost={config.reward.cost_weight:.0%}")
    logger.info("=== END CONFIGURATION ===")
    
    # Initialize service container for dependency injection
    global services
    services = ServiceContainer(
        config=config
    )
    
    services.prometheus_client = PrometheusClient()
    
    # Only initialize LLM if validation is enabled
    if config.ai.enable_llm_validation:
        try:
            services.llm = ChatOpenAI(model=config.ai.model, temperature=config.ai.temperature)
            logger.info(f"LLM: initialized_successfully model={config.ai.model}")
        except Exception as e:
            logger.error(f"LLM: initialization_failed error={e}")
            logger.warning("LLM: consider_disabling_validation ENABLE_LLM_VALIDATION=false")
            raise kopf.PermanentError(f"LLM initialization failed but validation is enabled: {e}")
    else:
        services.llm = None
        logger.info("LLM: skipped_initialization validation_disabled")

    # Initialize MinIO client
    try:
        minio_url = urlparse(config.minio.endpoint).netloc
        services.minio_client = Minio(
            minio_url,
            access_key=config.minio.access_key,
            secret_key=config.minio.secret_key,
            secure=False
        )
        if not services.minio_client.bucket_exists(config.minio.bucket_name):
            services.minio_client.make_bucket(config.minio.bucket_name)
            logger.info(f"MINIO: bucket_created name={config.minio.bucket_name}")
        logger.info("MINIO: connection_established")
    except Exception as e:
        logger.error(f"Failed to connect to MinIO: {e}")
        raise kopf.PermanentError(f"MinIO connection failed: {e}")

    # Load scaler from MinIO (upload from local file if not exists)
    try:
        logger.info(f"SCALER: loading from_minio={config.minio.scaler_name}")
        response = services.minio_client.get_object(config.minio.bucket_name, config.minio.scaler_name)
        buffer = BytesIO(response.read())
        services.scaler = joblib.load(buffer)
        logger.info(f"SCALER: loaded_successfully type=RobustScaler features={len(config.feature_order)}")
    except Exception as e:
        logger.warning(f"SCALER: not_found_in_minio error={e}")
        logger.info("SCALER: attempting_local_upload")
        
        try:
            # Upload local feature scaler to MinIO
            local_scaler_path = config.minio.scaler_path
            if os.path.exists(local_scaler_path):
                with open(local_scaler_path, 'rb') as f:
                    scaler_data = f.read()
                
                # Upload to MinIO
                services.minio_client.put_object(
                    config.minio.bucket_name, 
                    config.minio.scaler_name, 
                    BytesIO(scaler_data), 
                    len(scaler_data),
                    content_type='application/gzip'
                )
                logger.info(f"SCALER: uploaded_to_minio name={config.minio.scaler_name} size={len(scaler_data)}")
                
                # Now load it
                services.scaler = joblib.load(BytesIO(scaler_data))
                logger.info(f"SCALER: local_upload_loaded type=RobustScaler features={len(config.feature_order)}")
            else:
                logger.error(f"SCALER: local_file_missing path={local_scaler_path}")
                raise kopf.PermanentError(f"Feature scaler not available locally or in MinIO")
                
        except Exception as upload_error:
            logger.error(f"SCALER: upload_failed error={upload_error}")
            raise kopf.PermanentError(f"Feature scaler loading failed: {upload_error}")

    # Upload training dataset to MinIO if not exists (for research and retraining)
    try:
        # Check if dataset exists in MinIO
        services.minio_client.stat_object(config.minio.bucket_name, "dqn_features.parquet")
        logger.info("DATASET: already_exists_in_minio name=dqn_features.parquet")
    except Exception:
        logger.info("DATASET: uploading_to_minio")
        try:
            local_dataset_path = "/app/dqn_features.parquet"
            if os.path.exists(local_dataset_path):
                with open(local_dataset_path, 'rb') as f:
                    dataset_data = f.read()
                
                services.minio_client.put_object(
                    config.minio.bucket_name,
                    "dqn_features.parquet",
                    BytesIO(dataset_data),
                    len(dataset_data),
                    content_type='application/octet-stream'
                )
                logger.info(f"DATASET: uploaded_successfully name=dqn_features.parquet size={len(dataset_data)}")
            else:
                logger.warning(f"DATASET: local_file_missing path={local_dataset_path}")
        except Exception as dataset_upload_error:
            logger.warning(f"DATASET: upload_failed error={dataset_upload_error}")
            # Not a fatal error - the system can still operate without historical data

    # Load DQN model from MinIO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        logger.info("DQN_MODEL: loading from_minio=dqn_model.pt")
        response = services.minio_client.get_object(config.minio.bucket_name, "dqn_model.pt")
        buffer = BytesIO(response.read())
        checkpoint = torch.load(buffer, map_location=device)
        
        # Initialize model with scientifically-validated architecture (9 base features)
        state_dim = len(config.feature_order)  # 9 scientifically-selected features
        action_dim = 3
        services.dqn_model = EnhancedQNetwork(state_dim, action_dim, config.dqn.hidden_dims).to(device)
        logger.info(f"DQN_MODEL: architecture_initialized input_features={state_dim} output_actions={action_dim}")
        
        if isinstance(checkpoint, dict) and 'policy_net_state_dict' in checkpoint:
            services.dqn_model.load_state_dict(checkpoint['policy_net_state_dict'])
        else:
            services.dqn_model.load_state_dict(checkpoint)
            
        services.dqn_model.eval()  # Set to evaluation mode
        logger.info(f"DQN_MODEL: loaded_successfully device={device}")
        
        # Initialize combined trainer for real-time learning
        services.dqn_trainer = DQNTrainer(services.dqn_model, device, config.feature_order, services)
        
        # Ensure model is saved to MinIO (in case it was loaded but trainer is new)
        try:
            await services.dqn_trainer._save_model()
            logger.info("DQN_MODEL: saved_to_minio")
        except Exception as save_error:
            logger.error(f"DQN_MODEL: save_failed error={save_error}")
        
        # Start background training loop
        asyncio.create_task(services.dqn_trainer.continuous_training_loop())
        logger.info("DQN_TRAINER: background_loop_started")
        
    except Exception as e:
        logger.warning(f"DQN_MODEL: load_failed error={e}")
        logger.info("DQN_MODEL: using_fallback_logic")
        
        # Even without a pre-trained model, we can start with a fresh model
        try:
            state_dim = len(config.feature_order)  # Scientifically-validated: 9 base features
            action_dim = 3
            services.dqn_model = EnhancedQNetwork(state_dim, action_dim, config.dqn.hidden_dims).to(device)
            logger.info(f"DQN_MODEL: fresh_model_created input_features={state_dim} output_actions={action_dim}")
            
            # Initialize trainer with fresh model
            logger.info("DQN_TRAINER: initializing")
            services.dqn_trainer = DQNTrainer(services.dqn_model, device, config.feature_order, services)
            logger.info("DQN_TRAINER: initialized_successfully")
            
            # Save fresh model to MinIO immediately (synchronous to catch errors)
            logger.info("DQN_MODEL: saving_fresh_model")
            try:
                await services.dqn_trainer._save_model()
                logger.info("DQN_MODEL: fresh_model_saved")
            except Exception as save_error:
                logger.error(f"DQN_MODEL: save_failed error={save_error}")
                # Continue anyway - the system can still function without saving
            
            asyncio.create_task(services.dqn_trainer.continuous_training_loop())
            logger.info("DQN_TRAINER: fresh_model_loop_started")
            
        except Exception as fresh_model_error:
            logger.error(f"DQN_MODEL: fresh_creation_failed error={fresh_model_error}")

    try:
        services.redis_client = redis.from_url(config.redis.url, decode_responses=True)
        logger.info("REDIS: connection_established")
    except Exception as e:
        logger.error(f"REDIS: connection_failed error={e}")
        # Not raising a permanent error, as the operator might still function for decision-making
        
    # Initialize MCP validation with LLM supervisor (only if LLM validation is enabled)
    if config.ai.enable_llm_validation and services.llm is not None:
        if config.ai.mcp_server_url:
            try:
                logger.info(f"MCP: connecting url={config.ai.mcp_server_url}")
                mcp_client = MultiServerMCPClient({
                    "kubernetes": {
                        "url": f"{config.ai.mcp_server_url}/sse/",
                        "transport": "sse"  # Use SSE transport instead of streamable_http
                    }
                })
                tools = await mcp_client.get_tools()
                services.validator_agent = create_react_agent(services.llm, tools=tools)
                logger.info(f"MCP: validator_initialized tools_count={len(tools)}")
                logger.info(f"MCP: available_tools=[{','.join(tool.name for tool in tools)}]")
            except Exception as e:
                logger.error(f"MCP: initialization_failed error={e}")
                logger.info("MCP: fallback_to_llm_only")
                services.validator_agent = create_react_agent(services.llm, tools=[])
        else:
            logger.info("MCP: url_not_set using_llm_only")
            services.validator_agent = create_react_agent(services.llm, tools=[])
    else:
        services.validator_agent = None
        logger.info("VALIDATOR: skipped_initialization llm_validation_disabled")
    
    # Initialize evaluator
    if config.enable_evaluation_outputs:
        try:
            services.evaluator = DQNEvaluator(services.minio_client, bucket_name="evaluation-outputs")
            logger.info("EVALUATOR: initialized")
        except Exception as e:
            logger.error(f"EVALUATOR: initialization_failed error={e}")
            services.evaluator = None
    else:
        logger.info("EVALUATOR: disabled")
        services.evaluator = None


    # Initialize gauges to a default value
    CURRENT_REPLICAS_GAUGE.set(1)
    DESIRED_REPLICAS_GAUGE.set(1)
    
    # Initialize DQN metrics
    DQN_TRAINING_LOSS_GAUGE.set(0.0)
    DQN_EPSILON_GAUGE.set(config.dqn.epsilon_start)
    DQN_BUFFER_SIZE_GAUGE.set(0)
    DQN_TRAINING_STEPS_GAUGE.set(0)
    DQN_DECISION_CONFIDENCE_GAUGE.set(0.0)
    DQN_Q_VALUE_SCALE_UP_GAUGE.set(0.0)
    DQN_Q_VALUE_SCALE_DOWN_GAUGE.set(0.0)
    DQN_Q_VALUE_KEEP_SAME_GAUGE.set(0.0)
    DQN_REWARD_TOTAL_GAUGE.set(0.0)
    DQN_REWARD_PERFORMANCE_GAUGE.set(0.0)
    DQN_REWARD_RESOURCE_GAUGE.set(0.0)
    DQN_REWARD_HEALTH_GAUGE.set(0.0)
    DQN_REWARD_COST_GAUGE.set(0.0)
    
    # Run initial DQN decision to get a baseline
    try:
        logger.info("DECISION: running_initial_baseline")
        await run_intelligent_scaling_decision()
        logger.info("DECISION: initial_completed")
    except Exception as e:
        logger.warning(f"DECISION: initial_failed error={e} will_retry_with_timer")
    
    # Setup HTTP/2 error log suppression
    setup_logging_filters()
    
    # Set services reference in controller so it can access dependencies
    set_services(services)
    
    # Setup and start HTTP server from controller
    app = setup_http_server()
    metrics_server = web.AppRunner(app)
    await metrics_server.setup()
    site = web.TCPSite(metrics_server, '0.0.0.0', 8080)
    await site.start()
    logger.info("HTTP_SERVER: started port=8080 endpoints=[metrics,health,evaluate,decide]")
    
    # Prometheus trigger approach - no gRPC server needed
    # DQN decisions are exposed via nimbusguard_dqn_desired_replicas metric
    # KEDA reads this metric directly via Prometheus trigger
        
    logger.info("STARTUP: complete watching_scaledobject_events")





async def run_intelligent_scaling_decision():
    """Execute the intelligent scaling decision using DQN"""
    logger.info("DECISION: starting_intelligent_scaling")
    graph = create_graph(services)
    try:
        final_state = await graph.ainvoke({}, {"recursion_limit": 15})
        final_replicas = final_state.get('final_decision', 'N/A')
        logger.info(f"DECISION: completed final_replicas={final_replicas}")
    except Exception as e:
        logger.error(f"DECISION: critical_error error={e}", exc_info=True)

# Set the decision function reference in controller
set_decision_function(run_intelligent_scaling_decision)

# Initialize decision reasoning system
decision_reasoning = DecisionReasoning()

# --- Event-Driven DQN Decision Making (Direct Scaling) ---

# Since we're using direct scaling (bypassing KEDA/HPA), we only need to monitor
# the actual deployment replica changes to trigger additional learning

@kopf.timer('apps', 'v1', 'deployments', 
           when=lambda name, namespace, **_: name == config.kubernetes.target_deployment and namespace == config.kubernetes.target_namespace,
                       interval=config.kubernetes.stabilization_period_seconds)  # Configurable interval (default: 30s)
async def continuous_dqn_monitoring(name, namespace, **kwargs):
    """
    🎯 CONTINUOUS DQN MONITORING: Proactive decision-making every STABILIZATION_PERIOD_SECONDS
    
    This replaces the old ScaledObject timer with direct deployment monitoring.
    The DQN continuously monitors metrics and makes intelligent scaling decisions.
    Interval is configurable via STABILIZATION_PERIOD_SECONDS env var (default: 30s).
    """
    logger.info(f"TIMER: continuous_monitoring_triggered deployment={name} namespace={namespace} interval={config.kubernetes.stabilization_period_seconds}s")
    
    try:
        # Run intelligent scaling decision - this is our core DQN loop
        await run_intelligent_scaling_decision()
        logger.info("TIMER: decision_completed")
    except Exception as e:
        logger.error(f"TIMER: decision_failed error={e}", exc_info=True)

@kopf.on.field('apps', 'v1', 'deployments', 
               field='status.replicas')
async def on_replica_count_change(old, new, meta, **kwargs):
    """
    React to actual replica count changes in our target deployment.
    
    This provides immediate feedback when scaling actions complete,
    enabling faster reward calculation and experience generation.
    """
    # Only monitor our target deployment in the nimbusguard namespace
    if meta.get('namespace') != 'nimbusguard' or meta.get('name') != config.kubernetes.target_deployment:
        return
        
    logger.info(f"EVENT: replica_change_detected deployment={meta['name']} "
                               f"namespace={meta.get('namespace')} replicas_changed={old}=>{new}")
    
    # Update our current replicas gauge immediately  
    CURRENT_REPLICAS_GAUGE.set(new or 0)
    
    # This could trigger additional learning if the change was significant
    if old and abs((new or 0) - old) >= 2:
        logger.info(f"EVENT: significant_scaling_detected trigger_additional_learning")
        try:
            await run_intelligent_scaling_decision()
        except Exception as e:
            logger.error(f"EVENT: additional_learning_failed error={e}")

# Direct scaling architecture: DQN → Kubernetes API (no KEDA/HPA middleware)
@kopf.on.cleanup()
async def cleanup(**kwargs):
    logger.info("SHUTDOWN: operator_stopping")
    if services and services.redis_client:
        try:
            services.redis_client.close()
        except Exception as e:
            logger.error(f"SHUTDOWN: redis_close_failed error={e}")

    # Note: metrics_server is still handled separately as it's not in the services container yet
    # This could be refactored further if needed