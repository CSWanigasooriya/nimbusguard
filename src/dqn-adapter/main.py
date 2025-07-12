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
    logger.info(f"IMPROVEMENTS: stabilization_period={config.kubernetes.stabilization_period_seconds}s")
    
    # Log all configurable hyperparameters for research transparency
    logger.info("=== MODULAR CONFIGURATION ===")
    logger.info(f"DQN_EXPLORATION: epsilon_start={config.dqn.epsilon_start} epsilon_end={config.dqn.epsilon_end} epsilon_decay={config.dqn.epsilon_decay}")
    logger.info(f"DQN_TRAINING: gamma={config.dqn.gamma} lr={config.dqn.learning_rate} memory_capacity={config.dqn.memory_capacity} batch_size={config.dqn.batch_size}")
    logger.info(f"DQN_ARCHITECTURE: hidden_dims={config.dqn.hidden_dims}")
    logger.info("REWARD_SYSTEM: pure_llm_based_rewards no_hardcoded_weights")
    logger.info("=== END CONFIGURATION ===")
    
    # Initialize service container for dependency injection
    global services
    services = ServiceContainer(
        config=config
    )
    
    services.prometheus_client = PrometheusClient()
    
    # Always try to initialize LLM if OpenAI API key is available (needed for both safety and rewards)
    try:
        services.llm = ChatOpenAI(model=config.ai.model, temperature=config.ai.temperature)
        logger.info(f"LLM: initialized_successfully model={config.ai.model}")
    except Exception as e:
        logger.error(f"LLM: initialization_failed error={e}")
        
        # Check if this is an API key issue (graceful fallback)
        error_str = str(e).lower()
        if 'api_key' in error_str or 'openai_api_key' in error_str:
            logger.warning("LLM: api_key_missing - llm_features_disabled")
            logger.info("LLM: dqn_will_operate_without_llm_features (safety_monitor_and_rewards)")
            services.llm = None
        else:
            # For other errors (network issues, etc.), still fail hard
            raise kopf.PermanentError(f"LLM initialization failed with non-API-key error: {e}")

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
    
    # Check if we should force fresh model to fix Q-value bias
    force_fresh_model = os.getenv("FORCE_FRESH_MODEL", "false").lower() == "true"
    clear_experience_buffer = os.getenv("CLEAR_EXPERIENCE_BUFFER", "false").lower() == "true"
    
    if force_fresh_model:
        logger.warning("DQN_MODEL: force_fresh_model_enabled - creating_fresh_weights_to_fix_bias")
        # Skip loading existing model and go directly to fresh model creation
        services.dqn_model = None
        services.dqn_trainer = None
    else:
        try:
            # Try to load existing model from MinIO
            logger.info("DQN_MODEL: loading_from_minio")
            response = services.minio_client.get_object(config.minio.bucket_name, "dqn_model.pt")
            model_bytes = response.read()
            
            checkpoint = torch.load(BytesIO(model_bytes), map_location=device)
            
            # Initialize model with current architecture
            state_dim = len(config.feature_order)  # Scientifically-validated: 9 consumer performance features
            action_dim = 3
            services.dqn_model = EnhancedQNetwork(state_dim, action_dim, config.dqn.hidden_dims).to(device)
            
            # Try to load state dict with graceful handling of architecture mismatches
            if 'policy_net_state_dict' in checkpoint:
                try:
                    services.dqn_model.load_state_dict(checkpoint['policy_net_state_dict'], strict=False)
                    logger.info("DQN_MODEL: loaded_with_partial_compatibility (policy_net)")
                except Exception as load_error:
                    logger.warning(f"DQN_MODEL: policy_net_load_failed error={load_error}")
                    logger.info("DQN_MODEL: using_fresh_weights_with_checkpoint_metadata")
            else:
                try:
                    services.dqn_model.load_state_dict(checkpoint, strict=False)
                    logger.info("DQN_MODEL: loaded_with_partial_compatibility (direct)")
                except Exception as load_error:
                    logger.warning(f"DQN_MODEL: direct_load_failed error={load_error}")
                    logger.info("DQN_MODEL: using_fresh_weights")
                
            services.dqn_model.eval()  # Set to evaluation mode
            logger.info(f"DQN_MODEL: loaded_successfully device={device}")
            
            # Initialize combined trainer for real-time learning
            try:
                services.dqn_trainer = DQNTrainer(services.dqn_model, device, config.feature_order, services)
                logger.info("DQN_TRAINER: initialized_successfully")
                
                # Clear experience buffer if requested to ensure fresh learning
                if clear_experience_buffer:
                    services.dqn_trainer.memory.buffer.clear()
                    logger.warning("DQN_TRAINER: experience_buffer_cleared - fresh_learning_enabled")
                
                # Start background training loop
                asyncio.create_task(services.dqn_trainer.continuous_training_loop())
                logger.info("DQN_TRAINER: background_loop_started")
                
                # Ensure model is saved to MinIO (in case it was loaded but trainer is new)
                try:
                    await services.dqn_trainer._save_model()
                    logger.info("DQN_MODEL: saved_to_minio")
                except Exception as save_error:
                    logger.warning(f"DQN_MODEL: save_failed error={save_error}")
                    
            except Exception as trainer_error:
                logger.error(f"DQN_TRAINER: initialization_failed error={trainer_error}")
                # Clear the trainer to trigger fallback
                services.dqn_trainer = None
            
        except Exception as e:
            logger.warning(f"DQN_MODEL: load_failed error={e}")
            logger.info("DQN_MODEL: using_fallback_logic")
            # Clear variables to trigger fresh model creation below
            services.dqn_model = None
            services.dqn_trainer = None
    
    # Create fresh model if loading failed or was forced
    if services.dqn_model is None or services.dqn_trainer is None:
        try:
            state_dim = len(config.feature_order)  # Scientifically-validated: 9 consumer performance features
            action_dim = 3
            services.dqn_model = EnhancedQNetwork(state_dim, action_dim, config.dqn.hidden_dims).to(device)
            logger.info(f"DQN_MODEL: fresh_model_created input_features={state_dim} output_actions={action_dim}")
            
            # Initialize trainer with fresh model
            logger.info("DQN_TRAINER: initializing_fresh")
            services.dqn_trainer = DQNTrainer(services.dqn_model, device, config.feature_order, services)
            logger.info("DQN_TRAINER: initialized_successfully")
            
            # Clear experience buffer if requested to ensure fresh learning  
            if clear_experience_buffer:
                services.dqn_trainer.memory.buffer.clear()
                logger.warning("DQN_TRAINER: experience_buffer_cleared - fresh_learning_enabled")
            
            # Start background training loop
            asyncio.create_task(services.dqn_trainer.continuous_training_loop())
            logger.info("DQN_TRAINER: fresh_model_loop_started")
            
            # Save fresh model to MinIO immediately (synchronous to catch errors)
            logger.info("DQN_MODEL: saving_fresh_model")
            try:
                await services.dqn_trainer._save_model()
                logger.info("DQN_MODEL: fresh_model_saved")
            except Exception as save_error:
                logger.warning(f"DQN_MODEL: save_failed error={save_error}")
                # Continue anyway - the system can still function without saving
            
        except Exception as fresh_model_error:
            logger.error(f"DQN_MODEL: fresh_creation_failed error={fresh_model_error}")
            # Set trainer to None to indicate failure
            services.dqn_trainer = None

    try:
        services.redis_client = redis.from_url(config.redis.url, decode_responses=True)
        logger.info("REDIS: connection_established")
    except Exception as e:
        logger.error(f"REDIS: connection_failed error={e}")
        # Not raising a permanent error, as the operator might still function for decision-making
        
    # Initialize LLM agents separately for validation and rewards
    if services.llm is not None:
        # Create MCP tools if available
        tools = []
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
                logger.info(f"MCP: tools_loaded count={len(tools)}")
                logger.info(f"MCP: available_tools=[{','.join(tool.name for tool in tools)}]")
            except Exception as e:
                logger.error(f"MCP: initialization_failed error={e}")
                logger.info("MCP: fallback_to_llm_only")
                tools = []
        else:
            logger.info("MCP: url_not_set using_llm_only")
        
        # Initialize safety validation agent (only if validation enabled)
        if config.ai.enable_llm_validation:
            services.validator_agent = create_react_agent(services.llm, tools=tools)
            logger.info("LLM_SAFETY: validation_agent_created")
        else:
            services.validator_agent = None
            logger.info("LLM_SAFETY: validation_disabled")
        
        # Initialize reward agent (always when LLM available, separate from validation)
        if config.ai.enable_llm_rewards:
            services.reward_agent = create_react_agent(services.llm, tools=tools)
            logger.info("LLM_REWARDS: reward_agent_created")
        else:
            services.reward_agent = None
            logger.info("LLM_REWARDS: rewards_disabled")
        
        # Log final LLM capabilities
        validation_status = "enabled" if config.ai.enable_llm_validation else "disabled"
        reward_status = "enabled" if config.ai.enable_llm_rewards else "disabled"
        logger.info(f"LLM_FEATURES: safety_monitor={validation_status} rewards={reward_status}")
    else:
        services.validator_agent = None
        services.reward_agent = None
        logger.info("LLM_FEATURES: safety_monitor=disabled rewards=disabled (no_api_key)")
    
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
                       interval=config.kubernetes.stabilization_period_seconds) 
async def continuous_dqn_monitoring(name, namespace, **kwargs):
    """
    🎯 CONTINUOUS DQN MONITORING: Decision-making with proper stabilization waiting
    
    This replaces the old ScaledObject timer with direct deployment monitoring.
    The DQN makes decisions, waits for system stabilization, then calculates rewards.
    Interval is 3x STABILIZATION_PERIOD_SECONDS to allow full decision cycle (default: 90s).
    """
    logger.info(f"TIMER: continuous_monitoring_triggered deployment={name} namespace={namespace} interval={config.kubernetes.stabilization_period_seconds * 3}s")
    
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