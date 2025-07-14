"""
NimbusGuard Proactive Scaling Operator
=====================================

Production-ready Kubernetes operator that combines LSTM forecasting with DQN 
for intelligent, proactive autoscaling decisions.

Based on the proven architecture of the existing DQN adapter with enhancements:
- LSTM-based load forecasting for proactive scaling
- Enhanced DQN with forecast integration  
- Direct Kubernetes API scaling (bypasses KEDA/HPA)
- MinIO model persistence
- Comprehensive monitoring and health checks
"""

import asyncio
import logging
import time
from urllib.parse import urlparse

import kopf
import torch
from aiohttp import web
from minio import Minio

from config.settings import load_config
from dqn.agent import ProactiveDQNAgent
from forecasting.predictor import LoadForecaster
from k8s.client import KubernetesClient
from k8s.scaler import DirectScaler
from metrics.collector import MetricsCollector
from prometheus.client import PrometheusClient
from workflow.graph import create_workflow
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# Global configuration and services
config = load_config()
services = {}

logger = logging.getLogger("NimbusGuard-Operator")


# Setup Kopf probes
@kopf.on.probe(id='health')
def health_probe(**kwargs):
    """Main health probe for operator liveness."""
    return {
        "status": "healthy",
        "service": "nimbusguard-operator",
        "components": {
            "prometheus": services.get('prometheus') is not None,
            "forecaster": services.get('forecaster') is not None,
            "dqn_agent": services.get('dqn_agent') is not None,
            "k8s_client": services.get('k8s_client') is not None,
            "feature_scaler": (services.get('dqn_agent') and
                               services['dqn_agent'].scaler is not None) if services.get('dqn_agent') else False
        }
    }


@kopf.on.probe(id='model_status')
def model_status_probe(**kwargs):
    """Check ML model status."""
    try:
        if 'dqn_agent' in services:
            stats = services['dqn_agent'].get_training_stats()
            return {
                "dqn_model": "loaded",
                "training_steps": stats.get('training_steps', 0),
                "epsilon": stats.get('epsilon', 0.0),
                "buffer_size": stats.get('buffer_size', 0)
            }
        return {"dqn_model": "not_loaded"}
    except Exception as e:
        return {"dqn_model": f"error: {str(e)}"}


@kopf.on.probe(id='storage')
def storage_probe(**kwargs):
    """Check MinIO storage connectivity."""
    try:
        if 'minio_client' in services:
            # Simple connectivity test
            services['minio_client'].bucket_exists("models")
            return {"minio": "connected"}
        return {"minio": "not_configured"}
    except Exception as e:
        return {"minio": f"error: {str(e)}"}


async def setup_langgraph_agent():
    """Initialize LangGraph agent (LLM + MCP tools) if LLM validation is enabled."""
    if not getattr(config.ai, "enable_llm_validation", False):
        logger.info("LLM validation is disabled by config; skipping LangGraph agent setup.")
        return
    try:
        llm = ChatOpenAI(
            model=config.ai.model_name,
            temperature=config.ai.temperature,
            api_key=config.ai.openai_api_key
        )
        services['llm'] = llm

        mcp_client = MultiServerMCPClient(
            connections={
                "kubernetes": {
                    "url": f"{config.ai.mcp_server_url}/sse",
                    "transport": "sse"
                }
            }
        )
        services['mcp_client'] = mcp_client

        tools = await mcp_client.get_tools()
        agent = create_react_agent(llm, tools)
        services['langgraph_agent'] = agent
        logger.info("✅ LangGraph agent initialized (LLM + MCP tools)")
    except Exception as e:
        logger.error(f"❌ Failed to initialize LangGraph agent: {e}")


@kopf.on.startup()
async def startup_handler(settings: kopf.OperatorSettings, **kwargs):
    """Initialize the operator with all components."""

    # Configure Kopf settings
    settings.networking.health_listening_port = config.server.health_port
    settings.persistence.progress_storage = kopf.AnnotationsProgressStorage()
    settings.persistence.diffbase_storage = kopf.AnnotationsDiffBaseStorage()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger.info("🚀 NimbusGuard Proactive Scaling Operator starting...")
    logger.info(f"📊 Configuration: forecasting={config.forecasting.enabled}, "
                f"target={config.scaling.target_deployment}")

    # Initialize MinIO client
    await setup_minio_storage()

    # Initialize Prometheus client
    services['prometheus'] = PrometheusClient()
    logger.info("✅ Prometheus client initialized")

    # Initialize Kubernetes client
    services['k8s_client'] = KubernetesClient(config.scaling)
    services['scaler'] = DirectScaler(services['k8s_client'], config.scaling)
    logger.info("✅ Kubernetes client initialized")

    # Initialize DQN agent first (so forecaster can use its scaler)
    await setup_dqn_agent()

    # Initialize LSTM forecaster
    if config.forecasting.enabled:
        await setup_forecaster()
    else:
        logger.info("⏭️ LSTM forecasting disabled")

    # Initialize workflow
    services['workflow'] = create_workflow(services, config)
    logger.info("✅ LangGraph workflow created")

    # Setup metrics collection
    services['metrics'] = MetricsCollector(config.metrics)
    logger.info("✅ Metrics collector initialized")

    # Initialize LangGraph agent if needed
    await setup_langgraph_agent()

    # Start HTTP server for metrics and health
    await start_http_server()

    # Run initial scaling decision
    try:
        logger.info("🎯 Running initial scaling decision...")
        await execute_scaling_decision()
        logger.info("✅ Initial scaling decision completed")
    except Exception as e:
        logger.warning(f"⚠️ Initial scaling decision failed: {e}")

    logger.info("🎉 NimbusGuard operator startup complete!")


async def setup_minio_storage():
    """Initialize MinIO client and ensure buckets exist."""
    try:
        minio_url = urlparse(config.minio.endpoint).netloc
        services['minio_client'] = Minio(
            minio_url,
            access_key=config.minio.access_key,
            secret_key=config.minio.secret_key,
            secure=False
        )

        # Ensure models bucket exists
        if not services['minio_client'].bucket_exists("models"):
            services['minio_client'].make_bucket("models")
            logger.info("📦 Created MinIO models bucket")

        logger.info("✅ MinIO storage initialized")

    except Exception as e:
        logger.error(f"❌ MinIO initialization failed: {e}")
        raise kopf.PermanentError(f"MinIO storage required: {e}")


async def setup_forecaster():
    """Initialize LSTM forecaster."""
    try:
        services['forecaster'] = LoadForecaster()
        # Initialize forecaster (synchronous method)
        if services['forecaster'].initialize():
            logger.info("✅ LSTM forecaster initialized")
        else:
            logger.error("❌ LSTM forecaster initialization failed")
            services['forecaster'] = None
    except Exception as e:
        logger.error(f"❌ LSTM forecaster initialization failed: {e}")
        logger.warning("⚠️ Continuing without forecasting capabilities")
        services['forecaster'] = None


async def setup_dqn_agent():
    """Initialize DQN agent without loading feature scaler from storage."""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        services['dqn_agent'] = ProactiveDQNAgent(
            config=config.scaling,
            device=device,
            minio_client=services['minio_client']
        )
        # No scaler loading, always fit at runtime
        logger.info("✅ DQN agent created (scaler will be fit at runtime)")
        # Try to load existing model
        model_loaded = await services['dqn_agent'].load_model()
        if model_loaded:
            logger.info("✅ DQN model loaded from MinIO")
        else:
            logger.info("🆕 Using fresh DQN model")
        # Start training loop
        asyncio.create_task(continuous_training_loop())
        logger.info(f"✅ DQN agent initialized (device: {device})")
    except Exception as e:
        logger.error(f"❌ DQN agent initialization failed: {e}")
        raise kopf.PermanentError(f"DQN agent required: {e}")


async def start_http_server():
    """Start HTTP server for metrics and health endpoints."""
    app = web.Application()

    # Metrics endpoint
    async def metrics_handler(request):
        if 'metrics' in services:
            metrics_data = await services['metrics'].get_prometheus_metrics()
            return web.Response(text=metrics_data, content_type='text/plain')
        return web.Response(text="# No metrics available\n", content_type='text/plain')

    # Manual scaling trigger
    async def trigger_scaling_handler(request):
        try:
            await execute_scaling_decision()
            return web.json_response({"status": "success", "message": "Scaling decision triggered"})
        except Exception as e:
            logger.error(f"Manual scaling trigger failed: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    # Model save trigger
    async def save_model_handler(request):
        try:
            if 'dqn_agent' in services:
                success = await services['dqn_agent'].save_model()
                if success:
                    return web.json_response({"status": "success", "message": "Model saved"})
                else:
                    return web.json_response({"status": "error", "message": "Model save failed"}, status=500)
            return web.json_response({"status": "error", "message": "DQN agent not available"}, status=400)
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    # Only add /healthz for Kopf liveness if needed (Kopf handles this itself)
    app.router.add_get('/metrics', metrics_handler)
    app.router.add_post('/trigger-scaling', trigger_scaling_handler)
    app.router.add_post('/save-model', save_model_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', config.server.port)
    await site.start()

    logger.info(f"🌐 HTTP server started on port {config.server.port}")


async def execute_scaling_decision():
    """Execute a single scaling decision using the workflow."""
    try:
        logger.info("🎯 Executing scaling decision workflow")

        # Get current replica count
        current_replicas = 1
        if 'k8s_client' in services and 'scaler' in services:
            try:
                deployment_info = await services['k8s_client'].get_deployment(
                    config.scaling.target_deployment,
                    config.scaling.target_namespace
                )
                if deployment_info:
                    current_replicas = deployment_info['replicas']
            except Exception as e:
                logger.warning(f"Failed to get current replicas: {e}")

        # Execute workflow
        workflow_result = await services['workflow'].ainvoke(
            {'current_replicas': current_replicas},
            config_dict={}
        )

        # Update comprehensive metrics from workflow result
        if 'metrics' in services and workflow_result:
            await services['metrics'].update_decision_metrics(workflow_result)

            # Update component health
            await services['metrics'].update_component_status('workflow', workflow_result.get('success', False))
            await services['metrics'].update_component_status('prometheus', 'prometheus' in services)
            await services['metrics'].update_component_status('forecaster', 'forecaster' in services)
            await services['metrics'].update_component_status('dqn_agent', 'dqn_agent' in services)
            await services['metrics'].update_component_status('k8s_client', 'k8s_client' in services)

            # Record workflow duration
            if 'performance' in workflow_result:
                total_time_seconds = workflow_result['performance']['total_time_ms'] / 1000.0
                await services['metrics'].record_decision_duration(total_time_seconds)

        # Trigger model save if needed
        if ('dqn_agent' in services and
                hasattr(services['dqn_agent'], 'should_save_model') and
                services['dqn_agent'].should_save_model()):
            try:
                start_time = time.time()
                success = await services['dqn_agent'].save_model()
                save_duration = time.time() - start_time

                if 'metrics' in services:
                    await services['metrics'].record_model_save_duration(save_duration)

                if success:
                    logger.info(f"✅ Model saved successfully in {save_duration:.2f}s")
                else:
                    logger.warning("❌ Model save failed")
            except Exception as e:
                logger.error(f"Model save error: {e}")

        logger.info(f"✅ Scaling decision completed: {workflow_result.get('success', False)}")
        return workflow_result

    except Exception as e:
        logger.error(f"❌ Scaling decision failed: {e}")

        # Record error in metrics
        if 'metrics' in services:
            await services['metrics'].record_error('workflow', 'execution_error')
            await services['metrics'].update_component_status('workflow', False)

        return {
            'success': False,
            'error': str(e),
            'execution_id': 'failed',
            'final_decision': {
                'replicas': current_replicas if 'current_replicas' in locals() else 1,
                'action': 'keep_same',
                'confidence': 0.0,
                'reason': 'workflow_error'
            }
        }


async def continuous_training_loop():
    """Background training loop for DQN agent with async training."""
    logger.info("🔄 Starting continuous DQN async training loop...")

    while True:
        try:
            if 'dqn_agent' in services and len(services['dqn_agent'].replay_buffer) > 0:
                # Use async training method
                loss = await services['dqn_agent'].train_async()
                if loss is not None:
                    logger.info(
                        f"🧠 DQN async training completed: loss={loss:.6f}, steps={services['dqn_agent'].training_steps}")

                # Periodic model saving
                if services['dqn_agent'].should_save_model():
                    await services['dqn_agent'].save_model()
                    logger.info("💾 DQN model saved to MinIO")

            # Configurable stabilization period between training cycles
            await asyncio.sleep(config.scaling.stabilization_period_seconds)

        except Exception as e:
            logger.error(f"Training loop error: {e}")
            await asyncio.sleep(60)  # Back off longer on error


# Kopf event handlers
@kopf.timer('apps', 'v1', 'deployments', interval=config.scaling.stabilization_period_seconds)
async def periodic_scaling_decision(meta, **kwargs):
    """Periodic scaling decisions based on timer."""
    if (meta.get('namespace') != config.scaling.target_namespace or
            meta.get('name') != config.scaling.target_deployment):
        return

    logger.info(f"⏰ Periodic scaling decision triggered for {meta['name']}")

    try:
        result = await execute_scaling_decision()

        # Update periodic execution metrics
        if 'metrics' in services:
            await services['metrics'].update_component_status('periodic_timer', result.get('success', False))

            # Update system health based on periodic execution
            system_health = 1.0 if result.get('success', False) else 0.3
            services['metrics'].system_health_score.set(system_health)

        logger.info(f"✅ Periodic scaling decision completed: {result.get('success', False)}")

    except Exception as e:
        logger.error(f"❌ Periodic scaling decision failed: {e}")

        if 'metrics' in services:
            await services['metrics'].record_error('periodic_timer', 'execution_error')
            await services['metrics'].update_component_status('periodic_timer', False)


@kopf.on.field('apps', 'v1', 'deployments', field='status.replicas')
async def on_replica_change(old, new, meta, **kwargs):
    """React to replica count changes in target deployment."""
    if (meta.get('namespace') != config.scaling.target_namespace or
            meta.get('name') != config.scaling.target_deployment):
        return

    logger.info(f"📊 Replica change detected: {old} → {new} "
                f"(deployment: {meta['name']})")

    # Update metrics
    if 'metrics' in services:
        await services['metrics'].update_replica_metrics(old or 0, new or 0)

    # Trigger additional learning if significant change
    if old and abs((new or 0) - old) >= 2:
        logger.info("🎯 Significant scaling detected, triggering additional decision")
        try:
            await execute_scaling_decision()
        except Exception as e:
            logger.error(f"Additional scaling decision failed: {e}")


@kopf.on.cleanup()
async def cleanup_handler(**kwargs):
    """Cleanup handler for graceful shutdown."""
    logger.info("🧹 Starting operator cleanup...")

    try:
        # Cleanup DQN agent
        if 'dqn_agent' in services:
            await services['dqn_agent'].cleanup()

        # Cleanup other services
        if 'minio_client' in services:
            logger.info("💾 MinIO client cleanup completed")

        if 'metrics' in services:
            logger.info("📊 Metrics collector cleanup completed")

        logger.info("✅ Operator cleanup completed successfully")

    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")

    logger.info("👋 NimbusGuard Operator shutdown complete")


if __name__ == "__main__":
    # This allows running the operator directly for development
    kopf.run(
        [__file__],
        namespace=config.scaling.target_namespace,
        liveness=f"http://0.0.0.0:{config.server.health_port}/healthz"
    )
