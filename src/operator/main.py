"""
NimbusGuard Kubernetes Operator
"""

import asyncio
import logging
import os
from pathlib import Path

import kopf
from kubernetes import client, config as k8s_config

from config.settings import load_config
from prometheus.client import PrometheusClient
from forecasting.predictor import ModelPredictor
from workflow.graph import create_scaling_workflow
from metrics.collector import metrics

# Global configuration
config = load_config()

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.logging.level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
prometheus_client = None
predictor = None
scaling_workflow = None


async def initialize_components():
    """Initialize all operator components."""
    global prometheus_client, predictor, scaling_workflow
    
    try:
        # Initialize MinIO storage
        logger.info("Initializing MinIO storage...")
        from storage.minio_client import minio_storage
        minio_initialized = await minio_storage.initialize()
        if minio_initialized:
            logger.info("✅ MinIO storage ready for model persistence")
        else:
            logger.warning("⚠️ MinIO storage initialization failed - models will only be saved locally")
        
        # Initialize Prometheus client
        logger.info("Initializing Prometheus client...")
        prometheus_client = PrometheusClient(
            prometheus_url=config.prometheus.url,
            timeout=config.prometheus.timeout
        )
        
        # Initialize model predictor with config
        logger.info("Initializing dual-model predictor...")
        predictor = ModelPredictor(models_path="/tmp", config=config)
        await predictor.load_models()
        
        # Initialize scaling workflow
        logger.info("Initializing scaling workflow...")
        scaling_workflow = create_scaling_workflow(config, prometheus_client, predictor)
        
        logger.info("All components initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise





@kopf.on.startup()
async def startup_handler(**kwargs):
    """Handler called when the operator starts."""
    logger.info("NimbusGuard Operator starting up...")
    
    # Load Kubernetes configuration
    try:
        k8s_config.load_incluster_config()
        logger.info("Loaded in-cluster Kubernetes configuration")
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()
        logger.info("Loaded local Kubernetes configuration")
    
    # Initialize all components
    await initialize_components()
    
    # Start metrics server
    try:
        metrics.start_metrics_server(port=config.server.port)
        logger.info(f"Metrics server started on port {config.server.port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")
        # Don't fail startup if metrics server fails
    
    logger.info("NimbusGuard Operator startup complete!")


@kopf.timer('apps', 'v1', 'deployments', interval=config.scaling.stabilization_period_seconds)
async def monitor_deployments(spec, status, meta, **kwargs):
    """
    Timer handler for monitoring deployments and making scaling decisions.
    Triggered every stabilization_period_seconds for all deployments.
    """
    deployment_name = meta.get('name')
    namespace = meta.get('namespace')
    
    # Only process the target deployment
    if deployment_name != config.scaling.target_deployment:
        return
    
    if namespace != config.scaling.target_namespace:
        return
    
    # Ensure all components are initialized
    if scaling_workflow is None:
        logger.warning("Scaling workflow not initialized, skipping")
        return
    
    logger.info(f"Monitoring deployment: {deployment_name} in namespace: {namespace}")
    
    try:
        # Get current replica count
        current_replicas = status.get('replicas', 0)
        ready_replicas = status.get('readyReplicas', 0)
        
        logger.info(f"Current replicas: {current_replicas}, Ready replicas: {ready_replicas}")
        
        # Execute the scaling workflow
        workflow_result = await scaling_workflow.execute_scaling_workflow(
            deployment_name=deployment_name,
            namespace=namespace,
            current_replicas=current_replicas,
            ready_replicas=ready_replicas
        )
        
        # Log workflow summary
        if workflow_result.error_occurred:
            logger.error(f"Workflow failed for {deployment_name}: {workflow_result.error_message}")
        else:
            logger.info(f"Workflow completed for {deployment_name} - Action: {workflow_result.recommended_action}, "
                       f"Executed: {workflow_result.scaling_executed}, Reward: {workflow_result.reward_value}")
        
    except Exception as e:
        logger.error(f"Error monitoring deployment {deployment_name}: {e}")


@kopf.on.cleanup()
async def cleanup_handler(**kwargs):
    """Handler called when the operator is shutting down."""
    logger.info("NimbusGuard Operator shutting down...")


if __name__ == "__main__":
    logger.info("Starting NimbusGuard Operator...")
    kopf.run() 