"""
Kopf event handlers for NimbusGuard operator.
"""

import kopf
import logging
from datetime import datetime

from .config import Config
from .operator import NimbusGuardOperator
from .health import set_component_health
from .utils import parse_intelligent_scaling, create_status_patch

LOG = logging.getLogger(__name__)

# ============================================================================
# Global operator instance
# ============================================================================

operator = NimbusGuardOperator()

# ============================================================================
# Kopf Event Handlers
# ============================================================================

@kopf.on.startup()
async def startup(settings: kopf.OperatorSettings, **_):
    """Operator startup configuration"""
    try:
        # Initialize operator
        await operator.initialize()
        
        LOG.info("NimbusGuard operator started successfully")
        LOG.info("CRDs installed manually for deployment ordering, kubecrd provides validation")
        LOG.info("NimbusGuardHPA resources can now be processed")
        
        # Log environment info
        LOG.info(f"Log level: {Config.LOG_LEVEL}")
        LOG.info(f"OpenAI available: {bool(Config.OPENAI_API_KEY)}")
        LOG.info(f"Metrics port: {Config.METRICS_PORT}")
        
    except Exception as e:
        LOG.error(f"Startup failed: {e}")
        set_component_health("startup", False)
        raise

@kopf.on.create(Config.CRD_GROUP, Config.CRD_VERSION, Config.CRD_PLURAL)
async def on_create(body, name, namespace, **kwargs):
    """Handle NimbusGuardHPA resource creation"""
    LOG.info(f"Created NimbusGuardHPA '{name}' in '{namespace}'")
    return await evaluate_scaling(body, name, namespace, **kwargs)

@kopf.timer(Config.CRD_GROUP, Config.CRD_VERSION, Config.CRD_PLURAL, interval=Config.EVALUATION_INTERVAL)
async def evaluate_scaling(body, name, namespace, **kwargs):
    """Periodic scaling evaluation"""
    LOG.debug(f"Evaluating scaling for '{name}' in namespace '{namespace}'")
    
    try:
        # Ensure we have valid body structure
        if not body or 'spec' not in body:
            LOG.warning(f"Invalid resource body for '{name}' - missing spec")
            return
            
        # Evaluate scaling
        result = await operator.evaluate_scaling(body, name, namespace)
        
        # Create status patch using utility
        status_patch = create_status_patch(
            last_evaluation=result["last_evaluation"],
            current_replicas=result["current_replicas"],
            target_replicas=result["target_replicas"],
            decision_reason=result["decision_reason"],
            action=result["action"]
        )
        
        # Log decision with more details
        LOG.info(f"[{name}] Current: {result['current_replicas']} replicas")
        LOG.info(f"[{name}] Decision: {result['decision_reason']}")
        if result["action"] != "none":
            LOG.info(f"[{name}] Action: {result['action']} -> {result['target_replicas']} replicas")
        
        # Update health status on successful evaluation
        set_component_health("decision_engine", True)
        
        return status_patch
        
    except Exception as e:
        LOG.error(f"Evaluation failed for '{name}': {e}")
        set_component_health("kubernetes", False)
        set_component_health("decision_engine", False)
        # Return empty status to avoid kopf errors
        return {}

@kopf.on.delete(Config.CRD_GROUP, Config.CRD_VERSION, Config.CRD_PLURAL)
async def on_delete(body, name, namespace, **kwargs):
    """Handle resource deletion"""
    LOG.info(f"Deleted NimbusGuardHPA '{name}' from '{namespace}'")

@kopf.on.update(Config.CRD_GROUP, Config.CRD_VERSION, Config.CRD_PLURAL)
async def on_update(old, new, diff, name, namespace, **kwargs):
    """Handle resource updates"""
    LOG.info(f"Updated NimbusGuardHPA '{name}' in '{namespace}'")
    # Trigger immediate evaluation on update
    return await evaluate_scaling(new, name, namespace, **kwargs)

@kopf.on.field(Config.CRD_GROUP, Config.CRD_VERSION, Config.CRD_PLURAL, field='spec.metrics_config')
async def on_metrics_config_change(old, new, name, namespace, **kwargs):
    """Handle metrics configuration changes"""
    LOG.info(f"Metrics config changed for '{name}' in '{namespace}'")
    # Could implement specific logic for metrics config changes here

# ============================================================================
# Error Handlers
# ============================================================================

@kopf.on.event(Config.CRD_GROUP, Config.CRD_VERSION, Config.CRD_PLURAL)
async def on_event(event, **kwargs):
    """Handle all events for debugging"""
    if LOG.level <= logging.DEBUG:
        LOG.debug(f"Event: {event}")

# ============================================================================
# Cleanup
# ============================================================================

@kopf.on.cleanup()
async def cleanup(**kwargs):
    """Cleanup on operator shutdown"""
    LOG.info("Cleaning up operator resources...")
    operator.cleanup()  # No await since it's not async anymore
