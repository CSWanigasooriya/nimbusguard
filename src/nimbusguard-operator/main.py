# engine/main.py - DQN Implementation
# ============================================================================
# Kopf Operator Entrypoint and Event Handlers for DQN Mode
# ============================================================================

import logging
import signal
import sys
from datetime import datetime

import kopf

# Import components from our structured package
from config import setup_logging, health_status
from handler import OperatorHandler

# --- Initial Setup ---
setup_logging()
LOG = logging.getLogger(__name__)
operator_handler = OperatorHandler()


# ============================================================================
# Kopf Framework Handlers
# ============================================================================

def signal_handler(signum, frame):
    """Handle graceful shutdown signals for DQN mode."""
    LOG.info(f"Received signal {signum}, performing graceful shutdown...")
    try:
        # Log shutdown metrics if needed
        if operator_handler.dqn_agent:
            metrics = operator_handler.dqn_agent.get_performance_metrics()
            LOG.info(f"Shutdown metrics: {metrics}")
        LOG.info("DQN operator shutdown completed")
    except Exception as e:
        LOG.error(f"Error during graceful shutdown: {e}")
    sys.exit(0)


@kopf.on.startup()
async def startup(logger, **kwargs):
    """Runs once when the DQN operator starts."""
    try:
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        await operator_handler.initialize()
        LOG.info("NimbusGuard DQN Operator started successfully.")
        
        # Log DQN configuration
        if operator_handler.kserve_enabled:
            LOG.info(f"KServe endpoint: {operator_handler.kserve_endpoint}")
        else:
            LOG.error("KServe not enabled - operator will not function properly")
            
    except Exception as e:
        LOG.critical(f"DQN operator startup failed: {e}", exc_info=True)
        raise  # Stop the operator if startup fails


@kopf.on.probe()
def probe(**kwargs):
    """Kopf liveness probe to check DQN operator health."""
    return {
        "status": "healthy", 
        "components": health_status,
        "mode": "dqn",
        "kserve_enabled": operator_handler.kserve_enabled if operator_handler else False
    }


@kopf.timer('nimbusguard.io', 'v1alpha1', 'intelligentscaling', interval=30, sharp=True)
async def evaluate_scaling_on_timer(body, name, namespace, **kwargs):
    """Periodically evaluates the scaling requirements using DQN."""
    LOG.info(f"DQN timer evaluating '{name}' in namespace '{namespace}'.")
    try:
        result = await operator_handler.evaluate_scaling_logic(body, namespace)

        # Log DQN-specific decision info
        ml_decision = result.get('ml_decision', {})
        confidence = ml_decision.get('confidence', 0.0)
        prediction_source = ml_decision.get('prediction_source', 'unknown')
        
        LOG.info(
            f"[{name}] DQN Decision: {result['reason']} | "
            f"Action: {result['action']} -> {result['target_replicas']} replicas | "
            f"Confidence: {confidence:.3f} | Source: {prediction_source}")
        
        health_status["decision_engine"] = True
        health_status["kserve"] = True

        # Patch the resource's status with DQN evaluation results
        return {
            "last_evaluation": datetime.now().isoformat(),
            "decision_reason": result["reason"],
            "current_replicas": result["current_replicas"],
            "target_replicas": result["target_replicas"],
            "action": result["action"],
            "ml_status": {
                "model_type": "DQN",
                "confidence_score": confidence,
                "prediction_source": prediction_source,
                "decision_type": ml_decision.get('action_name', 'unknown')
            },
            "kserve_status": result.get("kserve_status", {})
        }
    except Exception as e:
        LOG.error(f"DQN evaluation failed for '{name}': {e}", exc_info=True)
        health_status["decision_engine"] = False
        health_status["kserve"] = False
        # Update status to reflect the error
        return {
            "last_evaluation_error": str(e),
            "ml_status": {
                "model_type": "DQN",
                "decision_type": "error",
                "confidence_score": 0.0
            }
        }


@kopf.on.delete('nimbusguard.io', 'v1alpha1', 'intelligentscaling', optional=True)
async def on_delete(body, name, namespace, **kwargs):
    """Handles cleanup when a DQN resource is deleted."""
    LOG.info(f"IntelligentScaling resource '{name}' in namespace '{namespace}' has been deleted.")
    # No local cleanup needed - just log the deletion
