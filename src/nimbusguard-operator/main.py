# engine/main.py
# ============================================================================
# Kopf Operator Entrypoint and Event Handlers
# ============================================================================

import logging
import signal
import sys
from datetime import datetime

import kopf

# Import components from our structured package
from config import setup_logging, health_status
from handler import OperatorHandler
from rules import _dqn_agent

# --- Initial Setup ---
setup_logging()
LOG = logging.getLogger(__name__)
operator_handler = OperatorHandler()


# ============================================================================
# Kopf Framework Handlers
# ============================================================================

def signal_handler(signum, frame):
    """Handle graceful shutdown signals to save models."""
    LOG.info(f"Received signal {signum}, saving models before shutdown...")
    try:
        global _dqn_agent
        if _dqn_agent and _dqn_agent.model_path:
            _dqn_agent.save_model(_dqn_agent.model_path)
            LOG.info("Successfully saved DQN model before shutdown")
    except Exception as e:
        LOG.error(f"Failed to save model during shutdown: {e}")
    sys.exit(0)


@kopf.on.startup()
async def startup(logger, **kwargs):
    """Runs once when the operator starts."""
    try:
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        await operator_handler.initialize()
        LOG.info("NimbusGuard Operator started successfully.")
    except Exception as e:
        LOG.critical(f"Operator startup failed: {e}", exc_info=True)
        raise  # Stop the operator if startup fails


@kopf.on.probe()
def probe(**kwargs):
    """Kopf liveness probe to check operator health."""
    return {"status": "healthy", "components": health_status}


@kopf.timer('nimbusguard.io', 'v1alpha1', 'intelligentscaling', interval=30, sharp=True)
async def evaluate_scaling_on_timer(body, name, namespace, **kwargs):
    """Periodically evaluates the scaling requirements for each resource."""
    LOG.info(f"Timer evaluating '{name}' in namespace '{namespace}'.")
    try:
        result = await operator_handler.evaluate_scaling_logic(body, namespace)

        LOG.info(
            f"[{name}] Decision: {result['reason']} | Action: {result['action']} -> {result['target_replicas']} replicas")
        health_status["decision_engine"] = True

        # Patch the resource's status with the latest evaluation results
        return {
            "last_evaluation": datetime.now().isoformat(),
            "decision_reason": result["reason"],
            "current_replicas": result["current_replicas"],
            "target_replicas": result["target_replicas"],
            "action": result["action"],
        }
    except Exception as e:
        LOG.error(f"Evaluation failed for '{name}': {e}", exc_info=True)
        health_status["decision_engine"] = False
        # Update status to reflect the error
        return {"last_evaluation_error": str(e)}


@kopf.on.delete('nimbusguard.io', 'v1alpha1', 'intelligentscaling', optional=True)
async def on_delete(body, name, namespace, **kwargs):
    """Handles cleanup when a resource is deleted."""
    # 'namespace' is now available to be used directly.
    LOG.info(f"IntelligentScaling resource '{name}' in namespace '{namespace}' has been deleted.")
