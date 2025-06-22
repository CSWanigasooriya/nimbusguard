#!/usr/bin/env python3
# ============================================================================
# NimbusGuard Operator - LangGraph Edition (DQN Primary with LLM Validation)
# ============================================================================

import asyncio
import logging
import os
import signal
import sys

import kopf

from config import health_status
from langgraph_handler import LangGraphOperatorHandler

# ============================================================================
# Configuration
# ============================================================================

LOG = logging.getLogger(__name__)

# Global handler instance
langgraph_handler = LangGraphOperatorHandler()

# ============================================================================
# Kopf Event Handlers (DQN Primary with LangGraph Validation)
# ============================================================================

@kopf.on.startup()
async def startup_handler(**kwargs):
    """Initialize the LangGraph-based operator with DQN decision engine."""
    LOG.info("🚀 Starting NimbusGuard Operator - LangGraph Edition (DQN Primary)")
    LOG.info("🧠 DQN Decision Engine: Primary decision maker")
    LOG.info("🤖 LLM Validation: Risk assessment and validation")
    
    try:
        await langgraph_handler.initialize()
        health_status["operator"] = True
        LOG.info("✅ LangGraph operator initialized successfully")
        
        # Log configuration
        kserve_endpoint = os.getenv('KSERVE_ENDPOINT')
        openai_key = os.getenv('OPENAI_API_KEY')
        
        LOG.info(f"🔗 KServe Endpoint: {'✅ Configured' if kserve_endpoint else '❌ Missing (REQUIRED)'}")
        LOG.info(f"🤖 OpenAI API: {'✅ Configured' if openai_key else '⚠️ Missing (LLM validation disabled)'}")
        
        if not kserve_endpoint:
            LOG.error("❌ KSERVE_ENDPOINT is required for DQN decision making")
            sys.exit(1)
            
    except Exception as e:
        LOG.critical(f"❌ Failed to initialize LangGraph operator: {e}", exc_info=True)
        health_status["operator"] = False
        sys.exit(1)


@kopf.on.cleanup()
async def cleanup_handler(**kwargs):
    """Cleanup resources on shutdown."""
    LOG.info("🛑 Shutting down NimbusGuard LangGraph Operator")
    health_status["operator"] = False


@kopf.on.create('nimbusguard.ai', 'v1', 'autoscalers')
@kopf.on.update('nimbusguard.ai', 'v1', 'autoscalers')
async def autoscaler_handler(body, spec, meta, namespace, **kwargs):
    """
    Handle AutoScaler resource events using LangGraph workflow with DQN decisions.
    This preserves the DQN as primary decision maker while adding LLM validation.
    """
    resource_name = meta.get('name', 'unknown')
    resource_uid = meta.get('uid', 'unknown')
    
    LOG.info(f"🎯 Processing AutoScaler '{resource_name}' - DQN Primary Decision")
    
    try:
        # Use LangGraph handler with DQN primary decision
        result = await langgraph_handler.evaluate_scaling_logic(body, namespace)
        
        # Log decision summary
        action = result.get("action", "none")
        current_replicas = result.get("current_replicas", 0)
        target_replicas = result.get("target_replicas", 0)
        reason = result.get("reason", "No reason provided")
        
        # Extract DQN and LangGraph info
        ml_decision = result.get("ml_decision", {})
        langgraph_status = result.get("langgraph_status", {})
        
        dqn_action = ml_decision.get("action_name", "NO_ACTION")
        dqn_confidence = ml_decision.get("confidence", 0.0)
        workflow_success = langgraph_status.get("workflow_completed", False)
        llm_validation = langgraph_status.get("llm_validation", False)
        
        if action != "none":
            LOG.info(f"🎯 Scaling Decision for '{resource_name}':")
            LOG.info(f"   📊 DQN Action: {dqn_action} (confidence: {dqn_confidence:.3f})")
            LOG.info(f"   🔄 Scaling: {current_replicas} → {target_replicas} replicas")
            LOG.info(f"   🤖 LLM Validation: {'✅ Applied' if llm_validation else '⚠️ Skipped'}")
            LOG.info(f"   💭 Reason: {reason}")
        else:
            LOG.info(f"🎯 No Action for '{resource_name}': {reason}")
            LOG.info(f"   📊 DQN Action: {dqn_action} (confidence: {dqn_confidence:.3f})")
            LOG.info(f"   🤖 LLM Validation: {'✅ Applied' if llm_validation else '⚠️ Skipped'}")
        
        # Return result for Kopf status update
        return {
            "autoscaler_status": "processed",
            "last_decision": {
                "action": action,
                "current_replicas": current_replicas,
                "target_replicas": target_replicas,
                "timestamp": result.get("langgraph_status", {}).get("workflow_duration", 0),
                "dqn_primary": True,
                "llm_validation": llm_validation
            },
            "ml_engine": {
                "dqn_action": dqn_action,
                "dqn_confidence": dqn_confidence,
                "prediction_source": ml_decision.get("prediction_source", "unknown")
            },
            "langgraph_workflow": {
                "completed": workflow_success,
                "reasoning_steps": langgraph_status.get("reasoning_steps", 0),
                "execution_success": langgraph_status.get("execution_success", False)
            }
        }
        
    except Exception as e:
        LOG.error(f"❌ AutoScaler processing failed for '{resource_name}': {e}", exc_info=True)
        return {
            "autoscaler_status": "error",
            "error": str(e),
            "last_decision": {
                "action": "none",
                "error": True,
                "timestamp": 0
            }
        }


@kopf.on.timer('nimbusguard.ai', 'v1', 'autoscalers', interval=30.0)
async def periodic_evaluation(body, spec, meta, namespace, **kwargs):
    """
    Periodic evaluation timer for continuous monitoring.
    This maintains the DQN-based decision making with LangGraph workflow enhancement.
    """
    resource_name = meta.get('name', 'unknown')
    
    # Check if periodic evaluation is enabled
    if not spec.get('enablePeriodicEvaluation', True):
        return
    
    LOG.debug(f"⏰ Periodic evaluation for '{resource_name}' - DQN Primary")
    
    try:
        result = await langgraph_handler.evaluate_scaling_logic(body, namespace)
        
        action = result.get("action", "none")
        if action != "none":
            LOG.info(f"⏰ Periodic scaling action for '{resource_name}': {action}")
            
        return {
            "periodic_evaluation": "completed",
            "action_taken": action,
            "dqn_primary": True
        }
        
    except Exception as e:
        LOG.error(f"❌ Periodic evaluation failed for '{resource_name}': {e}")
        return {
            "periodic_evaluation": "error",
            "error": str(e)
        }


# ============================================================================
# Health Check Endpoint
# ============================================================================

@kopf.on.probe(id='health')
def health_check(**kwargs):
    """Health check endpoint for Kubernetes probes."""
    if health_status.get("operator", False) and health_status.get("kubernetes", False):
        return {"status": "healthy", "dqn_primary": True, "langgraph_enabled": True}
    else:
        raise kopf.PermanentError("Operator not healthy")


@kopf.on.probe(id='metrics')
async def metrics_endpoint(**kwargs):
    """Metrics endpoint for monitoring."""
    try:
        metrics = await langgraph_handler.get_workflow_metrics()
        return metrics
    except Exception as e:
        LOG.error(f"Failed to get metrics: {e}")
        return {"error": str(e)}


# ============================================================================
# Signal Handlers
# ============================================================================

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    LOG.info(f"🛑 Received signal {signum}, shutting down...")
    health_status["operator"] = False
    sys.exit(0)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the LangGraph-based NimbusGuard operator."""
    
    # Set up logging
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    LOG.info("=" * 80)
    LOG.info("🚀 NimbusGuard Operator - LangGraph Edition")
    LOG.info("🧠 DQN Decision Engine: Primary AI decision maker")
    LOG.info("🤖 LLM Validation: Intelligent risk assessment and validation")
    LOG.info("🔄 LangGraph Workflow: Enhanced decision orchestration")
    LOG.info("=" * 80)
    
    # Validate environment
    required_env_vars = ['KSERVE_ENDPOINT']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        LOG.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        LOG.error("   KSERVE_ENDPOINT is required for DQN decision making")
        sys.exit(1)
    
    optional_env_vars = ['OPENAI_API_KEY', 'PROMETHEUS_URL']
    missing_optional = [var for var in optional_env_vars if not os.getenv(var)]
    
    if missing_optional:
        LOG.warning(f"⚠️ Missing optional environment variables: {', '.join(missing_optional)}")
        LOG.warning("   Some LangGraph features may be limited")
    
    try:
        # Run the Kopf operator
        kopf.run(
            clusterwide=True,
            liveness_endpoint='http://0.0.0.0:8080/health',
            ready_endpoint='http://0.0.0.0:8080/ready'
        )
    except KeyboardInterrupt:
        LOG.info("🛑 Operator stopped by user")
    except Exception as e:
        LOG.critical(f"❌ Operator failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 