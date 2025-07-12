import logging
from datetime import datetime
from aiohttp import web
from prometheus_client import generate_latest
import kopf

# Import required metrics
from monitoring.metrics import *
from .services import ServiceContainer

# Function reference and services to be set by main.py
_run_intelligent_scaling_decision = None
_services: ServiceContainer = None

def set_decision_function(func):
    """Set the run_intelligent_scaling_decision function reference"""
    global _run_intelligent_scaling_decision
    _run_intelligent_scaling_decision = func

def set_services(services: ServiceContainer):
    """Set the services container reference"""
    global _services
    _services = services


class HTTP2ErrorFilter(logging.Filter):
    """Filter to suppress HTTP/2 protocol error logs"""
    def filter(self, record):
        if record.name == "aiohttp.server":
            message = record.getMessage()
            if "PRI/Upgrade" in message or "BadHttpMessage" in message:
                return False  # Suppress this log
        return True  # Allow other logs


def setup_probes():
    """Setup all Kopf health check probes"""
    
    @kopf.on.probe(id='status')
    def health_status(**kwargs):
        """Health status probe for Kopf liveness check"""
        return {"status": "healthy", "service": "dqn-adapter"}

    @kopf.on.probe(id='redis_connection')
    def redis_health(**kwargs):
        """Check Redis connection health"""
        try:
            if _services and _services.redis_client and _services.redis_client.ping():
                return {"redis": "connected"}
            else:
                return {"redis": "disconnected"}
        except Exception as e:
            return {"redis": f"error: {str(e)}"}

    @kopf.on.probe(id='scaler_loaded')
    def scaler_health(**kwargs):
        """Check if the feature scaler is loaded"""
        try:
            return {"scaler": "loaded" if _services and _services.scaler else "not_loaded"}
        except Exception as e:
            return {"scaler": f"error: {str(e)}"}


async def metrics_handler(request):
    """Provide Prometheus metrics for KEDA"""
    metrics_data = generate_latest()
    return web.Response(text=metrics_data.decode('utf-8'), content_type='text/plain; version=0.0.4')


async def root_handler(request):
    """Root endpoint with service information"""
    return web.json_response({
        "message": "NimbusGuard DQN Adapter", 
        "status": "running",
        "architecture": "direct_scaling",
        "services": {
            "http": "health+metrics+decision_triggers"
        },
        "endpoints": {
            "GET /": "Service info",
            "GET /healthz": "Health check",
            "GET /metrics": "Prometheus metrics",
            "POST /evaluate": "Trigger evaluation",
            "POST /decide": "Trigger DQN decision",
            
        }
    })


async def health_handler(request):
    """Health check endpoint with component status"""
    try:
        health_status = {
            "status": "healthy",
            "service": "dqn-adapter",
            "components": {
                "dqn_trainer": _services and _services.dqn_trainer is not None,
                "evaluator": _services and _services.evaluator is not None,
                "redis": _services and _services.redis_client is not None,
                "prometheus": _services and _services.prometheus_client is not None,
        
                "scaler": _services and _services.scaler is not None,
                "dqn_model": _services and _services.dqn_model is not None
            },
            "metrics": {
                "decisions_total": int(DQN_DECISIONS_COUNTER._value._value),
                "training_steps": int(DQN_TRAINING_STEPS_GAUGE._value._value),
                "buffer_size": int(DQN_BUFFER_SIZE_GAUGE._value._value),
                "current_epsilon": float(DQN_EPSILON_GAUGE._value._value)
            }
        }
        return web.json_response(health_status)
    except Exception as e:
        return web.json_response({
            "status": "error",
            "error": str(e)
        }, status=500)


async def evaluation_trigger_handler(request):
    """Manual trigger for evaluation"""
    try:
        if not _services or not _services.evaluator:
            return web.json_response({"error": "Evaluation not enabled or services not initialized"}, status=400)
        
        if not _services.dqn_trainer:
            return web.json_response({"error": "DQN trainer not initialized"}, status=400)
        
        # Trigger evaluation
        await _services.dqn_trainer._generate_evaluation_outputs()
        
        return web.json_response({
            "message": "Evaluation triggered successfully",
            "timestamp": datetime.now().isoformat(),
            "experiences": len(_services.evaluator.experiences),
            "training_metrics": len(_services.evaluator.training_metrics)
        })
        
    except Exception as e:
        logger = logging.getLogger("Controller")
        logger.error(f"Evaluation trigger failed: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def decision_trigger_handler(request):
    """Manual trigger for DQN scaling decision"""
    try:
        logger = logging.getLogger("Controller")
        logger.info("HTTP_TRIGGER: manual_decision_requested")
        
        if not _run_intelligent_scaling_decision:
            return web.json_response({"error": "Decision function not initialized"}, status=500)
        
        # Run intelligent scaling decision
        await _run_intelligent_scaling_decision()
        
        # Get current metrics for response
        current_replicas = int(CURRENT_REPLICAS_GAUGE._value._value)
        desired_replicas = int(DESIRED_REPLICAS_GAUGE._value._value)
        decisions_total = int(DQN_DECISIONS_COUNTER._value._value)
        
        return web.json_response({
            "message": "DQN decision triggered successfully",
            "timestamp": datetime.now().isoformat(),
            "current_replicas": current_replicas,
            "desired_replicas": desired_replicas,
            "decisions_total": decisions_total,
            "epsilon": float(DQN_EPSILON_GAUGE._value._value)
        })
        
    except Exception as e:
        logger = logging.getLogger("Controller")
        logger.error(f"Decision trigger failed: {e}")
        return web.json_response({"error": str(e)}, status=500)





def setup_http_server():
    """Setup and configure the HTTP server with all routes"""
    app = web.Application()
    
    # Add all routes
    app.router.add_get('/metrics', metrics_handler)
    app.router.add_get('/', root_handler)
    app.router.add_get('/healthz', health_handler)
    app.router.add_post('/evaluate', evaluation_trigger_handler)
    app.router.add_post('/decide', decision_trigger_handler)

    
    return app


def setup_logging_filters():
    """Setup logging filters to suppress HTTP/2 errors"""
    aiohttp_server_logger = logging.getLogger("aiohttp.server")
    aiohttp_server_logger.addFilter(HTTP2ErrorFilter())



