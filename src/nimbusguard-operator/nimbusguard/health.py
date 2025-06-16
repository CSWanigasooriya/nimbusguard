"""
Health check functions for NimbusGuard operator.
"""

import kopf
from .config import health_status
from .metrics import update_health_metrics

# ============================================================================
# Health Check Functions
# ============================================================================

@kopf.on.probe(id='health')
def health_check(**kwargs):
    """Health check for kopf liveness probe"""
    # Update health metrics
    update_health_metrics()
    
    return {
        "status": "healthy" if all(health_status.values()) else "degraded",
        "components": health_status
    }

@kopf.on.probe(id='ready')  
def readiness_check(**kwargs):
    """Readiness check for kopf"""
    ready = health_status["kubernetes"] and health_status["decision_engine"]
    return {
        "status": "ready" if ready else "not_ready",
        "kubernetes": health_status["kubernetes"],
        "decision_engine": health_status["decision_engine"]
    }

# ============================================================================
# Health Status Management
# ============================================================================

def set_component_health(component: str, status: bool):
    """Set health status for a specific component"""
    health_status[component] = status

def get_overall_health() -> bool:
    """Get overall health status"""
    return all(health_status.values())

def get_component_health(component: str) -> bool:
    """Get health status for a specific component"""
    return health_status.get(component, False)
