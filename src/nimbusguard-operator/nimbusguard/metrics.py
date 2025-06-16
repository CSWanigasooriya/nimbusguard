"""
Prometheus metrics definitions for NimbusGuard operator.
"""

from prometheus_client import Counter, Gauge, start_http_server
from .config import health_status

# ============================================================================
# Prometheus Metrics
# ============================================================================

# Scaling operation metrics
SCALING_OPERATIONS = Counter(
    'nimbusguard_scaling_operations_total', 
    'Total scaling operations', 
    ['operation_type', 'namespace']
)

# Current replica metrics
CURRENT_REPLICAS = Gauge(
    'nimbusguard_current_replicas', 
    'Current replicas', 
    ['name', 'namespace']
)

# Decision metrics
DECISIONS_MADE = Counter(
    'nimbusguard_decisions_total', 
    'Total decisions made', 
    ['action']
)

# Operator health metrics
OPERATOR_HEALTH = Gauge(
    'nimbusguard_operator_health', 
    'Operator health', 
    ['component']
)

# ============================================================================
# Metrics Server Management
# ============================================================================

def start_metrics_server(port: int = 8000):
    """Start the Prometheus metrics server"""
    start_http_server(port)
    
    # Initialize some default metrics
    OPERATOR_HEALTH.labels(component='startup').set(1)
    SCALING_OPERATIONS.labels(operation_type='startup', namespace='system').inc(0)

def update_health_metrics():
    """Update health metrics based on current health status"""
    for component, status in health_status.items():
        OPERATOR_HEALTH.labels(component=component).set(1 if status else 0)
