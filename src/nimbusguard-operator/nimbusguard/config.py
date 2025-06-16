"""
Configuration and logging setup for NimbusGuard operator.
"""

import logging
import os
import warnings
from typing import Dict, Any

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*urllib3.*")

# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging():
    """Configure logging to reduce noise"""
    logging.getLogger('kubernetes').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

    logging.basicConfig(
        level=os.getenv('LOG_LEVEL', 'INFO').upper(),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# ============================================================================
# Configuration Constants
# ============================================================================

class Config:
    """Configuration constants for the operator"""
    
    # Environment variables
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    METRICS_PORT = int(os.getenv('METRICS_PORT', '8000'))
    
    # Operator settings
    EVALUATION_INTERVAL = 30  # seconds
    DEFAULT_MIN_REPLICAS = 1
    DEFAULT_MAX_REPLICAS = 10
    
    # Kubernetes settings
    CRD_GROUP = 'nimbusguard.io'
    CRD_VERSION = 'v1alpha1'
    CRD_PLURAL = 'nimbusguardhpa'
    
    # Prometheus settings
    PROMETHEUS_TIMEOUT = 10  # seconds
    DEFAULT_PROMETHEUS_URL = "http://prometheus:9090"
    
    # Health check settings
    LIVENESS_PORT = 8080

# ============================================================================
# Health Status (Global State)
# ============================================================================

health_status: Dict[str, bool] = {
    "prometheus": True,
    "kubernetes": True,
    "openai": bool(Config.OPENAI_API_KEY),
    "decision_engine": True
}

# ============================================================================
# Logger instance
# ============================================================================

setup_logging()
LOG = logging.getLogger(__name__)
