# engine/config.py
# ============================================================================
# Configuration and Shared State
# ============================================================================

import logging
import os
import warnings

# --- Suppress noisy warnings ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*urllib3.*")

# --- Logging Configuration ---
def setup_logging():
    """Configures the application's logging format and level."""
    # Reduce noise from underlying libraries
    logging.getLogger('kubernetes').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

    logging.basicConfig(
        level=os.getenv('LOG_LEVEL', 'INFO').upper(),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# --- Global Health Status ---
# This dictionary tracks the real-time health of various operator components.
health_status = {
    "kubernetes": True,
    "prometheus": False,  # Updated by PrometheusClient
    "tempo": False,       # Updated by TempoClient
    "loki": False,        # Updated by LokiClient
    "openai": bool(os.getenv("OPENAI_API_KEY")), # Maintained from original logic
    "ml_decision_engine": False,  # Updated by DQN Agent
    "decision_engine": True,
    "observability_collector": True
}
