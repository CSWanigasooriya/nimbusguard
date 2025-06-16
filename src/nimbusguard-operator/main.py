"""
Main entry point for NimbusGuard Intelligent Scaling Operator.
"""

import kopf
import os
import sys
import urllib3
import logging

from nimbusguard.config import Config, LOG
from nimbusguard.metrics import start_metrics_server
from nimbusguard import health, controller  # Import to register controller

# Suppress urllib3 warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def main():
    """Main entry point"""
    LOG.info("Starting NimbusGuard Intelligent Scaling Operator...")
    LOG.info(f"Python version: {sys.version}")
    LOG.info(f"Operator mode: {'Development' if Config.LOG_LEVEL == 'DEBUG' else 'Production'}")
    
    try:
        # Start metrics server
        start_metrics_server(Config.METRICS_PORT)
        LOG.info(f"Metrics server started on :{Config.METRICS_PORT}")
        LOG.info(f"Access metrics at: http://localhost:{Config.METRICS_PORT}/metrics")
        LOG.info(f"Health checks available via kopf probes")
        
        # Run operator
        kopf.run(
            clusterwide=True
        )
        
    except KeyboardInterrupt:
        LOG.info("Operator shutdown requested")
    except Exception as e:
        LOG.error(f"Operator failed: {e}")
        raise

if __name__ == "__main__":
    main()
