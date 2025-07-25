import kopf
import logging
from prometheus_client import (
    start_http_server
)

# Set up logging
logging.basicConfig(
    level=getattr(logging, 'DEBUG'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@kopf.on.startup()
async def startup_handler(**kwargs):
    logger.info("Startup handler called")
    start_http_server(8080)
    logger.info("Metrics server started on port 8080")

@kopf.timer('apps', 'v1', 'deployments', interval='5')
def timer_handler(spec, status, meta, **kwargs):
  # Check if the deployment's name is 'consumer'
    if meta.get('name') == 'consumer':
        replicas = spec.get('replicas', 'N/A')
        
        # Initialize resource values to 'N/A'
        cpu_requests, mem_requests = 'N/A', 'N/A'
        cpu_limits, mem_limits = 'N/A', 'N/A'

        # Safely access nested resource values
        try:
            # Path: spec -> template -> spec -> containers[0] -> resources
            resources = spec['template']['spec']['containers'][0]['resources']
            cpu_requests = resources.get('requests', {}).get('cpu', 'N/A')
            mem_requests = resources.get('requests', {}).get('memory', 'N/A')
            cpu_limits = resources.get('limits', {}).get('cpu', 'N/A')
            mem_limits = resources.get('limits', {}).get('memory', 'N/A')
        except (KeyError, IndexError):
            # This will catch errors if 'template', 'spec', 'containers', etc., don't exist
            print("Could not find resource specifications for 'consumer'.")

        print(f"--- App: consumer ---")
        print(f"  Replicas: {replicas}")
        print(f"  Requests -> CPU: {cpu_requests}, Memory: {mem_requests}")
        print(f"  Limits   -> CPU: {cpu_limits}, Memory: {mem_limits}")
