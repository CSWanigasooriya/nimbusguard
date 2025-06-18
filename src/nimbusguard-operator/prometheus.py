# engine/prometheus.py
# ============================================================================
# Prometheus Client
# ============================================================================

import asyncio
import logging
from typing import Optional
import aiohttp

# Import shared health status from the config module
from config import health_status

LOG = logging.getLogger(__name__)


class PrometheusClient:
    """A simple async client for querying Prometheus."""

    def __init__(self, url: str):
        """
        Initializes the client.
        Args:
            url: The base URL of the Prometheus server.
        """
        self.url = url.rstrip('/')
        self.session_timeout = aiohttp.ClientTimeout(total=10)  # 10-second timeout for requests

    async def query(self, query: str) -> Optional[float]:
        """
        Executes a PromQL query and returns the first result's value.
        Handles errors gracefully and updates the global health status.
        """
        try:
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                async with session.get(
                        f"{self.url}/api/v1/query",
                        params={"query": query},
                        headers={'Accept': 'application/json'}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = data.get("data", {}).get("result", [])
                        if result:
                            # Extract value from a standard vector result
                            value_pair = result[0].get("value", [None, None])
                            value = float(value_pair[1]) if value_pair[1] is not None else None
                            health_status["prometheus"] = True
                            return value
                        LOG.warning(f"Prometheus query returned no result for: {query}")
                    else:
                        LOG.warning(f"Prometheus returned status {resp.status} for query: {query}")

        except asyncio.TimeoutError:
            LOG.error(f"Prometheus query timed out for: {query}")
            health_status["prometheus"] = False
        except aiohttp.ClientError as e:
            LOG.error(f"Prometheus client error: {e}")
            health_status["prometheus"] = False
        except Exception as e:
            LOG.error(f"An unexpected error occurred during Prometheus query: {e}")
            health_status["prometheus"] = False

        return None
