"""
Prometheus client for metrics collection.
"""

import asyncio
import aiohttp
import logging
from typing import Optional
from ..health import set_component_health

LOG = logging.getLogger(__name__)

# ============================================================================
# Prometheus Client
# ============================================================================

class PrometheusClient:
    """Simple Prometheus client with improved error handling"""
    
    def __init__(self, url: str):
        self.url = url.rstrip('/')
        self.session_timeout = aiohttp.ClientTimeout(total=10)
    
    async def query(self, query: str) -> Optional[float]:
        """Execute PromQL query with proper error handling"""
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
                            value = result[0].get("value", [None, None])[1]
                            set_component_health("prometheus", True)
                            return float(value) if value is not None else None
                    else:
                        LOG.warning(f"Prometheus returned status {resp.status} for query: {query}")
                        
        except asyncio.TimeoutError:
            LOG.error(f"Prometheus query timeout for: {query}")
            set_component_health("prometheus", False)
        except aiohttp.ClientError as e:
            LOG.error(f"Prometheus client error: {e}")
            set_component_health("prometheus", False)
        except Exception as e:
            LOG.error(f"Prometheus query failed: {e}")
            set_component_health("prometheus", False)
        return None
    
    async def query_range(self, query: str, start: str, end: str, step: str) -> Optional[dict]:
        """Execute PromQL range query"""
        try:
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                async with session.get(
                    f"{self.url}/api/v1/query_range",
                    params={
                        "query": query,
                        "start": start,
                        "end": end,
                        "step": step
                    },
                    headers={'Accept': 'application/json'}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        set_component_health("prometheus", True)
                        return data.get("data", {})
                    else:
                        LOG.warning(f"Prometheus range query returned status {resp.status}")
                        
        except Exception as e:
            LOG.error(f"Prometheus range query failed: {e}")
            set_component_health("prometheus", False)
        return None
    
    async def health_check(self) -> bool:
        """Check if Prometheus is healthy"""
        try:
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                async with session.get(f"{self.url}/-/healthy") as resp:
                    healthy = resp.status == 200
                    set_component_health("prometheus", healthy)
                    return healthy
        except Exception as e:
            LOG.error(f"Prometheus health check failed: {e}")
            set_component_health("prometheus", False)
            return False
