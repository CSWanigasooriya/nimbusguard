"""
Refactored Prometheus client with proper type hints, error handling, and async patterns.
"""

import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import asynccontextmanager

from config import PrometheusConfig, get_config
from utils.exceptions import PrometheusError, ErrorContext
from utils.logging_config import get_logger


class PrometheusClient:
    """
    Async Prometheus client with proper error handling and type safety.
    """
    
    def __init__(self, config: Optional[PrometheusConfig] = None):
        self.config = config or get_config().prometheus
        self.logger = get_logger("PrometheusClient")
        self._session: Optional[aiohttp.ClientSession] = None
        self._base_url = f"{self.config.url}/api/v1"
    
    async def __aenter__(self) -> 'PrometheusClient':
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self) -> None:
        """Ensure HTTP session is created."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"User-Agent": "DQN-Adapter/1.0"}
            )
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def query(self, promql: str, time: Optional[datetime] = None) -> float:
        """
        Execute a Prometheus query and return a single float value.
        
        Args:
            promql: The PromQL query string
            time: Optional timestamp for the query
            
        Returns:
            float: The query result value
            
        Raises:
            PrometheusError: If the query fails or returns invalid data
        """
        with ErrorContext("prometheus_query", "PrometheusClient").add_context(query=promql):
            try:
                await self._ensure_session()
                
                params = {"query": promql}
                if time:
                    params["time"] = time.isoformat()
                
                self.logger.debug("Executing Prometheus query", query=promql, params=params)
                
                async with self._session.get(f"{self._base_url}/query", params=params) as response:
                    if response.status != 200:
                        raise PrometheusError(
                            f"Prometheus query failed with status {response.status}",
                            context={"query": promql, "status": response.status}
                        )
                    
                    data = await response.json()
                    
                    if data.get("status") != "success":
                        raise PrometheusError(
                            f"Prometheus query returned error: {data.get('error', 'Unknown error')}",
                            context={"query": promql, "response": data}
                        )
                    
                    result = data.get("data", {}).get("result", [])
                    
                    if not result:
                        self.logger.warning("Prometheus query returned no data", query=promql)
                        return 0.0
                    
                    # Get the first result's value
                    value = result[0].get("value", [None, "0"])[1]
                    
                    try:
                        return float(value)
                    except (ValueError, TypeError) as e:
                        raise PrometheusError(
                            f"Invalid value from Prometheus query: {value}",
                            context={"query": promql, "value": value}
                        ) from e
                        
            except aiohttp.ClientError as e:
                raise PrometheusError(
                    f"HTTP client error during Prometheus query: {e}",
                    context={"query": promql}
                ) from e
            except asyncio.TimeoutError as e:
                raise PrometheusError(
                    f"Timeout during Prometheus query: {e}",
                    context={"query": promql, "timeout": self.config.timeout}
                ) from e
    
    async def query_range(
        self, 
        promql: str, 
        start: datetime, 
        end: datetime, 
        step: str = "1m"
    ) -> List[Dict[str, Any]]:
        """
        Execute a Prometheus range query.
        
        Args:
            promql: The PromQL query string
            start: Start time for the range
            end: End time for the range
            step: Step size for the range query
            
        Returns:
            List of query results
            
        Raises:
            PrometheusError: If the query fails
        """
        with ErrorContext("prometheus_range_query", "PrometheusClient").add_context(
            query=promql, start=start, end=end, step=step
        ):
            try:
                await self._ensure_session()
                
                params = {
                    "query": promql,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "step": step
                }
                
                self.logger.debug("Executing Prometheus range query", **params)
                
                async with self._session.get(f"{self._base_url}/query_range", params=params) as response:
                    if response.status != 200:
                        raise PrometheusError(
                            f"Prometheus range query failed with status {response.status}",
                            context={"query": promql, "status": response.status}
                        )
                    
                    data = await response.json()
                    
                    if data.get("status") != "success":
                        raise PrometheusError(
                            f"Prometheus range query returned error: {data.get('error', 'Unknown error')}",
                            context={"query": promql, "response": data}
                        )
                    
                    return data.get("data", {}).get("result", [])
                    
            except aiohttp.ClientError as e:
                raise PrometheusError(
                    f"HTTP client error during Prometheus range query: {e}",
                    context={"query": promql}
                ) from e
            except asyncio.TimeoutError as e:
                raise PrometheusError(
                    f"Timeout during Prometheus range query: {e}",
                    context={"query": promql, "timeout": self.config.timeout}
                ) from e
    
    async def query_raw(self, promql: str, time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Execute a Prometheus query and return the raw result structure.
        
        Args:
            promql: The PromQL query string
            time: Optional timestamp for the query
            
        Returns:
            List[Dict[str, Any]]: The raw query results with metric labels
            
        Raises:
            PrometheusError: If the query fails
        """
        with ErrorContext("prometheus_raw_query", "PrometheusClient").add_context(query=promql):
            try:
                await self._ensure_session()
                
                params = {"query": promql}
                if time:
                    params["time"] = time.isoformat()
                
                self.logger.debug("Executing Prometheus raw query", query=promql, params=params)
                
                async with self._session.get(f"{self._base_url}/query", params=params) as response:
                    if response.status != 200:
                        raise PrometheusError(
                            f"Prometheus query failed with status {response.status}",
                            context={"query": promql, "status": response.status}
                        )
                    
                    data = await response.json()
                    
                    if data.get("status") != "success":
                        raise PrometheusError(
                            f"Prometheus query returned error: {data.get('error', 'Unknown error')}",
                            context={"query": promql, "response": data}
                        )
                    
                    return data.get("data", {}).get("result", [])
                        
            except aiohttp.ClientError as e:
                raise PrometheusError(
                    f"HTTP client error during Prometheus query: {e}",
                    context={"query": promql}
                ) from e
            except asyncio.TimeoutError as e:
                raise PrometheusError(
                    f"Timeout during Prometheus query: {e}",
                    context={"query": promql, "timeout": self.config.timeout}
                ) from e

    async def query_multiple(self, queries: Dict[str, str]) -> Dict[str, float]:
        """
        Execute multiple Prometheus queries concurrently.
        
        Args:
            queries: Dictionary mapping query names to PromQL strings
            
        Returns:
            Dictionary mapping query names to results
            
        Raises:
            PrometheusError: If any query fails
        """
        with ErrorContext("prometheus_multiple_queries", "PrometheusClient").add_context(
            query_count=len(queries)
        ):
            try:
                self.logger.debug("Executing multiple Prometheus queries", count=len(queries))
                
                # Execute all queries concurrently
                tasks = {name: self.query(query) for name, query in queries.items()}
                results = await asyncio.gather(*tasks.values(), return_exceptions=True)
                
                # Process results
                final_results = {}
                for (name, _), result in zip(queries.items(), results):
                    if isinstance(result, Exception):
                        raise PrometheusError(
                            f"Query '{name}' failed: {result}",
                            context={"query_name": name, "query": queries[name]}
                        ) from result
                    final_results[name] = result
                
                return final_results
                
            except Exception as e:
                if isinstance(e, PrometheusError):
                    raise
                raise PrometheusError(
                    f"Multiple query execution failed: {e}",
                    context={"queries": list(queries.keys())}
                ) from e
    
    async def health_check(self) -> bool:
        """
        Check if Prometheus is healthy.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            await self._ensure_session()
            
            async with self._session.get(f"{self.config.url}/-/healthy") as response:
                return response.status == 200
                
        except Exception as e:
            self.logger.error("Prometheus health check failed", error=str(e))
            return False
    
    async def get_targets(self) -> List[Dict[str, Any]]:
        """
        Get active targets from Prometheus.
        
        Returns:
            List of target information
            
        Raises:
            PrometheusError: If the request fails
        """
        try:
            await self._ensure_session()
            
            async with self._session.get(f"{self._base_url}/targets") as response:
                if response.status != 200:
                    raise PrometheusError(
                        f"Failed to get targets: HTTP {response.status}",
                        context={"status": response.status}
                    )
                
                data = await response.json()
                return data.get("data", {}).get("activeTargets", [])
                
        except aiohttp.ClientError as e:
            raise PrometheusError(f"HTTP error getting targets: {e}") from e


@asynccontextmanager
async def prometheus_client(config: Optional[PrometheusConfig] = None):
    """
    Context manager for Prometheus client.
    
    Args:
        config: Optional Prometheus configuration
        
    Yields:
        PrometheusClient: Configured Prometheus client
    """
    client = PrometheusClient(config)
    try:
        await client.__aenter__()
        yield client
    finally:
        await client.__aexit__(None, None, None)


# Factory function for dependency injection
def create_prometheus_client(config: Optional[PrometheusConfig] = None) -> PrometheusClient:
    """
    Factory function to create a Prometheus client.
    
    Args:
        config: Optional Prometheus configuration
        
    Returns:
        PrometheusClient: Configured client instance
    """
    return PrometheusClient(config) 