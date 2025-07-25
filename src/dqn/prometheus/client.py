"""
Prometheus client for fetching metrics from Prometheus server.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)


class PrometheusClient:
    """Client for fetching metrics from Prometheus."""
    
    def __init__(self, prometheus_url: str, timeout: int = 30):
        """
        Initialize Prometheus client.
        
        Args:
            prometheus_url: Prometheus server URL
            timeout: Request timeout in seconds
        """
        self.prometheus_url = prometheus_url.rstrip("/")
        self.timeout = timeout
        
        # Prometheus queries for the specific metrics needed
        self.queries = {
            # CPU usage rate - current CPU utilization per second
            "process_cpu_seconds_total_rate":
                'sum(rate(process_cpu_seconds_total{job=~"prometheus.scrape.annotated_pods", instance=~".*:8000"}[15s])) or vector(0)',

            # Memory usage - current instantaneous memory (gauge)
            "process_resident_memory_bytes":
                'sum(process_resident_memory_bytes{job=~"prometheus.scrape.annotated_pods", instance=~".*:8000"}) or vector(0)',
            
            # Current replica count - essential for DQN state awareness
            "kube_deployment_status_replicas":
                'sum(kube_deployment_status_replicas{deployment="consumer", job="prometheus.scrape.annotated_pods", namespace="nimbusguard"}) or vector(0)',
        }
    
    async def fetch_metric_range(
        self, 
        metric_name: str, 
        start_time: datetime, 
        end_time: datetime, 
        step: str = "15s"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch metric data over a time range.
        
        Args:
            metric_name: Name of the metric (key in self.queries)
            start_time: Start time for the query
            end_time: End time for the query
            step: Step size for the query
            
        Returns:
            DataFrame with timestamp and value columns, or None if failed
        """
        if metric_name not in self.queries:
            logger.error(f"Unknown metric: {metric_name}")
            return None
        
        query = self.queries[metric_name]
        
        # Convert times to Unix timestamps
        start_timestamp = start_time.timestamp()
        end_timestamp = end_time.timestamp()
        
        # Build the range query URL
        url = f"{self.prometheus_url}/api/v1/query_range"
        params = {
            "query": query,
            "start": start_timestamp,
            "end": end_timestamp,
            "step": step
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Prometheus query failed with status {response.status}")
                        return None
                    
                    data = await response.json()
                    
                    if data.get("status") != "success":
                        logger.error(f"Prometheus query failed: {data.get('error', 'Unknown error')}")
                        return None
                    
                    # Parse the response
                    result = data.get("data", {}).get("result", [])
                    if not result:
                        logger.warning(f"No data returned for metric {metric_name}")
                        return pd.DataFrame(columns=["timestamp", "value"])
                    
                    # Extract values (assuming single series result)
                    values = result[0].get("values", [])
                    if not values:
                        logger.warning(f"No values in result for metric {metric_name}")
                        return pd.DataFrame(columns=["timestamp", "value"])
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(values, columns=["timestamp", "value"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    
                    logger.info(f"Fetched {len(df)} data points for {metric_name}")
                    return df
                    
        except Exception as e:
            logger.error(f"Error fetching metric {metric_name}: {e}")
            return None
    
    async def fetch_current_metrics(self) -> Dict[str, float]:
        """
        Fetch current values for all metrics.
        
        Returns:
            Dictionary with metric names as keys and current values as values
        """
        results = {}
        
        # Build instant query URL
        url = f"{self.prometheus_url}/api/v1/query"
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                for metric_name, query in self.queries.items():
                    params = {"query": query}
                    
                    async with session.get(url, params=params) as response:
                        if response.status != 200:
                            logger.error(f"Prometheus query failed with status {response.status} for {metric_name}")
                            results[metric_name] = 0.0
                            continue
                        
                        data = await response.json()
                        
                        if data.get("status") != "success":
                            logger.error(f"Prometheus query failed for {metric_name}: {data.get('error', 'Unknown error')}")
                            results[metric_name] = 0.0
                            continue
                        
                        # Parse the response
                        result = data.get("data", {}).get("result", [])
                        if not result:
                            logger.warning(f"No data returned for current metric {metric_name}")
                            results[metric_name] = 0.0
                            continue
                        
                        # Extract current value
                        value = result[0].get("value", [None, "0"])[1]
                        results[metric_name] = float(value) if value else 0.0
                        
                        logger.debug(f"Current {metric_name}: {results[metric_name]}")
                        
        except Exception as e:
            logger.error(f"Error fetching current metrics: {e}")
            # Return zeros for all metrics on error
            return {metric_name: 0.0 for metric_name in self.queries.keys()}
        
        return results
    
    async def fetch_historical_data(self, lookback_minutes: int = 15) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all metrics.
        
        Args:
            lookback_minutes: How many minutes of history to fetch
            
        Returns:
            Dictionary with metric names as keys and DataFrames as values
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=lookback_minutes)
        
        results = {}
        
        # Fetch data for all metrics concurrently
        tasks = []
        for metric_name in self.queries.keys():
            task = self.fetch_metric_range(metric_name, start_time, end_time)
            tasks.append((metric_name, task))
        
        # Wait for all tasks to complete
        for metric_name, task in tasks:
            df = await task
            results[metric_name] = df if df is not None else pd.DataFrame(columns=["timestamp", "value"])
            
        return results 