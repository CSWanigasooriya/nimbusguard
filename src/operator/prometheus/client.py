"""Simple Prometheus client for fetching selected metrics."""
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np

from config.settings import load_config
config = load_config()
from prometheus.queries import PrometheusQueries


logger = logging.getLogger(__name__)


class PrometheusClient:
    """Simple client for fetching Prometheus metrics."""
    
    def __init__(self):
        self.base_url = config.prometheus.url
        self.timeout = config.prometheus.timeout
        self.queries = PrometheusQueries()
        
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict]:
        """Make HTTP request to Prometheus API."""
        try:
            url = f"{self.base_url}/api/v1/{endpoint}"
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            if data.get("status") != "success":
                logger.error(f"Prometheus API error: {data.get('error', 'Unknown error')}")
                return None
                
            return data.get("data", {})
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Prometheus request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error querying Prometheus: {e}")
            return None
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current values for all selected features."""
        metrics = {}
        
        for feature_name in config.scaling.selected_features:
            try:
                # Use aggregated query for current metrics
                query = self.queries.get_aggregated_query(feature_name)
                
                data = self._make_request("query", {"query": query})
                if not data or not data.get("result"):
                    logger.warning(f"No data returned for feature: {feature_name}")
                    metrics[feature_name] = 0.0
                    continue
                
                # Extract the metric value
                result = data["result"][0]  # Take first result
                value = float(result["value"][1])  # [timestamp, value]
                metrics[feature_name] = value
                
                # Debug logging for memory metrics specifically
                if "memory" in feature_name.lower():
                    logger.info(f"🔍 DEBUG: {feature_name} = {value} bytes ({value/(1024*1024):.1f} MB)")
                else:
                    logger.debug(f"Feature {feature_name}: {value}")
                
            except (IndexError, KeyError, ValueError) as e:
                logger.warning(f"Failed to parse {feature_name}: {e}")
                metrics[feature_name] = 0.0
            except Exception as e:
                logger.error(f"Error fetching {feature_name}: {e}")
                metrics[feature_name] = 0.0
        
        logger.info(f"Fetched {len(metrics)} current metrics")
        return metrics
    
    def get_historical_metrics(self, duration_minutes: int) -> Dict[str, List[float]]:
        """Get historical time series data for all selected features."""
        historical_data = {}
        
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=duration_minutes)
        
        for feature_name in config.scaling.selected_features:
            try:
                # Use aggregated query for historical data
                query = self.queries.get_aggregated_query(feature_name)
                
                params = {
                    "query": query,
                    "start": start_time.isoformat() + "Z",
                    "end": end_time.isoformat() + "Z", 
                    "step": "1m"  # 1-minute resolution
                }
                
                data = self._make_request("query_range", params)
                if not data or not data.get("result"):
                    logger.warning(f"No historical data for feature: {feature_name}")
                    historical_data[feature_name] = []
                    continue
                
                # Extract time series values
                result = data["result"][0]  # Take first result
                values = []
                for timestamp, value in result.get("values", []):
                    try:
                        values.append(float(value))
                    except (ValueError, TypeError):
                        values.append(0.0)
                
                historical_data[feature_name] = values
                logger.debug(f"Feature {feature_name}: {len(values)} historical points")
                
            except (IndexError, KeyError) as e:
                logger.warning(f"Failed to parse historical {feature_name}: {e}")
                historical_data[feature_name] = []
            except Exception as e:
                logger.error(f"Error fetching historical {feature_name}: {e}")
                historical_data[feature_name] = []
        
        logger.info(f"Fetched historical data for {len(historical_data)} features")
        return historical_data
    
    def get_feature_matrix(self, duration_minutes: int) -> Optional[np.ndarray]:
        """Get historical data as a numpy matrix for ML processing."""
        historical_data = self.get_historical_metrics(duration_minutes)
        
        if not historical_data:
            logger.error("No historical data available")
            return None
        
        # Find the minimum length across all features
        non_empty_lengths = [len(values) for values in historical_data.values() if values]
        if not non_empty_lengths:
            logger.error("All historical data is empty")
            return None
        
        min_length = min(non_empty_lengths)
        if min_length == 0:
            logger.error("All historical data is empty")
            return None
        
        # Create matrix: [time_steps, features]
        matrix = []
        for feature_name in config.scaling.selected_features:
            values = historical_data.get(feature_name, [])
            if len(values) >= min_length:
                matrix.append(values[-min_length:])  # Take last min_length values
            else:
                # Pad with zeros if needed
                padded = [0.0] * (min_length - len(values)) + values
                matrix.append(padded)
        
        # Transpose to get [time_steps, features] shape
        feature_matrix = np.array(matrix).T
        logger.info(f"Created feature matrix: {feature_matrix.shape}")
        return feature_matrix
    
    def health_check(self) -> bool:
        """Check if Prometheus is accessible."""
        try:
            data = self._make_request("query", {"query": "up"})
            return data is not None
        except Exception as e:
            logger.error(f"Prometheus health check failed: {e}")
            return False 