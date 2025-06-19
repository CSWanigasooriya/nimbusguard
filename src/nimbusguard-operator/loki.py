# engine/loki.py
# ============================================================================
# Loki Client for Log Analysis
# ============================================================================

import asyncio
import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List

import aiohttp
import numpy as np

from config import health_status

LOG = logging.getLogger(__name__)


class LokiClient:
    """Async client for querying Loki and extracting log-based features."""

    def __init__(self, url: str):
        """
        Initializes the Loki client.
        Args:
            url: The base URL of the Loki server (e.g., http://loki.default.svc:3100)
        """
        self.url = url.rstrip('/')
        self.session_timeout = aiohttp.ClientTimeout(total=15)
        
        # Common error patterns to detect
        self.error_patterns = {
            "error": re.compile(r'\berror\b|\bfailed\b|\bexception\b', re.IGNORECASE),
            "timeout": re.compile(r'\btimeout\b|\btimed out\b', re.IGNORECASE),
            "connection": re.compile(r'\bconnection refused\b|\bconnection lost\b', re.IGNORECASE),
            "http_5xx": re.compile(r'\b5[0-9]{2}\b'),
            "http_4xx": re.compile(r'\b4[0-9]{2}\b'),
            "oom": re.compile(r'\bout of memory\b|\boom\b', re.IGNORECASE),
        }

    async def get_log_analysis_features(self, app_labels: Dict[str, str], lookback_minutes: int = 5) -> Dict[str, float]:
        """
        Analyzes recent logs to extract error patterns and anomaly features.
        """
        features = {
            "error_log_frequency": 0.0,
            "critical_error_rate": 0.0,
            "timeout_error_rate": 0.0,
            "http_error_rate": 0.0,
            "log_anomaly_score": 0.0,
        }

        try:
            # Build LogQL query for the application
            label_selector = self._build_label_selector(app_labels)
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=lookback_minutes)
            
            # Query logs from Loki
            logs = await self._query_loki_logs(label_selector, start_time, end_time)
            
            if logs:
                features = self._analyze_log_patterns(logs, lookback_minutes)
                health_status["loki"] = True
            else:
                LOG.debug(f"No logs found for labels: {app_labels}")
                health_status["loki"] = True  # Still healthy, just no data
                
        except Exception as e:
            LOG.error(f"Error in log analysis: {e}", exc_info=True)
            health_status["loki"] = False

        return features

    async def get_business_kpi_features(self, app_labels: Dict[str, str], lookback_minutes: int = 10) -> Dict[str, float]:
        """
        Extracts business KPI trends from structured logs.
        """
        features = {
            "business_kpi_trend": 0.0,
            "user_activity_anomaly": 0.0,
            "transaction_error_rate": 0.0,
        }

        try:
            label_selector = self._build_label_selector(app_labels)
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=lookback_minutes)
            
            # Query for business-related logs
            business_query = f'{label_selector} |~ "transaction|user|order|payment"'
            logs = await self._query_loki_logs(business_query, start_time, end_time)
            
            if logs:
                features = self._extract_business_metrics(logs, lookback_minutes)
                health_status["loki"] = True
                
        except Exception as e:
            LOG.error(f"Error in business KPI analysis: {e}", exc_info=True)
            health_status["loki"] = False

        return features

    def _build_label_selector(self, app_labels: Dict[str, str]) -> str:
        """Builds a LogQL label selector from Kubernetes labels."""
        selectors = []
        for key, value in app_labels.items():
            selectors.append(f'{key}="{value}"')
        return "{" + ",".join(selectors) + "}"

    async def _query_loki_logs(self, query: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Executes a LogQL query and returns log entries."""
        logs = []
        
        try:
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                params = {
                    "query": query,
                    "start": int(start_time.timestamp() * 1_000_000_000),  # nanoseconds
                    "end": int(end_time.timestamp() * 1_000_000_000),
                    "limit": 1000,
                    "direction": "backward"
                }
                
                async with session.get(f"{self.url}/loki/api/v1/query_range", params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = data.get("data", {}).get("result", [])
                        
                        for stream in result:
                            stream_labels = stream.get("stream", {})
                            values = stream.get("values", [])
                            
                            for timestamp_ns, log_line in values:
                                logs.append({
                                    "timestamp": int(timestamp_ns) // 1_000_000,  # Convert to milliseconds
                                    "message": log_line,
                                    "labels": stream_labels
                                })
                    else:
                        LOG.warning(f"Loki returned status {resp.status} for query: {query}")
                        
        except asyncio.TimeoutError:
            LOG.error(f"Loki query timed out for: {query}")
            health_status["loki"] = False
        except aiohttp.ClientError as e:
            LOG.error(f"Loki client error: {e}")
            health_status["loki"] = False

        return logs

    def _analyze_log_patterns(self, logs: List[Dict], lookback_minutes: int) -> Dict[str, float]:
        """Analyzes log entries to extract error patterns and anomaly scores."""
        total_logs = len(logs)
        if total_logs == 0:
            return {
                "error_log_frequency": 0.0,
                "critical_error_rate": 0.0,
                "timeout_error_rate": 0.0,
                "http_error_rate": 0.0,
                "log_anomaly_score": 0.0,
            }

        error_counts = defaultdict(int)
        critical_errors = 0
        timeout_errors = 0
        http_errors = 0
        
        # Analyze each log message
        for log_entry in logs:
            message = log_entry.get("message", "").lower()
            
            # Count different types of errors
            for error_type, pattern in self.error_patterns.items():
                if pattern.search(message):
                    error_counts[error_type] += 1
                    
                    if error_type in ["error", "oom"]:
                        critical_errors += 1
                    elif error_type == "timeout":
                        timeout_errors += 1
                    elif error_type in ["http_5xx", "http_4xx"]:
                        http_errors += 1

        # Calculate rates
        error_log_frequency = sum(error_counts.values()) / lookback_minutes
        critical_error_rate = critical_errors / total_logs if total_logs > 0 else 0.0
        timeout_error_rate = timeout_errors / total_logs if total_logs > 0 else 0.0
        http_error_rate = http_errors / total_logs if total_logs > 0 else 0.0
        
        # Calculate anomaly score based on error concentrations
        anomaly_score = 0.0
        if total_logs > 10:  # Only calculate if we have sufficient data
            error_rate = sum(error_counts.values()) / total_logs
            if error_rate > 0.1:  # More than 10% error logs
                anomaly_score = min(error_rate * 2, 1.0)

        return {
            "error_log_frequency": error_log_frequency,
            "critical_error_rate": critical_error_rate,
            "timeout_error_rate": timeout_error_rate,
            "http_error_rate": http_error_rate,
            "log_anomaly_score": anomaly_score,
        }

    def _extract_business_metrics(self, logs: List[Dict], lookback_minutes: int) -> Dict[str, float]:
        """Extracts business KPI trends from business-related logs."""
        business_events = []
        transaction_errors = 0
        total_transactions = 0
        
        # Business event patterns
        success_pattern = re.compile(r'\bsuccess\b|\bcompleted\b|\bok\b', re.IGNORECASE)
        error_pattern = re.compile(r'\bfailed\b|\berror\b|\brejected\b', re.IGNORECASE)
        transaction_pattern = re.compile(r'\btransaction\b|\border\b|\bpayment\b', re.IGNORECASE)
        
        for log_entry in logs:
            message = log_entry.get("message", "")
            
            if transaction_pattern.search(message):
                total_transactions += 1
                if error_pattern.search(message):
                    transaction_errors += 1
                elif success_pattern.search(message):
                    business_events.append(1)  # Success
                else:
                    business_events.append(0)  # Neutral

        # Calculate business metrics
        business_kpi_trend = 0.0
        if business_events:
            business_kpi_trend = np.mean(business_events)

        transaction_error_rate = 0.0
        if total_transactions > 0:
            transaction_error_rate = transaction_errors / total_transactions

        # User activity anomaly (simplified)
        user_activity_anomaly = 0.0
        if len(logs) > 0:
            logs_per_minute = len(logs) / lookback_minutes
            if logs_per_minute > 100:  # High activity threshold
                user_activity_anomaly = min(logs_per_minute / 1000, 1.0)

        return {
            "business_kpi_trend": business_kpi_trend,
            "user_activity_anomaly": user_activity_anomaly,
            "transaction_error_rate": transaction_error_rate,
        }
