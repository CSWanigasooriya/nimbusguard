# engine/tempo.py
# ============================================================================
# Tempo Client for Trace Analysis
# ============================================================================

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import aiohttp
import numpy as np

from config import health_status

LOG = logging.getLogger(__name__)


class TempoClient:
    """Async client for querying Tempo and extracting trace-based features."""

    def __init__(self, url: str):
        """
        Initializes the Tempo client.
        Args:
            url: The base URL of the Tempo server (e.g., http://tempo.default.svc:3100)
        """
        self.url = url.rstrip('/')
        self.session_timeout = aiohttp.ClientTimeout(total=15)

    async def get_service_graph_metrics(self, service_name: str, lookback_minutes: int = 5) -> Dict[str, float]:
        """
        Extracts service graph metrics from Tempo.
        Returns features related to service dependencies and communication patterns.
        """
        features = {
            "service_graph_latency_p50": 0.0,
            "service_graph_latency_p95": 0.0,
            "service_graph_latency_p99": 0.0,
            "dependency_error_rate": 0.0,
            "dependency_request_rate": 0.0,
            "service_bottleneck_score": 0.0,
            "dependency_health_score": 1.0,
        }

        try:
            # Query service graph for the specified service
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=lookback_minutes)
            
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                # Tempo's service graph API endpoint
                async with session.get(
                    f"{self.url}/api/v2/search",
                    params={
                        "service.name": service_name,
                        "start": int(start_time.timestamp()),
                        "end": int(end_time.timestamp()),
                        "limit": 100
                    }
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        traces = data.get("traces", [])
                        
                        if traces:
                            latencies = []
                            error_count = 0
                            total_spans = 0
                            
                            for trace in traces:
                                trace_id = trace.get("traceID", "")
                                if trace_id:
                                    # Get detailed trace to analyze spans
                                    trace_details = await self._get_trace_details(session, trace_id)
                                    if trace_details:
                                        span_metrics = self._analyze_trace_spans(trace_details)
                                        latencies.extend(span_metrics["latencies"])
                                        error_count += span_metrics["errors"]
                                        total_spans += span_metrics["total_spans"]
                            
                            # Calculate latency percentiles
                            if latencies:
                                features["service_graph_latency_p50"] = np.percentile(latencies, 50)
                                features["service_graph_latency_p95"] = np.percentile(latencies, 95)
                                features["service_graph_latency_p99"] = np.percentile(latencies, 99)
                            
                            # Calculate error rate
                            if total_spans > 0:
                                features["dependency_error_rate"] = error_count / total_spans
                                features["dependency_request_rate"] = len(traces) / lookback_minutes
                                
                                # Calculate bottleneck score (higher latency = higher bottleneck)
                                if features["service_graph_latency_p95"] > 0:
                                    features["service_bottleneck_score"] = min(
                                        features["service_graph_latency_p95"] / 1000.0, 1.0
                                    )
                                
                                # Health score (inverse of error rate)
                                features["dependency_health_score"] = max(
                                    1.0 - features["dependency_error_rate"], 0.0
                                )
                        
                        health_status["tempo"] = True
                    else:
                        LOG.warning(f"Tempo returned status {resp.status} for service: {service_name}")
                        
        except asyncio.TimeoutError:
            LOG.error(f"Tempo query timed out for service: {service_name}")
            health_status["tempo"] = False
        except aiohttp.ClientError as e:
            LOG.error(f"Tempo client error: {e}")
            health_status["tempo"] = False
        except Exception as e:
            LOG.error(f"Unexpected error in Tempo query: {e}", exc_info=True)
            health_status["tempo"] = False

        return features

    async def _get_trace_details(self, session: aiohttp.ClientSession, trace_id: str) -> Optional[Dict]:
        """Fetches detailed trace information for a specific trace ID."""
        try:
            async with session.get(f"{self.url}/api/traces/{trace_id}") as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            LOG.debug(f"Failed to get trace details for {trace_id}: {e}")
        return None

    def _analyze_trace_spans(self, trace_data: Dict) -> Dict[str, Any]:
        """Analyzes spans within a trace to extract metrics."""
        latencies = []
        errors = 0
        total_spans = 0
        
        # Navigate trace structure - this may vary based on your Tempo configuration
        batches = trace_data.get("batches", [])
        for batch in batches:
            spans = batch.get("resource", {}).get("spans", [])
            for span in spans:
                total_spans += 1
                
                # Extract duration (usually in microseconds)
                start_time = span.get("startTimeUnixNano", 0)
                end_time = span.get("endTimeUnixNano", 0)
                if start_time and end_time:
                    duration_ms = (end_time - start_time) / 1_000_000  # Convert to milliseconds
                    latencies.append(duration_ms)
                
                # Check for errors in span status
                status = span.get("status", {})
                if status.get("code") == 2:  # ERROR status code
                    errors += 1
        
        return {
            "latencies": latencies,
            "errors": errors,
            "total_spans": total_spans
        }

    async def get_trace_anomaly_features(self, service_name: str, lookback_minutes: int = 10) -> Dict[str, float]:
        """
        Detects anomalies in trace patterns and returns anomaly-related features.
        """
        features = {
            "trace_anomaly_score": 0.0,
            "unusual_latency_patterns": 0.0,
            "trace_volume_anomaly": 0.0,
        }

        try:
            # This is a simplified anomaly detection
            # In production, you'd want to use more sophisticated methods
            current_metrics = await self.get_service_graph_metrics(service_name, lookback_minutes)
            
            # Simple anomaly scoring based on latency thresholds
            p95_latency = current_metrics.get("service_graph_latency_p95", 0)
            if p95_latency > 1000:  # 1 second threshold
                features["trace_anomaly_score"] = min(p95_latency / 5000.0, 1.0)
                features["unusual_latency_patterns"] = 1.0
            
            # Volume anomaly (simplified)
            request_rate = current_metrics.get("dependency_request_rate", 0)
            if request_rate > 100:  # requests per minute threshold
                features["trace_volume_anomaly"] = min(request_rate / 1000.0, 1.0)
            
            health_status["tempo"] = True
            
        except Exception as e:
            LOG.error(f"Error in trace anomaly detection: {e}", exc_info=True)
            health_status["tempo"] = False

        return features
