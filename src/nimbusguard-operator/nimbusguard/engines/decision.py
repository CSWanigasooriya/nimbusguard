"""
Decision engine for intelligent scaling decisions.
"""

import os
import logging
from typing import Dict, Any, List
from ..health import set_component_health

LOG = logging.getLogger(__name__)

# ============================================================================
# Decision Engine
# ============================================================================

class DecisionEngine:
    """Simple decision engine with basic scaling logic"""
    
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        set_component_health("openai", bool(self.openai_key))
    
    def make_decision(self, metrics: Dict[str, float], current_replicas: int, 
                     metric_configs: List[Dict], min_replicas: int = 1, max_replicas: int = 10) -> Dict[str, Any]:
        """Make scaling decision based on metrics"""
        
        alerts = []
        scale_up_needed = False
        scale_down_needed = True  # Assume we can scale down unless metrics say otherwise
        
        # Evaluate each metric
        for metric_config in metric_configs:
            query = metric_config.get("query", "")
            threshold = metric_config.get("threshold", 0)
            condition = metric_config.get("condition", "gt")
            
            # Get metric name from query
            metric_name = self._extract_metric_name(query)
            metric_value = metrics.get(metric_name, 0)
            
            # Evaluate condition
            if condition == "gt" and metric_value > threshold:
                alerts.append(f"{metric_name}: {metric_value:.1f} > {threshold}")
                scale_up_needed = True
                scale_down_needed = False
            elif condition == "lt" and metric_value < threshold:
                alerts.append(f"{metric_name}: {metric_value:.1f} < {threshold}")
                scale_up_needed = True
                scale_down_needed = False
            elif condition == "eq" and abs(metric_value - threshold) < 0.1:
                alerts.append(f"{metric_name}: {metric_value:.1f} ≈ {threshold}")
                scale_down_needed = False
        
        # Make scaling decision
        if scale_up_needed and current_replicas < max_replicas:
            target_replicas = min(current_replicas + 1, max_replicas)
            return {
                "action": "scale_up",
                "target_replicas": target_replicas,
                "reason": f"Metrics exceeded thresholds: {', '.join(alerts)}",
                "alerts": alerts
            }
        elif scale_down_needed and current_replicas > min_replicas and not alerts:
            target_replicas = max(current_replicas - 1, min_replicas)
            return {
                "action": "scale_down", 
                "target_replicas": target_replicas,
                "reason": "All metrics below thresholds, scaling down",
                "alerts": []
            }
        else:
            return {
                "action": "none",
                "target_replicas": current_replicas,
                "reason": "All metrics within acceptable range" if not alerts else "At scaling limits",
                "alerts": alerts
            }
    
    def _extract_metric_name(self, query: str) -> str:
        """Extract metric name from PromQL query"""
        if "cpu" in query.lower():
            return "cpu_usage"
        elif "memory" in query.lower():
            return "memory_usage"
        elif "request" in query.lower():
            return "request_rate"
        else:
            return "metric"
    
    async def make_ai_decision(self, metrics: Dict[str, float], current_replicas: int, 
                              metric_configs: List[Dict], historical_data: Dict = None) -> Dict[str, Any]:
        """Make AI-powered scaling decision (placeholder for future OpenAI integration)"""
        # This is where you could integrate with OpenAI API for more sophisticated decision making
        # For now, fall back to the simple decision engine
        return self.make_decision(metrics, current_replicas, metric_configs)
    
    def validate_decision(self, decision: Dict[str, Any]) -> bool:
        """Validate that a decision is safe to execute"""
        # Add validation logic here
        required_fields = ["action", "target_replicas", "reason"]
        return all(field in decision for field in required_fields)
