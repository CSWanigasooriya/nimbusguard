# engine/rules.py
# ============================================================================
# Scaling Decision Logic
# ============================================================================

import os
from typing import Dict, Any, List


def extract_metric_name(query: str) -> str:
    """Extracts a simple, readable metric name from a PromQL query."""
    query_lower = query.lower()
    if "cpu" in query_lower:
        return "cpu_usage"
    elif "memory" in query_lower:
        return "memory_usage"
    elif "request" in query_lower:
        return "request_rate"
    else:
        return "custom_metric"


def make_decision(
        metrics: Dict[str, float],
        current_replicas: int,
        metric_configs: List[Dict],
        min_replicas: int,
        max_replicas: int
) -> Dict[str, Any]:
    """
    Makes a pure, stateless scaling decision based on metric values.
    """
    alerts = []
    scale_up_needed = False
    # By default, assume scaling down is possible. Any metric above its threshold will prevent this.
    can_scale_down = True

    # Evaluate each metric against its configured threshold
    for config in metric_configs:
        query = config.get("query", "")
        threshold = float(config.get("threshold", 0.0))
        condition = config.get("condition", "gt")

        metric_name = extract_metric_name(query)
        metric_value = metrics.get(metric_name, 0.0)

        # Any metric value exceeding its threshold prevents a scale-down action.
        if metric_value > threshold:
            can_scale_down = False

        # Check conditions that would trigger a scale-up
        if condition == "gt" and metric_value > threshold:
            alerts.append(f"{metric_name}: {metric_value:.2f} > {threshold}")
            scale_up_needed = True
        elif condition == "lt" and metric_value < threshold:
            alerts.append(f"{metric_name}: {metric_value:.2f} < {threshold}")
            scale_up_needed = True

    # --- Determine Action ---

    # 1. Scale Up: This has the highest priority.
    if scale_up_needed and current_replicas < max_replicas:
        return {
            "action": "scale_up",
            "target_replicas": min(current_replicas + 1, max_replicas),
            "reason": f"Metrics exceeded thresholds: {', '.join(alerts)}",
            "alerts": alerts
        }

    # 2. Scale Down: Only if no scale-up is needed and all metrics are calm.
    if can_scale_down and current_replicas > min_replicas:
        return {
            "action": "scale_down",
            "target_replicas": max(current_replicas - 1, min_replicas),
            "reason": "All metrics are below thresholds; scaling down to optimize resources.",
            "alerts": []
        }

    # 3. No Action: Default case if no scaling is needed or possible.
    reason = "All metrics are within acceptable ranges."
    if scale_up_needed and current_replicas >= max_replicas:
        reason = f"At max replica limit ({max_replicas}) but thresholds are breached: {', '.join(alerts)}"
    elif not can_scale_down and not scale_up_needed:
        reason = "Metrics are stable; maintaining current replica count."

    return {
        "action": "none",
        "target_replicas": current_replicas,
        "reason": reason,
        "alerts": alerts
    }


class DecisionEngine:
    """
    A simple wrapper for the decision engine.
    In a more complex scenario, this could hold state or connect to an AI model.
    """

    def __init__(self, global_health_status: Dict):
        """Initializes the engine and updates the global health status."""
        self.openai_key = os.getenv("OPENAI_API_KEY")
        global_health_status["openai"] = bool(self.openai_key)
