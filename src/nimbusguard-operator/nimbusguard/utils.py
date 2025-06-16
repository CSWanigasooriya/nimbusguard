"""
Utilities for working with NimbusGuardHPA resources.
"""

from typing import Dict, Any
from .models.nimbusguard_hpa_crd import NimbusGuardHPA


def parse_intelligent_scaling(body: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a dictionary and extract scaling configuration"""
    try:
        # Extract spec 
        spec = body.get("spec", {})
        
        # Extract basic fields
        config = {
            "namespace": spec.get("namespace"),
            "target_labels": spec.get("target_labels", {}),
            "min_replicas": spec.get("min_replicas", 1),
            "max_replicas": spec.get("max_replicas", 10),
            "decision_engine": spec.get("decision_engine", "basic"),
        }
        
        # Extract metrics configuration - convert flattened to nested structure
        metrics_config = {
            "prometheus_url": spec.get("prometheus_url"),
            "evaluation_interval": spec.get("evaluation_interval", 30),
            "decision_window": spec.get("decision_window", "5m"),
            "trace_decisions": spec.get("trace_decisions", False),
            "metrics": spec.get("metrics", [])
        }
        
        config["metrics_config"] = metrics_config
        return config
        
    except Exception as e:
        raise ValueError(f"Failed to parse NimbusGuardHPA resource: {e}")


def create_status_patch(
    last_evaluation: str,
    current_replicas: int,
    target_replicas: int,
    decision_reason: str,
    action: str
) -> Dict[str, Any]:
    """Create a status patch for NimbusGuardHPA resource"""
    return {
        "status": {
            "last_evaluation": last_evaluation,
            "current_replicas": current_replicas,
            "target_replicas": target_replicas,
            "decision_reason": decision_reason,
            "conditions": [
                {
                    "type": "Ready",
                    "status": "True" if action == "none" else "False",
                    "lastTransitionTime": last_evaluation,
                    "reason": "EvaluationComplete",
                    "message": decision_reason
                }
            ]
        }
    }
