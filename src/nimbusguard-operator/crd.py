from dataclasses import dataclass, field
from apischema import schema
from typing import Optional, Dict, Any, List
from datetime import datetime
from kubecrd import KubeResourceBase

@dataclass
class PrometheusQuery:
    """Prometheus query configuration"""
    query: str = field(
        metadata=schema(
            description="PromQL query to execute",
            unique=False
        )
    )
    threshold: float = field(
        metadata=schema(
            description="Threshold value for the metric",
            unique=False
        )
    )
    condition: str = field(
        default="gt",  # gt, lt, eq, ne, gte, lte
        metadata=schema(
            description="Condition to evaluate (gt, lt, eq, ne, gte, lte)",
            unique=False
        )
    )

@dataclass
class MetricsDecisionSpec:
    """Specification for metrics-based decision making"""
    prometheus_url: str = field(
        default="http://prometheus:9090",
        metadata=schema(
            description="Prometheus server URL",
            unique=False
        )
    )
    metrics: List[PrometheusQuery] = field(
        default_factory=list,
        metadata=schema(
            description="List of Prometheus queries to monitor",
            unique=False
        )
    )
    decision_window: str = field(
        default="5m",
        metadata=schema(
            description="Time window for decision making (e.g., 5m, 1h)",
            unique=False
        )
    )
    evaluation_interval: int = field(
        default=30,
        metadata=schema(
            description="Evaluation interval in seconds",
            unique=False
        )
    )
    trace_decisions: bool = field(
        default=True,
        metadata=schema(
            description="Enable OpenTelemetry tracing for decisions",
            unique=False
        )
    )
    action_config: Dict[str, Any] = field(
        default_factory=dict,
        metadata=schema(
            description="Configuration for actions to take",
            unique=False
        )
    )

@dataclass
class IntelligentScalingSpec:
    """Specification for the IntelligentScaling resource"""
    namespace: str = field(
        metadata=schema(
            description="Target namespace to monitor",
            unique=False
        )
    )
    target_labels: Dict[str, str] = field(
        default_factory=dict,
        metadata=schema(
            description="Labels to select target pods/deployments",
            unique=False
        )
    )
    metrics_config: MetricsDecisionSpec = field(
        metadata=schema(
            description="Metrics-based decision configuration",
            unique=False
        )
    )
    decision_engine: str = field(
        default="langgraph",
        metadata=schema(
            description="Decision engine to use (langgraph, simple)",
            unique=False
        )
    )

@dataclass
class DecisionStatus:
    """Status of decision making process"""
    last_evaluation: Optional[datetime] = field(
        default=None,
        metadata=schema(
            description="Last evaluation timestamp",
            unique=False
        )
    )
    current_metrics: Dict[str, float] = field(
        default_factory=dict,
        metadata=schema(
            description="Current metric values",
            unique=False
        )
    )
    last_action: Optional[str] = field(
        default=None,
        metadata=schema(
            description="Last action taken",
            unique=False
        )
    )
    decisions_made: int = field(
        default=0,
        metadata=schema(
            description="Total number of decisions made",
            unique=False
        )
    )
    trace_id: Optional[str] = field(
        default=None,
        metadata=schema(
            description="OpenTelemetry trace ID of last decision",
            unique=False
        )
    )

@dataclass
class IntelligentScaling(KubeResourceBase):
    """Custom Resource Definition for Intelligent Scaling"""
    __group__ = "nimbusguard.io"
    __version__ = "v1alpha1"
    spec: IntelligentScalingSpec
    status: DecisionStatus = field(default_factory=DecisionStatus)

    def __post_init__(self):
        super().__post_init__()
        if not self.metadata:
            self.metadata = {}
        if "name" not in self.metadata:
            self.metadata["name"] = f"intelligent-scaling-{self.spec.namespace}"
