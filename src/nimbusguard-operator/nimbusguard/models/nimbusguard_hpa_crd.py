from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class ConditionType(Enum):
    """Condition types for metric evaluation"""
    GT = "gt"  # greater than
    LT = "lt"  # less than
    EQ = "eq"  # equal


class DecisionEngine(Enum):
    """Available decision engines"""
    LANGGRAPH = "langgraph"
    BASIC = "basic"


@dataclass
class Metric:
    """Metric configuration for scaling decisions"""
    query: str
    threshold: float
    condition: ConditionType
    
    def __post_init__(self):
        # Convert string to enum if needed
        if isinstance(self.condition, str):
            self.condition = ConditionType(self.condition)


@dataclass
class Condition:
    """Status condition for the NimbusGuardHPA resource"""
    type: str
    status: str
    lastTransitionTime: datetime
    reason: str
    message: str
    
    def __post_init__(self):
        # Convert string to datetime if needed
        if isinstance(self.lastTransitionTime, str):
            self.lastTransitionTime = datetime.fromisoformat(self.lastTransitionTime.replace('Z', '+00:00'))


@dataclass
class NimbusGuardHPASpec:
    """Specification for NimbusGuardHPA resource"""
    namespace: str
    target_labels: Dict[str, str]
    prometheus_url: str
    evaluation_interval: int
    decision_window: str
    metrics: List[Metric]
    min_replicas: int = 1
    max_replicas: int = 10
    trace_decisions: bool = False
    decision_engine: DecisionEngine = DecisionEngine.BASIC
    
    def __post_init__(self):
        # Convert string to enum if needed
        if isinstance(self.decision_engine, str):
            self.decision_engine = DecisionEngine(self.decision_engine)
        
        # Convert dict metrics to Metric objects if needed
        if self.metrics and isinstance(self.metrics[0], dict):
            self.metrics = [Metric(**metric) if isinstance(metric, dict) else metric 
                          for metric in self.metrics]


@dataclass
class NimbusGuardHPAStatus:
    """Status for NimbusGuardHPA resource"""
    last_evaluation: Optional[datetime] = None
    current_replicas: Optional[int] = None
    target_replicas: Optional[int] = None
    decision_reason: Optional[str] = None
    conditions: List[Condition] = field(default_factory=list)
    
    def __post_init__(self):
        # Convert string to datetime if needed
        if isinstance(self.last_evaluation, str):
            self.last_evaluation = datetime.fromisoformat(self.last_evaluation.replace('Z', '+00:00'))
        
        # Convert dict conditions to Condition objects if needed
        if self.conditions and isinstance(self.conditions[0], dict):
            self.conditions = [Condition(**cond) if isinstance(cond, dict) else cond 
                             for cond in self.conditions]


@dataclass
class NimbusGuardHPA:
    """NimbusGuardHPA Custom Resource"""
    apiVersion: str = "nimbusguard.io/v1alpha1"
    kind: str = "NimbusGuardHPA"
    metadata: Dict[str, any] = field(default_factory=dict)
    spec: Optional[NimbusGuardHPASpec] = None
    status: Optional[NimbusGuardHPAStatus] = None
    
    def __post_init__(self):
        # Convert dict spec to NimbusGuardHPASpec if needed
        if isinstance(self.spec, dict):
            self.spec = NimbusGuardHPASpec(**self.spec)
        
        # Convert dict status to NimbusGuardHPAStatus if needed
        if isinstance(self.status, dict):
            self.status = NimbusGuardHPAStatus(**self.status)


# Helper functions for creating instances
def create_metric(query: str, threshold: float, condition: str) -> Metric:
    """Helper function to create a Metric instance"""
    return Metric(query=query, threshold=threshold, condition=ConditionType(condition))


def create_nimbusguard_hpa_spec(
    namespace: str,
    target_labels: Dict[str, str],
    prometheus_url: str,
    evaluation_interval: int,
    decision_window: str,
    metrics: List[Dict[str, any]],
    **kwargs
) -> NimbusGuardHPASpec:
    """Helper function to create a NimbusGuardHPASpec instance"""
    metric_objects = [Metric(**metric) for metric in metrics]
    
    return NimbusGuardHPASpec(
        namespace=namespace,
        target_labels=target_labels,
        prometheus_url=prometheus_url,
        evaluation_interval=evaluation_interval,
        decision_window=decision_window,
        metrics=metric_objects,
        **kwargs
    )

