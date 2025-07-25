"""
Workflow state definition for the NimbusGuard scaling workflow.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import pandas as pd


class WorkflowState(BaseModel):
    """State object that flows through the LangGraph workflow."""
    
    # Input parameters
    deployment_name: str
    namespace: str
    current_replicas: int
    ready_replicas: int
    
    # Metrics collection
    current_metrics: Dict[str, float] = {}
    historical_data: Dict[str, pd.DataFrame] = {}
    
    # Forecasting results
    cpu_forecast: Optional[float] = None
    memory_forecast: Optional[float] = None
    
    # DQN decision
    recommended_action: Optional[str] = None  # "scale_up", "scale_down", "no_action"
    recommended_replicas: Optional[int] = None
    action_confidence: Optional[float] = None
    
    # Validation
    validation_passed: bool = False
    validation_reasons: List[str] = []
    
    # Execution
    scaling_executed: bool = False
    execution_error: Optional[str] = None
    
    # Reward calculation
    reward_value: Optional[float] = None
    reward_components: Dict[str, Any] = {}
    
    # DQN-specific fields
    dqn_q_values: Dict[str, float] = {}
    dqn_reward: Optional[float] = None
    dqn_reward_components: Dict[str, Any] = {}
    dqn_decision_data: Dict[str, Any] = {}
    deployment_context: Dict[str, Any] = {}
    
    # DQN training fields
    training_loss: Optional[float] = None
    current_epsilon: Optional[float] = None
    buffer_size: Optional[int] = None
    
    # Workflow metadata
    workflow_start_time: Optional[str] = None
    workflow_duration: Optional[float] = None
    error_occurred: bool = False
    error_message: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True 