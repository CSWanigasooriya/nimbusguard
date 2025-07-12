from typing import TypedDict, Dict, Any

# --- LangGraph State ---
class Experience(TypedDict):
    state: Dict[str, Any]
    action: str
    reward: float
    next_state: Dict[str, Any]

class ScalingState(TypedDict):
    current_metrics: Dict[str, float]
    current_replicas: int
    dqn_prediction: Dict[str, Any]
    llm_validation_response: Dict[str, Any]
    final_decision: int
    experience: Experience
    error: str