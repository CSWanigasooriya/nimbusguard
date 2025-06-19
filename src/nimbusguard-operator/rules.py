# engine/rules.py - Corrected to use action history and proper state dimension
# ============================================================================
# DQN-Based Decision Engine
# ============================================================================

import logging
import os
from typing import Dict, Any, Optional, List

import numpy as np
from ml.dqn_agent import DQNAgent
from ml.state_representation import EnvironmentState

LOG = logging.getLogger(__name__)

# Global DQN agent instance
_dqn_agent: Optional[DQNAgent] = None


def _initialize_dqn_agent(spec: Dict[str, Any]) -> DQNAgent:
    """Initialize the global DQN agent instance based on the CRD spec."""
    global _dqn_agent

    if _dqn_agent is None:
        ml_config = spec.get("ml_config", {})
        model_path = ml_config.get("model_path", "/models/nimbusguard_dqn.pth")

        state_dimension = 11

        _dqn_agent = DQNAgent(
            state_dim=state_dimension,
            action_dim=5,
            epsilon_start=ml_config.get("epsilon_start", 0.1),
            save_frequency=ml_config.get("save_frequency", 100),
            model_path=model_path
        )

        training_mode = ml_config.get("training_mode", False)
        _dqn_agent.set_training_mode(training_mode)
        LOG.info(f"Initialized DQN agent (state_dim={state_dimension}, training_mode={training_mode})")

    return _dqn_agent


def make_decision(
        current_replicas: int,
        min_replicas: int,
        max_replicas: int,
        spec: Dict[str, Any],
        unified_state: Dict[str, Any],
        recent_actions: List[int]
) -> Dict[str, Any]:
    """
    Uses the DQN agent to make a scaling decision, now considering recent action history.
    """
    agent = _initialize_dqn_agent(spec)

    confidence = unified_state.get("confidence_score", 0.0)
    ml_config = spec.get("ml_config", {})
    confidence_threshold = ml_config.get("confidence_threshold", 0.3)

    if confidence < confidence_threshold:
        reason = f"Observability confidence ({confidence:.2f}) is below threshold ({confidence_threshold})."
        return _fallback_to_safety_decision(current_replicas, min_replicas, max_replicas, reason)

    try:
        # Create environment state, now passing the history from the handler
        current_state = EnvironmentState.from_observability_data(
            unified_state=unified_state,
            current_replicas=current_replicas,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            recent_actions=recent_actions
        )

        # Get DQN decision
        action, decision_metadata = agent.select_action(
            state=current_state,
            force_valid=True
        )

        replica_change = action.get_replica_change()
        target_replicas = max(min_replicas, min(current_replicas + replica_change, max_replicas))

        action_str = "none"
        if target_replicas > current_replicas:
            action_str = "scale_up"
        elif target_replicas < current_replicas:
            action_str = "scale_down"

        # The agent provides a descriptive decision type ("exploration" or "exploitation")
        decision_type_str = "Exploit" if decision_metadata.get('decision_type', True) else "Explore"
        q_values = decision_metadata.get("q_values", [])
        max_q = max(q_values) if q_values else 0.0
        
        # Ensure max_q is a scalar value, not a list
        if isinstance(max_q, (list, np.ndarray)):
            max_q = float(max_q[0]) if len(max_q) > 0 else 0.0
        
        reason = f"DQN decision: {action.name} ({decision_type_str}) | Max Q-value: {max_q:.3f} | Confidence: {confidence:.2f}"

        return {
            "action": action_str,
            "target_replicas": target_replicas,
            "reason": reason,
            # Add ML context for logging and for the handler to update its history
            "ml_decision": {
                "decision_type": decision_type_str,
                "action_value": action.value,
                "q_values": decision_metadata.get("q_values"),
                "max_q_value": max_q
            }
        }

    except Exception as e:
        LOG.error(f"DQN decision failed: {e}", exc_info=True)
        return _fallback_to_safety_decision(current_replicas, min_replicas, max_replicas, str(e))


def _fallback_to_safety_decision(current_replicas: int, min_replicas: int, max_replicas: int, reason: str) -> Dict[
    str, Any]:
    """Safety fallback to maintain current state, respecting bounds."""
    LOG.warning(f"Using safety fallback decision: {reason}")
    target_replicas = max(min_replicas, min(current_replicas, max_replicas))
    action = "none"
    if target_replicas != current_replicas:
        action = "scale_up" if target_replicas > current_replicas else "scale_down"
    return {"action": action, "target_replicas": target_replicas, "reason": f"Safety Fallback: {reason}"}

