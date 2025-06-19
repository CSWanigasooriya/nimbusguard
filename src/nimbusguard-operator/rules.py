# engine/rules.py - Optimized for a 7-state DQN model
# ============================================================================
# DQN-Based Decision Engine
# ============================================================================

import logging
import os
from typing import Dict, Any, Optional

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

        # --- CRITICAL CHANGE ---
        # The state dimension is now 7, matching our core feature vector.
        # We add +4 for current_replicas, min/max replicas, and time since last scale,
        # if your DQNAgent implementation expects them as part of the input vector.
        # If the agent only takes the observability features, use state_dim=7.
        # Let's assume the agent is designed to take the 7 features plus replica info.
        state_dimension = 7 + 4  # 7 observability features + 4 context features

        _dqn_agent = DQNAgent(
            state_dim=state_dimension,
            action_dim=5,  # 5 scaling actions: strong down, down, none, up, strong up
            epsilon_start=ml_config.get("epsilon_start", 0.1),
            model_path=model_path if os.path.exists(model_path) else None
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
        unified_state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Uses the DQN agent to make a scaling decision based on the 7-feature state.
    """
    agent = _initialize_dqn_agent(spec)

    # Check data quality
    confidence = unified_state.get("confidence_score", 0.0)
    ml_config = spec.get("ml_config", {})
    confidence_threshold = ml_config.get("confidence_threshold", 0.3)

    if confidence < confidence_threshold:
        reason = f"Observability confidence ({confidence:.2f}) is below threshold ({confidence_threshold})."
        return _fallback_to_safety_decision(current_replicas, min_replicas, max_replicas, reason)

    try:
        # Create environment state from the new, focused unified_state
        current_state = EnvironmentState.from_observability_data(
            unified_state=unified_state,
            current_replicas=current_replicas,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            # Other context can be added here if the state representation needs it
        )

        # Get DQN decision
        action, decision_metadata = agent.select_action(
            state=current_state,
            force_valid=True  # Enforce replica constraints
        )

        replica_change = action.get_replica_change()
        target_replicas = max(min_replicas, min(current_replicas + replica_change, max_replicas))

        action_str = "none"
        if target_replicas > current_replicas:
            action_str = "scale_up"
        elif target_replicas < current_replicas:
            action_str = "scale_down"

        # Fix for boolean decision_type - interpret and convert to descriptive string
        decision_type = decision_metadata.get('decision_type', False)  # Default to False if missing
        decision_type_str = "automatic" if decision_type else "standard"

        reason = f"DQN decision: {action.name} ({decision_type_str}) with confidence {confidence:.2f}"

        return {
            "action": action_str,
            "target_replicas": target_replicas,
            "reason": reason,
            # You can add more detailed ml_decision context if needed
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

    return {
        "action": action,
        "target_replicas": target_replicas,
        "reason": f"Safety Fallback: {reason}",
    }