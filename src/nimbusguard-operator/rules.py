# engine/rules.py
# ============================================================================
# DQN-Based Scaling Decision Engine (Replaces Rule-Based System)
# ============================================================================

import os
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ml.dqn_agent import DQNAgent
from ml.state_representation import EnvironmentState, ScalingActions
from ml.reward_system import RewardComponents

LOG = logging.getLogger(__name__)

# Global DQN agent instance
_dqn_agent: Optional[DQNAgent] = None
_last_state: Optional[EnvironmentState] = None
_last_action: Optional[ScalingActions] = None
_last_decision_time: float = 0

def extract_metric_name(query: str) -> str:
    """Legacy function kept for backward compatibility."""
    query_lower = query.lower()
    if "cpu" in query_lower:
        return "cpu_usage"
    elif "memory" in query_lower:
        return "memory_usage"
    elif "request" in query_lower:
        return "request_rate"
    else:
        return "custom_metric"

def _initialize_dqn_agent() -> DQNAgent:
    """Initialize the global DQN agent instance."""
    global _dqn_agent
    
    if _dqn_agent is None:
        # Check for pre-trained model
        model_path = os.getenv("DQN_MODEL_PATH", "/models/nimbusguard_dqn.pth")
        
        # Initialize DQN agent
        _dqn_agent = DQNAgent(
            state_dim=43,  # Your observability feature dimension
            action_dim=5,  # 5 scaling actions
            learning_rate=float(os.getenv("DQN_LEARNING_RATE", "0.0001")),
            epsilon_start=float(os.getenv("DQN_EPSILON_START", "0.1")),
            epsilon_end=float(os.getenv("DQN_EPSILON_END", "0.01")),
            epsilon_decay=float(os.getenv("DQN_EPSILON_DECAY", "0.995")),
            model_path=model_path if os.path.exists(model_path) else None
        )
        
        # Set to evaluation mode for production
        training_mode = os.getenv("DQN_TRAINING_MODE", "false").lower() == "true"
        _dqn_agent.set_training_mode(training_mode)
        
        LOG.info(f"Initialized DQN agent (training_mode={training_mode})")
    
    return _dqn_agent

def make_decision(
        metrics: Dict[str, float],
        current_replicas: int,
        metric_configs: List[Dict],
        min_replicas: int,
        max_replicas: int,
        enhanced_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    **COMPLETELY REPLACED: DQN-Based Scaling Decision Engine**
    
    Uses Deep Q-Learning instead of rule-based thresholds for intelligent scaling decisions.
    
    Args:
        metrics: Legacy Prometheus metrics (kept for compatibility)
        current_replicas: Current number of replicas
        metric_configs: Metric configuration from CRD (kept for compatibility)
        min_replicas: Minimum allowed replicas
        max_replicas: Maximum allowed replicas
        enhanced_context: Enhanced observability context with feature vector,
                         health scores, and multi-source data availability
    
    Returns:
        Dictionary containing scaling decision and reasoning
    """
    global _last_state, _last_action, _last_decision_time
    
    # Initialize DQN agent if needed
    agent = _initialize_dqn_agent()
    
    # Validate enhanced context
    if not enhanced_context or "feature_vector" not in enhanced_context:
        LOG.error("Enhanced context with feature_vector is required for DQN decisions")
        return _fallback_to_safety_decision(current_replicas, min_replicas, max_replicas, 
                                          "Missing observability data for ML decision")
    
    try:
        # Create current state from observability data
        current_time = time.time()
        time_since_last_scale = current_time - _last_decision_time if _last_decision_time > 0 else 0
        
        # Get recent actions history
        recent_actions = getattr(agent, 'recent_actions_history', [2] * 5)  # Default to NO_ACTION
        
        # Create environment state
        current_state = EnvironmentState.from_observability_data(
            unified_state={
                "feature_vector": enhanced_context["feature_vector"],
                "feature_names": enhanced_context.get("feature_names", []),
                "health_score": enhanced_context.get("health_score", 0.0),
                "confidence_score": enhanced_context.get("confidence_score", 0.0)
            },
            resource_name=enhanced_context.get("resource_name", "unknown"),
            namespace=enhanced_context.get("namespace", "default"),
            current_replicas=current_replicas,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            time_since_last_scale=time_since_last_scale,
            recent_actions=recent_actions
        )
        
        # Evaluate previous decision if we have history
        if _last_state and _last_action:
            reward_components = agent.evaluate_decision(
                previous_state=_last_state,
                action=_last_action,
                current_state=current_state,
                execution_success=True  # Assume success if we're making new decision
            )
            
            # Store experience for learning
            training_mode = os.getenv("DQN_TRAINING_MODE", "false").lower() == "true"
            if training_mode:
                agent.store_experience(
                    state=_last_state,
                    action=_last_action,
                    reward_components=reward_components,
                    next_state=current_state,
                    done=False
                )
                
                # Perform training step
                train_metrics = agent.train_step()
                if train_metrics:
                    LOG.debug(f"Training step completed: {train_metrics}")
        
        # Data quality check
        data_sources = enhanced_context.get("data_sources_available", {})
        available_sources = sum(1 for available in data_sources.values() if available)
        
        if available_sources == 0:
            return _fallback_to_safety_decision(current_replicas, min_replicas, max_replicas,
                                              "No observability data sources available")
        
        if current_state.health_score < 0.3:
            return _fallback_to_safety_decision(current_replicas, min_replicas, max_replicas,
                                              f"Low observability health score: {current_state.health_score:.2f}")
        
        # Get DQN decision
        action, decision_metadata = agent.select_action(
            state=current_state,
            training=training_mode,
            force_valid=True  # Always enforce replica constraints
        )
        
        # Calculate target replicas
        replica_change = action.get_replica_change()
        target_replicas = max(min_replicas, min(current_replicas + replica_change, max_replicas))
        
        # Determine action string
        if target_replicas > current_replicas:
            action_str = "scale_up"
        elif target_replicas < current_replicas:
            action_str = "scale_down"
        else:
            action_str = "none"
        
        # Create comprehensive decision reason
        reason = _create_decision_reason(action, decision_metadata, enhanced_context, current_state)
        
        # Store state and action for next iteration
        _last_state = current_state
        _last_action = action
        _last_decision_time = current_time
        
        # Update agent's action history
        if not hasattr(agent, 'recent_actions_history'):
            agent.recent_actions_history = [2] * 5  # Initialize with NO_ACTION
        agent.recent_actions_history.append(action.value)
        if len(agent.recent_actions_history) > 5:
            agent.recent_actions_history.pop(0)
        
        # Create decision response
        decision = {
            "action": action_str,
            "target_replicas": target_replicas,
            "reason": reason,
            "alerts": [],  # DQN doesn't use traditional alerts
            
            # Enhanced DQN context
            "ml_decision": {
                "model_type": "DQN",
                "action_name": action.name,
                "replica_change": replica_change,
                "decision_type": decision_metadata["decision_type"],
                "epsilon": decision_metadata["epsilon"],
                "state_health_score": current_state.health_score,
                "confidence_score": current_state.confidence_score,
                "q_values": decision_metadata.get("q_values"),
                "valid_actions": decision_metadata["valid_actions"],
                "data_sources_available": list(data_sources.keys()) if data_sources else [],
                "feature_vector_size": len(current_state.feature_vector),
                "agent_performance": agent.get_performance_metrics()
            }
        }
        
        LOG.info(f"DQN Decision: {action.name} -> {target_replicas} replicas "
                f"(change: {replica_change:+d}, type: {decision_metadata['decision_type']})")
        
        return decision
        
    except Exception as e:
        LOG.error(f"DQN decision failed: {e}", exc_info=True)
        return _fallback_to_safety_decision(current_replicas, min_replicas, max_replicas,
                                          f"DQN decision error: {str(e)}")

def _create_decision_reason(action: ScalingActions, 
                          decision_metadata: Dict[str, Any],
                          enhanced_context: Dict[str, Any],
                          current_state: EnvironmentState) -> str:
    """Create comprehensive human-readable decision reason."""
    
    decision_type = decision_metadata["decision_type"]
    health_score = current_state.health_score
    confidence_score = current_state.confidence_score
    
    # Base reason
    if action == ScalingActions.NO_ACTION:
        base_reason = "DQN model recommends maintaining current replica count"
    elif action.get_replica_change() > 0:
        base_reason = f"DQN model recommends scaling up by {action.get_replica_change()} replica(s)"
    else:
        base_reason = f"DQN model recommends scaling down by {abs(action.get_replica_change())} replica(s)"
    
    # Add decision context
    context_parts = [
        f"decision_type: {decision_type}",
        f"health_score: {health_score:.2f}",
        f"confidence: {confidence_score:.2f}"
    ]
    
    # Add Q-value information if available
    if "q_values" in decision_metadata and decision_metadata["q_values"]:
        max_q = max(decision_metadata["q_values"])
        context_parts.append(f"max_q_value: {max_q:.3f}")
    
    # Add data source information
    data_sources = enhanced_context.get("data_sources_available", {})
    available_sources = [k for k, v in data_sources.items() if v]
    if available_sources:
        context_parts.append(f"data_sources: {', '.join(available_sources)}")
    
    return f"{base_reason} [{', '.join(context_parts)}]"

def _fallback_to_safety_decision(current_replicas: int, 
                                min_replicas: int, 
                                max_replicas: int, 
                                reason: str) -> Dict[str, Any]:
    """
    Safety fallback decision when DQN cannot be used.
    Maintains current state unless constraints are violated.
    """
    
    # Ensure replicas are within bounds
    if current_replicas < min_replicas:
        target_replicas = min_replicas
        action = "scale_up"
        safety_reason = f"Safety fallback: scaling to minimum replicas ({min_replicas})"
    elif current_replicas > max_replicas:
        target_replicas = max_replicas
        action = "scale_down"
        safety_reason = f"Safety fallback: scaling to maximum replicas ({max_replicas})"
    else:
        target_replicas = current_replicas
        action = "none"
        safety_reason = "Safety fallback: maintaining current replicas"
    
    full_reason = f"{safety_reason}. Original reason: {reason}"
    
    LOG.warning(f"Using safety fallback decision: {full_reason}")
    
    return {
        "action": action,
        "target_replicas": target_replicas,
        "reason": full_reason,
        "alerts": [f"DQN decision fallback: {reason}"],
        "ml_decision": {
            "model_type": "safety_fallback",
            "fallback_reason": reason
        }
    }

class DecisionEngine:
    """
    Enhanced decision engine wrapper for DQN-based autoscaling.
    """

    def __init__(self, global_health_status: Dict):
        """Initializes the DQN-based decision engine."""
        self.global_health_status = global_health_status
        
        # Initialize DQN agent
        try:
            _initialize_dqn_agent()
            self.global_health_status["ml_decision_engine"] = True
            LOG.info("DQN Decision Engine initialized successfully")
        except Exception as e:
            LOG.error(f"Failed to initialize DQN Decision Engine: {e}")
            self.global_health_status["ml_decision_engine"] = False
    
    def get_agent_metrics(self) -> Dict[str, Any]:
        """Get DQN agent performance metrics."""
        global _dqn_agent
        if _dqn_agent:
            return _dqn_agent.get_performance_metrics()
        return {}
    
    def save_model(self, filepath: str):
        """Save DQN model to file."""
        global _dqn_agent
        if _dqn_agent:
            _dqn_agent.save_model(filepath)
            LOG.info(f"DQN model saved to {filepath}")
    
    def set_training_mode(self, training: bool):
        """Set DQN agent training mode."""
        global _dqn_agent
        if _dqn_agent:
            _dqn_agent.set_training_mode(training)
