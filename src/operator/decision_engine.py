import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional
from collections import deque
from state_manager import state
import metrics

class DecisionEngine:
    """Handles DQN agent decision making and learning, managing state vectors and action execution."""
    
    def __init__(self):
        """Initialize the decision engine."""
        self.action_names = {
            0: "Keep Same",
            1: "Scale Up", 
            2: "Scale Down"
        }
        logging.info("DecisionEngine initialized")
    
    def construct_state_vector(self, current_cpu_util: float, current_mem_util: float, 
                             predicted_mem_util: float, current_replicas: int) -> np.ndarray:
        """
        Construct the state vector for DQN input.
        
        Args:
            current_cpu_util: Current CPU utilization percentage
            current_mem_util: Current memory utilization percentage  
            predicted_mem_util: Predicted memory utilization percentage
            current_replicas: Current number of replicas
            
        Returns:
            State vector as numpy array
        """
        state_list = [predicted_mem_util, current_cpu_util, current_mem_util, current_replicas]
        state_vector = np.reshape(state_list, [1, state.dqn_agent.state_size])
        
        logging.debug(f"Constructed state vector: {state_list}")
        return state_vector
    
    def make_decision(self, state_vector: np.ndarray) -> int:
        """
        Make a scaling decision using the DQN agent.
        
        Args:
            state_vector: State vector for DQN input
            
        Returns:
            Action integer (0: keep same, 1: scale up, 2: scale down)
        """
        # Use DQN agent to select action - agent handles its own metrics internally
        action = state.dqn_agent.act(state_vector)
        
        # Update decision-specific metrics (decision engine responsibility)
        if action == 0:
            metrics.DQN_ACTION_KEEP_SAME_TOTAL.inc()
        elif action == 1:
            metrics.DQN_ACTION_SCALE_UP_TOTAL.inc()
        elif action == 2:
            metrics.DQN_ACTION_SCALE_DOWN_TOTAL.inc()
        
        action_name = self.get_action_name(action)
        logging.info(f"DQN decision: {action} ({action_name})")
        
        return action
    
    def learn_from_experience(self, reward: float, current_state_vector: np.ndarray):
        """
        Update the DQN agent with the calculated reward.
        
        Args:
            reward: Calculated reward for the previous action
            current_state_vector: Current state vector
        """
        if state.last_state is not None and state.last_action is not None:
            # Store experience in agent memory - agent handles buffer metrics internally
            state.dqn_agent.remember(state.last_state, state.last_action, reward, 
                                   current_state_vector, False)
            
            # Update reward metric (decision engine responsibility)
            metrics.DQN_REWARD_TOTAL.set(reward)
            
            # Train the agent - agent handles training metrics internally, pass reward for save-on-improvement
            loss = state.dqn_agent.replay(current_reward=reward)
            
            if loss is not None:
                logging.debug(f"DQN training loss: {loss:.4f}")
            
            logging.info(f"DQN learning: reward={reward:.2f}, training_steps={state.dqn_agent.training_steps}")
    
    def prepare_for_next_cycle(self, current_state_vector: np.ndarray, current_action: int):
        """
        Prepare state for the next decision cycle.
        
        Args:
            current_state_vector: Current state vector to store as last_state
            current_action: Current action to store as last_action
        """
        state.last_state = current_state_vector
        state.last_action = current_action
        
        logging.debug(f"Prepared for next cycle: action={current_action}, state_shape={current_state_vector.shape}")
    
    def get_action_name(self, action: int) -> str:
        """
        Get human-readable name for action.
        
        Args:
            action: Action integer
            
        Returns:
            Human-readable action name
        """
        return self.action_names.get(action, f"Unknown({action})")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about the DQN agent state.
        
        Returns:
            Dictionary with agent information
        """
        return {
            'epsilon': state.dqn_agent.epsilon,
            'memory_size': len(state.dqn_agent.memory),
            'memory_capacity': state.dqn_agent.memory.maxlen,
            'training_steps': getattr(state.dqn_agent, 'training_steps', 0),
            'best_avg_reward': getattr(state.dqn_agent, 'best_avg_reward', float('-inf')),
            'config': state.dqn_agent.get_config()
        }
    
    def get_decision_history(self, limit: int = 10) -> list:
        """
        Get recent decision history.
        
        Args:
            limit: Maximum number of recent decisions to return
            
        Returns:
            List of recent decisions
        """
        # This could be enhanced to track decision history
        # For now, return basic state information
        return [{
            'last_action': state.last_action,
            'last_state': state.last_state.tolist() if state.last_state is not None else None,
            'current_epsilon': state.dqn_agent.epsilon,
            'training_steps': getattr(state.dqn_agent, 'training_steps', 0)
        }]
    
    def record_decision(self, state_vector: np.ndarray, action: int, reason: str = ""):
        """
        Record a decision for history tracking.
        
        Args:
            state_vector: State vector used for decision
            action: Action taken
            reason: Reason for the decision (optional)
        """
        # This could be enhanced to maintain a decision history
        logging.debug(f"Decision recorded: action={action}, reason={reason}")
    
    def is_ready(self) -> Tuple[bool, str]:
        """
        Check if the decision engine is ready to make decisions.
        
        Returns:
            Tuple of (is_ready, message)
        """
        if state.dqn_agent is None:
            return False, "DQN agent not initialized"
        
        if state.dqn_agent.model is None:
            return False, "DQN model not built"
        
        return True, "Decision engine ready"
    
    def reset_episode(self):
        """Reset episode state."""
        state.last_state = None
        state.last_action = None
        logging.info("Episode state reset")
    
    def validate_state_vector(self, state_vector: np.ndarray) -> Tuple[bool, str]:
        """
        Validate the state vector for correctness.
        
        Args:
            state_vector: State vector to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if state_vector is None:
            return False, "State vector is None"
        
        if not isinstance(state_vector, np.ndarray):
            return False, "State vector is not a numpy array"
        
        expected_shape = (1, state.dqn_agent.state_size)
        if state_vector.shape != expected_shape:
            return False, f"State vector shape {state_vector.shape} != expected {expected_shape}"
        
        # Check for invalid values (NaN, inf)
        if np.any(np.isnan(state_vector)) or np.any(np.isinf(state_vector)):
            return False, "State vector contains NaN or infinite values"
        
        return True, "State vector is valid"
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training statistics directly from the agent.
        
        Returns:
            Dictionary with training statistics
        """
        try:
            agent = state.dqn_agent
            
            # Get stats directly from agent properties to avoid metric conflicts
            stats = {
                'training_steps': getattr(agent, 'training_steps', 0),
                'epsilon': agent.epsilon,
                'memory_size': len(getattr(agent, 'memory', [])),
                'memory_capacity': getattr(agent, 'memory', deque(maxlen=1)).maxlen,
                'best_avg_reward': getattr(agent, 'best_avg_reward', float('-inf')),
                'recent_rewards_count': len(getattr(agent, 'recent_rewards', [])),
                'storage_status': agent.get_storage_status() if hasattr(agent, 'get_storage_status') else None
            }
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting training stats: {e}")
            return {'error': str(e)} 