# engine/ml/state_representation.py
# ============================================================================
# State Representation for DQN using Existing Observability Data
# ============================================================================

import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np
from enum import Enum

class ScalingActions(Enum):
    """Available scaling actions for the DQN agent"""
    SCALE_DOWN_2 = 0    # -2 replicas
    SCALE_DOWN_1 = 1    # -1 replica  
    NO_ACTION = 2       # 0 replicas (maintain current)
    SCALE_UP_1 = 3      # +1 replica
    SCALE_UP_2 = 4      # +2 replicas
    
    def get_replica_change(self) -> int:
        """Returns the replica change for this action"""
        return {
            ScalingActions.SCALE_DOWN_2: -2,
            ScalingActions.SCALE_DOWN_1: -1,
            ScalingActions.NO_ACTION: 0,
            ScalingActions.SCALE_UP_1: 1,
            ScalingActions.SCALE_UP_2: 2,
        }[self]

@dataclass
class EnvironmentState:
    """
    Enhanced state representation that works with your existing ObservabilityCollector
    """
    
    # Core metadata
    timestamp: float
    resource_name: str
    namespace: str
    current_replicas: int
    min_replicas: int
    max_replicas: int
    
    # Raw observability data (from your existing collector)
    feature_vector: List[float]
    feature_names: List[str] 
    health_score: float
    confidence_score: float
    
    # Additional ML context
    time_since_last_scale: float = 0.0
    recent_scaling_actions: List[int] = None  # Last 5 actions
    
    def __post_init__(self):
        """Initialize derived features"""
        if self.recent_scaling_actions is None:
            self.recent_scaling_actions = [2] * 5  # Default to NO_ACTION
            
        # Ensure feature vector is the right size
        if len(self.feature_vector) < 30:
            # Pad with zeros if needed
            self.feature_vector.extend([0.0] * (30 - len(self.feature_vector)))
        
        # Clip to exactly 30 features for consistency
        self.feature_vector = self.feature_vector[:30]
    
    def to_dqn_input(self) -> np.ndarray:
        """
        Convert to DQN-ready input tensor including temporal and scaling context
        """
        # Start with your existing 30-dimensional feature vector
        ml_features = list(self.feature_vector)
        
        # Add temporal features (4 features)
        current_time = time.time()
        time_of_day = (current_time % (24 * 3600)) / (24 * 3600)
        day_of_week = (current_time % (7 * 24 * 3600)) / (7 * 24 * 3600)
        
        ml_features.extend([
            time_of_day,
            day_of_week,
            min(self.time_since_last_scale / 3600.0, 1.0),  # Hours since last scale
            self.health_score,
        ])
        
        # Add scaling context features (8 features)
        ml_features.extend([
            self.current_replicas / 100.0,  # Normalized current replicas
            self.min_replicas / 100.0,      # Normalized min replicas
            self.max_replicas / 100.0,      # Normalized max replicas
            self.confidence_score,
        ])
        
        # Add recent scaling action history (5 features)
        normalized_actions = [action / 4.0 for action in self.recent_scaling_actions[-5:]]
        ml_features.extend(normalized_actions)
        
        # Total: 30 + 4 + 4 + 5 = 43 features
        return np.array(ml_features, dtype=np.float32)
    
    @classmethod
    def from_observability_data(cls, 
                               unified_state: Dict[str, Any],
                               resource_name: str,
                               namespace: str, 
                               current_replicas: int,
                               min_replicas: int = 1,
                               max_replicas: int = 10,
                               time_since_last_scale: float = 0.0,
                               recent_actions: Optional[List[int]] = None) -> 'EnvironmentState':
        """
        Create EnvironmentState from your existing ObservabilityCollector output
        """
        return cls(
            timestamp=time.time(),
            resource_name=resource_name,
            namespace=namespace,
            current_replicas=current_replicas,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            feature_vector=unified_state.get("feature_vector", [0.0] * 30),
            feature_names=unified_state.get("feature_names", []),
            health_score=unified_state.get("health_score", 0.0),
            confidence_score=unified_state.get("confidence_score", 0.0),
            time_since_last_scale=time_since_last_scale,
            recent_scaling_actions=recent_actions or [2] * 5  # Default to NO_ACTION
        )
    
    def get_feature_dimension(self) -> int:
        """Returns the total feature dimension for DQN input"""
        return 43  # 30 observability + 4 temporal + 4 context + 5 history
    
    def is_action_valid(self, action: ScalingActions) -> bool:
        """Check if the proposed action is within replica bounds"""
        new_replicas = self.current_replicas + action.get_replica_change()
        return self.min_replicas <= new_replicas <= self.max_replicas
    
    def get_valid_actions(self) -> List[ScalingActions]:
        """Get list of valid actions given current constraints"""
        valid_actions = []
        for action in ScalingActions:
            if self.is_action_valid(action):
                valid_actions.append(action)
        return valid_actions
    
    def update_recent_actions(self, new_action: ScalingActions):
        """Update the recent actions history"""
        self.recent_scaling_actions.append(new_action.value)
        if len(self.recent_scaling_actions) > 5:
            self.recent_scaling_actions.pop(0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging"""
        return {
            "timestamp": self.timestamp,
            "resource_name": self.resource_name,
            "namespace": self.namespace,
            "current_replicas": self.current_replicas,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "health_score": self.health_score,
            "confidence_score": self.confidence_score,
            "feature_dimension": self.get_feature_dimension(),
            "valid_actions": [action.name for action in self.get_valid_actions()],
            "time_since_last_scale": self.time_since_last_scale
        }
