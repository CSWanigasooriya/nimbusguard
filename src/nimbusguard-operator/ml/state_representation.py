# engine/ml/state_representation.py - Optimized for a 7-feature observability vector
# ============================================================================

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List

import numpy as np


class ScalingActions(Enum):
    """Available scaling actions for the DQN agent."""
    SCALE_DOWN_2 = 0
    SCALE_DOWN_1 = 1
    NO_ACTION = 2
    SCALE_UP_1 = 3
    SCALE_UP_2 = 4

    def get_replica_change(self) -> int:
        """Returns the replica change for this action."""
        return self.value - 2


def get_feature_dimension() -> int:
    """Returns the total feature dimension for the DQN input."""
    return 11


@dataclass
class EnvironmentState:
    """
    A lean state representation based on 7 core observability features plus context.
    The total dimension of the DQN input is 11.
    """
    # Core metadata
    timestamp: float
    current_replicas: int
    min_replicas: int
    max_replicas: int

    # Core 7 observability features
    feature_vector: List[float]
    feature_names: List[str] = field(default_factory=list)

    # Additional ML context
    time_since_last_scale: float = 0.0
    recent_scaling_actions: List[int] = field(default_factory=lambda: [2] * 5)

    def to_dqn_input(self) -> np.ndarray:
        """
        Convert to an 11-dimensional DQN-ready input vector.
        Contains 7 core observability features plus 4 scaling context features.
        """
        # --- (7 features) --- Core observability metrics
        ml_features = list(self.feature_vector)

        # --- (4 features) --- Add scaling context
        ml_features.extend([
            self.current_replicas / max(self.max_replicas, 1),  # Normalized current replicas
            self.min_replicas / max(self.max_replicas, 1),  # Normalized min replicas
            self.max_replicas / 100.0,  # Normalize against a hypothetical max of 100
            (self.current_replicas - self.min_replicas) / (self.max_replicas - self.min_replicas + 1e-6)
        ])

        # Total: 7 + 4 = 11 features
        return np.array(ml_features, dtype=np.float32)

    @classmethod
    def from_observability_data(cls,
                                unified_state: Dict[str, Any],
                                current_replicas: int,
                                min_replicas: int = 1,
                                max_replicas: int = 10,
                                time_since_last_scale: float = 0.0,
                                recent_actions: List[int] = None) -> 'EnvironmentState':
        """Create EnvironmentState from the new focused ObservabilityCollector output."""
        # Ensure feature vector has exactly 7 elements, padding if necessary
        feature_vector = unified_state.get("feature_vector", [0.0] * 7)
        if len(feature_vector) != 7:
            feature_vector = (feature_vector + [0.0] * 7)[:7]

        return cls(
            timestamp=time.time(),
            current_replicas=current_replicas,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            feature_vector=feature_vector,
            feature_names=unified_state.get("feature_names", []),
            time_since_last_scale=time_since_last_scale,
            recent_scaling_actions=recent_actions or [2] * 5
        )

    def is_action_valid(self, action: ScalingActions) -> bool:
        """Check if the proposed action is within replica bounds."""
        new_replicas = self.current_replicas + action.get_replica_change()
        return self.min_replicas <= new_replicas <= self.max_replicas

    def get_valid_actions(self) -> List[ScalingActions]:
        """Get list of valid actions given current constraints."""
        return [action for action in ScalingActions if self.is_action_valid(action)]
