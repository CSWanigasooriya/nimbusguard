from collections import deque
from config import TrainingExperience

class ReplayBuffer:
    """Simple replay buffer for combined DQN training."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, experience: TrainingExperience):
        """Add experience to buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        """Sample random batch of experiences."""
        import random
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return batch

    def __len__(self):
        return len(self.buffer)
