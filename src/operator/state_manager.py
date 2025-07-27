import threading
from collections import deque
# ✨ Import the DQNAgent class and configuration
from dqn_agent import DQNAgent
from config import system_config

class OperatorState:
    """
    A singleton class to hold and manage the operator's shared state.
    """
    def __init__(self):
        # --- Reinforcement Learning ---
        # Define the state and action sizes for the DQN
        STATE_SIZE = 4  # [pred_mem, curr_cpu, curr_mem, replicas]
        ACTION_SIZE = 3 # 0: Do Nothing, 1: Scale Up, 2: Scale Down
        # ✨ Instantiate the DQN agent and store it in the state (config is loaded automatically)
        self.dqn_agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
        self.last_state = None
        self.last_action = None

        # --- Forecasting ---
        # Use configurable LSTM sequence length
        self.sequence_length = system_config.lstm_sequence_length
        self.pod_history = {}  # Per-pod CPU history (still needed for CPU rate calculations)
        # Store structured data: [{'timestamp': str, 'pod_name': str, 'memory_bytes': int}, ...]
        self.global_memory_history = deque(maxlen=self.sequence_length * 5)  # Store more data points for multiple pods
        self.polling_threads = {}
        self.memory_model = None       # Raw Keras model
        self.memory_scaler = None      # Feature scaler
        self.target_scaler = None      # Target scaler
        self.models_loaded = threading.Event()
        self.prometheus_server_started = False

# Create a singleton instance
state = OperatorState()