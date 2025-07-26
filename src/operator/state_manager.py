import threading
from collections import deque
# ✨ Import the DQNAgent class
from dqn_agent import DQNAgent

class OperatorState:
    """
    A singleton class to hold and manage the operator's shared state.
    """
    def __init__(self):
        # --- Reinforcement Learning ---
        # Define the state and action sizes for the DQN
        STATE_SIZE = 4  # [pred_mem, curr_cpu, curr_mem, replicas]
        ACTION_SIZE = 3 # 0: Do Nothing, 1: Scale Up, 2: Scale Down
        # ✨ Instantiate the DQN agent and store it in the state
        self.dqn_agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
        self.last_state = None
        self.last_action = None

        # --- Forecasting ---
        self.sequence_length = 10
        self.pod_history = {}
        self.polling_threads = {}
        self.cpu_model = None
        self.cpu_scaler = None
        self.memory_model = None
        self.memory_scaler = None
        self.models_loaded = threading.Event()
        self.prometheus_server_started = False
# Create the single, global instance of the state.
state = OperatorState()