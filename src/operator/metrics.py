from prometheus_client import Gauge, Counter

# --- Metric Definitions for NimbusGuard DQN Autoscaler ---

# Note: Descriptions are added for clarity in the /metrics endpoint.

# --- State & Replica Metrics ---
NIMBUSGUARD_DESIRED_REPLICAS = Gauge(
    'nimbusguard_dqn_desired_replicas',
    'The number of replicas the DQN agent has decided upon.'
)
NIMBUSGUARD_CURRENT_REPLICAS = Gauge(
    'nimbusguard_current_replicas',
    'The current number of replicas for the deployment.'
)

# --- DQN Agent & Training Metrics ---
DQN_TRAINING_LOSS = Gauge(
    'dqn_training_loss',
    'The cumulative total of all training loss values from DQN agent training steps.'
)
DQN_EPSILON_VALUE = Gauge(
    'dqn_epsilon_value',
    'The current value of epsilon, representing the exploration rate.'
)
DQN_REPLAY_BUFFER_SIZE = Gauge(
    'dqn_replay_buffer_size',
    'The current number of experiences stored in the replay buffer.'
)
DQN_REWARD_TOTAL = Gauge(
    'dqn_reward_total',
    'The cumulative total of all reward values (positive and negative) received by the DQN agent.'
)

# --- Action & Decision Counters ---
DQN_ACTION_SCALE_UP_TOTAL = Counter(
    'dqn_action_scale_up_total',
    'Total number of times the agent chose to scale up.'
)
DQN_ACTION_SCALE_DOWN_TOTAL = Counter(
    'dqn_action_scale_down_total',
    'Total number of times the agent chose to scale down.'
)
DQN_ACTION_KEEP_SAME_TOTAL = Counter(
    'dqn_action_keep_same_total',
    'Total number of times the agent chose to do nothing.'
)
DQN_EXPLORATION_ACTIONS_TOTAL = Counter(
    'dqn_exploration_actions_total',
    'Total number of actions taken randomly (exploration).'
)
DQN_EXPLOITATION_ACTIONS_TOTAL = Counter(
    'dqn_exploitation_actions_total',
    'Total number of actions taken based on policy (exploitation).'
)
DQN_EXPERIENCES_ADDED_TOTAL = Counter(
    'dqn_experiences_added_total',
    'Total number of experiences added to the replay buffer.'
)
DQN_TRAINING_STEPS_TOTAL = Counter(
    'dqn_training_steps_total',
    'Total number of times the agent has trained on a batch.'
)

# --- Forecasting Metrics ---
LSTM_FORECAST_MEMORY_BYTES = Gauge(
    'lstm_forecast_memory_bytes',
    'The predicted memory usage in bytes from the LSTM model.'
)

# --- Q-Value Metrics ---
DQN_Q_VALUE_SCALE_UP = Gauge(
    'dqn_q_value_scale_up',
    'The predicted Q-value for the scale-up action.'
)
DQN_Q_VALUE_SCALE_DOWN = Gauge(
    'dqn_q_value_scale_down',
    'The predicted Q-value for the scale-down action.'
)
DQN_Q_VALUE_KEEP_SAME = Gauge(
    'dqn_q_value_keep_same',
    'The predicted Q-value for the keep-same action.'
)

