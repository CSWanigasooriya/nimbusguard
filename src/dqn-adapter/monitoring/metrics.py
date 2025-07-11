from prometheus_client import Counter, Gauge


# --- Prometheus Metrics ---
# These are defined globally and will be exposed by kopf's built-in metrics server.
DESIRED_REPLICAS_GAUGE = Gauge('nimbusguard_dqn_desired_replicas', 'Desired replicas calculated by the DQN adapter')
CURRENT_REPLICAS_GAUGE = Gauge('nimbusguard_current_replicas', 'Current replicas of the target deployment as seen by the adapter')

# DQN Training Metrics
DQN_TRAINING_LOSS_GAUGE = Gauge('dqn_training_loss', 'Current DQN training loss')
DQN_EPSILON_GAUGE = Gauge('dqn_epsilon_value', 'Current exploration epsilon value')
DQN_BUFFER_SIZE_GAUGE = Gauge('dqn_replay_buffer_size', 'Current replay buffer size')
DQN_TRAINING_STEPS_GAUGE = Gauge('dqn_training_steps_total', 'Total training steps completed')

# DQN Decision Metrics
DQN_DECISION_CONFIDENCE_GAUGE = Gauge('dqn_decision_confidence_avg', 'Average decision confidence')
DQN_Q_VALUE_SCALE_UP_GAUGE = Gauge('dqn_q_value_scale_up', 'Q-value for scale up action')
DQN_Q_VALUE_SCALE_DOWN_GAUGE = Gauge('dqn_q_value_scale_down', 'Q-value for scale down action')
DQN_Q_VALUE_KEEP_SAME_GAUGE = Gauge('dqn_q_value_keep_same', 'Q-value for keep same action')

# DQN Action Counters
DQN_ACTION_SCALE_UP_COUNTER = Counter('dqn_action_scale_up_total', 'Total scale up actions taken')
DQN_ACTION_SCALE_DOWN_COUNTER = Counter('dqn_action_scale_down_total', 'Total scale down actions taken')
DQN_ACTION_KEEP_SAME_COUNTER = Counter('dqn_action_keep_same_total', 'Total keep same actions taken')
DQN_EXPLORATION_COUNTER = Counter('dqn_exploration_actions_total', 'Total exploration actions taken')
DQN_EXPLOITATION_COUNTER = Counter('dqn_exploitation_actions_total', 'Total exploitation actions taken')
DQN_EXPERIENCES_COUNTER = Counter('dqn_experiences_added_total', 'Total experiences added to replay buffer')
DQN_DECISIONS_COUNTER = Counter('dqn_decisions_total', 'Total decisions made by DQN')

# Reward Component Metrics
DQN_REWARD_TOTAL_GAUGE = Gauge('dqn_reward_total', 'Total reward received')
DQN_REWARD_PERFORMANCE_GAUGE = Gauge('dqn_reward_performance_component', 'Performance component of reward')
DQN_REWARD_RESOURCE_GAUGE = Gauge('dqn_reward_resource_component', 'Resource component of reward')
DQN_REWARD_HEALTH_GAUGE = Gauge('dqn_reward_health_component', 'Health component of reward')
DQN_REWARD_COST_GAUGE = Gauge('dqn_reward_cost_component', 'Cost component of reward')
