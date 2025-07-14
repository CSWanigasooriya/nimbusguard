# NimbusGuard Proactive Scaling Operator

A clean, focused Kubernetes operator for proactive scaling using **Kopf + LangGraph + LSTM + DQN** architecture.

## Architecture

```
src/operator/
├── prometheus/          # Prometheus data collection
│   ├── client.py       # Simple Prometheus client
│   └── queries.py      # Predefined queries for 9 selected features
├── forecasting/         # LSTM-based prediction
│   ├── model.py        # LSTM model implementation
│   ├── predictor.py    # Main forecasting logic
│   └── data_prep.py    # Data preprocessing utilities
├── dqn/                # DQN agent with forecast integration
│   ├── agent.py        # DQN agent using forecast + current metrics
│   ├── model.py        # DQN neural network
│   ├── training.py     # DQN training logic
│   └── rewards.py      # Reward system for scaling decisions
├── workflow/           # LangGraph decision workflow
│   ├── graph.py        # Main LangGraph workflow definition
│   ├── nodes.py        # Individual workflow nodes
│   └── state.py        # Workflow state management
├── k8s/                # Kubernetes integration
│   ├── client.py       # Kubernetes API client
│   ├── operator.py     # Kopf-based operator
│   └── scaler.py       # Deployment scaling logic
├── metrics/            # Operator monitoring
│   ├── server.py       # Metrics HTTP server
│   └── collector.py    # Metrics collection
├── config/             # Simple configuration
│   └── settings.py     # Environment-based config
└── main.py             # Main operator entry point
```

## Intelligent Decision Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Kopf Operator  │───▶│  LangGraph Flow  │───▶│ Kubernetes API  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            ┌──────────────┐    ┌──────────────┐
            │ LSTM Forecast│    │  DQN Agent   │
            │              │───▶│              │
            │ • 10min ahead│    │ • Uses       │
            │ • 9 features │    │   forecast   │
            │ • Confidence │    │ • Current    │
            └──────────────┘    │   metrics    │
                               │ • Proactive  │
                               │   decisions  │
                               └──────────────┘
```

## LangGraph Workflow Nodes

1. **`collect_metrics`**: Fetch current Prometheus metrics
2. **`generate_forecast`**: LSTM prediction for next 10 minutes
3. **`dqn_decision`**: DQN agent makes scaling decision using forecast + current state
4. **`validate_decision`**: Safety checks and constraints
5. **`execute_scaling`**: Apply scaling to Kubernetes deployment
6. **`update_rewards`**: Train DQN based on scaling outcomes

## Enhanced DQN Features

- **Forecast-Enhanced State**: DQN receives both current metrics AND forecast predictions
- **Proactive Training**: Learns to scale before issues occur
- **Confidence-Based Exploration**: Higher forecast confidence = more decisive actions
- **Multi-horizon Rewards**: Rewards based on both immediate and predicted future performance

## Selected Features (9 metrics)

1. `process_cpu_seconds_total_rate` - CPU usage rate
2. `process_resident_memory_bytes` - Memory usage
3. `process_virtual_memory_bytes` - Virtual memory
4. `http_request_duration_seconds_sum_rate` - Request latency
5. `http_requests_total_rate` - Request rate
6. `http_request_duration_seconds_count_rate` - Request count rate
7. `process_open_fds` - File descriptors
8. `http_response_size_bytes_sum_rate` - Response size rate
9. `http_request_size_bytes_count_rate` - Request size rate

## Operator Workflow

1. **Kopf Event**: Deployment changes trigger operator
2. **LangGraph Execution**:
    - Collect current metrics
    - Generate LSTM forecast
    - DQN makes proactive scaling decision
    - Validate and execute
3. **Continuous Learning**: DQN trains on outcomes
4. **Background Retraining**: LSTM model updates every 60 minutes

## Goals

- **Proactive**: Scale before issues occur using forecasts
- **Intelligent**: DQN learns optimal scaling patterns
- **Simple**: Clean code, minimal complexity
- **Focused**: Use only proven metrics
- **Reliable**: Robust error handling and fallbacks
- **Event-Driven**: Kopf handles Kubernetes events efficiently 