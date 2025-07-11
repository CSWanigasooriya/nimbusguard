# NimbusGuard - AI-Powered Kubernetes Autoscaling

An intelligent cloud-native autoscaling system that uses **Deep Q-Learning (DQN)** and **Large Language Model (LLM) validation** to make smart scaling decisions for Kubernetes workloads.

## 🧠 What Makes NimbusGuard Special?

Unlike traditional autoscaling that relies on simple CPU/memory thresholds, NimbusGuard uses:
- **🤖 AI Decision Engine**: DQN neural network trained on 11 scientifically selected features
- **🛡️ LLM Safety Validator**: GPT-powered validation of scaling decisions
- **📊 Explainable AI**: Detailed reasoning for every scaling decision
- **🔄 Continuous Learning**: Real-time model improvement from system feedback

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[User/Load Generator] -->|HTTP Requests| B[Consumer Service]
    B -->|Metrics| C[Prometheus]
    C -->|Query Metrics| D[DQN Adapter]
    D -->|AI Decision| E[LLM Validator]
    E -->|Validation| F[KEDA ScaledObject]
    F -->|Scale Decision| G[Kubernetes HPA]
    G -->|Pod Scaling| H[Consumer Pods]
    
    subgraph "Monitoring Stack"
        I[Beyla] -->|Auto-instrumentation| B
        J[Alloy] -->|Collect| C
        K[Grafana] -->|Visualize| C
    end
    
    subgraph "AI Decision Engine"
        D -->|Feature Engineering| L[11 Selected Features]
        L -->|DQN Model| M[Q-Values]
        M -->|Action| N[Scale Up/Down/Same]
        N -->|Validation| E
    end
    
    style D fill:#e1f5fe
    style E fill:#f3e5f5
    style F fill:#e8f5e8
    style B fill:#fff3e0
```

---

## 🔄 Intelligent Decision Workflow

```mermaid
sequenceDiagram
    participant U as User/Load
    participant C as Consumer Service
    participant P as Prometheus
    participant D as DQN Adapter
    participant L as LLM Validator
    participant K as KEDA
    participant H as Kubernetes HPA
    
    Note over D: Timer triggers every 30s
    
    D->>P: Query 11 selected metrics
    P-->>D: Return current metrics
    
    Note over D: Feature Engineering
    D->>D: Scale & process features
    D->>D: DQN model inference
    D->>D: Generate Q-values
    
    Note over D: Explainable AI
    D->>D: Calculate confidence & risk
    D->>D: Generate reasoning factors
    
    D->>L: Send DQN recommendation
    Note over L: LLM validates using<br/>Kubernetes tools
    L->>L: Check cluster state
    L->>L: Assess safety & risks
    L-->>D: Validation result
    
    D->>D: Final decision (1-3 replicas)
    D->>K: Update desired_replicas metric
    
    K->>P: Query DQN metrics API
    P-->>K: Return desired_replicas
    K->>H: Create/Update HPA
    H->>H: Scale pods accordingly
    
    Note over D: Learning Loop
    D->>D: Wait 60s for stabilization
    D->>P: Observe new metrics
    D->>D: Calculate reward
    D->>D: Store experience for training
```

---

## 🧠 DQN Neural Network Architecture

```mermaid
graph LR
    subgraph "Input Layer"
        A1[avg_response_time]
        A2[http_request_duration_sum]
        A3[process_memory_bytes]
        A4[network_queue_length]
        A5[...7 more features]
    end
    
    subgraph "DQN Neural Network"
        B1[Hidden Layer 1<br/>512 neurons]
        B2[Hidden Layer 2<br/>256 neurons]
        B3[Hidden Layer 3<br/>128 neurons]
    end
    
    subgraph "Output Layer"
        C1[Scale Down<br/>Q-Value]
        C2[Keep Same<br/>Q-Value]
        C3[Scale Up<br/>Q-Value]
    end
    
    subgraph "Decision Logic"
        D1[Epsilon-Greedy<br/>Exploration]
        D2[Confidence<br/>Calculation]
        D3[Risk<br/>Assessment]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    A4 --> B1
    A5 --> B1
    
    B1 --> B2
    B2 --> B3
    B3 --> C1
    B3 --> C2
    B3 --> C3
    
    C1 --> D1
    C2 --> D1
    C3 --> D1
    
    D1 --> D2
    D2 --> D3
    
    style B1 fill:#e3f2fd
    style B2 fill:#e3f2fd
    style B3 fill:#e3f2fd
    style C1 fill:#ffebee
    style C2 fill:#fff3e0
    style C3 fill:#e8f5e8
```

---

## 🔍 Explainable AI Pipeline

```mermaid
graph TD
    subgraph "Data Collection"
        A[Prometheus Metrics] -->|11 Features| B[Feature Engineering]
        B -->|Scaled Features| C[DQN Model]
    end
    
    subgraph "AI Decision Pipeline"
        C -->|Q-Values| D[Epsilon-Greedy Strategy]
        D -->|Action + Confidence| E[Explainable AI Engine]
        E -->|Reasoning + Risk| F[LLM Validator]
    end
    
    subgraph "Validation & Safety"
        F -->|Safety Check| G{Approved?}
        G -->|Yes| H[Execute Scaling]
        G -->|No| I[Override with Caution]
        H --> J[Update KEDA Metrics]
        I --> J
    end
    
    subgraph "Learning & Feedback"
        J -->|Wait 60s| K[Observe Results]
        K -->|Calculate Reward| L[Store Experience]
        L -->|Background Training| M[Update DQN Model]
        M -->|Improved Model| C
    end
    
    subgraph "Explainable AI Features"
        N[Decision Confidence]
        O[Risk Assessment]
        P[Reasoning Factors]
        Q[Q-Value Analysis]
        R[Audit Trail]
    end
    
    E --> N
    E --> O
    E --> P
    E --> Q
    E --> R
    
    style C fill:#e1f5fe
    style E fill:#f3e5f5
    style F fill:#fff3e0
    style M fill:#e8f5e8
```

---

## 🚀 Quick Start

### Prerequisites
- Docker Desktop with Kubernetes enabled
- `make`, `helm`, `kubectl` installed

### Development Setup

**One-command deployment:**
```bash
make dev
```

This command will:
1. 🔨 Build all Docker images
2. 🔍 Check if KEDA is installed (auto-install if missing)
3. 🚀 Deploy the complete AI-powered stack

**First-time setup:**
```bash
make setup  # Install/update all required tools
make dev    # Deploy the application
```

### What Gets Deployed

- **🤖 DQN Adapter**: AI decision engine with explainable reasoning
- **🛡️ LLM Validator**: GPT-powered safety validation using MCP
- **📊 Consumer Service**: FastAPI application with load simulation
- **⚖️ KEDA Autoscaling**: Event-driven scaling based on AI decisions
- **📈 Monitoring Stack**: Prometheus, Grafana, Loki, Tempo, Alloy
- **🔍 Observability**: Beyla for automatic instrumentation

---

## 🎯 Key Features

### 🧠 **Intelligent Decision Making**
- **Deep Q-Learning**: Neural network trained on 11 scientifically selected features
- **Feature Engineering**: Advanced statistical analysis of system metrics
- **Exploration vs Exploitation**: Epsilon-greedy strategy for optimal learning

### 🛡️ **Safety & Validation**
- **LLM Validator**: GPT-4-turbo validates every scaling decision using real cluster data
- **Risk Assessment**: Automatic risk level calculation (low/medium/high)
- **Safety Override**: System can proceed with caution if LLM is unavailable

### 📊 **Explainable AI**
- **Decision Transparency**: Detailed reasoning for every scaling action
- **Confidence Metrics**: Q-value analysis and decision confidence scoring
- **Audit Trails**: Complete decision history with unique IDs
- **Real-time Logging**: Comprehensive reasoning logs for monitoring

### 🔄 **Continuous Learning**
- **Experience Replay**: Stores and learns from past scaling decisions
- **Reward Function**: Multi-factor reward considering latency, cost, and stability
- **Model Updates**: Background training with MinIO model persistence
- **Research Outputs**: Automated generation of training analytics

### 💰 **Cost Optimization**
- **Powerful LLM**: Uses GPT-4-turbo with 128K context for complex reasoning
- **Smart Scaling**: Considers resource costs in scaling decisions
- **Replica Optimization**: Balances performance and resource efficiency

---

## 🔧 Advanced Configuration

### Environment Variables

**AI Model Configuration:**
```bash
AI_MODEL=gpt-4-turbo                # Large context LLM model (128K tokens)
AI_TEMPERATURE=0.1                  # Low temperature for consistent reasoning
ENABLE_DETAILED_REASONING=true      # Enable comprehensive AI reasoning logs
REASONING_LOG_LEVEL=INFO            # Log level for AI reasoning
```

**DQN Training Parameters:**
```bash
EPSILON_START=0.3                   # Initial exploration rate
EPSILON_END=0.05                    # Final exploration rate
EPSILON_DECAY=0.995                 # Exploration decay rate
BATCH_SIZE=32                       # Training batch size
GAMMA=0.99                          # Discount factor
```

**System Configuration:**
```bash
STABILIZATION_PERIOD_SECONDS=60     # Wait time after scaling
EVALUATION_INTERVAL=300             # Research output generation interval
ENABLE_EVALUATION_OUTPUTS=true      # Enable research analytics
```

---

## 📊 Monitoring & Observability

### Access Services
```bash
make forward    # Port forward all services
```

- **🤖 DQN Adapter Metrics**: http://localhost:8001/api/v1/dqn-metrics
- **📊 Consumer Service**: http://localhost:8000
- **📈 Prometheus**: http://localhost:9090  
- **📊 Grafana**: http://localhost:3000 (admin/admin)

### Key Metrics to Monitor

**AI Decision Metrics:**
- `nimbusguard_dqn_desired_replicas`: Current DQN recommendation
- Decision confidence levels in logs
- LLM validation success rate
- Q-value distributions

**System Performance:**
- Response time trends
- Memory usage patterns
- Scaling frequency
- Reward function values

---

## 🧪 Load Testing & Validation

```bash
make load-test-light    # Quick validation (gentle load)
make load-test-medium   # Moderate scaling test
make load-test-heavy    # Immediate scaling trigger
make load-status        # Check autoscaling status
```

### Understanding AI Decisions

Watch the DQN adapter logs to see:
```bash
kubectl logs -n nimbusguard deployment/dqn-adapter -f
```

**Example AI Decision Log:**
```
🧠 AI DECISION REASONING ANALYSIS
⏰ Timestamp: 2025-06-27T14:57:33.919702
🎯 Recommended Action: Scale Up
🔍 Exploration Strategy: exploitation
⚠️ Risk Assessment: HIGH
📊 Decision Confidence: HIGH (gap: 363.184)
🎲 Exploration Rate: 0.291
🧮 Q-Value Analysis:
   Scale Down: 560.751 (confidence: high)
   Keep Same: -173.705 (confidence: low)
   Scale Up: 197.566 (confidence: high)
💭 Key Reasoning Factors:
   • Current system state: 1 risk factors detected
   • Latency status: HIGH LATENCY DETECTED
   • Memory status: MEMORY USAGE NORMAL
```

---

## 🔬 Research & Development

### Feature Selection
The system uses 11 scientifically selected features from advanced statistical analysis:
- `avg_response_time` - Primary performance indicator
- `http_request_duration_*` - Request processing metrics
- `process_resident_memory_bytes` - Memory utilization
- `node_network_*` - Network performance indicators
- Statistical derivatives (moving averages, deviations)

### Model Architecture
- **Input Layer**: 11 features (scientifically selected)
- **Hidden Layers**: 512 → 256 → 128 neurons with dropout and batch normalization
- **Output Layer**: 3 actions (Scale Down, Keep Same, Scale Up)
- **Training**: Double DQN with experience replay and target networks

---

## 📋 Available Commands

```bash
make help           # Show all available commands
make dev            # Build and deploy to development  
make clean          # Delete all resources
make keda-install   # Manually install KEDA
make load-status    # View autoscaling status
make logs-dqn       # View DQN adapter logs
make logs-validator # View LLM validator logs
```

---

## 🎓 Educational Value

NimbusGuard demonstrates:
- **Reinforcement Learning** in production systems
- **Explainable AI** for critical infrastructure decisions
- **LLM integration** for safety validation
- **MLOps practices** with model persistence and continuous learning
- **Cloud-native AI** architecture patterns

Perfect for:
- 🎓 **Students** learning AI/ML in cloud environments
- 🔬 **Researchers** studying autoscaling algorithms
- 🏗️ **Engineers** implementing intelligent infrastructure
- 🚀 **Organizations** exploring AI-powered operations

---

*Ready to experience the future of intelligent autoscaling! 🚀🤖* 