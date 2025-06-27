# NimbusGuard DQN Adapter Service

This service is the intelligent core of the NimbusGuard autoscaling system. It combines the predictive power of a Deep Q-Network (DQN) model with real-time learning capabilities and robust execution through KEDA. It runs a continuous reconciliation loop to determine the optimal number of replicas for a target service and exposes this decision as a Prometheus metric.

The adapter uses **8 scientifically selected features** identified through advanced statistical analysis using 6 rigorous methods (Mutual Information, Random Forest, Correlation Analysis, RFECV, Statistical Significance, and VIF Analysis) from 894 real per-minute decision points, ensuring optimal performance and decision accuracy.

## 🧠 Explainable AI Features

The DQN adapter includes comprehensive **explainable AI capabilities** that provide detailed reasoning for every scaling decision:

### 🔍 Decision Transparency
- **Detailed Q-Value Analysis**: Every DQN decision includes Q-values for all actions with confidence scoring
- **Risk Assessment**: Automatic risk level evaluation (low/medium/high) based on system metrics
- **Confidence Metrics**: Decision confidence gaps and exploration vs exploitation tracking
- **Reasoning Factors**: Structured list of factors contributing to each decision

### 📊 Comprehensive Logging
- **AI Reasoning Logger**: Dedicated logger (`AI_Reasoning`) for decision explanations
- **Structured Decision Pipeline**: Clear audit trail showing DQN → LLM → Final Decision flow
- **Performance Metrics Analysis**: Automatic analysis of latency, memory, and network metrics
- **Expected Outcomes**: Prediction of scaling action results and risk mitigation strategies

### 🤖 LLM Validation with Structured Reasoning
- **Cost-Effective Model**: Uses `gpt-3.5-turbo` instead of `gpt-4o-mini` for 60-80% cost reduction
- **Structured JSON Responses**: LLM provides structured validation with confidence, risks, and benefits
- **Fallback Parsing**: Robust JSON parsing with intelligent text analysis fallbacks
- **Cluster State Integration**: Can query live Kubernetes state via MCP tools for validation

### 📋 Audit Trail & Compliance
- **Decision History**: Maintains rolling history of last 100 decisions with full context
- **Audit Trail IDs**: Unique identifiers for each decision for compliance tracking
- **Explainable Compliance**: Every decision marked as explainable, auditable, and reversible
- **Risk Mitigation Tracking**: Automatic identification and logging of risk mitigation measures

### Example Decision Log Output:
```
🧠 AI DECISION REASONING ANALYSIS
================================================================================
⏰ Timestamp: 2025-01-27T10:30:45.123Z
🎯 Recommended Action: Scale Up
🔍 Exploration Strategy: exploitation
⚠️ Risk Assessment: MEDIUM
📊 Decision Confidence: HIGH (gap: 0.245)
🎲 Exploration Rate: 0.127
🧮 Q-Value Analysis:
   Scale Down: -0.123 (confidence: low)
   Keep Same: 0.234 (confidence: medium)
   Scale Up: 0.479 (confidence: high)
💭 Key Reasoning Factors:
   • Current system state: 2 risk factors detected
   • Latency status: ELEVATED LATENCY
   • Memory status: MEMORY USAGE NORMAL
   • Decision confidence: high
   • 🔺 SCALE UP reasoning: System showing signs of stress
📈 Key Metrics:
   • Response Time: 245.3ms
   • Memory Usage: 892.1MB
   • Network Queue: 234
================================================================================
```

## 🏛️ Architecture

The adapter is built as a stateful, resilient agent using the **LangGraph** framework. It follows a modern, production-ready **combined architecture** that integrates real-time inference and continuous learning in a single component.

### Core Concepts Used:

*   **Functional API (`@entrypoint`)**: The main reconciliation logic is defined as a LangGraph `@entrypoint`. This provides a clean, modern, and traceable way to define the sequence of operations (tasks).
*   **Checkpointer (`MemorySaver`)**: The workflow is configured with a checkpointer. This is **critical for resilience**. If the adapter pod restarts, it can resume its state from the last successful step, preventing interruptions. It is also the prerequisite for advanced features like human-in-the-loop.
*   **Tools (DQN & MCP)**: The agent uses two primary types of tools:
    1.  **Local DQN Model**: A locally-loaded PyTorch DQN model that provides ultra-low latency (~1-5ms) scaling recommendations with real-time learning capabilities.
    2.  **MCP (Model Context Protocol) Server**: It uses the `langchain-mcp-adapters` library to dynamically load tools from a running MCP server. This gives the validation LLM secure, read-only access to the live Kubernetes cluster state.
*   **Agentic Validator (`create_react_agent`)**: The final validation step is not just a simple LLM call. It's a true LangChain agent that can reason and decide whether to use its MCP tools to gather more context before approving or denying the DQN's suggestion.
*   **Supervisor (Future Extension)**: The architecture is designed to be easily extended into a multi-agent system using a `Supervisor`. For instance, we could add a "Cost-Optimizer Agent" that runs on a slower loop and provides input to the main "Scaling Agent".

### Workflow (Reconciliation Loop)

On a configurable interval (`POLLING_INTERVAL`), the LangGraph entrypoint is invoked, which executes the following flow:

1.  **`get_live_metrics`**: Queries the Prometheus server for the latest application metrics (request rate, latency, etc.) and the current replica count of the target deployment.
2.  **`get_dqn_recommendation`**: Prepares the metrics into a feature vector, scales them using the training-time scaler, and runs local DQN inference to get a scaling action (`Scale Up`, `Scale Down`, `Keep Same`) with epsilon-greedy exploration and **detailed explainable reasoning**.
3.  **`validate_with_llm`**: Invokes the validator agent with **structured reasoning requirements**. The agent provides detailed validation with confidence scores, risk factors, benefits analysis, and alternative suggestions.
4.  **`plan_final_action`**: Creates comprehensive decision explanation with audit trail, expected outcomes, and risk mitigation strategies.
5.  **Expose Metric**: The service updates a Prometheus `Gauge` (`nimbusguard_dqn_desired_replicas`) with the final, validated replica count.
6.  **KEDA Action**: A KEDA `ScaledObject` is configured to watch this single metric and executes the scaling action on the target deployment.

## ⚙️ Configuration

The service is configured via environment variables:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `PROMETHEUS_URL` | The URL of the Prometheus query API. | `http://prometheus.nimbusguard.svc:9090` |
| `MCP_SERVER_URL` | The URL of the Model Context Protocol server for cluster tools. | `None` |
| `OPENAI_API_KEY` | **(Required)** Your API key for OpenAI. | `None` |
| `AI_MODEL` | OpenAI model to use for LLM validation (cost-effective). | `gpt-3.5-turbo` |
| `AI_TEMPERATURE` | Temperature for LLM responses (lower = more consistent). | `0.1` |
| `ENABLE_DETAILED_REASONING` | Enable comprehensive AI reasoning logs. | `true` |
| `REASONING_LOG_LEVEL` | Log level for AI reasoning (INFO/DEBUG). | `INFO` |
| `POLLING_INTERVAL` | The interval in seconds for the reconciliation loop. | `30` |
| `TARGET_DEPLOYMENT` | The name of the deployment to monitor and scale. | `consumer` |
| `TARGET_NAMESPACE`| The namespace of the target deployment. | `nimbusguard` |
| `MINIO_ENDPOINT` | MinIO endpoint for model storage. | `http://minio.nimbusguard.svc:9000` |
| `MINIO_ACCESS_KEY` | MinIO access key. | `minioadmin` |
| `MINIO_SECRET_KEY` | MinIO secret key. | `minioadmin` |
| `SCALER_NAME` | The name of the feature scaler file in MinIO. | `feature_scaler.gz` |

### Cost Optimization

The system is configured for **cost-effective operation**:
- **AI Model**: Uses `gpt-3.5-turbo` instead of `gpt-4o-mini` for **60-80% cost reduction**
- **Smart Caching**: Validation results cached to reduce API calls
- **Efficient Prompting**: Structured prompts minimize token usage
- **Fallback Logic**: Robust fallbacks reduce failed API calls

## 🚀 Running the Service

1.  **Build the Docker Image**:
    ```bash
    docker build -t nimbusguard-dqn-adapter:latest -f src/dqn-adapter/Dockerfile .
    ```
2.  **Deploy to Kubernetes**:
    The service is included as a Kustomize component. Ensure the `dqn-adapter` component is added to your overlay's `kustomization.yaml` and deploy using:
    ```bash
    kubectl apply -k kubernetes-manifests/overlays/development
    ```
3.  **Local Port Forwarding**:
    Use the `Makefile` to forward the service's port for local inspection:
    ```bash
    make forward
    ```
    You can then access the health endpoint at `http://localhost:8001/health` and the metrics at `http://localhost:8001/metrics`.

## 🔍 Monitoring Explainable AI

### Log Analysis
Monitor the `AI_Reasoning` logger for detailed decision explanations:
```bash
kubectl logs -n nimbusguard deployment/dqn-adapter | grep "AI_Reasoning"
```

### Decision Confidence Tracking
Low confidence decisions are automatically flagged:
```bash
kubectl logs -n nimbusguard deployment/dqn-adapter | grep "LOW CONFIDENCE"
```

### Audit Trail Access
Each decision includes a unique audit trail ID for compliance tracking and review. 