# NimbusGuard DQN Adapter Service

This service is the intelligent core of the NimbusGuard autoscaling system. It acts as a bridge between the predictive power of a Deep Q-Network (DQN) model and the robust execution engine of KEDA. It runs a continuous reconciliation loop to determine the optimal number of replicas for a target service and exposes this decision as a Prometheus metric.

## 🏛️ Architecture

The adapter is built as a stateful, resilient agent using the **LangGraph** framework. It follows a modern, production-ready architecture that separates a real-time "Actor" loop from an offline "Learner" system.

### Core Concepts Used:

*   **Functional API (`@entrypoint`)**: The main reconciliation logic is defined as a LangGraph `@entrypoint`. This provides a clean, modern, and traceable way to define the sequence of operations (tasks).
*   **Checkpointer (`MemorySaver`)**: The workflow is configured with a checkpointer. This is **critical for resilience**. If the adapter pod restarts, it can resume its state from the last successful step, preventing interruptions. It is also the prerequisite for advanced features like human-in-the-loop.
*   **Tools (DQN & MCP)**: The agent uses two primary types of tools:
    1.  **DQN Model**: A call to a KServe-hosted DQN model to get an initial, numerically-driven scaling recommendation.
    2.  **MCP (Model Context Protocol) Server**: It uses the `langchain-mcp-adapters` library to dynamically load tools from a running MCP server. This gives the validation LLM secure, read-only access to the live Kubernetes cluster state.
*   **Agentic Validator (`create_react_agent`)**: The final validation step is not just a simple LLM call. It's a true LangChain agent that can reason and decide whether to use its MCP tools to gather more context before approving or denying the DQN's suggestion.
*   **Supervisor (Future Extension)**: The architecture is designed to be easily extended into a multi-agent system using a `Supervisor`. For instance, we could add a "Cost-Optimizer Agent" that runs on a slower loop and provides input to the main "Scaling Agent".

### Workflow (Reconciliation Loop)

On a configurable interval (`POLLING_INTERVAL`), the LangGraph entrypoint is invoked, which executes the following flow:

1.  **`get_live_metrics`**: Queries the Prometheus server for the latest application metrics (request rate, latency, etc.) and the current replica count of the target deployment.
2.  **`get_dqn_recommendation`**: Prepares the metrics into a feature vector, scales them using the training-time scaler, and calls the KServe endpoint to get a scaling action (`Scale Up`, `Scale Down`, `Keep Same`).
3.  **`validate_with_llm`**: Invokes the validator agent. The agent is prompted with the current state and the DQN's suggestion. It can then use its MCP tools to query the live cluster state (e.g., "are any pods currently in a crash loop?") before making a final judgment.
4.  **`plan_final_action`**: If the LLM agent approves the action, this final node calculates the new target replica count.
5.  **Expose Metric**: The service updates a Prometheus `Gauge` (`nimbusguard_dqn_desired_replicas`) with the final, validated replica count.
6.  **KEDA Action**: A KEDA `ScaledObject` is configured to watch this single metric and executes the scaling action on the target deployment.

## ⚙️ Configuration

The service is configured via environment variables:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `PROMETHEUS_URL` | The URL of the Prometheus query API. | `http://prometheus.nimbusguard.svc:9090` |
| `KSERVE_URL` | **(Required)** The prediction URL of the deployed DQN model on KServe. | `None` |
| `MCP_SERVER_URL` | The URL of the Model Context Protocol server for cluster tools. | `None` |
| `OPENAI_API_KEY` | **(Required)** Your API key for OpenAI. | `None` |
| `POLLING_INTERVAL` | The interval in seconds for the reconciliation loop. | `30` |
| `TARGET_DEPLOYMENT` | The name of the deployment to monitor and scale. | `consumer` |
| `TARGET_NAMESPACE`| The namespace of the target deployment. | `nimbusguard` |
| `SCALER_PATH` | The file path to the saved `feature_scaler.joblib`. | `/app/models/feature_scaler.joblib` |

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