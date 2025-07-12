# NimbusGuard - AI-Powered Kubernetes Autoscaling

An intelligent cloud-native autoscaling system that uses **Deep Q-Learning (DQN)** and **Large Language Model (LLM) validation** to make smart scaling decisions for Kubernetes workloads.

## 🧠 What Makes NimbusGuard Special?

Unlike traditional autoscaling that relies on simple CPU/memory thresholds, NimbusGuard uses:
- **🤖 AI Decision Engine**: DQN neural network trained on scientifically selected features
- **🛡️ LLM Safety Validator**: GPT-powered validation of scaling decisions with real cluster access
- **🧠 Pure LLM Rewards**: Dynamic reward calculation using OpenAI GPT (no hardcoded thresholds)
- **📊 Explainable AI**: Detailed reasoning for every scaling decision
- **🔄 Continuous Learning**: Real-time model improvement from intelligent feedback

---

## 🚀 Quick Setup (3 Commands)

### Prerequisites
- **Docker Desktop** with Kubernetes enabled
- **Git** installed
- **OpenAI API Key** (required - no fallbacks available)

### Clone Repository
```bash
# Clone with submodules (required)
git clone --recursive https://github.com/CSWanigasooriya/nimbusguard.git
cd nimbusguard

# If already cloned without --recursive:
git submodule update --init --recursive
```

### Deploy Everything
```bash
# 1. Setup tools and environment (auto-installs everything)
make setup

# 2. Build and deploy the complete AI system
make dev

# 3. Access services via port forwarding
make ports
```

**That's it!** 🎉

---

## 🎯 What Just Happened?

### `make setup` automatically:
- 🔧 Installs kubectl, helm, docker (macOS/Linux detection)
- 📦 Configures Helm repositories
- 🔑 **REQUIRED**: Sets up OpenAI API key (prompted during setup - mandatory for LLM rewards)
- 📊 Installs metrics-server for monitoring
- 🏗️ Creates `nimbusguard` namespace

### `make dev` automatically:
- 🔨 Builds all Docker images
- 🔍 Installs KEDA (if not present)
- 🚀 Deploys the complete AI-powered stack

### `make ports` provides access to:
- 📈 **Prometheus**: http://localhost:9090
- 📊 **Grafana**: http://localhost:3000 (admin/admin)
- 🧠 **DQN Adapter**: http://localhost:8080
- 💾 **Redis**: redis-cli -p 6379
- 🗄️ **MinIO**: http://localhost:9000

---

## 🧪 Test the AI System

```bash
# Light load test (gentle scaling)
make load-test-light

# Heavy load test (trigger AI scaling)
make load-test-heavy

# Check scaling status
make load-status

# Watch AI decisions in real-time
make logs-dqn
```

---

## 🔧 Available Commands

| Command | Description |
|---------|-------------|
| `make setup` | Install all tools and configure environment |
| `make dev` | Build and deploy complete system |
| `make ports` | Port forward all services |
| `make load-test-*` | Run various load tests |
| `make logs-dqn` | Watch AI decision making |
| `make clean` | Nuclear cleanup (delete everything) |
| `make help` | Show all available commands |

---

## 🐛 Troubleshooting

**If `make setup` fails:**
- Ensure Docker Desktop is running with Kubernetes enabled
- On Windows: Install `make` via `choco install make` or use PowerShell to run commands manually

**If deployment fails:**
- Run `make clean` then try `make dev` again
- **REQUIRED**: Ensure you have a valid OpenAI API key configured (system cannot function without it)

**View all available commands:**
```bash
make help
```

---

## 🎓 What Gets Deployed

- **🤖 DQN Adapter**: AI decision engine with intelligent reasoning
- **🛡️ LLM Validator**: GPT-powered safety validation with cluster access  
- **📊 Consumer Service**: FastAPI application for load testing
- **⚖️ KEDA Autoscaling**: Event-driven scaling based on AI decisions
- **📈 Monitoring Stack**: Prometheus, Grafana, Alloy for observability
- **🔍 Auto-instrumentation**: Beyla for automatic metrics collection

**Watch the AI make intelligent scaling decisions in the logs!** 🤖

---

*Ready to experience the future of intelligent autoscaling! 🚀* 