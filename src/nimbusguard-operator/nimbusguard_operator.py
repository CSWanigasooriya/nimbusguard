#!/usr/bin/env python3
"""
NimbusGuard Intelligent Scaling Operator - Production Ready
A Kubernetes operator for intelligent auto-scaling using AI-driven decision making.
Works with the existing CRD defined in kubernetes-manifests.
"""

import asyncio
import logging
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, Any, List, Optional

import aiohttp
import kopf
import kubernetes
from prometheus_client import Counter, Gauge, start_http_server

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*urllib3.*")

# ============================================================================
# Logging Configuration
# ============================================================================

# Configure logging to reduce noise
logging.getLogger('kubernetes').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOG = logging.getLogger(__name__)

# ============================================================================
# Prometheus Metrics
# ============================================================================

SCALING_OPERATIONS = Counter('nimbusguard_scaling_operations_total', 'Total scaling operations',
                             ['operation_type', 'namespace'])
CURRENT_REPLICAS = Gauge('nimbusguard_current_replicas', 'Current replicas', ['name', 'namespace'])
DECISIONS_MADE = Counter('nimbusguard_decisions_total', 'Total decisions made', ['action'])
OPERATOR_HEALTH = Gauge('nimbusguard_operator_health', 'Operator health', ['component'])

# ============================================================================
# Health Status (for kopf probes)
# ============================================================================

health_status = {
    "prometheus": True,
    "kubernetes": True,
    "openai": bool(os.getenv("OPENAI_API_KEY")),
    "decision_engine": True
}


# ============================================================================
# Simple Prometheus Client
# ============================================================================

class PrometheusClient:
    """Simple Prometheus client with improved error handling"""

    def __init__(self, url: str):
        self.url = url.rstrip('/')
        self.session_timeout = aiohttp.ClientTimeout(total=10)

    async def query(self, query: str) -> Optional[float]:
        """Execute PromQL query with proper error handling"""
        try:
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                async with session.get(
                        f"{self.url}/api/v1/query",
                        params={"query": query},
                        headers={'Accept': 'application/json'}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = data.get("data", {}).get("result", [])
                        if result:
                            value = result[0].get("value", [None, None])[1]
                            health_status["prometheus"] = True
                            return float(value) if value is not None else None
                    else:
                        LOG.warning(f"Prometheus returned status {resp.status} for query: {query}")

        except asyncio.TimeoutError:
            LOG.error(f"Prometheus query timeout for: {query}")
            health_status["prometheus"] = False
        except aiohttp.ClientError as e:
            LOG.error(f"Prometheus client error: {e}")
            health_status["prometheus"] = False
        except Exception as e:
            LOG.error(f"Prometheus query failed: {e}")
            health_status["prometheus"] = False
        return None


# ============================================================================
# Decision Engine
# ============================================================================

class DecisionEngine:
    """Simple decision engine with basic scaling logic"""

    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        health_status["openai"] = bool(self.openai_key)

    def make_decision(self, metrics: Dict[str, float], current_replicas: int,
                      metric_configs: List[Dict], min_replicas: int = 1, max_replicas: int = 10) -> Dict[str, Any]:
        """Make scaling decision based on metrics"""

        alerts = []
        scale_up_needed = False
        scale_down_needed = True  # Assume we can scale down unless metrics say otherwise

        # Evaluate each metric
        for metric_config in metric_configs:
            query = metric_config.get("query", "")
            threshold = metric_config.get("threshold", 0)
            condition = metric_config.get("condition", "gt")

            # Get metric name from query
            metric_name = self._extract_metric_name(query)
            metric_value = metrics.get(metric_name, 0)

            # Evaluate condition
            if condition == "gt" and metric_value > threshold:
                alerts.append(f"{metric_name}: {metric_value:.1f} > {threshold}")
                scale_up_needed = True
                scale_down_needed = False
            elif condition == "lt" and metric_value < threshold:
                alerts.append(f"{metric_name}: {metric_value:.1f} < {threshold}")
                scale_up_needed = True
                scale_down_needed = False
            elif condition == "eq" and abs(metric_value - threshold) < 0.1:
                alerts.append(f"{metric_name}: {metric_value:.1f} ≈ {threshold}")
                scale_down_needed = False

        # Make scaling decision
        if scale_up_needed and current_replicas < max_replicas:
            target_replicas = min(current_replicas + 1, max_replicas)
            return {
                "action": "scale_up",
                "target_replicas": target_replicas,
                "reason": f"Metrics exceeded thresholds: {', '.join(alerts)}",
                "alerts": alerts
            }
        elif scale_down_needed and current_replicas > min_replicas and not alerts:
            target_replicas = max(current_replicas - 1, min_replicas)
            return {
                "action": "scale_down",
                "target_replicas": target_replicas,
                "reason": "All metrics below thresholds, scaling down",
                "alerts": []
            }
        else:
            return {
                "action": "none",
                "target_replicas": current_replicas,
                "reason": "All metrics within acceptable range" if not alerts else "At scaling limits",
                "alerts": alerts
            }

    def _extract_metric_name(self, query: str) -> str:
        """Extract metric name from PromQL query"""
        if "cpu" in query.lower():
            return "cpu_usage"
        elif "memory" in query.lower():
            return "memory_usage"
        elif "request" in query.lower():
            return "request_rate"
        else:
            return "metric"


# ============================================================================
# Main Operator
# ============================================================================

class NimbusGuardOperator:
    """Main operator class"""

    def __init__(self):
        self.decision_engine = DecisionEngine()
        self.k8s_client = None
        self.custom_api = None
        self.apps_api = None

    async def initialize(self):
        """Initialize operator"""
        try:
            # Load Kubernetes config
            try:
                kubernetes.config.load_incluster_config()
                LOG.info("Loaded in-cluster Kubernetes config")
            except:
                kubernetes.config.load_kube_config()
                LOG.info("Loaded local Kubernetes config")

            # Initialize API clients
            self.k8s_client = kubernetes.client.ApiClient()
            self.custom_api = kubernetes.client.CustomObjectsApi(self.k8s_client)
            self.apps_api = kubernetes.client.AppsV1Api(self.k8s_client)

            health_status["kubernetes"] = True
            LOG.info("Kubernetes client ready")
        except Exception as e:
            LOG.error(f"K8s init failed: {e}")
            health_status["kubernetes"] = False
            raise

    async def evaluate_scaling(self, body: Dict[str, Any], name: str, namespace: str) -> Dict[str, Any]:
        """Main scaling evaluation logic"""
        try:
            spec = body.get("spec", {})

            # Extract configuration from the CRD spec
            target_namespace = spec.get("namespace", namespace)
            target_labels = spec.get("target_labels", {})
            metrics_config = spec.get("metrics_config", {})

            # Fetch metrics from Prometheus
            metrics = await self._fetch_metrics(metrics_config)

            # Get current replica count
            current_replicas = await self._get_current_replicas(target_labels, target_namespace)

            # Make scaling decision
            decision = self.decision_engine.make_decision(
                metrics=metrics,
                current_replicas=current_replicas,
                metric_configs=metrics_config.get("metrics", []),
                min_replicas=1,  # Could be made configurable
                max_replicas=10  # Could be made configurable
            )

            # Execute scaling action if needed
            if decision.get("action") != "none":
                await self._execute_scaling(decision, target_labels, target_namespace)
                DECISIONS_MADE.labels(action=decision.get("action")).inc()

            return {
                "current_replicas": current_replicas,
                "target_replicas": decision.get("target_replicas", current_replicas),
                "decision_reason": decision.get("reason", ""),
                "last_evaluation": datetime.now().isoformat(),
                "action": decision.get("action", "none")
            }

        except Exception as e:
            LOG.error(f"Evaluation failed: {e}")
            raise

    async def _fetch_metrics(self, metrics_config: Dict[str, Any]) -> Dict[str, float]:
        """Fetch metrics from Prometheus"""
        prometheus_url = metrics_config.get("prometheus_url", "http://prometheus:9090")
        client = PrometheusClient(prometheus_url)
        metrics = {}

        for metric_config in metrics_config.get("metrics", []):
            query = metric_config.get("query", "")
            if query:
                value = await client.query(query)
                if value is not None:
                    name = self.decision_engine._extract_metric_name(query)
                    metrics[name] = value

        return metrics

    async def _get_current_replicas(self, target_labels: Dict[str, str], namespace: str) -> int:
        """Get current replica count from target deployment"""
        try:
            if not target_labels:
                LOG.warning("No target labels specified")
                return 1

            # Create label selector
            selector = ",".join([f"{k}={v}" for k, v in target_labels.items()])

            # Get deployments matching the labels
            deployments = self.apps_api.list_namespaced_deployment(
                namespace=namespace,
                label_selector=selector
            )

            if deployments.items:
                current = deployments.items[0].status.replicas or 1
                CURRENT_REPLICAS.labels(name=deployments.items[0].metadata.name, namespace=namespace).set(current)
                return current

        except Exception as e:
            LOG.warning(f"Failed to get current replicas: {e}")

        return 1

    async def _execute_scaling(self, decision: Dict[str, Any], target_labels: Dict[str, str], namespace: str):
        """Execute scaling action on target deployment"""
        try:
            target_replicas = decision.get("target_replicas", 1)
            action = decision.get("action", "none")

            if not target_labels:
                LOG.warning("No target labels - cannot scale")
                return

            # Create label selector
            selector = ",".join([f"{k}={v}" for k, v in target_labels.items()])

            # Get deployments matching the labels
            deployments = self.apps_api.list_namespaced_deployment(
                namespace=namespace,
                label_selector=selector
            )

            for deployment in deployments.items:
                # Update replica count
                deployment.spec.replicas = target_replicas

                # Patch the deployment
                self.apps_api.patch_namespaced_deployment(
                    name=deployment.metadata.name,
                    namespace=namespace,
                    body=deployment
                )

                LOG.info(f"Scaled deployment {deployment.metadata.name} to {target_replicas} replicas")
                SCALING_OPERATIONS.labels(operation_type=action, namespace=namespace).inc()

        except Exception as e:
            LOG.error(f"Scaling execution failed: {e}")
            raise


# ============================================================================
# Global operator instance
# ============================================================================

operator = NimbusGuardOperator()


# ============================================================================
# Kopf Health Probes
# ============================================================================

@kopf.on.probe(id='health')
def health_check(**kwargs):
    """Health check for kopf liveness probe"""
    # Update health metrics
    for component, status in health_status.items():
        OPERATOR_HEALTH.labels(component=component).set(1 if status else 0)

    return {
        "status": "healthy" if all(health_status.values()) else "degraded",
        "components": health_status
    }


@kopf.on.probe(id='ready')
def readiness_check(**kwargs):
    """Readiness check for kopf"""
    ready = health_status["kubernetes"] and health_status["decision_engine"]
    return {
        "status": "ready" if ready else "not_ready",
        "kubernetes": health_status["kubernetes"],
        "decision_engine": health_status["decision_engine"]
    }


# ============================================================================
# Kopf Event Handlers
# ============================================================================

@kopf.on.startup()
async def startup(settings: kopf.OperatorSettings, **_):
    """Operator startup configuration"""
    try:
        # Initialize operator
        await operator.initialize()

        # Get metrics port from environment (default to 8000)
        metrics_port = int(os.getenv('METRICS_PORT', '8000'))

        # Start Prometheus metrics server  
        start_http_server(metrics_port)

        # Initialize some default metrics
        OPERATOR_HEALTH.labels(component='startup').set(1)
        SCALING_OPERATIONS.labels(operation_type='startup', namespace='system').inc(0)

        LOG.info(f"Metrics server started on :{metrics_port}")
        LOG.info(f"Access metrics at: http://localhost:{metrics_port}/metrics")
        LOG.info(f"Health endpoint: http://localhost:8080/healthz")

        LOG.info("NimbusGuard operator started successfully")
        LOG.info("Operator will work with existing IntelligentScaling CRD")

        # Log environment info
        LOG.info(f"Log level: {os.getenv('LOG_LEVEL', 'INFO')}")
        LOG.info(f"OpenAI available: {bool(os.getenv('OPENAI_API_KEY'))}")

    except Exception as e:
        LOG.error(f"Startup failed: {e}")
        OPERATOR_HEALTH.labels(component='startup').set(0)
        raise


@kopf.on.create('nimbusguard.io', 'v1alpha1', 'intelligentscaling')
async def on_create(body, name, namespace, **kwargs):
    """Handle IntelligentScaling resource creation"""
    LOG.info(f"Created IntelligentScaling '{name}' in '{namespace}'")
    return await evaluate_scaling(body, name, namespace, **kwargs)


@kopf.timer('nimbusguard.io', 'v1alpha1', 'intelligentscaling', interval=30)
async def evaluate_scaling(body, name, namespace, **kwargs):
    """Periodic scaling evaluation"""
    LOG.debug(f"Evaluating scaling for '{name}' in namespace '{namespace}'")

    try:
        # Ensure we have valid body structure
        if not body or 'spec' not in body:
            LOG.warning(f"Invalid resource body for '{name}' - missing spec")
            return

        # Evaluate scaling
        result = await operator.evaluate_scaling(body, name, namespace)

        # Prepare status patch matching the CRD schema
        status_patch = {
            "status": {
                "last_evaluation": result["last_evaluation"],
                "current_replicas": result["current_replicas"],
                "target_replicas": result["target_replicas"],
                "decision_reason": result["decision_reason"],
                "conditions": [
                    {
                        "type": "Ready",
                        "status": "True" if result["action"] == "none" else "False",
                        "lastTransitionTime": result["last_evaluation"],
                        "reason": "EvaluationComplete",
                        "message": result["decision_reason"]
                    }
                ]
            }
        }

        # Log decision with more details
        LOG.info(f"[{name}] Current: {result['current_replicas']} replicas")
        LOG.info(f"[{name}] Decision: {result['decision_reason']}")
        if result["action"] != "none":
            LOG.info(f"[{name}] Action: {result['action']} -> {result['target_replicas']} replicas")

        # Update health status on successful evaluation
        health_status["decision_engine"] = True

        return status_patch

    except Exception as e:
        LOG.error(f"Evaluation failed for '{name}': {e}")
        health_status["kubernetes"] = False
        health_status["decision_engine"] = False
        # Return empty status to avoid kopf errors
        return {}


@kopf.on.delete('nimbusguard.io', 'v1alpha1', 'intelligentscaling')
async def on_delete(body, name, namespace, **kwargs):
    """Handle resource deletion"""
    LOG.info(f"Deleted IntelligentScaling '{name}' from '{namespace}'")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Suppress additional warnings at startup
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    LOG.info("Starting NimbusGuard Intelligent Scaling Operator...")
    LOG.info(f"Python version: {sys.version}")
    LOG.info(f"Operator mode: {'Development' if os.getenv('LOG_LEVEL') == 'DEBUG' else 'Production'}")

    try:
        # Run operator with built-in liveness endpoint
        kopf.run(
            clusterwide=True,
            liveness_endpoint=("0.0.0.0", 8080),  # kopf's built-in liveness probe
            verbose=os.getenv('LOG_LEVEL') == 'DEBUG'
        )
    except KeyboardInterrupt:
        LOG.info("Operator shutdown requested")
    except Exception as e:
        LOG.error(f"Operator failed: {e}")
        raise
