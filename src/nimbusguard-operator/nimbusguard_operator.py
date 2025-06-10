import kopf
import asyncio
import logging
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import aiohttp
from fastapi import FastAPI, BackgroundTasks
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from langgraph.graph import StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="NimbusGuard Operator")

# Prometheus metrics
SCALING_OPERATIONS = Counter(
    'nimbusguard_scaling_operations_total',
    'Total number of scaling operations',
    ['operation_type']
)

CURRENT_REPLICAS = Gauge(
    'nimbusguard_current_replicas',
    'Current number of replicas',
    ['name', 'namespace']
)

SCALING_DURATION = Histogram(
    'nimbusguard_scaling_duration_seconds',
    'Time spent on scaling operations',
    ['operation_type']
)

OPERATOR_HEALTH = Gauge(
    'nimbusguard_operator_health',
    'Health status of the operator',
    ['component']
)

# Health check state
health_status = {
    "prometheus": True,
    "kubernetes": True,
    "openai": True
}

# Configure OpenTelemetry if endpoint is available
tempo_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
if tempo_endpoint:
    try:
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer(__name__)
        otlp_exporter = OTLPSpanExporter(endpoint=tempo_endpoint, insecure=True)
        span_processor = BatchSpanProcessor(otlp_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        AioHttpClientInstrumentor().instrument()
        LOG.info("OpenTelemetry tracing configured successfully")
    except Exception as e:
        LOG.warning(f"Failed to configure OpenTelemetry: {e}")
        tracer = trace.get_tracer(__name__)
else:
    LOG.info("OpenTelemetry tracing disabled")
    tracer = trace.get_tracer(__name__)

FINALIZER_NAME = "nimbusguard.finalizers.nimbusguard.io"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy" if all(health_status.values()) else "unhealthy", "components": health_status}

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    return {"status": "ready" if all(health_status.values()) else "not_ready", "components": health_status}

@app.get("/metrics")
async def metrics():
    """Metrics endpoint"""
    return {
        "scaling_operations": SCALING_OPERATIONS._value.get(),
        "current_replicas": CURRENT_REPLICAS._value.get(),
        "operator_health": OPERATOR_HEALTH._value.get()
    }

class PrometheusClient:
    """Client for querying Prometheus metrics"""
    
    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url.rstrip('/')
        OPERATOR_HEALTH.labels(component='prometheus').set(1)
        
    async def query(self, query: str, session: aiohttp.ClientSession) -> Optional[float]:
        """Execute a PromQL query and return the result"""
        url = f"{self.prometheus_url}/api/v1/query"
        params = {"query": query}
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get("data", {}).get("result", [])
                    if result and len(result) > 0:
                        value = result[0].get("value", [None, None])[1]
                        return float(value) if value is not None else None
                OPERATOR_HEALTH.labels(component='prometheus').set(0)
                return None
        except Exception as e:
            LOG.error(f"Prometheus query failed: {e}")
            OPERATOR_HEALTH.labels(component='prometheus').set(0)
            return None

class DecisionEngine:
    """LangGraph-based decision engine for intelligent scaling"""
    
    def __init__(self):
        self.graph = self._build_decision_graph()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            LOG.warning("OpenAI API key not found. Some decision-making capabilities may be limited.")
    
    def _build_decision_graph(self) -> StateGraph:
        """Build the decision-making graph using LangGraph"""
        builder = StateGraph(Dict[str, Any])
        builder.add_node("analyze", self._analyze_metrics)
        builder.add_node("decide", self._make_decision)
        builder.add_node("execute", self._execute_action)
        # Add edges and entry point as before
        # Example:
        builder.add_edge("analyze", "decide")
        builder.add_edge("decide", "execute")
        builder.set_entry_point("analyze")
        return builder.compile()
    
    def _analyze_metrics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current metrics and thresholds using LangGraph"""
        metrics = state.get("metrics", {})
        thresholds = state.get("thresholds", {})
        
        if not self.openai_api_key:
            # Fallback to basic analysis if no API key
            return self._basic_metric_analysis(state)
        
        try:
            # Create the prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a Kubernetes scaling expert. Analyze metrics and provide scaling recommendations."),
                ("user", """Analyze the following Kubernetes metrics and provide a scaling recommendation:
                Current Metrics:
                {metrics}
                
                Thresholds:
                {thresholds}
                
                Current Replicas: {current_replicas}
                Max Replicas: {max_replicas}
                Min Replicas: {min_replicas}
                
                Provide a JSON response with:
                1. analysis: {{
                    alerts: list of alerts,
                    severity: "low"|"medium"|"high",
                    resource_pressure: boolean
                 }}
                2. recommendation: {{
                    action: "scale_up"|"scale_down"|"none",
                    reason: string,
                    target_replicas: number
                 }}
                """)
            ])
            
            # Create the LLM
            llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.3,
                api_key=self.openai_api_key
            )
            
            # Format the prompt
            formatted_prompt = prompt.format_messages(
                metrics=json.dumps(metrics, indent=2),
                thresholds=json.dumps(thresholds, indent=2),
                current_replicas=state.get('current_replicas', 1),
                max_replicas=state.get('max_replicas', 10),
                min_replicas=state.get('min_replicas', 1)
            )
            
            # Get the response
            response = llm.invoke(formatted_prompt)
            
            # Parse the response
            result = json.loads(response.content)
            state["analysis"] = result["analysis"]
            state["recommendation"] = result["recommendation"]
            
        except Exception as e:
            LOG.error(f"LangGraph analysis failed: {e}")
            return self._basic_metric_analysis(state)
        
        return state
    
    def _make_decision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Make scaling decision based on analysis"""
        if "recommendation" in state:
            # Use LangGraph's recommendation
            state["decision"] = state["recommendation"]
        else:
            # Fallback to basic decision making
            analysis = state.get("analysis", {})
            current_replicas = state.get("current_replicas", 1)
            max_replicas = state.get("max_replicas", 10)
            min_replicas = state.get("min_replicas", 1)
            
            decision = {
                "action": "none",
                "reason": "All metrics within normal range",
                "target_replicas": current_replicas
            }
            
            if analysis.get("resource_pressure", False):
                severity = analysis.get("severity", "low")
                
                if severity == "high" and current_replicas < max_replicas:
                    scale_factor = 2
                    target = min(current_replicas * scale_factor, max_replicas)
                    decision = {
                        "action": "scale_up",
                        "reason": f"High resource pressure detected: {', '.join(analysis.get('alerts', []))}",
                        "target_replicas": target
                    }
                elif severity == "medium" and current_replicas < max_replicas:
                    target = min(current_replicas + 1, max_replicas)
                    decision = {
                        "action": "scale_up",
                        "reason": f"Medium resource pressure: {', '.join(analysis.get('alerts', []))}",
                        "target_replicas": target
                    }
            
            state["decision"] = decision
        
        return state
    
    def _execute_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare action for execution"""
        decision = state.get("decision", {})
        action = decision.get("action", "none")
        
        execution_plan = {
            "action": action,
            "details": decision,
            "timestamp": datetime.now().isoformat()
        }
        
        state["execution_plan"] = execution_plan
        return state
    
    def _basic_metric_analysis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Basic metric analysis when OpenAI is not available"""
        metrics = state.get("metrics", {})
        thresholds = state.get("thresholds", {})
        
        analysis = {
            "alerts": [],
            "severity": "low",
            "resource_pressure": False
        }
        
        # Analyze CPU metrics
        cpu_usage = metrics.get("cpu_usage", 0)
        cpu_threshold = thresholds.get("cpu_threshold", 80)
        
        if cpu_usage > cpu_threshold:
            analysis["alerts"].append(f"High CPU usage: {cpu_usage}%")
            analysis["resource_pressure"] = True
            analysis["severity"] = "high" if cpu_usage > 90 else "medium"
        
        # Analyze memory metrics
        memory_usage = metrics.get("memory_usage", 0)
        memory_threshold = thresholds.get("memory_threshold", 80)
        
        if memory_usage > memory_threshold:
            analysis["alerts"].append(f"High memory usage: {memory_usage}%")
            analysis["resource_pressure"] = True
            if analysis["severity"] == "low":
                analysis["severity"] = "medium"
        
        state["analysis"] = analysis
        return state
    
    async def make_decision(self, metrics: Dict[str, float], config: Dict[str, Any]) -> Dict[str, Any]:
        """Make a scaling decision based on current metrics"""
        
        # Prepare initial state
        state = {
            "metrics": metrics,
            "thresholds": config.get("thresholds", {}),
            "current_replicas": config.get("current_replicas", 1),
            "max_replicas": config.get("max_replicas", 10),
            "min_replicas": config.get("min_replicas", 1)
        }
        
        # Run the decision graph
        result = await asyncio.to_thread(self.graph.invoke, state)
        
        return result.get("execution_plan", {})

class IntelligentOperator:
    """Main operator for intelligent scaling decisions"""
    
    def __init__(self):
        self.prometheus_client = None
        self.decision_engine = DecisionEngine()
        
    async def initialize_prometheus_client(self, prometheus_url: str):
        """Initialize Prometheus client"""
        self.prometheus_client = PrometheusClient(prometheus_url)
    
    @tracer.start_as_current_span("evaluate_metrics")
    async def evaluate_metrics(self, body: Dict[str, Any], name: str, namespace: str) -> Dict[str, Any]:
        """Evaluate metrics and make scaling decisions"""
        
        with tracer.start_as_current_span("fetch_metrics") as span:
            span.set_attribute("resource.name", name)
            span.set_attribute("resource.namespace", namespace)
            
            spec = body.get("spec", {})
            metrics_config = spec.get("metrics_config", {})
            
            # Initialize Prometheus client if needed
            prometheus_url = metrics_config.get("prometheus_url", "http://prometheus:9090")
            if not self.prometheus_client:
                await self.initialize_prometheus_client(prometheus_url)
            
            # Fetch current metrics
            current_metrics = {}
            
            async with aiohttp.ClientSession() as session:
                for metric_query in metrics_config.get("metrics", []):
                    query = metric_query.get("query", "")
                    if query:
                        value = await self.prometheus_client.query(query, session)
                        if value is not None:
                            # Extract metric name from query for storage
                            metric_name = query.split("_")[0] if "_" in query else "metric"
                            current_metrics[metric_name] = value
                            
                            # Set span attributes for observability
                            span.set_attribute(f"metric.{metric_name}", value)
        
        # Make decision using LangGraph
        with tracer.start_as_current_span("make_decision") as decision_span:
            config = {
                "thresholds": {
                    "cpu_threshold": 80,
                    "memory_threshold": 80
                },
                "current_replicas": 1,  # You'd fetch this from K8s API
                "max_replicas": 10,
                "min_replicas": 1
            }
            
            # Extract thresholds from metrics config
            for metric_query in metrics_config.get("metrics", []):
                threshold = metric_query.get("threshold")
                condition = metric_query.get("condition", "gt")
                query = metric_query.get("query", "")
                
                if "cpu" in query.lower():
                    config["thresholds"]["cpu_threshold"] = threshold
                elif "memory" in query.lower():
                    config["thresholds"]["memory_threshold"] = threshold
            
            decision = await self.decision_engine.make_decision(current_metrics, config)
            
            decision_span.set_attribute("decision.action", decision.get("action", "none"))
            decision_span.set_attribute("decision.reason", decision.get("reason", ""))
            
        return {
            "metrics": current_metrics,
            "decision": decision,
            "trace_id": span.get_span_context().trace_id
        }

# Global operator instance
intelligent_operator = IntelligentOperator()

@kopf.on.startup()
async def configure(settings: kopf.OperatorSettings, **_):
    """Configure the operator on startup"""
    # Start Prometheus metrics server
    start_http_server(8000)
    
    # Initialize health metrics
    OPERATOR_HEALTH.labels(component='prometheus').set(1)
    OPERATOR_HEALTH.labels(component='kubernetes').set(1)
    OPERATOR_HEALTH.labels(component='openai').set(1)
    
    # Start FastAPI server in background
    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=8081, log_level="info")
    server = uvicorn.Server(config)
    asyncio.create_task(server.serve())
    LOG.info("FastAPI server starting on port 8081")
    
    LOG.info("NimbusGuard operator started successfully")

@kopf.on.timer('intelligentscaling', group='nimbusguard.io', version='v1alpha1', interval=30)
async def evaluate_intelligent_scaling(body, name, namespace, **kwargs):
    """Evaluate and perform intelligent scaling"""
    with SCALING_DURATION.labels(operation_type='evaluation').time():
        try:
            # Update current replicas metric
            current_replicas = body.get('spec', {}).get('replicas', 1)
            CURRENT_REPLICAS.labels(name=name, namespace=namespace).set(current_replicas)
            
            # Evaluate metrics and make decisions
            result = await intelligent_operator.evaluate_metrics(body, name, namespace)
            
            # Update the resource status
            current_time = datetime.now()
            status_patch = {
                "status": {
                    "last_evaluation": current_time.isoformat(),
                    "current_metrics": result["metrics"],
                    "last_action": result["decision"].get("action", "none"),
                    "decisions_made": body.get("status", {}).get("decisions_made", 0) + 1,
                    "trace_id": str(result["trace_id"])
                }
            }
            
            # Log the decision
            LOG.info(f"Decision for {name}: {result['decision']}")
            
            # Here you would implement the actual scaling action
            # For now, just log what would happen
            decision = result["decision"]
            if decision.get("action") != "none":
                LOG.info(f"Would execute: {decision['action']} to {decision.get('target_replicas')} replicas")
                LOG.info(f"Reason: {decision.get('reason')}")
            
            # Record successful scaling operation
            SCALING_OPERATIONS.labels(operation_type='evaluation').inc()
            
            return status_patch
            
        except Exception as e:
            LOG.error(f"Scaling evaluation failed: {e}")
            OPERATOR_HEALTH.labels(component='kubernetes').set(0)
            raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    LOG.info("Starting NimbusGuard Operator...")
    
    # Run the operator
    kopf.run(
        clusterwide=True
    )
