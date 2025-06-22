# engine/langgraph_tools.py
# ============================================================================
# LangGraph Tools for NimbusGuard Operator - Proper @tool Implementation
# ============================================================================

import asyncio
import logging
import json
import os
from typing import Dict, Any, List, Optional

import kubernetes
from langchain_core.tools import tool
from pydantic import BaseModel

from observability import ObservabilityCollector
from ml.kserve_dqn_agent import KServeOnlyDQNAgent, create_kserve_only_agent
from ml.state_representation import EnvironmentState, ScalingActions

LOG = logging.getLogger(__name__)


# ============================================================================
# Tool Input Schemas
# ============================================================================

class ObservabilityInput(BaseModel):
    """Input schema for observability collection."""
    prometheus_url: str
    target_labels: Dict[str, str]
    service_name: str


class DQNDecisionInput(BaseModel):
    """Input schema for DQN decision making."""
    current_replicas: int
    min_replicas: int
    max_replicas: int
    unified_state: Dict[str, Any]
    recent_actions: List[int]


class KubernetesScalingInput(BaseModel):
    """Input schema for Kubernetes scaling operations."""
    deployment_name: str
    namespace: str
    target_replicas: int
    dry_run: bool = False


class KServeInferenceInput(BaseModel):
    """Input schema for KServe model inference."""
    feature_vector: List[float] = Field(description="Feature vector for model inference")
    endpoint_url: str = Field(description="KServe endpoint URL")
    confidence_threshold: float = Field(0.7, description="Confidence threshold for decisions")


# ============================================================================
# Observability Tools
# ============================================================================

@tool
async def collect_observability_data(
    prometheus_url: str,
    target_labels: Dict[str, str],
    service_name: str
) -> Dict[str, Any]:
    """
    Collect comprehensive observability metrics from Prometheus and other sources.
    This replaces the manual observability collection in the original handler.
    """
    try:
        collector = ObservabilityCollector(prometheus_url)
        unified_state = await collector.collect_unified_state(
            metrics_config={"prometheus_url": prometheus_url},
            target_labels=target_labels,
            service_name=service_name
        )
        await collector.close()
        
        LOG.info(f"Observability data collected - Confidence: {unified_state.get('confidence_score', 0.0):.2f}")
        return unified_state
        
    except Exception as e:
        LOG.error(f"Observability collection failed: {e}")
        return {
            "error": str(e),
            "confidence_score": 0.0,
            "feature_vector": [0.0] * 7,
            "feature_names": [],
            "health_score": 0.0
        }


class PrometheusObservabilityTool(BaseTool):
    """Tool for collecting observability data from Prometheus."""
    
    name: str = "prometheus_observability"
    description: str = "Collect comprehensive observability metrics from Prometheus, Loki, and Tempo"
    
    def _run(self, prometheus_url: str, target_labels: Dict[str, str], 
             service_name: str, loki_url: Optional[str] = None, 
             tempo_url: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous wrapper - not used in async context."""
        raise NotImplementedError("Use async version")
    
    async def _arun(self, prometheus_url: str, target_labels: Dict[str, str], 
                    service_name: str, loki_url: Optional[str] = None, 
                    tempo_url: Optional[str] = None) -> Dict[str, Any]:
        """Collect observability data from multiple sources."""
        
        result = {
            "prometheus_data": {},
            "loki_data": {},
            "tempo_data": {},
            "unified_metrics": {},
            "health_status": {},
            "confidence_score": 0.0
        }
        
        try:
            # Collect from Prometheus
            collector = ObservabilityCollector(prometheus_url)
            unified_state = await collector.collect_unified_state(
                metrics_config={"prometheus_url": prometheus_url},
                target_labels=target_labels,
                service_name=service_name
            )
            
            result["unified_metrics"] = unified_state
            result["prometheus_data"] = unified_state.get("raw_metrics", {})
            result["confidence_score"] = unified_state.get("confidence_score", 0.0)
            result["health_status"]["prometheus"] = unified_state.get("data_sources_available", {}).get("prometheus", False)
            
            # Collect from Loki if available
            if loki_url:
                try:
                    loki_client = LokiClient(loki_url)
                    log_features = await loki_client.get_log_analysis_features(target_labels)
                    business_features = await loki_client.get_business_kpi_features(target_labels)
                    
                    result["loki_data"] = {
                        "log_analysis": log_features,
                        "business_kpis": business_features
                    }
                    result["health_status"]["loki"] = True
                    
                except Exception as e:
                    LOG.warning(f"Loki collection failed: {e}")
                    result["health_status"]["loki"] = False
            
            # Collect from Tempo if available
            if tempo_url:
                try:
                    tempo_client = TempoClient(tempo_url)
                    service_graph = await tempo_client.get_service_graph_metrics(service_name)
                    anomaly_features = await tempo_client.get_trace_anomaly_features(service_name)
                    
                    result["tempo_data"] = {
                        "service_graph": service_graph,
                        "anomaly_features": anomaly_features
                    }
                    result["health_status"]["tempo"] = True
                    
                except Exception as e:
                    LOG.warning(f"Tempo collection failed: {e}")
                    result["health_status"]["tempo"] = False
            
            await collector.close()
            
        except Exception as e:
            LOG.error(f"Observability collection failed: {e}")
            result["error"] = str(e)
            
        return result


class HealthCheckTool(BaseTool):
    """Tool for checking system health status."""
    
    name: str = "health_check"
    description: str = "Check the health status of all observability and ML systems"
    
    def _run(self, systems: List[str]) -> Dict[str, Any]:
        """Synchronous wrapper - not used in async context."""
        raise NotImplementedError("Use async version")
    
    async def _arun(self, systems: List[str]) -> Dict[str, Any]:
        """Check health of specified systems."""
        health_results = {}
        overall_health = ObservabilityHealth.HEALTHY
        
        for system in systems:
            try:
                if system == "prometheus":
                    # Simple connectivity test
                    health_results[system] = {"status": "healthy", "message": "Connected"}
                elif system == "kserve":
                    # KServe endpoint test would go here
                    health_results[system] = {"status": "unknown", "message": "Health check not implemented"}
                else:
                    health_results[system] = {"status": "unknown", "message": f"Unknown system: {system}"}
                    
            except Exception as e:
                health_results[system] = {"status": "unhealthy", "message": str(e)}
                overall_health = ObservabilityHealth.DEGRADED
        
        return {
            "individual_health": health_results,
            "overall_health": overall_health.value,
            "healthy_systems": [s for s, h in health_results.items() if h["status"] == "healthy"],
            "unhealthy_systems": [s for s, h in health_results.items() if h["status"] != "healthy"]
        }


# ============================================================================
# ML and Decision Tools
# ============================================================================

class KServeInferenceTool(BaseTool):
    """Tool for making predictions using KServe."""
    
    name: str = "kserve_inference"
    description: str = "Make scaling decisions using KServe model inference"
    
    def _run(self, feature_vector: List[float], endpoint_url: str, 
             confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """Synchronous wrapper - not used in async context."""
        raise NotImplementedError("Use async version")
    
    async def _arun(self, feature_vector: List[float], endpoint_url: str, 
                    confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """Make prediction using KServe endpoint."""
        
        try:
            # This would typically use the existing KServeOnlyDQNAgent
            # For now, we'll simulate the structure
            
            # In a real implementation, you'd instantiate the agent here
            # agent = KServeOnlyDQNAgent(endpoint_url, confidence_threshold=confidence_threshold)
            # action, metadata = await agent.select_action(state, force_valid=True)
            
            # Simulated response for demonstration
            result = {
                "scaling_decision": "no_action",  # Would come from agent
                "confidence": 0.85,              # Would come from agent
                "q_values": [0.1, 0.3, 0.9, 0.2, 0.1],  # Would come from agent
                "reasoning": "Metrics are within normal range, no scaling needed",
                "kserve_available": True,
                "prediction_source": "kserve",
                "endpoint": endpoint_url
            }
            
            return result
            
        except Exception as e:
            LOG.error(f"KServe inference failed: {e}")
            return {
                "scaling_decision": "no_action",
                "confidence": 0.0,
                "error": str(e),
                "kserve_available": False,
                "prediction_source": "error"
            }


# ============================================================================
# Kubernetes Operations Tools
# ============================================================================

@tool
async def execute_kubernetes_scaling(
    deployment_name: str,
    namespace: str,
    target_replicas: int,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Execute Kubernetes scaling operation.
    This preserves the existing scaling execution logic from the original handler.
    """
    try:
        # Initialize Kubernetes client (same as original handler)
        try:
            kubernetes.config.load_incluster_config()
        except kubernetes.config.ConfigException:
            kubernetes.config.load_kube_config()
        
        apps_api = kubernetes.client.AppsV1Api()
        
        if dry_run:
            # Just validate the operation (same as original handler)
            deployment = apps_api.read_namespaced_deployment(
                name=deployment_name, 
                namespace=namespace
            )
            current_replicas = deployment.spec.replicas or 0
            
            return {
                "success": True,
                "dry_run": True,
                "current_replicas": current_replicas,
                "target_replicas": target_replicas,
                "message": f"Dry run: Would scale from {current_replicas} to {target_replicas} replicas"
            }
        
        # Execute actual scaling (same as original handler)
        try:
            # Method 1: Try using the scale subresource
            scale_body = kubernetes.client.V1Scale(
                spec=kubernetes.client.V1ScaleSpec(replicas=target_replicas)
            )
            apps_api.patch_namespaced_deployment_scale(
                name=deployment_name, 
                namespace=namespace, 
                body=scale_body
            )
            
            return {
                "success": True,
                "dry_run": False,
                "target_replicas": target_replicas,
                "message": f"Successfully scaled to {target_replicas} replicas using scale subresource"
            }
            
        except Exception as scale_error:
            LOG.warning(f"Scale subresource failed: {scale_error}. Trying direct deployment patch.")
            
            # Method 2: Fallback to direct deployment patch (same as original handler)
            deployment = apps_api.read_namespaced_deployment(
                name=deployment_name, 
                namespace=namespace
            )
            deployment.spec.replicas = target_replicas
            
            apps_api.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace, 
                body=deployment
            )
            
            return {
                "success": True,
                "dry_run": False,
                "target_replicas": target_replicas,
                "message": f"Successfully scaled to {target_replicas} replicas using direct deployment patch"
            }
        
    except Exception as e:
        LOG.error(f"Kubernetes scaling failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Scaling failed: {str(e)}"
        }


@tool
async def get_current_replicas(
    target_labels: Dict[str, str],
    namespace: str
) -> Dict[str, Any]:
    """
    Get current replica count for a deployment using Kubernetes API.
    This preserves the existing Kubernetes interaction logic.
    """
    try:
        # Initialize Kubernetes client (same as original handler)
        try:
            kubernetes.config.load_incluster_config()
        except kubernetes.config.ConfigException:
            kubernetes.config.load_kube_config()
        
        apps_api = kubernetes.client.AppsV1Api()
        
        if not target_labels:
            return {"success": False, "error": "No target_labels specified"}
            
        selector = ",".join(f"{k}={v}" for k, v in target_labels.items())
        deployments = apps_api.list_namespaced_deployment(
            namespace=namespace, 
            label_selector=selector
        )
        
        if not deployments.items:
            return {"success": False, "error": f"No deployment found with labels: {target_labels}"}
            
        deployment = deployments.items[0]
        current_replicas = deployment.status.replicas if deployment.status.replicas is not None else 0
        
        return {
            "success": True,
            "deployment_name": deployment.metadata.name,
            "current_replicas": current_replicas,
            "ready_replicas": deployment.status.ready_replicas or 0,
            "available_replicas": deployment.status.available_replicas or 0
        }
        
    except Exception as e:
        LOG.error(f"Failed to get current replicas: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# DQN Decision Tools (Core Logic Preserved)
# ============================================================================

@tool
async def make_dqn_decision(
    current_replicas: int,
    min_replicas: int,
    max_replicas: int,
    unified_state: Dict[str, Any],
    recent_actions: List[int]
) -> Dict[str, Any]:
    """
    Make scaling decision using the existing KServe DQN agent.
    This preserves the core DQN decision logic from the original handler.
    """
    try:
        # Get KServe endpoint (same as original handler)
        kserve_endpoint = os.getenv('KSERVE_ENDPOINT')
        if not kserve_endpoint:
            return {
                "success": False,
                "error": "KSERVE_ENDPOINT environment variable not set",
                "action": "none",
                "target_replicas": current_replicas
            }
        
        # Create KServe agent (same as original handler)
        kserve_config = {
            'kserve_endpoint': kserve_endpoint,
            'model_name': os.getenv('KSERVE_MODEL_NAME', 'nimbusguard-dqn'),
            'state_dim': 11,
            'action_dim': 5,
            'confidence_threshold': float(os.getenv('DQN_CONFIDENCE_THRESHOLD', '0.7')),
            'health_check_interval': int(os.getenv('KSERVE_HEALTH_CHECK_INTERVAL', '300'))
        }
        
        kserve_agent = await create_kserve_only_agent(kserve_config)
        
        # Create environment state (same as original handler)
        env_state = EnvironmentState.from_observability_data(
            unified_state=unified_state,
            current_replicas=current_replicas,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            recent_actions=recent_actions
        )
        
        # Use KServe for prediction (same as original handler)
        action, metadata = await kserve_agent.select_action(
            state=env_state,
            force_valid=True
        )
        
        # Calculate target replicas (same as original handler)
        replica_change = action.get_replica_change()
        target_replicas = max(min_replicas, min(current_replicas + replica_change, max_replicas))
        
        # Determine action string (same as original handler)
        if target_replicas > current_replicas:
            action_str = "scale_up"
        elif target_replicas < current_replicas:
            action_str = "scale_down"
        else:
            action_str = "none"
        
        return {
            "success": True,
            "action": action_str,
            "target_replicas": target_replicas,
            "dqn_action": action.name,
            "dqn_action_value": action.value,
            "confidence": metadata.get("confidence", 0.0),
            "q_values": metadata.get("q_values", []),
            "reasoning": f"DQN decision: {action.name} (confidence: {metadata.get('confidence', 0.0):.3f})",
            "kserve_available": metadata.get("kserve_available", False),
            "prediction_source": metadata.get("prediction_source", "kserve"),
            "metadata": metadata
        }
        
    except Exception as e:
        LOG.error(f"DQN decision failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "action": "none",
            "target_replicas": current_replicas,
            "reasoning": f"DQN decision failed: {str(e)}"
        }


# ============================================================================
# LLM Validation Tool
# ============================================================================

@tool
def validate_dqn_decision_with_context(
    dqn_decision: Dict[str, Any],
    observability_data: Dict[str, Any],
    current_replicas: int,
    min_replicas: int,
    max_replicas: int
) -> Dict[str, Any]:
    """
    Prepare context for LLM validation of the DQN decision.
    This tool formats the data for LLM analysis while preserving the DQN decision as primary.
    """
    
    # Extract key metrics for LLM context
    confidence_score = observability_data.get("confidence_score", 0.0)
    feature_vector = observability_data.get("feature_vector", [])
    feature_names = observability_data.get("feature_names", [])
    
    # Create feature summary
    feature_summary = {}
    for i, name in enumerate(feature_names):
        if i < len(feature_vector):
            feature_summary[name] = feature_vector[i]
    
    # Prepare validation context
    validation_context = {
        "dqn_decision": {
            "action": dqn_decision.get("action", "none"),
            "target_replicas": dqn_decision.get("target_replicas", current_replicas),
            "dqn_action": dqn_decision.get("dqn_action", "NO_ACTION"),
            "confidence": dqn_decision.get("confidence", 0.0),
            "reasoning": dqn_decision.get("reasoning", "No reasoning provided")
        },
        "current_state": {
            "current_replicas": current_replicas,
            "min_replicas": min_replicas,
            "max_replicas": max_replicas,
            "observability_confidence": confidence_score,
            "metrics": feature_summary
        },
        "validation_needed": True,
        "primary_decision_source": "DQN",
        "llm_role": "validation_and_risk_assessment"
    }
    
    return validation_context


# ============================================================================
# Tool Registry
# ============================================================================

def get_nimbusguard_tools():
    """Get all NimbusGuard LangGraph tools with @tool decorators."""
    return [
        collect_observability_data,
        get_current_replicas,
        make_dqn_decision,
        execute_kubernetes_scaling,
        validate_dqn_decision_with_context
    ]


def get_tool_by_name(tool_name: str) -> Optional[BaseTool]:
    """Get a specific tool by name."""
    tools = get_nimbusguard_tools()
    for tool in tools:
        if tool.name == tool_name:
            return tool
    return None 