"""
MCP Integration Tools for LangGraph

This module provides LangGraph tools that interact with MCP servers for
Kubernetes operations, Prometheus metrics, and other external systems.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for communicating with MCP servers."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MCP client.
        
        Args:
            config: MCP configuration from mcp_config.yaml
        """
        self.config = config
        self.servers = config.get("mcp_servers", {})
        self.tools_mapping = config.get("tools_mapping", {})
        
    async def call_mcp_tool(self, server_name: str, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on an MCP server.
        
        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool to call
            parameters: Parameters for the tool
            
        Returns:
            Tool execution result
        """
        try:
            # In a real implementation, this would establish a connection to the MCP server
            # and execute the tool. For now, we'll simulate the calls.
            logger.info(f"Calling MCP tool {tool_name} on server {server_name}")
            
            if server_name == "kubernetes":
                return await self._simulate_kubernetes_call(tool_name, parameters)
            elif server_name == "prometheus":
                return await self._simulate_prometheus_call(tool_name, parameters)
            elif server_name == "filesystem":
                return await self._simulate_filesystem_call(tool_name, parameters)
            else:
                raise ValueError(f"Unknown MCP server: {server_name}")
                
        except Exception as e:
            logger.error(f"MCP tool call failed: {e}")
            return {"error": str(e), "success": False}

    async def _simulate_kubernetes_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate Kubernetes MCP server calls."""
        
        if tool_name == "scale_deployment":
            deployment = parameters.get("deployment_name", "consumer-workload")
            namespace = parameters.get("namespace", "nimbusguard")
            replicas = parameters.get("replicas", 1)
            
            return {
                "success": True,
                "deployment": deployment,
                "namespace": namespace,
                "old_replicas": 2,  # Simulated
                "new_replicas": replicas,
                "message": f"Scaled {deployment} to {replicas} replicas"
            }
            
        elif tool_name == "get_deployment_status":
            deployment = parameters.get("deployment_name", "consumer-workload")
            namespace = parameters.get("namespace", "nimbusguard")
            
            return {
                "success": True,
                "deployment": deployment,
                "namespace": namespace,
                "replicas": {
                    "desired": 3,
                    "current": 3,
                    "ready": 3,
                    "available": 3
                },
                "status": "Running"
            }
            
        elif tool_name == "list_pods":
            namespace = parameters.get("namespace", "nimbusguard")
            
            return {
                "success": True,
                "pods": [
                    {"name": "consumer-workload-1", "status": "Running", "ready": True},
                    {"name": "consumer-workload-2", "status": "Running", "ready": True},
                    {"name": "consumer-workload-3", "status": "Running", "ready": True}
                ],
                "total_pods": 3
            }
        
        return {"error": f"Unknown Kubernetes tool: {tool_name}", "success": False}

    async def _simulate_prometheus_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate Prometheus MCP server calls."""
        import random
        
        if tool_name == "query_cpu_usage":
            return {
                "success": True,
                "metric": "cpu_utilization",
                "value": round(random.uniform(45, 85), 2),
                "timestamp": datetime.now().isoformat(),
                "query": parameters.get("query", "")
            }
            
        elif tool_name == "query_memory_usage":
            return {
                "success": True,
                "metric": "memory_utilization", 
                "value": round(random.uniform(40, 80), 2),
                "timestamp": datetime.now().isoformat(),
                "query": parameters.get("query", "")
            }
            
        elif tool_name == "query_request_rate":
            return {
                "success": True,
                "metric": "request_rate",
                "value": round(random.uniform(100, 1000), 2),
                "timestamp": datetime.now().isoformat(),
                "query": parameters.get("query", "")
            }
            
        elif tool_name == "query_error_rate":
            return {
                "success": True,
                "metric": "error_rate",
                "value": round(random.uniform(0.1, 5.0), 2),
                "timestamp": datetime.now().isoformat(),
                "query": parameters.get("query", "")
            }
        
        return {"error": f"Unknown Prometheus tool: {tool_name}", "success": False}

    async def _simulate_filesystem_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate filesystem MCP server calls."""
        
        if tool_name == "save_model":
            file_path = parameters.get("file_path", "/tmp/model.pkl")
            return {
                "success": True,
                "file_path": file_path,
                "message": f"Model saved to {file_path}"
            }
            
        elif tool_name == "load_model":
            file_path = parameters.get("file_path", "/tmp/model.pkl")
            return {
                "success": True,
                "file_path": file_path,
                "message": f"Model loaded from {file_path}",
                "model_data": {"episodes": 100, "avg_reward": 5.2}
            }
        
        return {"error": f"Unknown filesystem tool: {tool_name}", "success": False}


# Global MCP client instance
_mcp_client: Optional[MCPClient] = None

def get_mcp_client() -> MCPClient:
    """Get or create MCP client instance."""
    global _mcp_client
    if _mcp_client is None:
        # Load config from file or use defaults
        config = {
            "mcp_servers": {},
            "tools_mapping": {}
        }
        _mcp_client = MCPClient(config)
    return _mcp_client


# ============================================================================
# LangGraph Tools for Kubernetes Operations
# ============================================================================

@tool
async def scale_kubernetes_deployment(
    deployment_name: str,
    namespace: str,
    replicas: int
) -> str:
    """
    Scale a Kubernetes deployment to the specified number of replicas.
    
    Args:
        deployment_name: Name of the deployment to scale
        namespace: Kubernetes namespace
        replicas: Target number of replicas
        
    Returns:
        Result of the scaling operation
    """
    client = get_mcp_client()
    result = await client.call_mcp_tool(
        "kubernetes", 
        "scale_deployment",
        {
            "deployment_name": deployment_name,
            "namespace": namespace, 
            "replicas": replicas
        }
    )
    
    if result.get("success"):
        return f"Successfully scaled {deployment_name} to {replicas} replicas in namespace {namespace}"
    else:
        return f"Failed to scale deployment: {result.get('error', 'Unknown error')}"


@tool
async def get_kubernetes_deployment_status(
    deployment_name: str,
    namespace: str
) -> str:
    """
    Get the current status of a Kubernetes deployment.
    
    Args:
        deployment_name: Name of the deployment
        namespace: Kubernetes namespace
        
    Returns:
        Current deployment status
    """
    client = get_mcp_client()
    result = await client.call_mcp_tool(
        "kubernetes",
        "get_deployment_status", 
        {
            "deployment_name": deployment_name,
            "namespace": namespace
        }
    )
    
    if result.get("success"):
        replicas = result.get("replicas", {})
        return f"Deployment {deployment_name}: {replicas.get('ready', 0)}/{replicas.get('desired', 0)} replicas ready"
    else:
        return f"Failed to get deployment status: {result.get('error', 'Unknown error')}"


@tool
async def list_kubernetes_pods(namespace: str) -> str:
    """
    List all pods in a Kubernetes namespace.
    
    Args:
        namespace: Kubernetes namespace
        
    Returns:
        List of pods and their status
    """
    client = get_mcp_client()
    result = await client.call_mcp_tool(
        "kubernetes",
        "list_pods",
        {"namespace": namespace}
    )
    
    if result.get("success"):
        pods = result.get("pods", [])
        pod_summary = []
        for pod in pods:
            status = "✅" if pod.get("ready") else "❌"
            pod_summary.append(f"{status} {pod.get('name')} ({pod.get('status')})")
        
        return f"Pods in {namespace}:\n" + "\n".join(pod_summary)
    else:
        return f"Failed to list pods: {result.get('error', 'Unknown error')}"


# ============================================================================
# LangGraph Tools for Prometheus Metrics
# ============================================================================

@tool
async def get_cpu_utilization() -> str:
    """
    Get current CPU utilization from Prometheus.
    
    Returns:
        Current CPU utilization percentage
    """
    client = get_mcp_client()
    result = await client.call_mcp_tool(
        "prometheus",
        "query_cpu_usage",
        {"query": "avg(cpu_utilization)", "time_range": "5m"}
    )
    
    if result.get("success"):
        value = result.get("value", 0)
        return f"Current CPU utilization: {value}%"
    else:
        return f"Failed to get CPU metrics: {result.get('error', 'Unknown error')}"


@tool
async def get_memory_utilization() -> str:
    """
    Get current memory utilization from Prometheus.
    
    Returns:
        Current memory utilization percentage
    """
    client = get_mcp_client()
    result = await client.call_mcp_tool(
        "prometheus",
        "query_memory_usage",
        {"query": "avg(memory_utilization)", "time_range": "5m"}
    )
    
    if result.get("success"):
        value = result.get("value", 0)
        return f"Current memory utilization: {value}%"
    else:
        return f"Failed to get memory metrics: {result.get('error', 'Unknown error')}"


@tool
async def get_request_rate() -> str:
    """
    Get current request rate from Prometheus.
    
    Returns:
        Current request rate per second
    """
    client = get_mcp_client()
    result = await client.call_mcp_tool(
        "prometheus",
        "query_request_rate",
        {"query": "sum(rate(http_requests_total[5m]))", "time_range": "5m"}
    )
    
    if result.get("success"):
        value = result.get("value", 0)
        return f"Current request rate: {value} req/s"
    else:
        return f"Failed to get request rate: {result.get('error', 'Unknown error')}"


@tool
async def get_error_rate() -> str:
    """
    Get current error rate from Prometheus.
    
    Returns:
        Current error rate percentage
    """
    client = get_mcp_client()
    result = await client.call_mcp_tool(
        "prometheus",
        "query_error_rate",
        {"query": "sum(rate(http_errors_total[5m]))", "time_range": "5m"}
    )
    
    if result.get("success"):
        value = result.get("value", 0)
        return f"Current error rate: {value}%"
    else:
        return f"Failed to get error rate: {result.get('error', 'Unknown error')}"


@tool
async def get_cluster_metrics() -> str:
    """
    Get comprehensive cluster metrics from Prometheus.
    
    Returns:
        JSON string with all key metrics
    """
    client = get_mcp_client()
    
    # Gather all metrics in parallel
    tasks = [
        client.call_mcp_tool("prometheus", "query_cpu_usage", {"query": "cpu", "time_range": "5m"}),
        client.call_mcp_tool("prometheus", "query_memory_usage", {"query": "memory", "time_range": "5m"}),
        client.call_mcp_tool("prometheus", "query_request_rate", {"query": "requests", "time_range": "5m"}),
        client.call_mcp_tool("prometheus", "query_error_rate", {"query": "errors", "time_range": "5m"}),
        client.call_mcp_tool("kubernetes", "list_pods", {"namespace": "nimbusguard"})
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    metrics = {
        "cpu_utilization": results[0].get("value", 0) if isinstance(results[0], dict) else 0,
        "memory_utilization": results[1].get("value", 0) if isinstance(results[1], dict) else 0,
        "request_rate": results[2].get("value", 0) if isinstance(results[2], dict) else 0,
        "error_rate": results[3].get("value", 0) if isinstance(results[3], dict) else 0,
        "pod_count": len(results[4].get("pods", [])) if isinstance(results[4], dict) else 0,
        "timestamp": datetime.now().isoformat()
    }
    
    return json.dumps(metrics, indent=2)


# ============================================================================
# Tool Collections for Agents
# ============================================================================

# Tools for State Observer Agent
STATE_OBSERVER_TOOLS = [
    get_cpu_utilization,
    get_memory_utilization,
    get_request_rate,
    get_error_rate,
    get_cluster_metrics,
    list_kubernetes_pods
]

# Tools for Action Executor Agent
ACTION_EXECUTOR_TOOLS = [
    scale_kubernetes_deployment,
    get_kubernetes_deployment_status,
    list_kubernetes_pods
]

# Tools for all agents (monitoring and basic operations)
COMMON_TOOLS = [
    get_cluster_metrics,
    get_kubernetes_deployment_status
]


def get_tools_for_agent(agent_name: str) -> List:
    """
    Get the appropriate tools for a specific agent.
    
    Args:
        agent_name: Name of the agent
        
    Returns:
        List of tools for the agent
    """
    if agent_name == "state_observer":
        return STATE_OBSERVER_TOOLS
    elif agent_name == "action_executor":
        return ACTION_EXECUTOR_TOOLS
    else:
        return COMMON_TOOLS 