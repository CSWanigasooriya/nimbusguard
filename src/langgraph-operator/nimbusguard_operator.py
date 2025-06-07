#!/usr/bin/env python3
"""
NimbusGuard LangGraph Operator

AI-powered Kubernetes operator for intelligent autoscaling using LangGraph workflows
and Q-learning reinforcement learning. Follows proper Kubernetes operator patterns
with reconciliation loops and status management.
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import time

import kopf
import kubernetes
from kubernetes.client.rest import ApiException
import structlog
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Import our AI components
from workflows.scaling_state import (
    ScalingWorkflowState, 
    WorkflowStatus, 
    create_initial_state,
    MetricData,
    ActionResult
)
from agents.supervisor_agent import SupervisorAgent
from agents.state_observer_agent import StateObserverAgent  
from ml_models.q_learning import QLearningAgent

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class NimbusGuardController:
    """
    Main NimbusGuard controller that manages AI-powered scaling policies.
    Follows Kubernetes controller patterns with reconciliation loops.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.k8s_apps_v1 = kubernetes.client.AppsV1Api()
        self.k8s_core_v1 = kubernetes.client.CoreV1Api()
        
        # Initialize AI agents
        self.q_agent = QLearningAgent(config)
        self.supervisor = SupervisorAgent(config)
        self.state_observer = StateObserverAgent(config)
        
        # Track active policies and workflows
        self.active_policies: Dict[str, Dict[str, Any]] = {}
        self.active_workflows: Dict[str, ScalingWorkflowState] = {}
        
        # Metrics tracking
        self.start_time = time.time()
        
        # Prometheus metrics
        self.policies_total = Counter('nimbusguard_policies_total', 'Total scaling policies')
        self.workflows_total = Counter('nimbusguard_workflows_total', 'Total workflows created')
        self.workflows_successful = Counter('nimbusguard_workflows_successful_total', 'Total successful workflows')
        self.workflows_failed = Counter('nimbusguard_workflows_failed_total', 'Total failed workflows')
        self.scaling_actions_total = Counter('nimbusguard_scaling_actions_total', 'Total scaling actions executed')
        self.active_policies_gauge = Gauge('nimbusguard_active_policies', 'Current number of active policies')
        self.active_workflows_gauge = Gauge('nimbusguard_active_workflows', 'Current number of active workflows')
        self.controller_uptime = Gauge('nimbusguard_controller_uptime_seconds', 'Time since controller started')
        self.policies_by_phase = Gauge('nimbusguard_policies_by_phase', 'Policies by phase', ['phase'])
        self.q_learning_epsilon = Gauge('nimbusguard_q_learning_epsilon', 'Current epsilon value for exploration')
        self.q_learning_episodes = Counter('nimbusguard_q_learning_episodes_total', 'Total Q-learning episodes')
        
        logger.info("NimbusGuard controller initialized", 
                   version="0.1.0", 
                   agents=["supervisor", "state_observer", "q_learning"])

    def update_metrics(self):
        """Update live metrics."""
        # Update uptime
        uptime = time.time() - self.start_time
        self.controller_uptime.set(uptime)
        
        # Update active counts
        self.active_policies_gauge.set(len(self.active_policies))
        self.active_workflows_gauge.set(len(self.active_workflows))
        
        # Update policy phase metrics
        phase_counts = {}
        for policy in self.active_policies.values():
            phase = policy.get('status', {}).get('phase', 'Unknown')
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        # Clear and set phase metrics
        for phase in ['Initializing', 'Learning', 'Active', 'Paused', 'Error']:
            count = phase_counts.get(phase, 0)
            self.policies_by_phase.labels(phase=phase).set(count)
        
        # Update Q-learning metrics if available
        if hasattr(self.q_agent, 'epsilon'):
            self.q_learning_epsilon.set(self.q_agent.epsilon)
    
    def get_metrics(self) -> str:
        """
        Generate Prometheus-compatible metrics.
        
        Returns:
            str: Prometheus metrics format
        """
        # Update live metrics first
        self.update_metrics()
        
        # Return prometheus formatted metrics
        return generate_latest().decode('utf-8')

    async def reconcile_scaling_policy(self, spec: Dict[str, Any], status: Dict[str, Any], 
                                     name: str, namespace: str, uid: str) -> Dict[str, Any]:
        """
        Reconcile a scaling policy to ensure desired state matches actual state.
        This is the main controller reconciliation loop.
        
        Args:
            spec: ScalingPolicy specification
            status: Current status of the ScalingPolicy
            name: Name of the ScalingPolicy
            namespace: Namespace of the ScalingPolicy
            uid: UID of the ScalingPolicy
            
        Returns:
            Updated status dictionary
        """
        policy_key = f"{namespace}/{name}"
        logger.info("Reconciling scaling policy", policy=policy_key, uid=uid)
        
        try:
            # Initialize policy tracking
            if policy_key not in self.active_policies:
                self.active_policies[policy_key] = {
                    'spec': spec,
                    'status': status or {},
                    'uid': uid,
                    'last_reconcile': datetime.now().isoformat()
                }
                self.policies_total.inc()
            
            # Update policy spec if changed
            current_policy = self.active_policies[policy_key]
            current_policy['spec'] = spec
            current_policy['last_reconcile'] = datetime.now().isoformat()
            
            # Get target deployment
            target = spec['target']
            deployment_name = target['name']
            deployment_namespace = target.get('namespace', namespace)
            
            # Check if target deployment exists
            try:
                deployment = self.k8s_apps_v1.read_namespaced_deployment(
                    name=deployment_name,
                    namespace=deployment_namespace
                )
            except ApiException as e:
                if e.status == 404:
                    return await self._update_policy_status(
                        policy_key, "Error", 
                        error_message=f"Target deployment {deployment_namespace}/{deployment_name} not found"
                    )
                raise
            
            # Initialize status if new policy
            current_status = current_policy['status']
            if not current_status.get('phase'):
                current_status = await self._initialize_policy_status(policy_key, deployment)
            
            # Get current metrics and decide if scaling is needed
            current_replicas = deployment.spec.replicas
            current_status['currentReplicas'] = current_replicas
            
            # Check if we need to trigger a scaling decision
            needs_scaling_decision = await self._should_trigger_scaling_decision(
                policy_key, spec, current_status, deployment
            )
            
            if needs_scaling_decision:
                # Start scaling workflow
                workflow_id = await self._start_scaling_workflow(policy_key, spec, current_status)
                current_status['lastWorkflowId'] = workflow_id
                current_status['phase'] = "Learning"
            
            # Update policy status
            current_policy['status'] = current_status
            
            return current_status
            
        except Exception as e:
            logger.error("Error reconciling scaling policy", 
                        policy=policy_key, error=str(e), exc_info=True)
            
            return await self._update_policy_status(
                policy_key, "Error", 
                error_message=f"Reconciliation failed: {str(e)}"
            )

    async def _initialize_policy_status(self, policy_key: str, deployment) -> Dict[str, Any]:
        """Initialize status for a new scaling policy."""
        return {
            'phase': 'Initializing',
            'currentReplicas': deployment.spec.replicas,
            'targetReplicas': deployment.spec.replicas,
            'lastScalingTime': None,
            'lastDecision': None,
            'metrics': {
                'totalDecisions': 0,
                'successfulScalings': 0,
                'averageConfidence': 0.0,
                'learningProgress': 0.0
            },
            'conditions': [
                {
                    'type': 'Ready',
                    'status': 'True',
                    'reason': 'PolicyInitialized'
                }
            ]
        }

    async def _update_policy_status(self, policy_key: str, phase: str, 
                                  error_message: str = None) -> Dict[str, Any]:
        """Update policy status with new phase and optional error."""
        if policy_key in self.active_policies:
            status = self.active_policies[policy_key]['status']
            status['phase'] = phase
            
            if error_message:
                # Add error condition
                status['conditions'] = [
                    {
                        'type': 'Ready',
                        'status': 'False',
                        'reason': 'Error',
                        'message': error_message
                    }
                ]
            
            return status
        
        return {'phase': phase}

    async def _should_trigger_scaling_decision(self, policy_key: str, spec: Dict[str, Any], 
                                             status: Dict[str, Any], deployment) -> bool:
        """
        Determine if we should trigger a new scaling decision based on:
        - Cooldown period
        - Current metrics
        - Policy configuration
        """
        # Check cooldown period
        cooldown_seconds = spec.get('scaling', {}).get('cooldownPeriod', 300)
        last_scaling_time = status.get('lastScalingTime')
        
        if last_scaling_time:
            try:
                last_time = datetime.fromisoformat(last_scaling_time.replace('Z', '+00:00'))
                if (datetime.now() - last_time.replace(tzinfo=None)).total_seconds() < cooldown_seconds:
                    return False
            except:
                pass  # If time parsing fails, proceed with decision
        
        # TODO: Add metrics-based triggering logic here
        # For now, trigger every 5 minutes during active phase
        if status.get('phase') == 'Active':
            # Simple time-based triggering for demo
            return True
        
        # Always trigger for new policies
        return status.get('phase') == 'Initializing'

    async def _start_scaling_workflow(self, policy_key: str, spec: Dict[str, Any], 
                                    status: Dict[str, Any]) -> str:
        """Start a new scaling workflow for the policy."""
        workflow_id = f"workflow-{uuid.uuid4().hex[:8]}"
        
        # Create initial workflow state
        event_data = {
            'event_type': 'policy_reconciliation',
            'policy': policy_key,
            'spec': spec,
            'timestamp': datetime.now().isoformat()
        }
        
        initial_state = create_initial_state(
            workflow_id=workflow_id,
            trigger_event=event_data,
            config=self.config
        )
        
        # Store workflow
        self.active_workflows[workflow_id] = initial_state
        
        # Update metrics
        self.workflows_total.inc()
        
        logger.info("Started scaling workflow", 
                   workflow_id=workflow_id,
                   policy=policy_key)
        
        # Start the workflow execution
        asyncio.create_task(self._execute_workflow(workflow_id, policy_key))
        
        return workflow_id

    async def _execute_workflow(self, workflow_id: str, policy_key: str):
        """Execute a scaling workflow using the supervisor and specialized agents."""
        try:
            state = self.active_workflows[workflow_id]
            state["status"] = WorkflowStatus.OBSERVING
            
            # Execute workflow steps
            max_iterations = 20
            iteration = 0
            
            while not state["completed"] and iteration < max_iterations:
                iteration += 1
                
                # Route to appropriate agent via supervisor
                command = await self.supervisor.invoke(state)
                
                # Execute command based on supervisor decision
                if command.update:
                    state.update(command.update)
                
                if command.goto == "__end__":
                    state["completed"] = True
                    state["status"] = WorkflowStatus.COMPLETED
                    break
                elif command.goto == "state_observer":
                    await self._handle_state_observation(state)
                elif command.goto == "decision_agent":
                    await self._handle_decision_making(state)
                elif command.goto == "action_executor":
                    await self._handle_action_execution(state, policy_key)
                elif command.goto == "reward_calculator":
                    await self._handle_reward_calculation(state)
                
                await asyncio.sleep(0.1)
            
            # Clean up completed workflow
            if state["completed"]:
                # Update metrics based on final status
                if state["status"] == WorkflowStatus.COMPLETED:
                    self.workflows_successful.inc()
                    # Update policy status
                    await self._update_workflow_completion(workflow_id, policy_key, True)
                else:
                    self.workflows_failed.inc()
                    await self._update_workflow_completion(workflow_id, policy_key, False)
                
                logger.info("Workflow completed", 
                           workflow_id=workflow_id,
                           iterations=iteration)
                del self.active_workflows[workflow_id]
            
        except Exception as e:
            logger.error("Workflow execution error", 
                        workflow_id=workflow_id, 
                        error=str(e))
            
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id]["status"] = WorkflowStatus.FAILED
                self.active_workflows[workflow_id]["errors"].append(str(e))
                # Update metrics for failed workflows
                self.workflows_failed.inc()
                await self._update_workflow_completion(workflow_id, policy_key, False)

    async def _update_workflow_completion(self, workflow_id: str, policy_key: str, success: bool):
        """Update policy status when workflow completes."""
        if policy_key in self.active_policies:
            policy = self.active_policies[policy_key]
            status = policy['status']
            
            # Update metrics
            status['metrics']['totalDecisions'] += 1
            if success:
                status['metrics']['successfulScalings'] += 1
            
            # Update phase
            if success:
                status['phase'] = 'Active'
                status['lastScalingTime'] = datetime.now().isoformat()
            
            # Update conditions
            status['conditions'] = [
                {
                    'type': 'Ready',
                    'status': 'True' if success else 'False',
                    'reason': 'WorkflowCompleted' if success else 'WorkflowFailed'
                }
            ]

    async def _handle_state_observation(self, state: ScalingWorkflowState):
        """Handle state observation phase."""
        try:
            state["status"] = WorkflowStatus.OBSERVING
            state["current_agent"] = "state_observer"
            
            # Get current metrics
            metrics = await self.state_observer.collect_metrics()
            
            if metrics:
                metric_data = MetricData(
                    cpu_utilization=metrics.get("cpu_utilization", 0),
                    memory_utilization=metrics.get("memory_utilization", 0),
                    request_rate=metrics.get("request_rate", 0),
                    error_rate=metrics.get("error_rate", 0),
                    pod_count=metrics.get("pod_count", 1)
                )
                
                state["current_metrics"] = metric_data
                state["metrics_history"].append(metric_data)
                
                logger.info("Collected metrics", 
                           workflow_id=state["workflow_id"],
                           cpu=metric_data.cpu_utilization,
                           memory=metric_data.memory_utilization)
            
        except Exception as e:
            logger.error("State observation error", error=str(e))
            state["errors"].append(f"State observation failed: {str(e)}")

    async def _handle_decision_making(self, state: ScalingWorkflowState):
        """Handle AI decision making phase."""
        try:
            state["status"] = WorkflowStatus.DECIDING
            state["current_agent"] = "decision_agent"
            
            if not state["current_metrics"]:
                logger.warning("No metrics available for decision making")
                return
            
            # Convert metrics for Q-learning agent
            metrics_dict = {
                "cpu_utilization": state["current_metrics"].cpu_utilization,
                "memory_utilization": state["current_metrics"].memory_utilization,
                "request_rate": state["current_metrics"].request_rate,
                "error_rate": state["current_metrics"].error_rate,
                "pod_count": state["current_metrics"].pod_count
            }
            
            # Get Q-learning recommendation
            q_decision = self.q_agent.choose_scaling_action(metrics_dict)
            
            # Create scaling decision
            from workflows.scaling_state import ScalingDecision, ScalingAction
            
            action_map = {
                "scale_up": ScalingAction.SCALE_UP,
                "scale_down": ScalingAction.SCALE_DOWN,
                "no_action": ScalingAction.MAINTAIN
            }
            
            decision = ScalingDecision(
                action=action_map.get(q_decision["action"], ScalingAction.MAINTAIN),
                target_replicas=q_decision["target_replicas"],
                current_replicas=state["current_metrics"].pod_count,
                confidence=q_decision["confidence"],
                reasoning=q_decision["reasoning"],
                metrics_snapshot=state["current_metrics"]
            )
            
            state["scaling_decision"] = decision
            state["decisions_history"].append(decision)
            
            logger.info("AI decision made", 
                       workflow_id=state["workflow_id"],
                       action=decision.action.value,
                       target_replicas=decision.target_replicas)
            
        except Exception as e:
            logger.error("Decision making error", error=str(e))
            state["errors"].append(f"Decision making failed: {str(e)}")

    async def _handle_action_execution(self, state: ScalingWorkflowState, policy_key: str):
        """Handle scaling action execution."""
        try:
            state["status"] = WorkflowStatus.EXECUTING
            state["current_agent"] = "action_executor"
            
            decision = state["scaling_decision"]
            if not decision:
                logger.warning("No scaling decision available")
                return
            
            # Execute the scaling action
            result = await self._execute_scaling_action(decision, policy_key)
            
            state["last_action"] = result
            state["action_history"].append(result)
            
            logger.info("Action executed", 
                       workflow_id=state["workflow_id"],
                       success=result.success)
            
        except Exception as e:
            logger.error("Action execution error", error=str(e))
            state["errors"].append(f"Action execution failed: {str(e)}")

    async def _handle_reward_calculation(self, state: ScalingWorkflowState):
        """Handle reward calculation for Q-learning."""
        try:
            state["current_agent"] = "reward_calculator"
            
            # Get metrics after action
            post_action_metrics = await self.state_observer.collect_metrics()
            
            if state["last_action"] and post_action_metrics:
                # Calculate reward using Q-learning module
                from ml_models.q_learning import calculate_reward
                
                pre_metrics = {
                    "cpu_utilization": state["scaling_decision"].metrics_snapshot.cpu_utilization,
                    "memory_utilization": state["scaling_decision"].metrics_snapshot.memory_utilization,
                    "request_rate": state["scaling_decision"].metrics_snapshot.request_rate,
                    "error_rate": state["scaling_decision"].metrics_snapshot.error_rate,
                    "pod_count": state["scaling_decision"].metrics_snapshot.pod_count
                }
                
                reward, components = calculate_reward(
                    pre_metrics, 
                    post_action_metrics,
                    state["last_action"].action_taken.value,
                    self.config
                )
                
                # Provide feedback to Q-learning agent
                self.q_agent.provide_feedback(reward, post_action_metrics)
                
                # Store reward
                from workflows.scaling_state import RewardSignal
                reward_signal = RewardSignal(
                    reward=reward,
                    components=components,
                    metrics_before=state["scaling_decision"].metrics_snapshot,
                    metrics_after=None  # Will be populated later
                )
                
                state["last_reward"] = reward_signal
                state["reward_history"].append(reward_signal)
                state["cumulative_reward"] += reward
                
                logger.info("Reward calculated", 
                           workflow_id=state["workflow_id"],
                           reward=reward)
            
        except Exception as e:
            logger.error("Reward calculation error", error=str(e))
            state["errors"].append(f"Reward calculation failed: {str(e)}")

    async def _execute_scaling_action(self, decision, policy_key: str) -> ActionResult:
        """Execute the actual scaling action on Kubernetes."""
        try:
            # Get policy configuration
            policy = self.active_policies.get(policy_key)
            if not policy:
                raise Exception(f"Policy {policy_key} not found")
            
            target = policy['spec']['target']
            deployment_name = target['name']
            deployment_namespace = target.get('namespace', 'default')
            
            # Get current deployment
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=deployment_name, 
                namespace=deployment_namespace
            )
            
            current_replicas = deployment.spec.replicas
            target_replicas = decision.target_replicas
            
            # Update deployment if needed
            if current_replicas != target_replicas and decision.action.value != "maintain":
                deployment.spec.replicas = target_replicas
                
                self.k8s_apps_v1.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=deployment_namespace,
                    body=deployment
                )
                
                # Update scaling actions metric
                self.scaling_actions_total.inc()
                
                return ActionResult(
                    success=True,
                    action_taken=decision.action,
                    old_replicas=current_replicas,
                    new_replicas=target_replicas
                )
            else:
                return ActionResult(
                    success=True,
                    action_taken=decision.action,
                    old_replicas=current_replicas,
                    new_replicas=current_replicas
                )
                
        except ApiException as e:
            logger.error("Kubernetes API error", error=str(e))
            return ActionResult(
                success=False,
                action_taken=decision.action,
                old_replicas=decision.current_replicas,
                new_replicas=decision.current_replicas,
                error_message=str(e)
            )
        except Exception as e:
            logger.error("Scaling action error", error=str(e))
            return ActionResult(
                success=False,
                action_taken=decision.action,
                old_replicas=decision.current_replicas,
                new_replicas=decision.current_replicas,
                error_message=str(e)
            )


# Global operator instance
nimbusguard_controller: Optional[NimbusGuardController] = None


@kopf.on.startup()
async def startup_handler(settings: kopf.OperatorSettings, **kwargs):
    """Initialize the operator on startup."""
    global nimbusguard_controller
    
    # Configure kopf settings
    settings.posting.level = logging.INFO
    settings.watching.connect_timeout = 1 * 60
    settings.watching.server_timeout = 10 * 60
    
    # Load configuration
    config = {
        "agents": {
            "supervisor": {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 1000
            }
        },
        "q_learning": {
            "learning_rate": 0.1,
            "discount_factor": 0.95,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995
        }
    }
    
    # Initialize Kubernetes client
    try:
        kubernetes.config.load_incluster_config()
        logger.info("Loaded in-cluster Kubernetes config")
    except kubernetes.config.ConfigException:
        try:
            kubernetes.config.load_kube_config()
            logger.info("Loaded local Kubernetes config")
        except kubernetes.config.ConfigException:
            logger.error("Could not load Kubernetes config")
            raise
    
    # Initialize operator
    nimbusguard_controller = NimbusGuardController(config)
    
    logger.info("NimbusGuard controller started successfully")


@kopf.on.cleanup()
async def cleanup_handler(**kwargs):
    """Clean up on operator shutdown."""
    global nimbusguard_controller
    
    if nimbusguard_controller:
        # Save Q-learning model
        try:
            nimbusguard_controller.q_agent.save_model("/tmp/q_learning_model.pkl")
            logger.info("Saved Q-learning model on shutdown")
        except Exception as e:
            logger.error("Failed to save Q-learning model", error=str(e))
    
    logger.info("NimbusGuard controller shut down successfully")


# Proper Controller Pattern Handlers
@kopf.on.create('nimbusguard.io', 'v1', 'scalingpolicies')
async def create_scaling_policy(spec, status, name, namespace, uid, logger, **kwargs):
    """Handle creation of NimbusGuard scaling policies using controller pattern."""
    global nimbusguard_controller
    
    if not nimbusguard_controller:
        raise kopf.TemporaryError("Controller not ready", delay=10)
    
    logger.info(f"Creating scaling policy: {namespace}/{name}")
    
    # Reconcile the new policy
    updated_status = await nimbusguard_controller.reconcile_scaling_policy(
        spec, status, name, namespace, uid
    )
    
    return {"status": updated_status}


@kopf.on.update('nimbusguard.io', 'v1', 'scalingpolicies')
async def update_scaling_policy(spec, status, name, namespace, uid, logger, **kwargs):
    """Handle updates to NimbusGuard scaling policies using controller pattern."""
    global nimbusguard_controller
    
    if not nimbusguard_controller:
        raise kopf.TemporaryError("Controller not ready", delay=10)
    
    logger.info(f"Updating scaling policy: {namespace}/{name}")
    
    # Reconcile the updated policy
    updated_status = await nimbusguard_controller.reconcile_scaling_policy(
        spec, status, name, namespace, uid
    )
    
    return {"status": updated_status}


@kopf.on.resume('nimbusguard.io', 'v1', 'scalingpolicies')
async def resume_scaling_policy(spec, status, name, namespace, uid, logger, **kwargs):
    """Handle resuming of NimbusGuard scaling policies (e.g., operator restart)."""
    global nimbusguard_controller
    
    if not nimbusguard_controller:
        raise kopf.TemporaryError("Controller not ready", delay=10)
    
    logger.info(f"Resuming scaling policy: {namespace}/{name}")
    
    # Reconcile the resumed policy
    updated_status = await nimbusguard_controller.reconcile_scaling_policy(
        spec, status, name, namespace, uid
    )
    
    return {"status": updated_status}


@kopf.on.delete('nimbusguard.io', 'v1', 'scalingpolicies')
async def delete_scaling_policy(spec, name, namespace, logger, **kwargs):
    """Handle deletion of NimbusGuard scaling policies."""
    global nimbusguard_controller
    
    if not nimbusguard_controller:
        return  # Controller already shut down
    
    logger.info(f"Deleting scaling policy: {namespace}/{name}")
    
    # Clean up policy tracking
    policy_key = f"{namespace}/{name}"
    if policy_key in nimbusguard_controller.active_policies:
        del nimbusguard_controller.active_policies[policy_key]
    
    return {"status": "deleted", "message": f"Scaling policy {name} deleted successfully"}


@kopf.on.probe(id='liveness')
def liveness_probe(**kwargs):
    """Liveness probe endpoint."""
    return {'status': 'alive', 'timestamp': datetime.now().isoformat()}


@kopf.on.probe(id='readiness')
def readiness_probe(**kwargs):
    """Readiness probe endpoint."""
    global nimbusguard_controller
    
    if not nimbusguard_controller:
        raise kopf.TemporaryError("Controller not ready")
    
    return {'status': 'ready', 'timestamp': datetime.now().isoformat()}


@kopf.on.probe(id='metrics')
def metrics_probe(**kwargs):
    """Metrics probe endpoint for Prometheus."""
    global nimbusguard_controller
    
    try:
        if not nimbusguard_controller:
            return "# Controller not initialized\n"
        
        # Return Prometheus metrics
        return nimbusguard_controller.get_metrics()
        
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        return f"# Error generating metrics: {e}\n"


if __name__ == "__main__":
    # Run the operator with health endpoint enabled
    kopf.run(
        standalone=True,
        liveness_endpoint="0.0.0.0:8080/healthz"
    ) 