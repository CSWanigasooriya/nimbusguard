"""
MLflow Experiment Tracker for NimbusGuard

This module handles experiment tracking for AI-powered scaling decisions,
Q-learning performance, and system metrics.
"""

import logging
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    MLflow experiment tracker for NimbusGuard scaling experiments.
    
    This tracker:
    - Creates and manages MLflow experiments
    - Logs scaling decisions and outcomes
    - Tracks Q-learning performance metrics
    - Records system performance data
    - Manages experiment lifecycle
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the experiment tracker.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # MLflow configuration
        self.tracking_uri = self.config.get("mlflow_tracking_uri", "http://localhost:5000")
        self.experiment_name = self.config.get("experiment_name", "nimbusguard-scaling")
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create or get experiment
        self.experiment = self._get_or_create_experiment()
        
        # Current run tracking
        self.current_run_id: Optional[str] = None
        
        logger.info(f"MLflow Experiment Tracker initialized for experiment: {self.experiment_name}")

    def _get_or_create_experiment(self):
        """Get existing experiment or create new one."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name=self.experiment_name,
                    tags={
                        "project": "nimbusguard",
                        "component": "ai-scaling",
                        "created_at": datetime.now().isoformat()
                    }
                )
                experiment = mlflow.get_experiment(experiment_id)
            
            logger.info(f"Using MLflow experiment: {experiment.name} (ID: {experiment.experiment_id})")
            return experiment
            
        except Exception as e:
            logger.error(f"Error setting up MLflow experiment: {e}")
            return None

    def start_scaling_run(self, workflow_id: str, trigger_event: Dict[str, Any]) -> str:
        """
        Start a new MLflow run for a scaling workflow.
        
        Args:
            workflow_id: Unique workflow identifier
            trigger_event: Event that triggered the scaling
            
        Returns:
            str: MLflow run ID
        """
        try:
            # Start new MLflow run
            run = mlflow.start_run(
                experiment_id=self.experiment.experiment_id,
                run_name=f"scaling-{workflow_id}",
                tags={
                    "workflow_id": workflow_id,
                    "trigger_event_type": trigger_event.get("event_type", "unknown"),
                    "service": trigger_event.get("service", "unknown"),
                    "started_at": datetime.now().isoformat()
                }
            )
            
            self.current_run_id = run.info.run_id
            
            # Log initial parameters
            mlflow.log_params({
                "workflow_id": workflow_id,
                "trigger_event_type": trigger_event.get("event_type"),
                "trigger_service": trigger_event.get("service"),
                "trigger_value": trigger_event.get("value")
            })
            
            logger.info(f"Started MLflow run: {self.current_run_id} for workflow: {workflow_id}")
            return self.current_run_id
            
        except Exception as e:
            logger.error(f"Error starting MLflow run: {e}")
            return None

    def log_metrics_observation(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log cluster metrics observation.
        
        Args:
            metrics: Cluster metrics dictionary
            step: Optional step number
        """
        try:
            if not self.current_run_id:
                logger.warning("No active MLflow run to log metrics")
                return
            
            # Log individual metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"cluster_{metric_name}", value, step=step)
            
            logger.debug(f"Logged cluster metrics: {len(metrics)} metrics")
            
        except Exception as e:
            logger.error(f"Error logging metrics observation: {e}")

    def log_scaling_decision(self, decision: Dict[str, Any], q_learning_data: Optional[Dict[str, Any]] = None):
        """
        Log scaling decision and Q-learning data.
        
        Args:
            decision: Scaling decision dictionary
            q_learning_data: Optional Q-learning specific data
        """
        try:
            if not self.current_run_id:
                logger.warning("No active MLflow run to log decision")
                return
            
            # Log decision parameters
            mlflow.log_params({
                "decision_action": decision.get("action"),
                "decision_target_replicas": decision.get("target_replicas"),
                "decision_confidence": decision.get("confidence"),
                "decision_reasoning": decision.get("reasoning", "")[:100]  # Truncate long reasoning
            })
            
            # Log decision metrics
            mlflow.log_metrics({
                "decision_confidence": decision.get("confidence", 0.0),
                "decision_target_replicas": decision.get("target_replicas", 0)
            })
            
            # Log Q-learning specific data
            if q_learning_data:
                mlflow.log_metrics({
                    "q_value": q_learning_data.get("q_value", 0.0),
                    "epsilon": q_learning_data.get("epsilon", 0.0),
                    "exploration": 1 if q_learning_data.get("exploration", False) else 0
                })
            
            logger.debug("Logged scaling decision to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging scaling decision: {e}")

    def log_action_execution(self, action_result: Dict[str, Any]):
        """
        Log scaling action execution results.
        
        Args:
            action_result: Action execution result dictionary
        """
        try:
            if not self.current_run_id:
                logger.warning("No active MLflow run to log action")
                return
            
            # Log action parameters
            mlflow.log_params({
                "action_status": action_result.get("status"),
                "action_execution_time_ms": action_result.get("execution_time_ms"),
                "action_details": action_result.get("details", "")[:100]
            })
            
            # Log action metrics
            mlflow.log_metrics({
                "action_success": 1 if action_result.get("status") == "SUCCESS" else 0,
                "action_execution_time_ms": action_result.get("execution_time_ms", 0),
                "actual_replicas": action_result.get("actual_replicas", 0)
            })
            
            logger.debug("Logged action execution to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging action execution: {e}")

    def log_reward_calculation(self, reward_result: Dict[str, Any]):
        """
        Log reward calculation results.
        
        Args:
            reward_result: Reward calculation result dictionary
        """
        try:
            if not self.current_run_id:
                logger.warning("No active MLflow run to log reward")
                return
            
            # Log reward metrics
            mlflow.log_metrics({
                "total_reward": reward_result.get("total_reward", 0.0),
                "resource_efficiency_reward": reward_result.get("resource_efficiency_reward", 0.0),
                "sla_compliance_reward": reward_result.get("sla_compliance_reward", 0.0),
                "stability_reward": reward_result.get("stability_reward", 0.0)
            })
            
            # Log reward parameters
            mlflow.log_params({
                "reward_reasoning": reward_result.get("reasoning", "")[:200]
            })
            
            logger.debug("Logged reward calculation to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging reward calculation: {e}")

    def log_q_learning_stats(self, q_stats: Dict[str, Any]):
        """
        Log Q-learning model statistics.
        
        Args:
            q_stats: Q-learning statistics dictionary
        """
        try:
            if not self.current_run_id:
                logger.warning("No active MLflow run to log Q-learning stats")
                return
            
            # Log Q-learning metrics
            mlflow.log_metrics({
                "q_episodes": q_stats.get("episodes", 0),
                "q_total_reward": q_stats.get("total_reward", 0.0),
                "q_epsilon": q_stats.get("epsilon", 0.0),
                "q_table_size": q_stats.get("q_table_size", 0)
            })
            
            logger.debug("Logged Q-learning statistics to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging Q-learning stats: {e}")

    def end_scaling_run(self, final_status: str, workflow_summary: Optional[Dict[str, Any]] = None):
        """
        End the current MLflow run.
        
        Args:
            final_status: Final workflow status
            workflow_summary: Optional workflow summary data
        """
        try:
            if not self.current_run_id:
                logger.warning("No active MLflow run to end")
                return
            
            # Log final status
            mlflow.log_params({
                "final_status": final_status,
                "completed_at": datetime.now().isoformat()
            })
            
            # Log workflow summary if provided
            if workflow_summary:
                mlflow.log_metrics({
                    "workflow_duration_seconds": workflow_summary.get("duration_seconds", 0),
                    "workflow_success": 1 if final_status == "COMPLETED" else 0
                })
            
            # End the run
            mlflow.end_run()
            
            logger.info(f"Ended MLflow run: {self.current_run_id} with status: {final_status}")
            self.current_run_id = None
            
        except Exception as e:
            logger.error(f"Error ending MLflow run: {e}")

    def get_experiment_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent experiment metrics and runs.
        
        Args:
            limit: Maximum number of runs to retrieve
            
        Returns:
            List of run data dictionaries
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                max_results=limit,
                order_by=["start_time DESC"]
            )
            
            return runs.to_dict('records') if not runs.empty else []
            
        except Exception as e:
            logger.error(f"Error getting experiment metrics: {e}")
            return []

    def get_model_performance_summary(self) -> Dict[str, float]:
        """
        Get performance summary across all experiments.
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            runs_data = self.get_experiment_metrics(limit=50)
            
            if not runs_data:
                return {}
            
            # Calculate average performance metrics
            total_rewards = [run.get("metrics.total_reward", 0) for run in runs_data if "metrics.total_reward" in run]
            success_rates = [run.get("metrics.workflow_success", 0) for run in runs_data if "metrics.workflow_success" in run]
            
            summary = {
                "avg_total_reward": sum(total_rewards) / len(total_rewards) if total_rewards else 0.0,
                "success_rate": sum(success_rates) / len(success_rates) if success_rates else 0.0,
                "total_experiments": len(runs_data)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting model performance summary: {e}")
            return {} 