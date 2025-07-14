"""Main forecasting predictor combining data preprocessing and LSTM model."""
import logging
import threading
from datetime import datetime
from typing import Dict, Optional, Any

import numpy as np
from config.settings import load_config

config = load_config()

logger = logging.getLogger(__name__)


class LoadForecaster:
    """Main forecasting system for proactive scaling predictions using CPU and Memory metrics only."""

    def __init__(self):
        from prometheus.client import PrometheusClient
        from forecasting.data_prep import DataPreprocessor
        from forecasting.model import LSTMTrainer
        
        self.prometheus_client = PrometheusClient()
        self.data_preprocessor = DataPreprocessor()
        self.lstm_trainer = LSTMTrainer()

        self.is_initialized = False
        self.last_training_time = None
        self.last_prediction = None
        self.prediction_confidence = 0.0

        # Threading for background training
        self.training_lock = threading.Lock()
        self.is_training = False

        # Configurable forecasting parameters with fallbacks
        self._initialize_config()

        logger.info(f"LoadForecaster initialized with CPU and Memory metrics only: {self._get_config_summary()}")

    def _initialize_config(self):
        """Initialize configurable parameters with safe defaults for CPU and Memory only."""
        # Forecasting thresholds (per-replica basis) - Only CPU and Memory
        self.scaling_thresholds = {
            "load_trend_up": getattr(config, 'forecast_load_trend_up_threshold', 0.15),  # 15% per-replica increase
            "load_trend_down": getattr(config, 'forecast_load_trend_down_threshold', -0.15),  # 15% per-replica decrease
            "cpu_change_up": getattr(config, 'forecast_cpu_change_up_threshold', 25.0),  # 25% CPU increase
            "cpu_change_down": getattr(config, 'forecast_cpu_change_down_threshold', -20.0),  # 20% CPU decrease
            "memory_change_up": getattr(config, 'forecast_memory_change_up_threshold', 20.0),  # 20% memory increase
            "memory_change_down": getattr(config, 'forecast_memory_change_down_threshold', -20.0),  # 20% memory decrease
        }

        # Confidence calculation weights
        self.confidence_weights = {
            "variance": getattr(config, 'forecast_confidence_variance_weight', 0.5),
            "data_quality": getattr(config, 'forecast_confidence_data_weight', 0.3),
            "model_status": getattr(config, 'forecast_confidence_model_weight', 0.2)
        }

        # Timing and data parameters
        self.lookback_minutes = getattr(config, 'forecast_lookback_minutes', 60)
        self.forecast_horizon_minutes = getattr(config, 'forecast_horizon_minutes', 10)
        self.min_training_samples = getattr(config, 'forecast_min_training_samples', 40)
        self.retrain_interval_minutes = getattr(config, 'forecast_retrain_interval_minutes', 60)
        self.min_data_quality_samples = getattr(config, 'forecast_min_data_quality_samples', 5)

        # Target metrics for per-replica scaling decisions - Updated for CPU and Memory
        self.target_cpu_cores = getattr(config, 'target_cpu_cores', 0.5)  # 500m
        self.target_memory_mb = getattr(config, 'target_memory_mb', 1024)  # 1Gi
        self.target_cpu_utilization = getattr(config, 'target_cpu_utilization', 0.70)

        logger.debug(f"Forecasting thresholds: {self.scaling_thresholds}")
        logger.debug(f"Confidence weights: {self.confidence_weights}")

    def _get_config_summary(self) -> str:
        """Get summary of current configuration for logging."""
        return (f"thresholds={self.scaling_thresholds}, "
                f"lookback={self.lookback_minutes}min, "
                f"horizon={self.forecast_horizon_minutes}min, "
                f"min_samples={self.min_training_samples}")

    @staticmethod
    def _validate_metrics(feature_matrix: np.ndarray, expected_features: list) -> Dict[str, int]:
        """Validate that expected features are present and get their indices."""
        if feature_matrix is None or feature_matrix.size == 0:
            logger.warning("⚠️ Empty or None feature matrix provided")
            return {}

        if len(feature_matrix.shape) != 2:
            logger.warning(f"⚠️ Invalid feature matrix shape: {feature_matrix.shape}")
            return {}

        num_features = feature_matrix.shape[1]
        logger.debug(f"Feature matrix validation: shape={feature_matrix.shape}, expected_features={len(expected_features)}")

        # Map available features to indices
        feature_indices = {}
        missing_features = []

        for i, feature_name in enumerate(expected_features):
            if i < num_features:
                feature_indices[feature_name] = i
            else:
                missing_features.append(feature_name)

        if missing_features:
            logger.warning(f"⚠️ Missing features (will use defaults): {missing_features}")

        logger.debug(f"Available feature mapping: {feature_indices}")
        return feature_indices

    def _validate_replicas(self, replicas: int, context: str = "") -> int:
        """Validate replica count and handle edge cases."""
        if replicas < 0:
            logger.error(f"⚠️ CRITICAL: Negative replica count: {replicas} {context}")
            return 1
        elif replicas == 0:
            logger.warning(f"⚠️ CRITICAL: Zero replicas detected {context}")
            return 1
        else:
            return replicas

    def initialize(self) -> bool:
        """Initialize the forecasting system."""
        try:
            logger.debug("[initialize] Starting initialization of LoadForecaster")
            
            # Check Prometheus connectivity
            if not self.prometheus_client.health_check():
                logger.error("Prometheus health check failed")
                return False
            logger.debug("[initialize] Prometheus health check passed")

            # Try to load existing models
            scaler_loaded = self.data_preprocessor.load_scaler()
            logger.debug(f"[initialize] Scaler loaded: {scaler_loaded}")

            # Try to collect some initial data to determine input size
            feature_matrix = self.prometheus_client.get_feature_matrix(self.lookback_minutes)
            logger.debug(f"[initialize] Feature matrix type: {type(feature_matrix)}, shape: {getattr(feature_matrix, 'shape', None)}")

            # Fit scaler if not loaded and data is available
            if not scaler_loaded and feature_matrix is not None and feature_matrix.size > 0:
                logger.info("[initialize] Scaler not loaded, fitting scaler on initial feature matrix")
                self.data_preprocessor.fit_scaler(feature_matrix)
                logger.info("[initialize] Scaler fitted and saved on startup")

            if feature_matrix is not None and feature_matrix.size > 0:
                input_size = feature_matrix.shape[1]
                logger.debug(f"[initialize] Input size determined: {input_size}")
                self.lstm_trainer.load_model(input_size)
                logger.debug(f"[initialize] LSTM model loaded. is_trained: {self.lstm_trainer.is_trained}")

                # If we have enough data and no trained model, train one
                if not self.lstm_trainer.is_trained and len(feature_matrix) >= self.min_training_samples:
                    logger.info("No trained model found, training initial model")
                    self._train_model_async(feature_matrix)
                else:
                    logger.debug(f"[initialize] Skipping training: is_trained={self.lstm_trainer.is_trained}, data_len={len(feature_matrix)}")
            else:
                logger.warning("[initialize] Feature matrix is None or empty, skipping model loading and training")

            self.is_initialized = True
            logger.info("LoadForecaster initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize LoadForecaster: {e}")
            return False

    def generate_forecast(self, current_replicas: int = 1, deployment_info: Optional[Dict[str, Any]] = None) -> Dict[str, any]:
        """Generate load forecast for the next horizon with deployment context."""
        if not self.is_initialized:
            logger.warning("Forecaster not initialized")
            return self._get_fallback_forecast()

        # Validate inputs
        validated_replicas = self._validate_replicas(current_replicas, "in forecast generation")

        try:
            # Get recent data for prediction
            prediction_lookback = min(self.lookback_minutes, 30)  # Last 30 minutes for prediction
            feature_matrix = self.prometheus_client.get_feature_matrix(prediction_lookback)

            if feature_matrix is None or feature_matrix.size == 0:
                logger.warning("No recent data available for forecasting")
                return self._get_fallback_forecast()

            # Validate data quality
            if not self.data_preprocessor.validate_data_quality(feature_matrix):
                logger.warning("Poor data quality, using fallback")
                return self._get_fallback_forecast()

            # Get expected features - Only CPU and Memory
            expected_features = getattr(config, 'scaling_selected_features', [
                'process_cpu_seconds_total_rate',
                'process_resident_memory_bytes'
            ])

            # Validate feature matrix structure
            feature_indices = self._validate_metrics(feature_matrix, expected_features)
            if not feature_indices:
                logger.warning("No valid features found in matrix")
                return self._get_fallback_forecast()

            # Prepare input for prediction
            prediction_input = self.data_preprocessor.prepare_prediction_input(feature_matrix)

            if prediction_input is None:
                logger.warning("Failed to prepare prediction input")
                return self._get_fallback_forecast()

            # Check if we should train LSTM model if not already trained
            if not self.lstm_trainer.is_trained and len(feature_matrix) >= self.min_training_samples:
                logger.info(f"🔄 LSTM not trained but have {len(feature_matrix)} data points (min: {self.min_training_samples}) - starting training")
                self._schedule_retrain()

            # Make prediction if model is trained
            if self.lstm_trainer.is_trained:
                predictions = self.lstm_trainer.predict(prediction_input)

                if predictions is not None:
                    forecast_result = self._analyze_predictions(
                        predictions, feature_matrix, feature_indices, validated_replicas, deployment_info
                    )
                    forecast_result["method"] = "lstm"
                    forecast_result["confidence"] = min(0.9, forecast_result.get("confidence", 0.5))
                    self.last_prediction = forecast_result

                    # Check if we need to retrain
                    self._check_retrain_schedule()

                    logger.info(f"✅ LSTM forecast generated: confidence={forecast_result['confidence']:.3f}, recommendation={forecast_result['recommendation']}")
                    return forecast_result

            # If no trained model, use statistical fallback
            if len(feature_matrix) >= self.min_data_quality_samples:
                logger.info(f"📊 No trained LSTM model, using statistical forecast for {len(feature_matrix)} samples")
            else:
                logger.info(f"⚠️ Insufficient data ({len(feature_matrix)}/{self.min_data_quality_samples}), using statistical forecast")

            return self._get_statistical_forecast(feature_matrix, feature_indices, validated_replicas, deployment_info)

        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            return self._get_fallback_forecast()

    def _analyze_predictions(self, predictions: np.ndarray, current_data: np.ndarray, 
                           feature_indices: Dict[str, int], current_replicas: int,
                           deployment_info: Optional[Dict[str, Any]] = None) -> Dict[str, any]:
        """Analyze LSTM predictions and generate forecast result with deployment context."""
        try:
            # predictions shape: [1, forecast_horizon, num_features]
            forecast_values = predictions[0]  # Remove batch dimension

            # Get current values (last row of current data)
            current_values = current_data[-1] if len(current_data) > 0 else np.zeros(predictions.shape[2])

            # Calculate key metrics for scaling decision
            forecast_summary = self._calculate_forecast_metrics(
                forecast_values, current_values, feature_indices, current_replicas, deployment_info
            )

            # Determine scaling recommendation
            recommendation = self._get_scaling_recommendation(forecast_summary, current_replicas, deployment_info)

            # Calculate confidence based on prediction consistency
            confidence = self._calculate_confidence(forecast_values, current_values)

            self.prediction_confidence = confidence

            # Build current metrics dict safely
            current_metrics = {}
            for feature_name, idx in feature_indices.items():
                if idx < len(current_values):
                    current_metrics[feature_name] = float(current_values[idx])

            # Build predicted peaks dict safely
            predicted_peak = {}
            for feature_name, idx in feature_indices.items():
                if idx < forecast_values.shape[1]:
                    predicted_peak[feature_name] = float(np.max(forecast_values[:, idx]))

            result = {
                "timestamp": datetime.utcnow().isoformat(),
                "horizon_minutes": self.forecast_horizon_minutes,
                "recommendation": recommendation,
                "confidence": confidence,
                "forecast_summary": forecast_summary,
                "current_metrics": current_metrics,
                "predicted_peak": predicted_peak,
                "deployment_context": {
                    "current_replicas": current_replicas,
                    "has_resource_limits": bool(deployment_info and deployment_info.get('resource_limits'))
                }
            }

            logger.info(f"Generated forecast: {recommendation} (confidence: {confidence:.2f}, replicas: {current_replicas})")
            return result

        except Exception as e:
            logger.error(f"Failed to analyze predictions: {e}")
            return self._get_fallback_forecast()

    def _calculate_forecast_metrics(self, forecast_values: np.ndarray, current_values: np.ndarray,
                                  feature_indices: Dict[str, int], current_replicas: int,
                                  deployment_info: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Calculate key metrics from forecast for scaling decisions with per-replica awareness and health detection."""
        
        # Safely get feature indices with validation
        cpu_rate_idx = feature_indices.get("process_cpu_seconds_total_rate")
        memory_idx = feature_indices.get("process_resident_memory_bytes")

        # Initialize metrics with safe defaults
        metrics = {
            "cpu_change_percent": 0.0,
            "memory_change_percent": 0.0,
            "cpu_peak": 0.0,
            "memory_peak": 0.0,
            "overall_load_trend": 0.0,
            "per_replica_context": {},
            "health_emergency": False
        }
        
        # CRITICAL: Detect unhealthy pod scenario
        all_metrics_near_zero = (
            (cpu_rate_idx is None or current_values[cpu_rate_idx] < 1e-3) and
            (memory_idx is None or current_values[memory_idx] < 1000)  # < 1KB
        )
        
        # Check if we have health context
        ready_replicas = deployment_info.get('ready_replicas', current_replicas) if deployment_info else current_replicas
        
        if all_metrics_near_zero or ready_replicas == 0:
            logger.warning("🚨 HEALTH EMERGENCY: All metrics near zero or no ready replicas - unhealthy pods detected!")
            metrics["health_emergency"] = True
            # Return emergency metrics that will trigger scale-up
            metrics["cpu_change_percent"] = 100.0  # Force scale-up signal
            metrics["memory_change_percent"] = 100.0
            metrics["overall_load_trend"] = 1.0  # Strong positive trend
            return metrics

        # Calculate CPU metrics if available
        if cpu_rate_idx is not None and cpu_rate_idx < len(current_values) and cpu_rate_idx < forecast_values.shape[1]:
            current_cpu = current_values[cpu_rate_idx]
            predicted_cpu = np.mean(forecast_values[:, cpu_rate_idx])
            cpu_peak = np.max(forecast_values[:, cpu_rate_idx])
            
            # Per-replica CPU metrics
            current_cpu_per_replica = current_cpu / current_replicas
            predicted_cpu_per_replica = predicted_cpu / current_replicas
            cpu_peak_per_replica = cpu_peak / current_replicas
            
            # Handle near-zero current values properly
            if current_cpu_per_replica < 1e-3:  # Very low CPU (likely unhealthy pods)
                if predicted_cpu_per_replica > 0.1:  # Model predicts significant load
                    metrics["cpu_change_percent"] = 100.0  # Signal need for scale-up
                    logger.info("🚨 CPU: Near-zero current, positive prediction → scale-up signal")
                else:
                    metrics["cpu_change_percent"] = 0.0  # No significant change
                    logger.debug("CPU: Near-zero current and prediction → no change")
            else:
                # Normal calculation when current value is significant
                metrics["cpu_change_percent"] = ((predicted_cpu_per_replica - current_cpu_per_replica) / 
                                               current_cpu_per_replica) * 100
                # Clamp extreme values
                metrics["cpu_change_percent"] = max(-500, min(500, metrics["cpu_change_percent"]))
            
            metrics["cpu_peak"] = cpu_peak
            metrics["per_replica_context"]["cpu_per_replica_current"] = current_cpu_per_replica
            metrics["per_replica_context"]["cpu_per_replica_predicted"] = predicted_cpu_per_replica
            metrics["per_replica_context"]["cpu_per_replica_peak"] = cpu_peak_per_replica
            
            logger.debug(f"CPU metrics: current_per_replica={current_cpu_per_replica:.3f}, "
                        f"predicted_per_replica={predicted_cpu_per_replica:.3f}, "
                        f"change={metrics['cpu_change_percent']:.1f}%")
        else:
            logger.warning(f"⚠️ CPU metric not available: idx={cpu_rate_idx}, current_len={len(current_values)}, forecast_features={forecast_values.shape[1]}")

        # Calculate memory metrics if available
        if memory_idx is not None and memory_idx < len(current_values) and memory_idx < forecast_values.shape[1]:
            current_memory = current_values[memory_idx]
            predicted_memory = np.mean(forecast_values[:, memory_idx])
            memory_peak = np.max(forecast_values[:, memory_idx])
            
            # Per-replica memory metrics
            current_memory_per_replica = current_memory / current_replicas
            predicted_memory_per_replica = predicted_memory / current_replicas
            memory_peak_per_replica = memory_peak / current_replicas
            
            # Handle near-zero current values properly
            if current_memory_per_replica < 1024:  # Very low memory (< 1KB, likely unhealthy pods)
                if predicted_memory_per_replica > 10*1024*1024:  # Model predicts > 10MB
                    metrics["memory_change_percent"] = 100.0  # Signal need for scale-up
                    logger.info("🚨 Memory: Near-zero current, positive prediction → scale-up signal")
                else:
                    metrics["memory_change_percent"] = 0.0  # No significant change
                    logger.debug("Memory: Near-zero current and prediction → no change")
            else:
                # Normal calculation when current value is significant
                metrics["memory_change_percent"] = ((predicted_memory_per_replica - current_memory_per_replica) / 
                                                  current_memory_per_replica) * 100
                # Clamp extreme values
                metrics["memory_change_percent"] = max(-500, min(500, metrics["memory_change_percent"]))
            
            metrics["memory_peak"] = memory_peak
            metrics["per_replica_context"]["memory_per_replica_current"] = current_memory_per_replica
            metrics["per_replica_context"]["memory_per_replica_predicted"] = predicted_memory_per_replica
            metrics["per_replica_context"]["memory_per_replica_peak"] = memory_peak_per_replica
            
            logger.debug(f"Memory metrics: current_per_replica={current_memory_per_replica/1024/1024:.1f}MB, "
                        f"predicted_per_replica={predicted_memory_per_replica/1024/1024:.1f}MB, "
                        f"change={metrics['memory_change_percent']:.1f}%")
        else:
            logger.warning(f"⚠️ Memory metric not available: idx={memory_idx}")

        # Calculate overall load trend (average of normalized changes for CPU and Memory only)
        valid_changes = []
        if metrics["cpu_change_percent"] != 0.0:
            valid_changes.append(metrics["cpu_change_percent"] / 100)
        if metrics["memory_change_percent"] != 0.0:
            valid_changes.append(metrics["memory_change_percent"] / 100)

        if valid_changes:
            metrics["overall_load_trend"] = sum(valid_changes) / len(valid_changes)
            # Clamp overall trend to reasonable bounds
            metrics["overall_load_trend"] = max(-5.0, min(5.0, metrics["overall_load_trend"]))
        else:
            logger.warning("⚠️ No valid changes calculated for overall load trend")

        logger.info(f"📊 Forecast metrics calculated: cpu_change={metrics['cpu_change_percent']:.1f}%, "
                   f"memory_change={metrics['memory_change_percent']:.1f}%, "
                   f"overall_trend={metrics['overall_load_trend']:.3f}")

        return metrics

    def _get_scaling_recommendation(self, forecast_summary: Dict[str, float], current_replicas: int,
                                  deployment_info: Optional[Dict[str, Any]] = None) -> str:
        """Determine scaling recommendation based on configurable, per-replica thresholds with health awareness."""
        
        # CRITICAL: Check for health emergency first
        if forecast_summary.get("health_emergency", False):
            logger.warning("🚨 HEALTH EMERGENCY: Forcing scale-up recommendation due to unhealthy pods")
            return "scale_up"
        
        load_trend = forecast_summary["overall_load_trend"]
        cpu_change = forecast_summary["cpu_change_percent"]
        memory_change = forecast_summary["memory_change_percent"]

        # Get per-replica context
        per_replica_context = forecast_summary.get("per_replica_context", {})
        
        # Log current thresholds being used
        logger.debug(f"Using scaling thresholds: {self.scaling_thresholds}")
        logger.debug(f"Analyzing changes: load_trend={load_trend:.3f}, cpu={cpu_change:.1f}%, "
                    f"memory={memory_change:.1f}%")

        # Check resource limits for context-aware scaling
        resource_aware_scaling = False
        if deployment_info and deployment_info.get('resource_limits'):
            resource_limits = deployment_info['resource_limits']
            cpu_limit_total = resource_limits.get('cpu', 0.0)
            
            if cpu_limit_total > 0 and per_replica_context.get("cpu_per_replica_predicted"):
                cpu_limit_per_replica = cpu_limit_total / current_replicas
                predicted_cpu_util = per_replica_context["cpu_per_replica_predicted"] / cpu_limit_per_replica
                
                if predicted_cpu_util > self.target_cpu_utilization:
                    logger.info(f"🚀 Resource-aware scale up: predicted CPU utilization {predicted_cpu_util:.2f} > target {self.target_cpu_utilization}")
                    resource_aware_scaling = True

        # Scale up conditions (using configurable thresholds)
        scale_up_conditions = [
            load_trend > self.scaling_thresholds["load_trend_up"],
            cpu_change > self.scaling_thresholds["cpu_change_up"],
            memory_change > self.scaling_thresholds["memory_change_up"],
            resource_aware_scaling
        ]

        # Scale down conditions (using configurable thresholds)
        scale_down_conditions = [
            load_trend < self.scaling_thresholds["load_trend_down"],
            cpu_change < self.scaling_thresholds["cpu_change_down"] and 
            memory_change < self.scaling_thresholds["memory_change_down"]
        ]

        # Make recommendation
        if any(scale_up_conditions):
            triggered_conditions = []
            if load_trend > self.scaling_thresholds["load_trend_up"]:
                triggered_conditions.append(f"load_trend({load_trend:.3f})")
            if cpu_change > self.scaling_thresholds["cpu_change_up"]:
                triggered_conditions.append(f"cpu_change({cpu_change:.1f}%)")
            if memory_change > self.scaling_thresholds["memory_change_up"]:
                triggered_conditions.append(f"memory_change({memory_change:.1f}%)")
            if resource_aware_scaling:
                triggered_conditions.append("resource_limits")
                
            logger.info(f"🔼 Scale UP recommendation: {', '.join(triggered_conditions)}")
            return "scale_up"

        elif any(scale_down_conditions):
            triggered_conditions = []
            if load_trend < self.scaling_thresholds["load_trend_down"]:
                triggered_conditions.append(f"load_trend({load_trend:.3f})")
            if (cpu_change < self.scaling_thresholds["cpu_change_down"] and 
                memory_change < self.scaling_thresholds["memory_change_down"]):
                triggered_conditions.append(f"cpu_and_memory_decreasing")
                
            logger.info(f"🔽 Scale DOWN recommendation: {', '.join(triggered_conditions)}")
            return "scale_down"

        else:
            logger.info(f"➡️ KEEP SAME recommendation: no thresholds exceeded")
            return "keep_same"

    def _calculate_confidence(self, forecast_values: np.ndarray, current_values: np.ndarray) -> float:
        """Calculate confidence in the forecast using configurable weights."""
        try:
            # Calculate variance in predictions (lower variance = higher confidence)
            prediction_variance = np.mean(np.var(forecast_values, axis=0))

            # Normalize variance to confidence (0-1 scale)
            max_variance = np.mean(current_values ** 2) + 1e-6
            normalized_variance = min(prediction_variance / max_variance, 1.0)
            variance_confidence = 1.0 - normalized_variance

            # Factor in data quality (how much data we have)
            data_confidence = min(len(current_values) / 30.0, 1.0)  # 30 minutes = full confidence

            # Factor in model training status
            model_confidence = 1.0 if self.lstm_trainer.is_trained else 0.3

            # Combined confidence using configurable weights
            overall_confidence = (
                variance_confidence * self.confidence_weights["variance"] +
                data_confidence * self.confidence_weights["data_quality"] +
                model_confidence * self.confidence_weights["model_status"]
            )

            logger.debug(f"Confidence calculation: variance={variance_confidence:.3f}, "
                        f"data={data_confidence:.3f}, model={model_confidence:.3f}, "
                        f"overall={overall_confidence:.3f}")

            return max(0.1, min(1.0, overall_confidence))  # Clamp between 0.1 and 1.0

        except Exception as e:
            logger.warning(f"Failed to calculate confidence: {e}")
            return 0.3  # Default low confidence

    def _get_statistical_forecast(self, feature_matrix: np.ndarray, feature_indices: Dict[str, int], 
                                current_replicas: int, deployment_info: Optional[Dict[str, Any]] = None) -> Dict[str, any]:
        """Generate forecast using statistical methods with deployment awareness."""
        try:
            # Simple trend analysis on recent data
            if len(feature_matrix) < self.min_data_quality_samples:
                return self._get_fallback_forecast()

            # Calculate trends for recent data
            recent_data = feature_matrix[-self.min_data_quality_samples:]
            trends = {}

            # Calculate trends for available features (CPU and Memory only)
            for feature_name, idx in feature_indices.items():
                if idx < feature_matrix.shape[1]:
                    feature_data = recent_data[:, idx]
                    if len(feature_data) > 1:
                        trend = np.polyfit(range(len(feature_data)), feature_data, 1)[0]
                        trends[feature_name] = trend
                    else:
                        trends[feature_name] = 0.0

            # Calculate per-replica trends
            current_values = feature_matrix[-1]
            
            # Get per-replica metrics for CPU and Memory only
            cpu_trend = trends.get("process_cpu_seconds_total_rate", 0.0) / current_replicas
            memory_trend = trends.get("process_resident_memory_bytes", 0.0) / current_replicas

            # Calculate percentage changes for consistency with LSTM path
            cpu_idx = feature_indices.get("process_cpu_seconds_total_rate")
            memory_idx = feature_indices.get("process_resident_memory_bytes")

            cpu_change_percent = 0.0
            memory_change_percent = 0.0

            if cpu_idx is not None and cpu_idx < len(current_values):
                current_cpu_per_replica = current_values[cpu_idx] / current_replicas
                if current_cpu_per_replica > 1e-3:  # Avoid division by near-zero
                    cpu_change_percent = (cpu_trend / current_cpu_per_replica) * 100
                    cpu_change_percent = max(-500, min(500, cpu_change_percent))  # Clamp
                else:
                    cpu_change_percent = 0.0  # No significant change

            if memory_idx is not None and memory_idx < len(current_values):
                current_memory_per_replica = current_values[memory_idx] / current_replicas
                if current_memory_per_replica > 1024:  # Avoid division by near-zero (> 1KB)
                    memory_change_percent = (memory_trend / current_memory_per_replica) * 100
                    memory_change_percent = max(-500, min(500, memory_change_percent))  # Clamp
                else:
                    memory_change_percent = 0.0  # No significant change

            overall_trend = (cpu_change_percent + memory_change_percent) / 200  # Normalize to -1 to 1 for CPU+Memory
            overall_trend = max(-5.0, min(5.0, overall_trend))  # Clamp to reasonable bounds

            # Use same recommendation logic as LSTM path
            forecast_summary = {
                "cpu_change_percent": cpu_change_percent,
                "memory_change_percent": memory_change_percent,
                "overall_load_trend": overall_trend,
                "per_replica_context": {}
            }

            recommendation = self._get_scaling_recommendation(forecast_summary, current_replicas, deployment_info)

            logger.info(f"📊 Statistical forecast: cpu_change={cpu_change_percent:.1f}%, "
                       f"memory_change={memory_change_percent:.1f}%, "
                       f"recommendation={recommendation}")

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "horizon_minutes": self.forecast_horizon_minutes,
                "recommendation": recommendation,
                "confidence": 0.4,  # Lower confidence for statistical method
                "method": "statistical",
                "forecast_summary": forecast_summary,
                "deployment_context": {
                    "current_replicas": current_replicas,
                    "has_resource_limits": bool(deployment_info and deployment_info.get('resource_limits'))
                }
            }

        except Exception as e:
            logger.error(f"Statistical forecast failed: {e}")
            return self._get_fallback_forecast()

    def _get_fallback_forecast(self) -> Dict[str, any]:
        """Get fallback forecast when all else fails."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "horizon_minutes": self.forecast_horizon_minutes,
            "recommendation": "keep_same",
            "confidence": 0.1,
            "method": "fallback",
            "status": "insufficient_data",
            "deployment_context": {
                "current_replicas": 1,
                "has_resource_limits": False
            }
        }

    def _check_retrain_schedule(self) -> None:
        """Check if model needs retraining and schedule if needed."""
        forecasting_enabled = getattr(config, 'forecasting_enabled', True)
        if not forecasting_enabled:
            return

        current_time = datetime.utcnow()

        # Retrain if enough time has passed
        if (self.last_training_time is None or
                (current_time - self.last_training_time).total_seconds() > self.retrain_interval_minutes * 60):
            self._schedule_retrain()

    def _schedule_retrain(self) -> None:
        """Schedule model retraining in background."""
        if self.is_training:
            logger.debug("Training already in progress")
            return

        def retrain_worker():
            try:
                with self.training_lock:
                    self.is_training = True
                    logger.info("Starting background model retraining")

                    # Get training data
                    feature_matrix = self.prometheus_client.get_feature_matrix(self.lookback_minutes)

                    if feature_matrix is not None and len(feature_matrix) >= self.min_training_samples:
                        logger.info(f"🔄 Starting LSTM retraining with {len(feature_matrix)} samples (min: {self.min_training_samples})")
                        self._train_model_async(feature_matrix)
                        self.last_training_time = datetime.utcnow()
                        logger.info("✅ Background retraining completed successfully")
                    else:
                        actual_samples = len(feature_matrix) if feature_matrix is not None else 0
                        logger.warning(f"❌ Insufficient data for retraining: {actual_samples}/{self.min_training_samples} samples")

            except Exception as e:
                logger.error(f"Background retraining failed: {e}")
            finally:
                self.is_training = False

        # Start training in background thread
        training_thread = threading.Thread(target=retrain_worker, daemon=True)
        training_thread.start()

    def _train_model_async(self, feature_matrix: np.ndarray) -> None:
        """Train LSTM model asynchronously."""
        try:
            logger.info(f"🚀 Starting LSTM training with {len(feature_matrix)} data points")

            # Prepare training data
            X, y = self.data_preprocessor.prepare_training_data(feature_matrix)

            if X.size == 0 or y.size == 0:
                logger.warning(f"❌ No training data prepared: X.size={X.size}, y.size={y.size}")
                return

            logger.info(f"📊 Training data prepared: X.shape={X.shape}, y.shape={y.shape}")

            # Train the model with configurable epochs
            epochs = getattr(config, 'lstm_training_epochs', 30)
            success = self.lstm_trainer.train(X, y, epochs=epochs)

            if success:
                logger.info("✅ LSTM model training completed successfully")
                logger.info(f"📈 Model info: {self.lstm_trainer.get_model_info()}")
            else:
                logger.warning("❌ LSTM model training failed")

        except Exception as e:
            logger.error(f"❌ Async model training failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def get_status(self) -> Dict[str, any]:
        """Get current forecaster status."""
        return {
            "initialized": self.is_initialized,
            "prometheus_healthy": self.prometheus_client.health_check(),
            "scaler_fitted": self.data_preprocessor.is_fitted,
            "model_trained": self.lstm_trainer.is_trained,
            "is_training": self.is_training,
            "last_training": self.last_training_time.isoformat() if self.last_training_time else None,
            "last_prediction": self.last_prediction,
            "prediction_confidence": self.prediction_confidence,
            "model_info": self.lstm_trainer.get_model_info(),
            "config_summary": self._get_config_summary(),
            "scaling_thresholds": self.scaling_thresholds,
            "confidence_weights": self.confidence_weights,
            "metrics_used": ["process_cpu_seconds_total_rate", "process_resident_memory_bytes"]
        }