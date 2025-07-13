"""Main forecasting predictor combining data preprocessing and LSTM model."""
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import time

from config.settings import load_config
config = load_config()
from prometheus.client import PrometheusClient
from forecasting.data_prep import DataPreprocessor
from forecasting.model import LSTMTrainer


logger = logging.getLogger(__name__)


class LoadForecaster:
    """Main forecasting system for proactive scaling predictions."""
    
    def __init__(self):
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
        
        logger.info("LoadForecaster initialized")
    
    def initialize(self) -> bool:
        """Initialize the forecasting system."""
        try:
            # Check Prometheus connectivity
            if not self.prometheus_client.health_check():
                logger.error("Prometheus health check failed")
                return False
            
            # Try to load existing models
            self.data_preprocessor.load_scaler()
            
            # Try to collect some initial data to determine input size
            feature_matrix = self.prometheus_client.get_feature_matrix(
                config.forecasting.lookback_minutes
            )
            
            if feature_matrix is not None and feature_matrix.size > 0:
                input_size = feature_matrix.shape[1]  # Number of features
                self.lstm_trainer.load_model(input_size)
                
                # If we have enough data and no trained model, train one
                if not self.lstm_trainer.is_trained and len(feature_matrix) >= 40:
                    logger.info("No trained model found, training initial model")
                    self._train_model_async(feature_matrix)
            
            self.is_initialized = True
            logger.info("LoadForecaster initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LoadForecaster: {e}")
            return False
    
    def generate_forecast(self) -> Dict[str, any]:
        """Generate load forecast for the next horizon."""
        if not self.is_initialized:
            logger.warning("Forecaster not initialized")
            return self._get_fallback_forecast()
        
        try:
            # Get recent data for prediction
            feature_matrix = self.prometheus_client.get_feature_matrix(
                min(config.forecasting.lookback_minutes, 30)  # Last 30 minutes
            )
            
            if feature_matrix is None or feature_matrix.size == 0:
                logger.warning("No recent data available for forecasting")
                return self._get_fallback_forecast()
            
            # Validate data quality
            if not self.data_preprocessor.validate_data_quality(feature_matrix):
                logger.warning("Poor data quality, using fallback")
                return self._get_fallback_forecast()
            
            # Prepare input for prediction
            prediction_input = self.data_preprocessor.prepare_prediction_input(feature_matrix)
            
            if prediction_input is None:
                logger.warning("Failed to prepare prediction input")
                return self._get_fallback_forecast()
            
            min_samples = config.forecasting.min_training_samples
            
            # Check if we should train LSTM model if not already trained
            if not self.lstm_trainer.is_trained and len(feature_matrix) >= min_samples:
                logger.info(f"🔄 LSTM not trained but have {len(feature_matrix)} data points (min: {min_samples}) - starting training")
                self._schedule_retrain()
            
            # Make prediction if model is trained
            if self.lstm_trainer.is_trained:
                predictions = self.lstm_trainer.predict(prediction_input)
                
                if predictions is not None:
                    forecast_result = self._analyze_predictions(predictions, feature_matrix)
                    forecast_result["method"] = "lstm"
                    forecast_result["confidence"] = min(0.9, forecast_result.get("confidence", 0.5))  # Higher confidence for LSTM
                    self.last_prediction = forecast_result
                    
                    # Check if we need to retrain
                    self._check_retrain_schedule()
                    
                    logger.info(f"✅ LSTM forecast generated: confidence={forecast_result['confidence']:.3f}")
                    return forecast_result
            
            # If no trained model, use statistical fallback
            if len(feature_matrix) >= min_samples:
                logger.info(f"📊 No trained LSTM model, using statistical forecast for {len(feature_matrix)} samples")
            else:
                logger.info(f"⚠️ Insufficient data ({len(feature_matrix)}/{min_samples}), using statistical forecast")
            
            return self._get_statistical_forecast(feature_matrix)
            
        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            return self._get_fallback_forecast()
    
    def _analyze_predictions(self, predictions: np.ndarray, current_data: np.ndarray) -> Dict[str, any]:
        """Analyze LSTM predictions and generate forecast result."""
        try:
            # predictions shape: [1, forecast_horizon, num_features]
            forecast_values = predictions[0]  # Remove batch dimension
            
            # Get current values (last row of current data)
            current_values = current_data[-1] if len(current_data) > 0 else np.zeros(predictions.shape[2])
            
            # Calculate key metrics for scaling decision
            forecast_summary = self._calculate_forecast_metrics(forecast_values, current_values)
            
            # Determine scaling recommendation
            recommendation = self._get_scaling_recommendation(forecast_summary)
            
            # Calculate confidence based on prediction consistency
            confidence = self._calculate_confidence(forecast_values, current_values)
            
            self.prediction_confidence = confidence
            
            result = {
                "timestamp": datetime.utcnow().isoformat(),
                "horizon_minutes": config.forecasting.forecast_horizon_minutes,
                "recommendation": recommendation,
                "confidence": confidence,
                "forecast_summary": forecast_summary,
                "current_metrics": {
                    feature: float(current_values[i]) 
                    for i, feature in enumerate(config.scaling.selected_features)
                },
                "predicted_peak": {
                    feature: float(np.max(forecast_values[:, i])) 
                    for i, feature in enumerate(config.scaling.selected_features)
                }
            }
            
            logger.info(f"Generated forecast: {recommendation} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze predictions: {e}")
            return self._get_fallback_forecast()
    
    def _calculate_forecast_metrics(self, forecast_values: np.ndarray, current_values: np.ndarray) -> Dict[str, float]:
        """Calculate key metrics from forecast for scaling decisions."""
        # Map feature indices
        feature_indices = {name: i for i, name in enumerate(config.scaling.selected_features)}
        
        # Extract key metrics
        cpu_rate_idx = feature_indices.get("process_cpu_seconds_total_rate", 0)
        memory_idx = feature_indices.get("process_resident_memory_bytes", 1)
        request_rate_idx = feature_indices.get("http_requests_total_rate", 4)
        
        # Current vs predicted changes
        cpu_change = (np.mean(forecast_values[:, cpu_rate_idx]) - current_values[cpu_rate_idx]) / (current_values[cpu_rate_idx] + 1e-6)
        memory_change = (np.mean(forecast_values[:, memory_idx]) - current_values[memory_idx]) / (current_values[memory_idx] + 1e-6)
        request_change = (np.mean(forecast_values[:, request_rate_idx]) - current_values[request_rate_idx]) / (current_values[request_rate_idx] + 1e-6)
        
        # Peak values
        cpu_peak = np.max(forecast_values[:, cpu_rate_idx])
        memory_peak = np.max(forecast_values[:, memory_idx])
        request_peak = np.max(forecast_values[:, request_rate_idx])
        
        return {
            "cpu_change_percent": cpu_change * 100,
            "memory_change_percent": memory_change * 100,
            "request_change_percent": request_change * 100,
            "cpu_peak": cpu_peak,
            "memory_peak": memory_peak,
            "request_peak": request_peak,
            "overall_load_trend": (cpu_change + memory_change + request_change) / 3
        }
    
    def _get_scaling_recommendation(self, forecast_summary: Dict[str, float]) -> str:
        """Determine scaling recommendation based on forecast."""
        load_trend = forecast_summary["overall_load_trend"]
        cpu_change = forecast_summary["cpu_change_percent"]
        memory_change = forecast_summary["memory_change_percent"]
        request_change = forecast_summary["request_change_percent"]
        
        # Scale up if significant increases predicted
        if (load_trend > 0.2 or  # 20% overall increase
            cpu_change > 30 or    # 30% CPU increase
            memory_change > 25 or # 25% memory increase
            request_change > 50): # 50% request increase
            return "scale_up"
        
        # Scale down if significant decreases predicted
        elif (load_trend < -0.2 or  # 20% overall decrease
              (cpu_change < -20 and memory_change < -20 and request_change < -30)):
            return "scale_down"
        
        # Keep same for stable predictions
        else:
            return "keep_same"
    
    def _calculate_confidence(self, forecast_values: np.ndarray, current_values: np.ndarray) -> float:
        """Calculate confidence in the forecast."""
        try:
            # Calculate variance in predictions (lower variance = higher confidence)
            prediction_variance = np.mean(np.var(forecast_values, axis=0))
            
            # Normalize variance to confidence (0-1 scale)
            # Lower variance means higher confidence
            max_variance = np.mean(current_values ** 2) + 1e-6
            normalized_variance = min(prediction_variance / max_variance, 1.0)
            variance_confidence = 1.0 - normalized_variance
            
            # Factor in data quality (how much data we have)
            data_confidence = min(len(current_values) / 30.0, 1.0)  # 30 minutes = full confidence
            
            # Factor in model training status
            model_confidence = 1.0 if self.lstm_trainer.is_trained else 0.3
            
            # Combined confidence
            overall_confidence = (variance_confidence * 0.5 + 
                                data_confidence * 0.3 + 
                                model_confidence * 0.2)
            
            return max(0.1, min(1.0, overall_confidence))  # Clamp between 0.1 and 1.0
            
        except Exception as e:
            logger.warning(f"Failed to calculate confidence: {e}")
            return 0.3  # Default low confidence
    
    def _get_statistical_forecast(self, feature_matrix: np.ndarray) -> Dict[str, any]:
        """Generate forecast using statistical methods when LSTM is not available."""
        try:
            # Simple trend analysis on recent data
            if len(feature_matrix) < 5:
                return self._get_fallback_forecast()
            
            # Calculate trends for last 5 minutes
            recent_data = feature_matrix[-5:]
            trends = []
            
            for i in range(feature_matrix.shape[1]):
                feature_data = recent_data[:, i]
                if len(feature_data) > 1:
                    trend = np.polyfit(range(len(feature_data)), feature_data, 1)[0]
                    trends.append(trend)
                else:
                    trends.append(0.0)
            
            # Simple prediction: extend trends
            current_values = feature_matrix[-1]
            
            # Calculate overall load trend
            cpu_trend = trends[0] if len(trends) > 0 else 0
            memory_trend = trends[1] if len(trends) > 1 else 0
            request_trend = trends[4] if len(trends) > 4 else 0
            
            overall_trend = (cpu_trend + memory_trend + request_trend) / 3
            
            # Simple recommendation based on trends
            if overall_trend > 0.01:  # Increasing trend
                recommendation = "scale_up"
            elif overall_trend < -0.01:  # Decreasing trend
                recommendation = "scale_down"
            else:
                recommendation = "keep_same"
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "horizon_minutes": config.forecasting.forecast_horizon_minutes,
                "recommendation": recommendation,
                "confidence": 0.4,  # Lower confidence for statistical method
                "method": "statistical",
                "trend_analysis": {
                    "overall_trend": overall_trend,
                    "cpu_trend": cpu_trend,
                    "memory_trend": memory_trend,
                    "request_trend": request_trend
                }
            }
            
        except Exception as e:
            logger.error(f"Statistical forecast failed: {e}")
            return self._get_fallback_forecast()
    
    def _get_fallback_forecast(self) -> Dict[str, any]:
        """Get fallback forecast when all else fails."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "horizon_minutes": config.forecasting.forecast_horizon_minutes,
            "recommendation": "keep_same",
            "confidence": 0.1,
            "method": "fallback",
            "status": "insufficient_data"
        }
    
    def _check_retrain_schedule(self) -> None:
        """Check if model needs retraining and schedule if needed."""
        if not config.forecasting.enabled:
            return
        
        current_time = datetime.utcnow()
        
        # Retrain if enough time has passed
        if (self.last_training_time is None or 
            (current_time - self.last_training_time).total_seconds() > 
            config.forecasting.retrain_interval_minutes * 60):
            
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
                    feature_matrix = self.prometheus_client.get_feature_matrix(
                        config.forecasting.lookback_minutes
                    )
                    
                    min_samples = config.forecasting.min_training_samples
                    if feature_matrix is not None and len(feature_matrix) >= min_samples:
                        logger.info(f"🔄 Starting LSTM retraining with {len(feature_matrix)} samples (min: {min_samples})")
                        self._train_model_async(feature_matrix)
                        self.last_training_time = datetime.utcnow()
                        logger.info("✅ Background retraining completed successfully")
                    else:
                        actual_samples = len(feature_matrix) if feature_matrix is not None else 0
                        logger.warning(f"❌ Insufficient data for retraining: {actual_samples}/{min_samples} samples")
                    
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
            
            # Train the model
            success = self.lstm_trainer.train(X, y, epochs=30)
            
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
            "model_info": self.lstm_trainer.get_model_info()
        } 