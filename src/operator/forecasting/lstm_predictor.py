
import logging
import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from keras.models import load_model

from config.settings import ScalingConfig
from forecasting.base import BasePredictor

logger = logging.getLogger(__name__)


class LSTMPredictor(BasePredictor):
    """
    LSTM-based time series forecaster using a pre-trained Keras model.
    Handles data scaling, prediction, and inverse scaling.
    """

    def __init__(self, config: ScalingConfig):
        self.config = config
        # NOTE: In the container image the Dockerfile copies the pre-trained
        # model and its scaler into /tmp/ so we load from that location.
        # These can still be overridden via env vars if needed.
        self.model_path = os.getenv("FORECASTER_MODEL_PATH", "/tmp/forecaster.keras")
        self.scaler_path = os.getenv("FORECASTER_SCALER_PATH", "/tmp/forecaster_scaler.pkl")
        self.model = None
        self.scaler = None
        self.is_loaded = False
        self._load_model_and_scaler()

    def _load_model_and_scaler(self):
        """Load the Keras model and the feature scaler."""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = load_model(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.is_loaded = True
                logger.info(f"✅ LSTM model loaded from {self.model_path}")
                logger.info(f"✅ Scaler loaded from {self.scaler_path}")
            else:
                logger.warning("LSTM model or scaler not found, predictor disabled.")
        except Exception as e:
            logger.error(f"❌ Failed to load LSTM model or scaler: {e}")

    async def predict(
        self, historical_data: Dict[str, pd.DataFrame]
    ) -> Optional[Dict[str, float]]:
        """
        Generate forecast using historical data.
        Returns a dictionary of predicted metric values or None if prediction fails.
        """
        if not self.is_loaded:
            logger.warning("Predictor not loaded, cannot make predictions.")
            return None

        try:
            # Map Prometheus metric names to training data column names
            metric_mapping = {
                'process_cpu_seconds_total_rate': 'cpu',
                'process_resident_memory_bytes': 'memory'
            }

            # Convert historical data to the format expected by the model
            df_list = []
            for prometheus_metric, df in historical_data.items():
                if prometheus_metric in metric_mapping:
                    training_column = metric_mapping[prometheus_metric]
                    df_renamed = df.rename(columns={"value": training_column})
                    df_list.append(df_renamed[['timestamp', training_column]].set_index('timestamp'))
                    
            if not df_list:
                logger.warning("No matching metrics found for prediction.")
                return None

            # Merge all metrics into a single DataFrame (like your training script)
            df_merged = pd.concat(df_list, axis=1)
            
            # Apply 15-second resampling with forward fill (matching training preprocessing)
            df_merged = df_merged.resample('15s').mean().ffill()
            
            # Select only the features the model was trained on
            df_features = df_merged[['cpu', 'memory']].dropna()

            if df_features.shape[0] < self.config.sequence_length:
                logger.warning(
                    f"Not enough data for prediction: have {df_features.shape[0]}, need {self.config.sequence_length}"
                )
                return None

            # Scale and reshape data (exactly like training)
            sequence_data = df_features.tail(self.config.sequence_length).values
            scaled_data = self.scaler.transform(sequence_data)
            reshaped_data = np.reshape(
                scaled_data,
                (1, self.config.sequence_length, 2)  # 2 features: cpu, memory
            )

            # Make prediction (single step ahead)
            scaled_prediction = self.model.predict(reshaped_data, verbose=0)
            
            # Handle model output shape: should be (1, 1, 2) for single step prediction
            if scaled_prediction.shape != (1, 1, 2):
                logger.warning(f"Unexpected prediction shape: {scaled_prediction.shape}, expected (1, 1, 2)")
                # Reshape if needed
                scaled_prediction = scaled_prediction.reshape(1, 1, 2)
            
            # Calculate confidence based on data quality and prediction stability
            confidence = self._calculate_confidence(df_features, scaled_prediction, sequence_data)
            
            # Inverse transform to get actual values
            prediction = self.scaler.inverse_transform(scaled_prediction.reshape(1, 2))

            # Map back to Prometheus metric names for the operator
            forecast = {
                'process_cpu_seconds_total_rate': float(prediction[0, 0]),
                'process_resident_memory_bytes': float(prediction[0, 1]),
                'confidence': float(confidence)
            }
            
            # Ensure non-negative values (physical constraints)
            forecast['process_cpu_seconds_total_rate'] = max(0.0, forecast['process_cpu_seconds_total_rate'])
            forecast['process_resident_memory_bytes'] = max(0.0, forecast['process_resident_memory_bytes'])
            
            logger.info(f"🔮 LSTM Forecast Generated: CPU={forecast['process_cpu_seconds_total_rate']:.4f}, "
                       f"Memory={forecast['process_resident_memory_bytes']:.0f} bytes")
            return forecast

        except Exception as e:
            logger.error(f"❌ Error during LSTM prediction: {e}")
            return None 