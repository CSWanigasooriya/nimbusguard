"""Data preprocessing utilities for LSTM forecasting."""
import logging
import os
from typing import Tuple, Optional

import joblib
import numpy as np
from config.settings import load_config
from sklearn.preprocessing import StandardScaler

config = load_config()

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data preprocessing for LSTM model."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.scaler_path = "/tmp/feature_scaler.pkl"

    def fit_scaler(self, data: np.ndarray) -> None:
        """Fit the scaler on training data."""
        try:
            # Convert to DataFrame with feature names for consistency
            import pandas as pd
            data_df = pd.DataFrame(data, columns=config.scaling.selected_features)
            self.scaler.fit(data_df)
            self.is_fitted = True

            # Save scaler for persistence
            joblib.dump(self.scaler, self.scaler_path)
            logger.info(f"Fitted scaler on data shape: {data.shape}")

        except Exception as e:
            logger.error(f"Failed to fit scaler: {e}")
            raise

    def load_scaler(self) -> bool:
        """Load existing scaler from disk."""
        try:
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                self.is_fitted = True
                logger.info("Loaded existing scaler")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to load scaler: {e}")
            return False

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using the already-fitted scaler."""
        if not self.is_fitted:
            logger.error("Scaler not fitted, cannot transform data")
            return data
        try:
            import pandas as pd
            data_df = pd.DataFrame(data, columns=config.scaling.selected_features)
            scaled_data = self.scaler.transform(data_df)
            return scaled_data
        except Exception as e:
            logger.error(f"Failed to transform data: {e}")
            return data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data."""
        if not self.is_fitted:
            logger.warning("Scaler not fitted, returning original data")
            return data

        try:
            # Convert to DataFrame with feature names to match how scaler was fitted
            import pandas as pd
            data_df = pd.DataFrame(data, columns=config.scaling.selected_features)
            original_data = self.scaler.inverse_transform(data_df)
            return original_data
        except Exception as e:
            logger.error(f"Failed to inverse transform data: {e}")
            return data

    def create_sequences(self, data: np.ndarray, sequence_length: int,
                         forecast_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences and targets for LSTM training."""
        if len(data) < sequence_length + forecast_horizon:
            raise ValueError(f"Not enough data: need {sequence_length + forecast_horizon}, got {len(data)}")

        X, y = [], []

        for i in range(len(data) - sequence_length - forecast_horizon + 1):
            # Input sequence
            X.append(data[i:i + sequence_length])

            # Target: next forecast_horizon steps
            y.append(data[i + sequence_length:i + sequence_length + forecast_horizon])

        X = np.array(X)
        y = np.array(y)

        logger.debug(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
        return X, y

    def prepare_training_data(self, feature_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training with improved handling for small datasets."""
        try:
            # Check data quality
            if feature_matrix.size == 0:
                raise ValueError("Empty feature matrix")

            logger.info(f"📊 Preparing training data: input shape={feature_matrix.shape}")

            # Remove any infinite or NaN values
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

            # Validate data quality
            if not self.validate_data_quality(feature_matrix):
                logger.warning("⚠️ Data quality issues detected, but proceeding with training")

            # Fit scaler if not already fitted
            if not self.is_fitted:
                self.fit_scaler(feature_matrix)

            # Scale the data
            scaled_data = self.transform(feature_matrix)

            # Adaptive sequence length based on available data
            sequence_length = config.forecasting.sequence_length
            forecast_horizon = config.forecasting.forecast_horizon_minutes
            min_required = sequence_length + forecast_horizon

            # Adjust sequence length if we don't have enough data
            if len(scaled_data) < min_required:
                sequence_length = max(5, len(scaled_data) - forecast_horizon - 1)  # Minimum 5 time steps
                logger.info(f"📉 Adjusted sequence length to {sequence_length} due to limited data")

            if len(scaled_data) < sequence_length + forecast_horizon:
                raise ValueError(
                    f"Insufficient data for training: have {len(scaled_data)}, need {sequence_length + forecast_horizon}")

            X, y = self.create_sequences(scaled_data, sequence_length, forecast_horizon)

            if X.size == 0 or y.size == 0:
                raise ValueError("No sequences created - check data length and parameters")

            logger.info(f"✅ Prepared training data: X {X.shape}, y {y.shape}, seq_len={sequence_length}")
            return X, y

        except Exception as e:
            logger.error(f"❌ Failed to prepare training data: {e}")
            raise

    def prepare_prediction_input(self, recent_data: np.ndarray) -> Optional[np.ndarray]:
        """Prepare recent data for prediction."""
        try:
            if not self.is_fitted:
                logger.error("Scaler not fitted, cannot prepare prediction input")
                return None

            # Clean data
            recent_data = np.nan_to_num(recent_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Scale the data
            scaled_data = self.transform(recent_data)

            # Take the last 30 points (or available data) as input sequence
            sequence_length = min(30, len(scaled_data))
            input_sequence = scaled_data[-sequence_length:]

            # Add batch dimension: [1, sequence_length, features]
            input_tensor = np.expand_dims(input_sequence, axis=0)

            logger.debug(f"Prepared prediction input: {input_tensor.shape}")
            return input_tensor

        except Exception as e:
            logger.error(f"Failed to prepare prediction input: {e}")
            return None

    def validate_data_quality(self, data: np.ndarray) -> bool:
        """Validate data quality for training/prediction."""
        try:
            if data.size == 0:
                logger.error("Data is empty")
                return False

            if np.all(data == 0):
                logger.warning("All data points are zero")
                return False

            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                logger.warning("Data contains NaN or infinite values")
                # We can still proceed after cleaning

            # Check for reasonable variance
            if np.var(data) < 1e-10:
                logger.warning("Data has very low variance")

            logger.debug("Data quality validation passed")
            return True

        except Exception as e:
            logger.error(f"Data quality validation failed: {e}")
            return False
