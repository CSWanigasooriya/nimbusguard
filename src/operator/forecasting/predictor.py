"""
Forecasting module for CPU and Memory prediction using pre-trained models.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Handles loading and using pre-trained CPU (GRU) and Memory (LSTM) prediction models."""
    
    def __init__(self, models_path: str = "/tmp", config=None):
        """
        Initialize the predictor with model paths and configuration.
        
        Args:
            models_path: Directory containing the model files
            config: Configuration object with forecasting parameters
        """
        self.models_path = Path(models_path)
        self.config = config
        
        # Model components
        self.cpu_model = None
        self.cpu_scaler = None
        self.memory_model = None
        self.memory_scaler = None
        
        # Model specifications from config (with fallbacks)
        if config and hasattr(config, 'forecasting'):
            self.cpu_sequence_length = config.forecasting.cpu_sequence_length
            self.cpu_features = config.forecasting.cpu_features_count
            self.memory_sequence_length = config.forecasting.memory_sequence_length
            self.memory_features = config.forecasting.memory_features_count
            logger.info(f"Using config: CPU seq={self.cpu_sequence_length}, features={self.cpu_features}")
            logger.info(f"Using config: Memory seq={self.memory_sequence_length}, features={self.memory_features}")
        else:
            # Fallback to trained model defaults
            self.cpu_sequence_length = 8  # GRU model default
            self.cpu_features = 9  # GRU model features
            self.memory_sequence_length = 4  # LSTM model default
            self.memory_features = 4  # LSTM model features
            logger.warning("No config provided, using model defaults")
    
    async def load_models(self):
        """Load the pre-trained CPU and Memory prediction models."""
        try:
            # Load CPU model and scaler
            cpu_model_path = self.models_path / "cpu_forecaster.keras"
            cpu_scaler_path = self.models_path / "cpu_forecaster_scaler.pkl"
            
            logger.info(f"Loading CPU model from {cpu_model_path}")
            self.cpu_model = tf.keras.models.load_model(cpu_model_path)
            
            logger.info(f"Loading CPU scaler from {cpu_scaler_path}")
            self.cpu_scaler = joblib.load(cpu_scaler_path)
            
            # Load Memory model and scaler
            memory_model_path = self.models_path / "memory_forecaster.keras"
            memory_scaler_path = self.models_path / "memory_forecaster_scaler.pkl"
            
            logger.info(f"Loading Memory model from {memory_model_path}")
            self.memory_model = tf.keras.models.load_model(memory_model_path)
            
            logger.info(f"Loading Memory scaler from {memory_scaler_path}")
            self.memory_scaler = joblib.load(memory_scaler_path)
            
            logger.info("All forecasting models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load forecasting models: {e}")
            raise
    
    def _preprocess_time_series(self, data: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        """
        Robust preprocessing for time series data to handle real-world noise and outliers.
        
        Args:
            data: DataFrame with timestamp and value columns
            metric_name: Name of metric for logging
            
        Returns:
            Preprocessed DataFrame
        """
        if data is None or data.empty:
            return data
        
        processed_data = data.copy()
        original_values = processed_data['value'].values
        
        logger.info(f"🔬 {metric_name.upper()}: Preprocessing {len(original_values)} points, range=[{np.min(original_values):.4f}, {np.max(original_values):.4f}]")
        
        # === 1. OUTLIER DETECTION AND FILTERING ===
        if len(original_values) > 3:
            # Use IQR method for outlier detection
            q1, q3 = np.percentile(original_values, [25, 75])
            iqr = q3 - q1
            
            # Set reasonable bounds (more permissive for CPU since it can spike)
            multiplier = 3.0 if metric_name == 'cpu' else 2.5
            lower_bound = max(0, q1 - multiplier * iqr)
            upper_bound = q3 + multiplier * iqr
            
            # For CPU, ensure we don't filter too aggressively on normal load variations
            if metric_name == 'cpu':
                median_val = np.median(original_values)
                # Allow spikes up to 10x median for CPU (normal for load bursts)
                upper_bound = max(upper_bound, median_val * 10)
            
            outlier_mask = (original_values < lower_bound) | (original_values > upper_bound)
            outlier_count = np.sum(outlier_mask)
            
            if outlier_count > 0:
                logger.info(f"🔧 {metric_name.upper()}: Filtering {outlier_count}/{len(original_values)} outliers")
                logger.debug(f"   Bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
                
                # Replace outliers with rolling median (more robust than global median)
                rolling_median = processed_data['value'].rolling(window=5, center=True, min_periods=1).median()
                processed_data.loc[outlier_mask, 'value'] = rolling_median.loc[outlier_mask]
        
        # === 2. DATA SMOOTHING FOR NOISE REDUCTION ===
        if len(processed_data) > 3:
            # Calculate volatility ratio
            volatility_ratio = np.std(processed_data['value']) / (np.mean(processed_data['value']) + 1e-6)
            logger.info(f"🔬 {metric_name.upper()}: Volatility ratio = {volatility_ratio:.3f} (std={np.std(processed_data['value']):.4f}, mean={np.mean(processed_data['value']):.4f})")
            
            if volatility_ratio > 1.5:  # High volatility
                logger.info(f"📊 {metric_name.upper()}: High volatility detected (ratio={volatility_ratio:.2f}), applying smoothing")
                # Apply adaptive smoothing - stronger for more volatile data
                window_size = min(5, max(3, int(len(processed_data) / 3)))
                processed_data['value'] = processed_data['value'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
            elif volatility_ratio > 0.8:  # Moderate volatility
                logger.info(f"📊 {metric_name.upper()}: Moderate volatility detected (ratio={volatility_ratio:.2f}), applying light smoothing")
                # Light smoothing to preserve trends
                processed_data['value'] = processed_data['value'].rolling(
                    window=3, center=True, min_periods=1
                ).mean()
        
        # === 3. DATA QUALITY VALIDATION ===
        final_values = processed_data['value'].values
        improvement_ratio = np.std(original_values) / (np.std(final_values) + 1e-6)
        
        if improvement_ratio > 1.2:
            logger.debug(f"✅ {metric_name.upper()}: Reduced noise by {improvement_ratio:.1f}x "
                        f"(std: {np.std(original_values):.4f} → {np.std(final_values):.4f})")
        
        return processed_data
    
    def _extract_cpu_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract CPU features EXACTLY matching the UltraAccurateCPUForecaster training.
        This must match the training code PRECISELY to work correctly.
        Input: DataFrame with CPU data (8 time steps)
        Output: Array of shape (8, 9) with engineered features matching training
        """
        logger.info(f"🔬 CPU: Starting EXACT training-matched preprocessing with {len(data)} data points")
        
        if len(data) < self.cpu_sequence_length:
            logger.warning(f"Insufficient CPU data: {len(data)} < {self.cpu_sequence_length}")
            # Pad with zeros if not enough data
            padded_data = pd.DataFrame(
                np.zeros((self.cpu_sequence_length, len(data.columns))),
                columns=data.columns
            )
            if len(data) > 0:
                padded_data.iloc[-len(data):] = data.values
            data = padded_data
        
        # Take the last 8 time steps
        data = data.tail(self.cpu_sequence_length).copy()
        
        # Ensure we have a proper timestamp index for time-based features
        if 'timestamp' not in data.columns:
            if hasattr(data.index, 'to_pydatetime'):
                data['timestamp'] = data.index
            else:
                # Create synthetic timestamps if none available
                data['timestamp'] = pd.date_range(start='2024-01-01', periods=len(data), freq='15S')
        
        # === EXACT FEATURE ENGINEERING FROM UltraAccurateCPUForecaster ===
        
        # Multiple time scale moving averages (EXACT training match)
        data['cpu_ma_3'] = data['value'].rolling(window=3, center=True).mean()
        data['cpu_ma_5'] = data['value'].rolling(window=5, center=True).mean()
        data['cpu_ma_7'] = data['value'].rolling(window=7, center=True).mean()
        
        # Fill any remaining NaN from rolling operations
        data['cpu_ma_3'] = data['cpu_ma_3'].fillna(data['value'])
        data['cpu_ma_5'] = data['cpu_ma_5'].fillna(data['value'])
        data['cpu_ma_7'] = data['cpu_ma_7'].fillna(data['value'])
        
        # Statistical features (EXACT training match)
        data['cpu_std_5'] = data['value'].rolling(window=5, center=True).std().fillna(0)
        
        # Trend features (3-period difference)
        data['cpu_trend_3'] = data['value'].diff(3).fillna(0)
        
        # Momentum (5-period rate of change) - EXACT training match
        data['cpu_momentum'] = data['value'].rolling(window=5).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / len(x) if len(x) > 1 else 0, raw=False
        ).fillna(0)
        
        # Time-based features (normalized)
        timestamps = pd.to_datetime(data['timestamp'])
        data['cpu_hour'] = timestamps.dt.hour / 24.0  # Normalize hour [0, 1]
        
        # Activity level categorization (matching training bins exactly)
        data['cpu_activity_level'] = pd.cut(
            data['value'], 
            bins=[0, 0.1, 0.3, 0.6, 1.0, float('inf')], 
            labels=[0, 1, 2, 3, 4],
            include_lowest=True
        ).astype(float).fillna(0)
        
        # === EXTRACT FEATURES IN EXACT TRAINING ORDER ===
        features = []
        for i in range(len(data)):
            row_features = [
                data.iloc[i]['value'],              # 0: Main CPU feature (cores)
                data.iloc[i]['cpu_ma_3'],           # 1: 3-period moving average
                data.iloc[i]['cpu_ma_5'],           # 2: 5-period moving average  
                data.iloc[i]['cpu_ma_7'],           # 3: 7-period moving average
                data.iloc[i]['cpu_std_5'],          # 4: 5-period standard deviation
                data.iloc[i]['cpu_trend_3'],        # 5: 3-period trend
                data.iloc[i]['cpu_momentum'],       # 6: Momentum indicator
                data.iloc[i]['cpu_hour'],           # 7: Hour of day (normalized)
                data.iloc[i]['cpu_activity_level'], # 8: Activity level (0-4)
            ]
            features.append(row_features)
        
        features_array = np.array(features, dtype=np.float32)
        
        # === DEBUG: Feature Engineering Analysis ===
        logger.debug(f"CPU Feature Engineering Debug:")
        logger.debug(f"  Raw data range: [{data['value'].min():.6f}, {data['value'].max():.6f}]")
        logger.debug(f"  Moving averages: MA3=[{data['cpu_ma_3'].min():.6f}, {data['cpu_ma_3'].max():.6f}]")
        logger.debug(f"  Trends: [{data['cpu_trend_3'].min():.6f}, {data['cpu_trend_3'].max():.6f}]")
        logger.debug(f"  Activity levels: [{data['cpu_activity_level'].min():.0f}, {data['cpu_activity_level'].max():.0f}]")
        logger.debug(f"  Features array shape: {features_array.shape}")
        logger.debug(f"  Features array range: [{np.min(features_array):.6f}, {np.max(features_array):.6f}]")
        
        # === ROBUST CLEANING (MATCHING TRAINING) ===
        
        # Handle infinity and NaN values with training-matching approach
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Apply reasonable clipping (more conservative than before)
        features_array[:, 0] = np.clip(features_array[:, 0], 0, 10)      # CPU cores [0, 10]
        features_array[:, 1:4] = np.clip(features_array[:, 1:4], 0, 10)  # Moving averages [0, 10]
        features_array[:, 4] = np.clip(features_array[:, 4], 0, 5)       # Std deviation [0, 5]
        features_array[:, 5] = np.clip(features_array[:, 5], -2, 2)      # Trend [-2, 2]
        features_array[:, 6] = np.clip(features_array[:, 6], -1, 1)      # Momentum [-1, 1]
        features_array[:, 7] = np.clip(features_array[:, 7], 0, 1)       # Hour [0, 1]
        features_array[:, 8] = np.clip(features_array[:, 8], 0, 4)       # Activity level [0, 4]
        
        logger.debug(f"CPU features shape: {features_array.shape}, range: [{np.min(features_array):.4f}, {np.max(features_array):.4f}]")
        
        return features_array
    
    def _extract_memory_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract Memory features exactly matching the DualTargetLSTMForecaster training.
        Input: DataFrame with Memory data (4 time steps for memory model)
        Output: Array of shape (4, 4) with engineered features matching training
        """
        # === CONVERT AND PREPROCESS FOR MEMORY ===
        
        # Convert bytes to MB first (before preprocessing)
        data = data.copy()
        if data['value'].max() > 1000000:  # Likely in bytes
            data['value'] = data['value'] / (1024 * 1024)
            logger.debug(f"Converted memory from bytes to MB, new range: [{data['value'].min():.2f}, {data['value'].max():.2f}] MB")
        
        # === ROBUST PREPROCESSING FOR REAL-WORLD DATA ===
        data = self._preprocess_time_series(data, 'memory')
        
        if len(data) < self.memory_sequence_length:
            logger.warning(f"Insufficient Memory data: {len(data)} < {self.memory_sequence_length}")
            # Pad with zeros if not enough data
            padded_data = pd.DataFrame(
                np.zeros((self.memory_sequence_length, len(data.columns))),
                columns=data.columns
            )
            if len(data) > 0:
                padded_data.iloc[-len(data):] = data.values
            data = padded_data
        
        # Take the required sequence length
        data = data.tail(self.memory_sequence_length).copy()
        
        # === EXACT MEMORY PREPROCESSING FROM TRAINING ===
        
        # === FEATURE ENGINEERING (MATCHING TRAINING EXACTLY) ===
        
        # Moving averages with center=True matching training
        data['memory_ma_3'] = data['value'].rolling(window=3, min_periods=1, center=True).mean()
        data['memory_ma_7'] = data['value'].rolling(window=7, min_periods=1, center=True).mean()  # Longer window for memory
        
        # Fill any NaN values from rolling operations
        data['memory_ma_3'] = data['memory_ma_3'].fillna(data['value'])
        data['memory_ma_7'] = data['memory_ma_7'].fillna(data['value'])
        
        # Standard deviation (5-period with center=True)
        data['memory_std_5'] = data['value'].rolling(window=5, min_periods=1, center=True).std().fillna(0)
        
        # === EXTRACT FEATURES IN EXACT TRAINING ORDER ===
        features = []
        for i in range(len(data)):
            row_features = [
                data.iloc[i]['value'],         # 0: Main Memory feature (MB)
                data.iloc[i]['memory_ma_3'],   # 1: 3-period moving average
                data.iloc[i]['memory_ma_7'],   # 2: 7-period moving average  
                data.iloc[i]['memory_std_5'],  # 3: 5-period standard deviation
            ]
            features.append(row_features)
        
        features_array = np.array(features, dtype=np.float32)
        
        # === DEBUG: Memory Feature Engineering Analysis ===
        logger.debug(f"Memory Feature Engineering Debug:")
        logger.debug(f"  Raw data range (MB): [{data['value'].min():.2f}, {data['value'].max():.2f}]")
        logger.debug(f"  Moving averages: MA3=[{data['memory_ma_3'].min():.2f}, {data['memory_ma_3'].max():.2f}]")
        logger.debug(f"  Moving averages: MA7=[{data['memory_ma_7'].min():.2f}, {data['memory_ma_7'].max():.2f}]")
        logger.debug(f"  Std deviation: [{data['memory_std_5'].min():.2f}, {data['memory_std_5'].max():.2f}]")
        logger.debug(f"  Features array shape: {features_array.shape}")
        logger.debug(f"  Features array range: [{np.min(features_array):.2f}, {np.max(features_array):.2f}] MB")
        
        # === ROBUST CLEANING (MATCHING TRAINING) ===
        
        # Handle infinity and NaN values
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1000.0, neginf=0.0)
        
        # Apply reasonable clipping for memory (in MB)
        features_array[:, 0] = np.clip(features_array[:, 0], 0, 8192)    # Memory [0, 8GB]
        features_array[:, 1] = np.clip(features_array[:, 1], 0, 8192)    # MA_3 [0, 8GB]
        features_array[:, 2] = np.clip(features_array[:, 2], 0, 8192)    # MA_7 [0, 8GB]
        features_array[:, 3] = np.clip(features_array[:, 3], 0, 1024)    # Std [0, 1GB]
        
        logger.debug(f"Memory features shape: {features_array.shape}, range: [{np.min(features_array):.2f}, {np.max(features_array):.2f}] MB")
        
        return features_array
    
    def predict_cpu(self, cpu_data: pd.DataFrame) -> Optional[float]:
        """
        Predict CPU usage for the next interval using UltraAccurateCPUForecaster approach.
        
        Args:
            cpu_data: DataFrame with CPU time series data
            
        Returns:
            Predicted CPU usage (cores) or None if prediction failed
        """
        if self.cpu_model is None or self.cpu_scaler is None:
            logger.error("CPU model not loaded")
            return None
        
        try:
            # Extract features (shape: [8, 9])
            features = self._extract_cpu_features(cpu_data)
            logger.debug(f"CPU features extracted: shape={features.shape}")
            
            # Scale features
            logger.debug(f"=== CPU SCALING DEBUG ===")
            logger.debug(f"Scaler type: {type(self.cpu_scaler)}")
            if hasattr(self.cpu_scaler, 'scale_'):
                logger.debug(f"Scaler scale_: {self.cpu_scaler.scale_[:3]}...")  # First 3 elements
            if hasattr(self.cpu_scaler, 'mean_'):
                logger.debug(f"Scaler mean_: {self.cpu_scaler.mean_[:3]}...")   # First 3 elements
            
            scaled_features = self.cpu_scaler.transform(features)
            logger.debug(f"CPU features scaled: shape={scaled_features.shape}, range=[{np.min(scaled_features):.4f}, {np.max(scaled_features):.4f}]")
            
            # Check for scaling issues
            if np.any(np.abs(scaled_features) > 10):
                logger.warning(f"⚠️ CPU: Extreme scaled values detected! Max: {np.max(np.abs(scaled_features)):.2f}")
            if np.all(np.abs(scaled_features) < 0.01):
                logger.warning(f"⚠️ CPU: All scaled values very small! This might indicate scaling problems.")
            
            # Reshape for model input (1, sequence_length, features)
            model_input = scaled_features.reshape(1, self.cpu_sequence_length, self.cpu_features)
            logger.debug(f"CPU model input shape: {model_input.shape}")
            
            # Predict (model outputs single value)
            prediction_scaled = self.cpu_model.predict(model_input, verbose=0)
            logger.debug(f"CPU model output shape: {prediction_scaled.shape}")
            
            # === HANDLE INVERSE TRANSFORM FOR CPU ===
            
            # The CPU model outputs a single value, but the scaler was trained on 9 features
            # We need to create a full feature array to inverse transform properly
            
            if prediction_scaled.ndim == 2 and prediction_scaled.shape[1] == 1:
                # Model outputs shape (1, 1) - single prediction
                pred_value = prediction_scaled[0, 0]
            else:
                pred_value = prediction_scaled.flatten()[0]
            
            # Create full feature array for inverse transform
            # Put the prediction in the first position (main CPU feature)
            full_prediction = np.zeros((1, self.cpu_features))  # Shape: (1, 9)
            full_prediction[0, 0] = pred_value  # Main CPU prediction
            
            # Inverse transform
            prediction_unscaled = self.cpu_scaler.inverse_transform(full_prediction)
            result = float(prediction_unscaled[0, 0])  # Get the main CPU feature
            
            # Apply post-processing (matching training approach)
            # Handle log transformation if it was used (check if values seem log-transformed)
            if result < 0:
                # If negative, likely log-transformed, apply expm1
                try:
                    result = np.expm1(result)
                except:
                    result = max(0, result)
            
            # Ensure physical constraints
            result = max(0.0, result)  # CPU can't be negative
            result = min(result, 10.0)  # Reasonable upper bound
            
            logger.debug(f"CPU prediction final: {result:.6f} cores")
            return result
            
        except Exception as e:
            logger.error(f"CPU prediction failed: {e}")
            logger.debug(f"CPU prediction error details", exc_info=True)
            return None
    
    def predict_memory(self, memory_data: pd.DataFrame) -> Optional[float]:
        """
        Predict Memory usage for the next interval using DualTargetLSTMForecaster approach.
        
        Args:
            memory_data: DataFrame with Memory time series data
            
        Returns:
            Predicted Memory usage (MB) or None if prediction failed
        """
        if self.memory_model is None or self.memory_scaler is None:
            logger.error("Memory model not loaded")
            return None
        
        try:
            # Extract features (shape: [sequence_length, 4])
            features = self._extract_memory_features(memory_data)
            logger.debug(f"Memory features extracted: shape={features.shape}")
            
            # Scale features
            scaled_features = self.memory_scaler.transform(features)
            logger.debug(f"Memory features scaled: shape={scaled_features.shape}, range=[{np.min(scaled_features):.4f}, {np.max(scaled_features):.4f}]")
            
            # Reshape for model input (1, sequence_length, features)
            model_input = scaled_features.reshape(1, self.memory_sequence_length, self.memory_features)
            logger.debug(f"Memory model input shape: {model_input.shape}")
            
            # Predict (model outputs single value)
            prediction_scaled = self.memory_model.predict(model_input, verbose=0)
            logger.debug(f"Memory model output shape: {prediction_scaled.shape}")
            
            # === HANDLE INVERSE TRANSFORM FOR MEMORY ===
            
            # The Memory model outputs a single value, but the scaler was trained on 4 features
            # We need to create a full feature array to inverse transform properly
            
            if prediction_scaled.ndim == 2 and prediction_scaled.shape[1] == 1:
                # Model outputs shape (1, 1) - single prediction
                pred_value = prediction_scaled[0, 0]
            else:
                pred_value = prediction_scaled.flatten()[0]
            
            # Create full feature array for inverse transform
            # Put the prediction in the first position (main Memory feature)
            full_prediction = np.zeros((1, self.memory_features))  # Shape: (1, 4)
            full_prediction[0, 0] = pred_value  # Main Memory prediction
            
            # Inverse transform
            prediction_unscaled = self.memory_scaler.inverse_transform(full_prediction)
            result = float(prediction_unscaled[0, 0])  # Get the main Memory feature
            
            # Apply post-processing (matching training approach)
            # Ensure physical constraints for memory
            result = max(0.0, result)  # Memory can't be negative
            result = min(result, 8192.0)  # Reasonable upper bound (8GB in MB)
            
            logger.debug(f"Memory prediction final: {result:.2f} MB")
            return result
            
        except Exception as e:
            logger.error(f"Memory prediction failed: {e}")
            logger.debug(f"Memory prediction error details", exc_info=True)
            return None
    
    def predict_all(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Optional[float]]:
        """
        Make predictions for both CPU and Memory.
        
        Args:
            historical_data: Dictionary with 'process_cpu_seconds_total_rate' and 
                           'process_resident_memory_bytes' DataFrames
            
        Returns:
            Dictionary with predicted values for CPU and Memory
        """
        predictions = {}
        
        # === DEBUG: Data Quality Analysis ===
        logger.info("=== FORECASTING DEBUG: Data Quality Analysis ===")
        for metric_name, df in historical_data.items():
            if df is not None and not df.empty:
                values = df['value'].values
                logger.info(f"{metric_name}:")
                logger.info(f"  - Data points: {len(values)}")
                logger.info(f"  - Value range: [{np.min(values):.6f}, {np.max(values):.6f}]")
                logger.info(f"  - Mean: {np.mean(values):.6f}, Std: {np.std(values):.6f}")
                logger.info(f"  - Recent trend: {values[-1]:.6f} (latest)")
                if len(values) >= 2:
                    logger.info(f"  - Change from previous: {(values[-1] - values[-2]):.6f}")
            else:
                logger.warning(f"{metric_name}: No data available")
        
        # CPU prediction
        if 'process_cpu_seconds_total_rate' in historical_data:
            cpu_data = historical_data['process_cpu_seconds_total_rate']
            if cpu_data is not None and not cpu_data.empty:
                logger.info(f"=== CPU PREDICTION INPUT ===")
                logger.info(f"CPU data shape: {cpu_data.shape}")
                logger.info(f"CPU data columns: {list(cpu_data.columns)}")
                logger.info(f"CPU latest values: {cpu_data['value'].tail(3).tolist()}")
                
                cpu_pred = self.predict_cpu(cpu_data)
                predictions['cpu_forecast'] = cpu_pred
                
                if cpu_pred is not None:
                    current_cpu = cpu_data['value'].iloc[-1] if not cpu_data.empty else 0
                    logger.info(f"=== CPU PREDICTION RESULT ===")
                    logger.info(f"Current CPU: {current_cpu:.6f}")
                    logger.info(f"Predicted CPU: {cpu_pred:.6f}")
                    logger.info(f"Prediction change: {(cpu_pred - current_cpu):.6f}")
                    logger.info(f"Prediction change %: {((cpu_pred - current_cpu) / max(current_cpu, 0.001) * 100):.2f}%")
            else:
                predictions['cpu_forecast'] = None
                logger.warning("CPU data is empty")
        else:
            predictions['cpu_forecast'] = None
            logger.warning("No CPU data available for prediction")
        
        # Memory prediction
        if 'process_resident_memory_bytes' in historical_data:
            memory_data = historical_data['process_resident_memory_bytes']
            if memory_data is not None and not memory_data.empty:
                logger.info(f"=== MEMORY PREDICTION INPUT ===")
                logger.info(f"Memory data shape: {memory_data.shape}")
                logger.info(f"Memory data columns: {list(memory_data.columns)}")
                memory_mb = memory_data['value'].tail(3) / (1024 * 1024)  # Convert to MB for logging
                logger.info(f"Memory latest values (MB): {memory_mb.tolist()}")
                
                memory_pred = self.predict_memory(memory_data)
                predictions['memory_forecast'] = memory_pred
                
                if memory_pred is not None:
                    current_memory_mb = memory_data['value'].iloc[-1] / (1024 * 1024) if not memory_data.empty else 0
                    logger.info(f"=== MEMORY PREDICTION RESULT ===")
                    logger.info(f"Current Memory: {current_memory_mb:.2f} MB")
                    logger.info(f"Predicted Memory: {memory_pred:.2f} MB")
                    logger.info(f"Prediction change: {(memory_pred - current_memory_mb):.2f} MB")
                    logger.info(f"Prediction change %: {((memory_pred - current_memory_mb) / max(current_memory_mb, 1) * 100):.2f}%")
            else:
                predictions['memory_forecast'] = None
                logger.warning("Memory data is empty")
        else:
            predictions['memory_forecast'] = None
            logger.warning("No Memory data available for prediction")
        
        logger.info("=== FORECASTING DEBUG COMPLETE ===")
        return predictions 