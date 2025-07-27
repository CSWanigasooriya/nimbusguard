import logging
import pandas as pd
import numpy as np
from state_manager import state

class MemoryForecaster:
    """
    Handles LSTM-based memory forecasting for Kubernetes autoscaling.
    Uses shared state for model storage and configuration.
    """
    
    def __init__(self):
        self.sequence_length = state.sequence_length
        
    def load_models(self, model_path, scaler_path):
        """
        Load LSTM model and scalers into shared state.
        
        Args:
            model_path: Path to the .keras model file
            scaler_path: Path to the .pkl scaler file
        """
        import tensorflow as tf
        import joblib
        
        logging.info("--- Loading LSTM model and scalers ---")
        try:
            # Load the Keras model
            state.memory_model = tf.keras.models.load_model(model_path)
            logging.info("LSTM model loaded successfully.")
            
            # Load the scalers
            scalers_data = joblib.load(scaler_path)
            state.memory_scaler = scalers_data['feature_scaler']
            state.target_scaler = scalers_data['target_scaler']
            logging.info("Scalers loaded successfully.")
            
            # Verify prediction mode
            predict_both = scalers_data.get('predict_both', True)
            lookback_window = scalers_data.get('lookback_window', 20)
            logging.info(f"Model configuration: predict_both={predict_both}, lookback_window={lookback_window}")
            
            state.models_loaded.set()
            logging.info("LSTM forecaster ready.")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load LSTM model and scalers: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    def is_ready(self):
        """Check if models are loaded and ready for prediction."""
        return state.models_loaded.is_set()
    
    def predict_next_interval(self, pod_data):
        """
        Predict memory and pod count for next 15-second interval.
        
        Args:
            pod_data: List of dictionaries with format:
                [{'timestamp': str, 'pod_name': str, 'memory_bytes': int}, ...]
        
        Returns:
            Dictionary with prediction results, or None if prediction fails
        """
        if not self.is_ready():
            logging.error("LSTM models not loaded")
            return None
            
        if not pod_data:
            logging.warning("No pod data provided for prediction")
            return None
        
        try:
            # Check if we have enough data points
            if len(pod_data) < self.sequence_length:
                logging.info(f"Not enough data for LSTM prediction. Have {len(pod_data)}, need {self.sequence_length}")
                return None
            
            # Preprocess data
            aggregated = self._preprocess_pod_data(pod_data)
            if aggregated is None:
                return None
            
            # Filter for stable periods
            stable_data = self._filter_stable_periods(aggregated)
            if stable_data is None:
                return None
            
            # Make prediction
            prediction_result = self._make_lstm_prediction(stable_data, aggregated)
            return prediction_result
            
        except Exception as e:
            logging.error(f"Error during LSTM prediction: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None
    
    def _preprocess_pod_data(self, pod_data):
        """Convert raw pod data to aggregated time series."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(pod_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Round timestamps to nearest 15 seconds to group pods with slightly different timestamps
            df['timestamp_rounded'] = df['timestamp'].dt.round('15s')
            
            # Aggregate by rounded timestamp
            aggregated = df.groupby('timestamp_rounded').agg({
                'memory_bytes': 'sum',    # Total memory across all consumer pods
                'pod_name': 'count'       # Number of consumer pods
            }).reset_index()
            
            # Rename column back to timestamp
            aggregated = aggregated.rename(columns={'timestamp_rounded': 'timestamp'})
            
            # Rename columns to match training format
            aggregated.columns = ['timestamp', 'total_memory_bytes', 'pod_count']
            
            # Convert to MB (same as training)
            aggregated['total_memory_mb'] = aggregated['total_memory_bytes'] / (1024 * 1024)
            
            # Sort by timestamp
            aggregated = aggregated.sort_values('timestamp').reset_index(drop=True)
            
            # Debug: Show aggregation results
            logging.info(f"--- LSTM Aggregation: {len(aggregated)} time intervals from {len(pod_data)} pod entries ---")
            if len(aggregated) > 0:
                latest_agg = aggregated.iloc[-1]
                logging.info(f"    Latest aggregated: {latest_agg['timestamp']} | {latest_agg['total_memory_mb']:.1f} MB | {latest_agg['pod_count']} pods")
            
            return aggregated
            
        except Exception as e:
            logging.error(f"Error preprocessing pod data: {e}")
            return None
    
    def _filter_stable_periods(self, aggregated):
        """Filter data to only include periods with stable pod counts."""
        try:
            # Check if we have enough history after aggregation
            if len(aggregated) < self.sequence_length:
                logging.info(f"Not enough aggregated data for prediction. Need {self.sequence_length}, got {len(aggregated)}")
                logging.info(f"LSTM needs {self.sequence_length * 15} seconds ({self.sequence_length * 15 / 60:.1f} minutes) of data for prediction")
                return None
            
            # Filter out periods with pod count changes to get stable prediction data
            current_pod_count = aggregated['pod_count'].iloc[-1]
            stable_data = aggregated[aggregated['pod_count'] == current_pod_count]
            
            if len(stable_data) < self.sequence_length:
                logging.info(f"Not enough stable data (same pod count={current_pod_count}). Have {len(stable_data)}, need {self.sequence_length}")
                logging.info("LSTM requires stable pod count period for accurate prediction")
                return None
            
            logging.info(f"Found {len(stable_data)} stable intervals with {current_pod_count} pods")
            return stable_data
            
        except Exception as e:
            logging.error(f"Error filtering stable periods: {e}")
            return None
    
    def _make_lstm_prediction(self, stable_data, aggregated):
        """Make prediction using LSTM model."""
        try:
            # Extract features for last N intervals
            features = ['total_memory_mb', 'pod_count']
            recent_data = stable_data[features].tail(self.sequence_length).values
            
            # Debug: Show the actual sequence we're feeding to LSTM
            recent_timestamps = stable_data['timestamp'].tail(self.sequence_length)
            recent_memory = stable_data['total_memory_mb'].tail(self.sequence_length)
            recent_pods = stable_data['pod_count'].tail(self.sequence_length)
            current_pod_count = aggregated['pod_count'].iloc[-1]
            
            logging.info(f"--- LSTM Input Sequence (last {self.sequence_length} stable intervals with {current_pod_count} pods) ---")
            logging.info(f"Memory range: {recent_memory.min():.1f} - {recent_memory.max():.1f} MB")
            logging.info(f"Pod count: {current_pod_count} (constant)")
            
            # Scale using the scaler
            scaled_data = state.memory_scaler.transform(recent_data)
            
            # Reshape for LSTM (batch_size=1, sequence_length=N, features=2)
            model_input = scaled_data.reshape(1, self.sequence_length, len(features))
            
            # Make prediction using the raw Keras model
            prediction_scaled = state.memory_model.predict(model_input, verbose=0)
            
            # Debug: Show raw model output
            logging.info(f"--- LSTM Raw Output (scaled): {prediction_scaled[0]} ---")
            
            # Inverse transform prediction
            prediction = state.target_scaler.inverse_transform(prediction_scaled)[0]
            
            # Debug: Show final prediction
            logging.info(f"--- LSTM Final Prediction: Memory={prediction[0]:.1f} MB, Pods={prediction[1]:.1f} ---")
            
            # Get current values for comparison
            latest_stable = stable_data.iloc[-1]
            latest_all = aggregated.iloc[-1]
            
            # Debug logging
            logging.info(f"--- LSTM Debug: Latest stable data: {latest_stable['total_memory_mb']:.1f} MB ---")
            logging.info(f"--- LSTM Debug: Latest all data: {latest_all['total_memory_mb']:.1f} MB ---")
            logging.info(f"--- LSTM Raw Prediction vs Latest Stable: {prediction[0]:.1f} MB vs {latest_stable['total_memory_mb']:.1f} MB ---")
            
            # Return structured result (use latest_all for current comparison since DQN uses current total)
            result = {
                'predicted_memory_mb': round(prediction[0], 1),
                'predicted_pod_count': round(prediction[1], 0),
                'predicted_memory_bytes': int(prediction[0] * 1024 * 1024),
                'current_memory_bytes': int(latest_all['total_memory_bytes']),
                'memory_change_mb': round(prediction[0] - latest_all['total_memory_mb'], 1),
                'pod_change': round(prediction[1] - latest_all['pod_count'], 0)
            }
            
            # Additional debug: Check if prediction equals current
            if abs(result['memory_change_mb']) < 0.1:
                logging.warning(f"LSTM predicting essentially no change! Raw prediction: {prediction[0]:.6f} MB, Current: {latest_all['total_memory_mb']:.6f} MB")
                logging.warning(f"Raw model output was: {prediction_scaled[0]}")
            
            logging.info(f"LSTM prediction successful: Memory {result['predicted_memory_mb']:.0f}MB (+{result['memory_change_mb']:.0f}), Pods {result['predicted_pod_count']:.0f} (+{result['pod_change']:.0f})")
            return result
            
        except Exception as e:
            logging.error(f"Error making LSTM prediction: {e}")
            return None
    
    def get_prediction_requirements(self):
        """Get data requirements for making predictions."""
        return {
            'min_intervals': self.sequence_length,
            'min_time_minutes': (self.sequence_length * 15) / 60,
            'requires_stable_pod_count': True,
            'models_loaded': self.is_ready()
        } 