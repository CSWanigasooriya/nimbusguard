import numpy as np
import torch
import torch.nn as nn
from collections import deque
from datetime import datetime, timedelta
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
import json

logger = logging.getLogger("LSTM_Forecaster")

class LSTMForecaster(nn.Module):
    """LSTM model for predicting workload patterns and system behavior."""
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = None, num_layers: int = None, 
                 forecast_horizon: int = 5, dropout: float = None):
        super(LSTMForecaster, self).__init__()
        
        # Import configurable parameters from environment variables
        if hidden_dim is None:
            try:
                import os
                hidden_dim = int(os.getenv("LSTM_HIDDEN_DIM", 64))
            except:
                hidden_dim = 64  # Default fallback
        if num_layers is None:
            try:
                import os
                num_layers = int(os.getenv("LSTM_NUM_LAYERS", 2))
            except:
                num_layers = 2  # Default fallback
        if dropout is None:
            try:
                import os
                dropout = float(os.getenv("LSTM_DROPOUT", 0.2))
            except:
                dropout = 0.2  # Default fallback
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers for PURE TEMPORAL INTELLIGENCE (no current state redundancy)
        self.next_30sec_pressure_head = nn.Linear(hidden_dim, 1)       # PROACTIVE: Load pressure in next 30 seconds
        self.next_60sec_pressure_head = nn.Linear(hidden_dim, 1)       # PROACTIVE: Load pressure in next 60 seconds
        self.trend_velocity_head = nn.Linear(hidden_dim, 1)            # How fast is the trend changing
        self.pattern_type_head = nn.Linear(hidden_dim, 4)              # Pattern type: spike, gradual, cyclical, random
        self.time_to_peak_head = nn.Linear(hidden_dim, 1)              # Estimated seconds until next peak/trough
        self.optimal_replicas_head = nn.Linear(hidden_dim, 1)          # Optimal replicas based on predicted pattern
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.constant_(param, 0)
    
    def forward(self, x):
        """
        Forward pass through LSTM.
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
        Returns:
            Dictionary of predictions
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state for predictions
        last_hidden = hidden[-1]  # Shape: (batch_size, hidden_dim)
        
        # Generate PURE TEMPORAL INTELLIGENCE (no current state redundancy)
        predictions = {
            'next_30sec_pressure': torch.sigmoid(self.next_30sec_pressure_head(last_hidden)),         # PROACTIVE: 30sec ahead
            'next_60sec_pressure': torch.sigmoid(self.next_60sec_pressure_head(last_hidden)),         # PROACTIVE: 60sec ahead
            'trend_velocity': torch.tanh(self.trend_velocity_head(last_hidden)),                      # -1 to 1: how fast trend changes
            'pattern_type': torch.softmax(self.pattern_type_head(last_hidden), dim=1),               # spike/gradual/cyclical/random
            'time_to_peak': torch.sigmoid(self.time_to_peak_head(last_hidden)) * 300,                # 0-300 seconds to next peak
            'optimal_replicas': torch.sigmoid(self.optimal_replicas_head(last_hidden)) * 10 + 1      # 1-11 replicas
        }
        
        return predictions


class TimeSeriesBuffer:
    """Buffer for storing time series data and generating LSTM features."""
    
    def __init__(self, sequence_length: int = None, feature_names: List[str] = None):
        """
        Initialize time series buffer.
        Args:
            sequence_length: Number of time steps to store (configurable, default 24 = 2 minutes at 5-second intervals)
            feature_names: List of feature names to track
        """
        # Use configurable sequence length
        if sequence_length is None:
            try:
                # Try to import from main module if available
                import os
                sequence_length = int(os.getenv("LSTM_SEQUENCE_LENGTH", 24))
            except:
                sequence_length = 24  # Default fallback
        
        self.sequence_length = sequence_length
        self.feature_names = feature_names or [
            'http_request_duration_highr_seconds_bucket',   # HTTP Traffic patterns
            'process_resident_memory_bytes',                # Memory resource usage  
            'process_cpu_seconds_total',                    # CPU resource usage
            'scrape_samples_scraped',                       # Health monitoring
            'http_response_size_bytes_sum'                  # Response size patterns
        ]
        
        # Buffer for storing sequences
        self.buffer = deque(maxlen=sequence_length)
        self.timestamps = deque(maxlen=sequence_length)
        
        # Statistics for normalization
        self.feature_stats = {name: {'min': float('inf'), 'max': float('-inf'), 'mean': 0, 'count': 0} 
                             for name in self.feature_names}
        
        # Pattern detection
        self.pattern_history = deque(maxlen=288)  # 24 hours at 5-second intervals
        self.daily_patterns = {}
        self.weekly_patterns = {}
        
        logger.info(f"LSTM_BUFFER: initialized steps={sequence_length} features={len(self.feature_names)}")
    
    def add_observation(self, metrics: Dict[str, float], timestamp: Optional[datetime] = None) -> None:
        """Add new observation to the buffer."""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Extract relevant features and normalize
        observation = []
        for feature_name in self.feature_names:
            value = metrics.get(feature_name, 0.0)
            
            # Update statistics
            stats = self.feature_stats[feature_name]
            stats['min'] = min(stats['min'], value)
            stats['max'] = max(stats['max'], value)
            stats['count'] += 1
            stats['mean'] = (stats['mean'] * (stats['count'] - 1) + value) / stats['count']
            
            # Simple normalization (can be improved with robust scaling)
            if stats['max'] > stats['min']:
                normalized_value = (value - stats['min']) / (stats['max'] - stats['min'])
            else:
                normalized_value = 0.5  # Default middle value
            
            observation.append(normalized_value)
        
        self.buffer.append(observation)
        self.timestamps.append(timestamp)
        
        # Update pattern history for pattern detection
        self._update_pattern_history(observation, timestamp)
    
    def _update_pattern_history(self, observation: List[float], timestamp: datetime) -> None:
        """Update pattern history for cyclical pattern detection."""
        # Add to pattern history
        pattern_data = {
            'timestamp': timestamp,
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'http_bucket': observation[0] if len(observation) > 0 else 0.0,     # HTTP bucket
            'memory_usage': observation[1] if len(observation) > 1 else 0.0,    # Memory usage
            'scrape_health': observation[3] if len(observation) > 3 else 0.0    # Scrape samples
        }
        self.pattern_history.append(pattern_data)
        
        # Update daily patterns (every hour)
        hour_key = timestamp.hour
        if hour_key not in self.daily_patterns:
            self.daily_patterns[hour_key] = {'http_buckets': [], 'memory_usage': [], 'scrape_health': []}
        
        self.daily_patterns[hour_key]['http_buckets'].append(pattern_data['http_bucket'])
        self.daily_patterns[hour_key]['memory_usage'].append(pattern_data['memory_usage'])
        self.daily_patterns[hour_key]['scrape_health'].append(pattern_data['scrape_health'])
        
        # Keep only recent data (last 7 days)
        if len(self.daily_patterns[hour_key]['http_buckets']) > 7:
            self.daily_patterns[hour_key]['http_buckets'].pop(0)
            self.daily_patterns[hour_key]['memory_usage'].pop(0)
            self.daily_patterns[hour_key]['scrape_health'].pop(0)
    
    def get_sequence(self) -> Optional[np.ndarray]:
        """Get the current sequence as numpy array."""
        if len(self.buffer) < 2:  # Need at least 2 points
            return None
        
        return np.array(list(self.buffer), dtype=np.float32)
    
    def has_sufficient_data(self, min_length: int = 12) -> bool:
        """Check if we have sufficient data for predictions (default: 1 minute)."""
        return len(self.buffer) >= min_length
    
    def detect_patterns(self) -> Dict[str, float]:
        """Detect cyclical patterns and trends."""
        if len(self.pattern_history) < 12:  # Need at least 1 minute of data
            return {
                'daily_pattern_strength': 0.0,
                'weekly_pattern_strength': 0.0,
                'trend_direction': 0.0,  # -1: decreasing, 0: stable, 1: increasing
                'pattern_confidence': 0.0
            }
        
        # Analyze recent trend (last 5 minutes)
        recent_data = list(self.pattern_history)[-60:] if len(self.pattern_history) >= 60 else list(self.pattern_history)
        http_buckets = [d['http_bucket'] for d in recent_data]
        memory_usage = [d['memory_usage'] for d in recent_data]
        scrape_health = [d['scrape_health'] for d in recent_data]
        
        # Simple trend detection using linear regression on HTTP buckets
        if len(http_buckets) >= 3:
            x = np.arange(len(http_buckets))
            slope = np.polyfit(x, http_buckets, 1)[0]
            trend_direction = np.clip(slope * 100, -1, 1)  # Scale and clip
        else:
            trend_direction = 0.0
        
        # Daily pattern strength (check if current hour matches historical pattern)
        current_hour = datetime.now().hour
        daily_pattern_strength = 0.0
        
        if current_hour in self.daily_patterns:
            historical_avg = np.mean(self.daily_patterns[current_hour]['http_buckets'])
            current_avg = np.mean(http_buckets[-5:]) if len(http_buckets) >= 5 else http_buckets[-1]
            
            # Pattern strength based on how well current matches historical
            if historical_avg > 0:
                similarity = 1 - abs(current_avg - historical_avg) / (historical_avg + 0.001)
                daily_pattern_strength = max(0, similarity)
        
        # Overall pattern confidence
        pattern_confidence = (daily_pattern_strength + (1 - abs(trend_direction))) / 2
        
        return {
            'daily_pattern_strength': daily_pattern_strength,
            'weekly_pattern_strength': 0.0,  # TODO: Implement weekly patterns
            'trend_direction': trend_direction,
            'pattern_confidence': pattern_confidence
        }
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get buffer statistics."""
        return {
            'buffer_size': len(self.buffer),
            'max_size': self.sequence_length,
            'utilization': len(self.buffer) / self.sequence_length,
            'feature_stats': self.feature_stats,
            'time_span': (self.timestamps[-1] - self.timestamps[0]).total_seconds() if len(self.timestamps) >= 2 else 0
        }


class LSTMWorkloadPredictor:
    """Main class that coordinates LSTM forecasting with the DQN system."""
    
    def __init__(self, feature_names: List[str], device: str = 'cpu', 
                 sequence_length: int = 60, model_path: Optional[str] = None):
        """
        Initialize LSTM workload predictor.
        Args:
            feature_names: List of feature names from DQN system
            device: PyTorch device ('cpu' or 'cuda')
            sequence_length: Length of sequences to use for prediction
            model_path: Path to saved LSTM model (optional)
        """
        self.device = torch.device(device)
        self.feature_names = feature_names
        self.sequence_length = sequence_length
        
        # Initialize components with configurable parameters
        self.buffer = TimeSeriesBuffer(sequence_length, feature_names)
        self.model = LSTMForecaster(
            input_dim=len(feature_names),
            hidden_dim=None,  # Will use configurable value
            num_layers=None,  # Will use configurable value
            dropout=None,     # Will use configurable value
            forecast_horizon=1  # Reactive analysis of current state
        ).to(self.device)
        
        # Training components
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        
        # Prediction cache
        self.last_prediction = None
        self.last_prediction_time = None
        self.prediction_cache_duration = 10  # Cache predictions for 10 seconds
        
        # Load pre-trained model if available
        if model_path:
            self.load_model(model_path)
        
        logger.info(f"LSTM_PREDICTOR: initialized device={device}")
        logger.info(f"LSTM_PREDICTOR: features={len(feature_names)}")
        logger.info(f"LSTM_PREDICTOR: sequence_length={sequence_length} steps")
        logger.info(f"LSTM_PREDICTOR: forecast_horizon=5_steps_25_seconds")
    
    async def add_observation_async(self, metrics: Dict[str, float]) -> None:
        """Async wrapper for adding observations."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.buffer.add_observation, metrics)
    
    def add_observation(self, metrics: Dict[str, float]) -> None:
        """Add new metrics observation to the buffer."""
        self.buffer.add_observation(metrics)
    
    async def get_lstm_features_async(self) -> Dict[str, float]:
        """Async wrapper for getting LSTM features."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_lstm_features)
    
    def get_lstm_features(self) -> Dict[str, float]:
        """Generate LSTM-based features for the DQN system."""
        # Check cache first
        now = datetime.now()
        if (self.last_prediction is not None and 
            self.last_prediction_time is not None and
            (now - self.last_prediction_time).total_seconds() < self.prediction_cache_duration):
            return self.last_prediction
        
        # Default features for when LSTM is not ready
        default_features = {
            'next_30sec_pressure': 0.5,              # PROACTIVE: Assume stable in 30sec
            'next_60sec_pressure': 0.5,              # PROACTIVE: Assume stable in 60sec
            'trend_velocity': 0.0,                   # No acceleration/deceleration
            'pattern_type_spike': 0.25,              # Equal probability for all patterns
            'pattern_type_gradual': 0.25,
            'pattern_type_cyclical': 0.25,
            'pattern_type_random': 0.25,
            'time_to_peak': 150.0,                   # 2.5 minutes default
            'optimal_replicas_forecast': 2.0         # 2 replicas default
        }
        
        try:
            # Check if we have enough data
            if not self.buffer.has_sufficient_data():
                logger.debug("LSTM_PREDICTION: insufficient_data using_defaults")
                return default_features
            
            # Get sequence data
            sequence = self.buffer.get_sequence()
            if sequence is None:
                return default_features
            
            # Prepare input tensor
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # Add batch dimension
            
            # Run prediction
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(sequence_tensor)
            
            # Extract PURE TEMPORAL INTELLIGENCE (no current state redundancy)
            next_30sec_pressure = predictions['next_30sec_pressure'][0, 0].item()
            next_60sec_pressure = predictions['next_60sec_pressure'][0, 0].item()
            trend_velocity = predictions['trend_velocity'][0, 0].item()
            time_to_peak = predictions['time_to_peak'][0, 0].item()
            optimal_replicas = predictions['optimal_replicas'][0, 0].item()
            
            # Pattern type (get probabilities for each type)
            pattern_probs = predictions['pattern_type'][0].cpu().numpy()
            pattern_types = ['spike', 'gradual', 'cyclical', 'random']
            
            # Get additional pattern information from buffer
            pattern_info = self.buffer.detect_patterns()
            
            # Combine LSTM temporal intelligence with buffer pattern detection
            features = {
                'next_30sec_pressure': float(next_30sec_pressure),           # PROACTIVE: 30sec ahead
                'next_60sec_pressure': float(next_60sec_pressure),           # PROACTIVE: 60sec ahead
                'trend_velocity': float(trend_velocity),                     # How fast trend is changing
                'pattern_type_spike': float(pattern_probs[0]),               # Probability of spike pattern
                'pattern_type_gradual': float(pattern_probs[1]),             # Probability of gradual pattern
                'pattern_type_cyclical': float(pattern_probs[2]),            # Probability of cyclical pattern
                'pattern_type_random': float(pattern_probs[3]),              # Probability of random pattern
                'time_to_peak': float(time_to_peak),                         # Seconds to next peak/trough
                'optimal_replicas_forecast': float(optimal_replicas)         # LSTM's replica recommendation
            }
            
            # Cache the prediction
            self.last_prediction = features
            self.last_prediction_time = now
            
            logger.debug(f"LSTM_TEMPORAL: 30sec_pressure={next_30sec_pressure:.2f} "
                        f"60sec_pressure={next_60sec_pressure:.2f} "
                        f"trend_velocity={trend_velocity:+.2f} "
                        f"pattern={pattern_types[np.argmax(pattern_probs)]} "
                        f"time_to_peak={time_to_peak:.0f}s")
            
            return features
            
        except Exception as e:
            logger.error(f"LSTM_PREDICTION: failed error={e}")
            return default_features
    
    def save_model(self, path: str) -> None:
        """Save LSTM model to disk."""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'feature_names': self.feature_names,
                'sequence_length': self.sequence_length,
                'buffer_stats': self.buffer.get_statistics()
            }
            torch.save(checkpoint, path)
            logger.info(f"LSTM_MODEL: saved path={path}")
        except Exception as e:
            logger.error(f"LSTM_MODEL: save_failed error={e}")
    
    def load_model(self, path: str) -> bool:
        """Load LSTM model from disk."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"LSTM_MODEL: loaded path={path}")
            return True
        except Exception as e:
            logger.error(f"LSTM_MODEL: load_failed error={e}")
            return False
    
    def get_buffer_info(self) -> Dict[str, any]:
        """Get information about the current buffer state."""
        stats = self.buffer.get_statistics()
        pattern_info = self.buffer.detect_patterns()
        
        return {
            'buffer_stats': stats,
            'pattern_info': pattern_info,
            'model_ready': self.buffer.has_sufficient_data(),
            'last_prediction': self.last_prediction,
            'feature_names': self.feature_names
        }


# Factory function for easy integration
def create_lstm_predictor(feature_names: List[str], device: str = 'cpu') -> LSTMWorkloadPredictor:
    """Create and return an LSTM predictor instance with configurable parameters."""
    return LSTMWorkloadPredictor(
        feature_names=feature_names,
        device=device,
        sequence_length=None,  # Will use configurable LSTM_SEQUENCE_LENGTH
    ) 