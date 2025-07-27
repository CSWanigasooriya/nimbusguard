import os
import logging
from typing import Dict, Any

class DQNConfig:
    """Configuration class for DQN Agent hyperparameters."""
    
    def __init__(self):
        """Initialize DQN configuration from environment variables."""
        self.gamma = float(os.getenv('DQN_GAMMA', '0.95'))
        self.epsilon = float(os.getenv('DQN_EPSILON', '1.0'))
        self.epsilon_min = float(os.getenv('DQN_EPSILON_MIN', '0.01'))
        self.epsilon_decay = float(os.getenv('DQN_EPSILON_DECAY', '0.995'))
        self.learning_rate = float(os.getenv('DQN_LEARNING_RATE', '0.001'))
        self.batch_size = int(os.getenv('DQN_BATCH_SIZE', '32'))
        self.memory_size = int(os.getenv('DQN_MEMORY_SIZE', '2000'))
        self.update_target_freq = int(os.getenv('DQN_UPDATE_TARGET_FREQ', '10'))
        
        # Network architecture
        self.hidden_units = int(os.getenv('DQN_HIDDEN_UNITS', '24'))
        self.hidden_layers = int(os.getenv('DQN_HIDDEN_LAYERS', '2'))
        
        # Model persistence settings
        self.save_frequency = int(os.getenv('DQN_SAVE_FREQUENCY', '100'))  # Save every N training steps
        self.save_on_improvement = bool(os.getenv('DQN_SAVE_ON_IMPROVEMENT', 'true').lower() == 'true')
        self.auto_load_model = bool(os.getenv('DQN_AUTO_LOAD_MODEL', 'true').lower() == 'true')
        
        # Log the loaded configuration
        self._log_config()
    
    def _log_config(self):
        """Log the current DQN configuration."""
        logging.info("DQN Configuration loaded:")
        logging.info(f"  Gamma (discount factor): {self.gamma}")
        logging.info(f"  Epsilon (initial): {self.epsilon}")
        logging.info(f"  Epsilon min: {self.epsilon_min}")
        logging.info(f"  Epsilon decay: {self.epsilon_decay}")
        logging.info(f"  Learning rate: {self.learning_rate}")
        logging.info(f"  Batch size: {self.batch_size}")
        logging.info(f"  Memory size: {self.memory_size}")
        logging.info(f"  Target update frequency: {self.update_target_freq}")
        logging.info(f"  Hidden units: {self.hidden_units}")
        logging.info(f"  Hidden layers: {self.hidden_layers}")
        logging.info(f"  Save frequency: {self.save_frequency} training steps")
        logging.info(f"  Save on improvement: {self.save_on_improvement}")
        logging.info(f"  Auto load model: {self.auto_load_model}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'memory_size': self.memory_size,
            'update_target_freq': self.update_target_freq,
            'hidden_units': self.hidden_units,
            'hidden_layers': self.hidden_layers,
            'save_frequency': self.save_frequency,
            'save_on_improvement': self.save_on_improvement,
            'auto_load_model': self.auto_load_model
        }
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        try:
            assert 0.0 < self.gamma <= 1.0, f"Gamma must be between 0 and 1, got {self.gamma}"
            assert 0.0 <= self.epsilon <= 1.0, f"Epsilon must be between 0 and 1, got {self.epsilon}"
            assert 0.0 <= self.epsilon_min <= 1.0, f"Epsilon min must be between 0 and 1, got {self.epsilon_min}"
            assert 0.0 < self.epsilon_decay <= 1.0, f"Epsilon decay must be between 0 and 1, got {self.epsilon_decay}"
            assert self.learning_rate > 0.0, f"Learning rate must be positive, got {self.learning_rate}"
            assert self.batch_size > 0, f"Batch size must be positive, got {self.batch_size}"
            assert self.memory_size > 0, f"Memory size must be positive, got {self.memory_size}"
            assert self.update_target_freq > 0, f"Update target frequency must be positive, got {self.update_target_freq}"
            assert self.hidden_units > 0, f"Hidden units must be positive, got {self.hidden_units}"
            assert self.hidden_layers > 0, f"Hidden layers must be positive, got {self.hidden_layers}"
            assert self.epsilon_min <= self.epsilon, f"Epsilon min ({self.epsilon_min}) must be <= initial epsilon ({self.epsilon})"
            assert self.save_frequency > 0, f"Save frequency must be positive, got {self.save_frequency}"
            
            logging.info("DQN configuration validation passed")
            return True
            
        except AssertionError as e:
            logging.error(f"DQN configuration validation failed: {e}")
            return False

class SystemConfig:
    """Configuration class for system-wide settings."""
    
    def __init__(self):
        """Initialize system configuration from environment variables."""
        # Prometheus settings
        self.prometheus_url = os.getenv('PROMETHEUS_URL', 'http://prometheus.nimbusguard.svc:9090')
        self.prometheus_port = int(os.getenv('SERVER_PORT', '8080'))
        
        # LSTM settings
        self.lstm_sequence_length = int(os.getenv('LSTM_SEQUENCE_LENGTH', '20'))
        self.lstm_model_path = os.getenv('LSTM_MODEL_PATH', '/tmp/memory.keras')
        self.lstm_scaler_path = os.getenv('LSTM_SCALER_PATH', '/tmp/memory.pkl')
        
        # Scaling settings
        self.scaling_interval = int(os.getenv('SCALING_INTERVAL', '15'))  # seconds
        self.min_replicas_default = int(os.getenv('MIN_REPLICAS_DEFAULT', '1'))
        self.max_replicas_default = int(os.getenv('MAX_REPLICAS_DEFAULT', '10'))
        
        # Deployment settings
        self.target_deployment = os.getenv('TARGET_DEPLOYMENT', 'consumer')
        self.target_namespace = os.getenv('TARGET_NAMESPACE', 'default')
        
        # MinIO settings
        self.minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://minio.nimbusguard.svc:9000")
        self.minio_access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.minio_secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.minio_bucket = os.getenv("MINIO_BUCKET", "models")
        self.minio_secure = bool(os.getenv("MINIO_SECURE", "false").lower() == "true")
        
        # DQN model persistence settings
        self.dqn_model_key = os.getenv("DQN_MODEL_KEY", "dqn_model.weights.h5")
        self.dqn_metadata_key = os.getenv("DQN_METADATA_KEY", "dqn_metadata.json")
        
        # Log the loaded configuration
        self._log_config()
    
    def _log_config(self):
        """Log the current system configuration."""
        logging.info("System Configuration loaded:")
        logging.info(f"  Prometheus URL: {self.prometheus_url}")
        logging.info(f"  Prometheus port: {self.prometheus_port}")
        logging.info(f"  LSTM sequence length: {self.lstm_sequence_length}")
        logging.info(f"  LSTM model path: {self.lstm_model_path}")
        logging.info(f"  LSTM scaler path: {self.lstm_scaler_path}")
        logging.info(f"  Scaling interval: {self.scaling_interval}s")
        logging.info(f"  Target deployment: {self.target_deployment}")
        logging.info(f"  Target namespace: {self.target_namespace}")
        logging.info(f"  MinIO endpoint: {self.minio_endpoint}")
        logging.info(f"  MinIO bucket: {self.minio_bucket}")
        logging.info(f"  DQN model key: {self.dqn_model_key}")

# Global configuration instances
dqn_config = DQNConfig()
system_config = SystemConfig()

# Validate DQN configuration on import
if not dqn_config.validate():
    logging.warning("DQN configuration validation failed - using potentially invalid parameters") 