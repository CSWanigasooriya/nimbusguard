import json
import logging
import tempfile
import os
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from minio import Minio
from minio.error import S3Error
from config import system_config

class DQNModelStorage:
    """Handles saving and loading DQN models to/from MinIO storage."""
    
    def __init__(self):
        """Initialize MinIO client with configuration."""
        self.client = Minio(
            system_config.minio_endpoint.replace('http://', '').replace('https://', ''),
            access_key=system_config.minio_access_key,
            secret_key=system_config.minio_secret_key,
            secure=system_config.minio_secure
        )
        self.bucket_name = system_config.minio_bucket
        self.model_key = system_config.dqn_model_key
        self.metadata_key = system_config.dqn_metadata_key
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
        
        logging.info(f"DQN Model Storage initialized - Bucket: {self.bucket_name}")
    
    def _ensure_bucket_exists(self):
        """Ensure the models bucket exists."""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logging.info(f"Created MinIO bucket: {self.bucket_name}")
            else:
                logging.debug(f"MinIO bucket exists: {self.bucket_name}")
        except S3Error as e:
            logging.error(f"Failed to create/check MinIO bucket: {e}")
            raise
    
    def model_exists(self) -> bool:
        """Check if a DQN model exists in storage."""
        try:
            self.client.stat_object(self.bucket_name, self.model_key)
            return True
        except S3Error:
            return False
    
    def save_model(self, agent, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save DQN model weights and metadata to MinIO.
        
        Args:
            agent: DQN agent instance
            metadata: Additional metadata to save with the model
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Create temporary files for weights and metadata
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_weights:
                weights_path = temp_weights.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_metadata:
                metadata_path = temp_metadata.name
            
            try:
                # Save model weights to temporary file
                agent.model.save_weights(weights_path)
                logging.debug(f"Saved model weights to temporary file: {weights_path}")
                
                # Prepare metadata
                model_metadata = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'training_steps': getattr(agent, 'training_steps', 0),
                    'epsilon': agent.epsilon,
                    'config': agent.get_config(),
                    'model_architecture': {
                        'state_size': agent.state_size,
                        'action_size': agent.action_size,
                        'hidden_units': agent.hidden_units,
                        'hidden_layers': agent.hidden_layers
                    }
                }
                
                # Add user metadata if provided
                if metadata:
                    model_metadata.update(metadata)
                
                # Save metadata to temporary file
                with open(metadata_path, 'w') as f:
                    json.dump(model_metadata, f, indent=2)
                
                logging.debug(f"Saved metadata to temporary file: {metadata_path}")
                
                # Upload weights to MinIO
                self.client.fput_object(
                    self.bucket_name,
                    self.model_key,
                    weights_path,
                    content_type='application/octet-stream'
                )
                
                # Upload metadata to MinIO
                self.client.fput_object(
                    self.bucket_name,
                    self.metadata_key,
                    metadata_path,
                    content_type='application/json'
                )
                
                logging.info(f"DQN model saved to MinIO: {self.model_key}")
                logging.info(f"   Training steps: {model_metadata['training_steps']}")
                logging.info(f"   Epsilon: {model_metadata['epsilon']:.4f}")
                logging.info(f"   Timestamp: {model_metadata['timestamp']}")
                
                return True
                
            finally:
                # Clean up temporary files
                try:
                    os.unlink(weights_path)
                    os.unlink(metadata_path)
                except:
                    pass
                
        except Exception as e:
            logging.error(f"Failed to save DQN model: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    def load_model(self, agent) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Load DQN model weights and metadata from MinIO.
        
        Args:
            agent: DQN agent instance to load weights into
            
        Returns:
            Tuple of (success, metadata) where success is bool and metadata is dict or None
        """
        try:
            if not self.model_exists():
                logging.info("No existing DQN model found in storage")
                return False, None
            
            # Create temporary files for download
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_weights:
                weights_path = temp_weights.name
            
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_metadata:
                metadata_path = temp_metadata.name
            
            try:
                # Download weights from MinIO
                self.client.fget_object(
                    self.bucket_name,
                    self.model_key,
                    weights_path
                )
                
                # Download metadata from MinIO (if exists)
                metadata = None
                try:
                    self.client.fget_object(
                        self.bucket_name,
                        self.metadata_key,
                        metadata_path
                    )
                    
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        
                except S3Error:
                    logging.warning("No metadata found for DQN model")
                
                # Load weights into agent
                agent.model.load_weights(weights_path)
                agent.update_target_model()  # Update target model with loaded weights
                
                # Restore epsilon if available in metadata
                if metadata and 'epsilon' in metadata:
                    agent.epsilon = metadata['epsilon']
                    logging.info(f"Restored epsilon: {agent.epsilon:.4f}")
                
                # Restore training steps if available
                if metadata and 'training_steps' in metadata:
                    agent.training_steps = metadata['training_steps']
                    logging.info(f"Restored training steps: {agent.training_steps}")
                
                logging.info(f"DQN model loaded from MinIO: {self.model_key}")
                if metadata:
                    logging.info(f"   Saved at: {metadata.get('timestamp', 'unknown')}")
                    logging.info(f"   Training steps: {metadata.get('training_steps', 0)}")
                
                return True, metadata
                
            finally:
                # Clean up temporary files
                try:
                    os.unlink(weights_path)
                    os.unlink(metadata_path)
                except:
                    pass
                
        except Exception as e:
            logging.error(f"Failed to load DQN model: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False, None
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the stored model without loading it.
        
        Returns:
            Dictionary with model information or None if no model exists
        """
        try:
            if not self.model_exists():
                return None
            
            # Try to get metadata
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_metadata:
                metadata_path = temp_metadata.name
            
            try:
                self.client.fget_object(
                    self.bucket_name,
                    self.metadata_key,
                    metadata_path
                )
                
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                return metadata
                
            except S3Error:
                # If no metadata, just return basic info
                model_stat = self.client.stat_object(self.bucket_name, self.model_key)
                return {
                    'size': model_stat.size,
                    'last_modified': model_stat.last_modified.isoformat(),
                    'metadata': 'unavailable'
                }
            finally:
                try:
                    os.unlink(metadata_path)
                except:
                    pass
                
        except Exception as e:
            logging.error(f"Failed to get model info: {e}")
            return None
    
    def delete_model(self) -> bool:
        """
        Delete the stored DQN model and metadata.
        
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            if not self.model_exists():
                logging.info("No DQN model to delete")
                return True
            
            # Delete model weights
            self.client.remove_object(self.bucket_name, self.model_key)
            
            # Delete metadata (if exists)
            try:
                self.client.remove_object(self.bucket_name, self.metadata_key)
            except S3Error:
                pass  # Metadata might not exist
            
            logging.info(f"DQN model deleted from MinIO: {self.model_key}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to delete DQN model: {e}")
            return False
    
    def get_storage_status(self) -> Dict[str, Any]:
        """
        Get storage connectivity and status information.
        
        Returns:
            Dictionary with storage status
        """
        try:
            # Test MinIO connectivity
            bucket_exists = self.client.bucket_exists(self.bucket_name)
            model_exists = self.model_exists()
            
            status = {
                'connected': True,
                'bucket_exists': bucket_exists,
                'model_exists': model_exists,
                'endpoint': system_config.minio_endpoint,
                'bucket': self.bucket_name,
                'model_key': self.model_key
            }
            
            if model_exists:
                model_info = self.get_model_info()
                if model_info:
                    status['model_info'] = model_info
            
            return status
            
        except Exception as e:
            return {
                'connected': False,
                'error': str(e),
                'endpoint': system_config.minio_endpoint,
                'bucket': self.bucket_name
            } 