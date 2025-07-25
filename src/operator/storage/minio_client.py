"""
MinIO client for DQN model persistence.
"""

import logging
import tempfile
import os
from typing import Optional
from urllib.parse import urlparse
from minio import Minio
from minio.error import S3Error

from config.settings import load_config

logger = logging.getLogger(__name__)


class MinIOModelStorage:
    """
    MinIO client for saving and loading DQN models.
    """
    
    def __init__(self):
        self.config = load_config()
        self.client: Optional[Minio] = None
        self.bucket_name = "models"
        self.model_path = "dqn_model.keras"
        
    async def initialize(self):
        """Initialize MinIO client and ensure bucket exists."""
        try:
            # Parse MinIO endpoint
            minio_url = urlparse(self.config.minio.endpoint).netloc
            
            # Create MinIO client
            self.client = Minio(
                minio_url,
                access_key=self.config.minio.access_key,
                secret_key=self.config.minio.secret_key,
                secure=False  # Assuming internal cluster communication
            )
            
            # Ensure bucket exists
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"📦 Created MinIO bucket: {self.bucket_name}")
            
            logger.info("✅ MinIO storage initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ MinIO initialization failed: {e}")
            self.client = None
            return False
    
    def model_exists(self) -> bool:
        """Check if DQN model exists in MinIO."""
        if not self.client:
            return False
            
        try:
            self.client.stat_object(self.bucket_name, self.model_path)
            return True
        except S3Error:
            return False
        except Exception as e:
            logger.error(f"Error checking model existence: {e}")
            return False
    
    def save_model(self, local_model_path: str) -> bool:
        """
        Save DQN model to MinIO.
        
        Args:
            local_model_path: Path to the local model file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            logger.error("MinIO client not initialized")
            return False
            
        try:
            # Upload model to MinIO
            self.client.fput_object(
                self.bucket_name,
                self.model_path,
                local_model_path
            )
            
            logger.info(f"💾 DQN model saved to MinIO: {self.bucket_name}/{self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model to MinIO: {e}")
            return False
    
    def load_model(self, local_model_path: str) -> bool:
        """
        Load DQN model from MinIO.
        
        Args:
            local_model_path: Path where to save the downloaded model
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            logger.error("MinIO client not initialized")
            return False
            
        if not self.model_exists():
            logger.info("No existing DQN model found in MinIO")
            return False
            
        try:
            # Download model from MinIO
            self.client.fget_object(
                self.bucket_name,
                self.model_path,
                local_model_path
            )
            
            logger.info(f"📥 DQN model loaded from MinIO: {self.bucket_name}/{self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model from MinIO: {e}")
            return False
    
    def get_model_info(self) -> dict:
        """Get information about the stored model."""
        if not self.client or not self.model_exists():
            return {}
            
        try:
            stat = self.client.stat_object(self.bucket_name, self.model_path)
            return {
                "size": stat.size,
                "last_modified": stat.last_modified,
                "etag": stat.etag
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}


# Global instance
minio_storage = MinIOModelStorage() 