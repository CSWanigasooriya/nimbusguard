"""
Infrastructure configuration for external services.
"""

from typing import Optional
from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings


class PrometheusConfig(BaseSettings):
    """Prometheus configuration."""
    url: str = Field(default="http://prometheus.nimbusguard.svc:9090", env="PROMETHEUS_URL")
    timeout: int = Field(default=30, env="PROMETHEUS_TIMEOUT")
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Prometheus URL must start with http:// or https://')
        return v
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


class MinIOConfig(BaseSettings):
    """MinIO configuration."""
    endpoint: str = Field(default="http://minio.nimbusguard.svc:9000", env="MINIO_ENDPOINT")
    access_key: str = Field(default="minioadmin", env="MINIO_ACCESS_KEY")
    secret_key: str = Field(default="minioadmin", env="MINIO_SECRET_KEY")
    bucket_name: str = Field(default="models", env="BUCKET_NAME")
    scaler_name: str = Field(default="feature_scaler.gz", env="SCALER_NAME")
    scaler_path: str = Field(default="/app/feature_scaler.gz", env="SCALER_PATH")
    
    @field_validator('endpoint')
    @classmethod
    def validate_endpoint(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('MinIO endpoint must start with http:// or https://')
        return v
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


class RedisConfig(BaseSettings):
    """Redis configuration."""
    url: str = Field(default="redis://redis:6379", env="REDIS_URL")
    replay_buffer_key: str = Field(default="replay_buffer", env="REPLAY_BUFFER_KEY")
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        if not v.startswith('redis://'):
            raise ValueError('Redis URL must start with redis://')
        return v
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


class KubernetesConfig(BaseSettings):
    """Kubernetes configuration."""
    target_deployment: str = Field(default="consumer", env="TARGET_DEPLOYMENT")
    target_namespace: str = Field(default="nimbusguard", env="TARGET_NAMESPACE")
    stabilization_period_seconds: int = Field(default=30, env="STABILIZATION_PERIOD_SECONDS")
    
    @field_validator('stabilization_period_seconds')
    @classmethod
    def validate_stabilization_period(cls, v):
        if v < 5 or v > 300:
            raise ValueError('Stabilization period must be between 5 and 300 seconds')
        return v
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


class ServerConfig(BaseSettings):
    """Server configuration."""
    port: int = Field(default=8080, env="SERVER_PORT")
    host: str = Field(default="0.0.0.0", env="SERVER_HOST")
    kopf_health_port: int = Field(default=8081, env="KOPF_HEALTH_PORT")
    
    @field_validator('port', 'kopf_health_port')
    @classmethod
    def validate_port(cls, v):
        if v < 1024 or v > 65535:
            raise ValueError('Port must be between 1024 and 65535')
        return v
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    ) 