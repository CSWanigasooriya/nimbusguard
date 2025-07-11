"""
Infrastructure configuration for external services.
"""

from typing import Optional
from pydantic import ConfigDict, Field, field_validator, AliasChoices
from pydantic_settings import BaseSettings


class PrometheusConfig(BaseSettings):
    """Prometheus configuration."""
    url: str = Field(default="http://prometheus.nimbusguard.svc:9090", validation_alias=AliasChoices("PROMETHEUS_URL", "url"))
    timeout: int = Field(default=30, validation_alias=AliasChoices("PROMETHEUS_TIMEOUT", "timeout"))
    
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
    endpoint: str = Field(default="http://minio.nimbusguard.svc:9000", validation_alias=AliasChoices("MINIO_ENDPOINT", "endpoint"))
    access_key: str = Field(default="minioadmin", validation_alias=AliasChoices("MINIO_ACCESS_KEY", "access_key"))
    secret_key: str = Field(default="minioadmin", validation_alias=AliasChoices("MINIO_SECRET_KEY", "secret_key"))
    bucket_name: str = Field(default="models", validation_alias=AliasChoices("BUCKET_NAME", "bucket_name"))
    scaler_name: str = Field(default="feature_scaler.gz", validation_alias=AliasChoices("SCALER_NAME", "scaler_name"))
    scaler_path: str = Field(default="/app/feature_scaler.gz", validation_alias=AliasChoices("SCALER_PATH", "scaler_path"))
    
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
    url: str = Field(default="redis://redis:6379", validation_alias=AliasChoices("REDIS_URL", "redis_url"))
    replay_buffer_key: str = Field(default="replay_buffer", validation_alias=AliasChoices("REPLAY_BUFFER_KEY", "replay_buffer_key"))
    
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
    target_deployment: str = Field(default="consumer", validation_alias=AliasChoices("TARGET_DEPLOYMENT", "target_deployment"))
    target_namespace: str = Field(default="nimbusguard", validation_alias=AliasChoices("TARGET_NAMESPACE", "target_namespace"))
    stabilization_period_seconds: int = Field(default=30, validation_alias=AliasChoices("STABILIZATION_PERIOD_SECONDS", "stabilization_period_seconds"))
    
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
    port: int = Field(default=8080, validation_alias=AliasChoices("SERVER_PORT", "port"))
    host: str = Field(default="0.0.0.0", validation_alias=AliasChoices("SERVER_HOST", "host"))
    kopf_health_port: int = Field(default=8081, validation_alias=AliasChoices("KOPF_HEALTH_PORT", "kopf_health_port"))
    
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