"""
Configuration management for MaPle application
"""

from typing import Optional, List
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from pydantic import field_validator
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # Application
    APP_NAME: str = "MaPle Clustering API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # API
    API_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS - use str and parse manually to avoid JSON parsing issues
    ALLOW_ORIGINS: str = "*"
    ALLOW_CREDENTIALS: bool = True
    ALLOW_METHODS: str = "*"
    ALLOW_HEADERS: str = "*"
    
    # Data
    MAX_UPLOAD_SIZE_MB: int = 50
    SUPPORTED_FILE_TYPES: str = ".csv"
    
    # Clustering defaults
    DEFAULT_MIN_SUPPORT: float = 0.05
    DEFAULT_MAX_PATTERN_LENGTH: int = 10
    DEFAULT_N_BINS: int = 5
    DEFAULT_DISCRETIZATION_STRATEGY: str = "quantile"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    @property
    def allow_origins_list(self) -> List[str]:
        """Get ALLOW_ORIGINS as a list."""
        if self.ALLOW_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.ALLOW_ORIGINS.split(",")]
    
    @property
    def allow_methods_list(self) -> List[str]:
        """Get ALLOW_METHODS as a list."""
        if self.ALLOW_METHODS == "*":
            return ["*"]
        return [method.strip() for method in self.ALLOW_METHODS.split(",")]
    
    @property
    def allow_headers_list(self) -> List[str]:
        """Get ALLOW_HEADERS as a list."""
        if self.ALLOW_HEADERS == "*":
            return ["*"]
        return [header.strip() for header in self.ALLOW_HEADERS.split(",")]

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()
