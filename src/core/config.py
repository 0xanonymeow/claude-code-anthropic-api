"""
Configuration management for the Claude Code Anthropic API compatibility server.

This module provides centralized configuration management using Pydantic Settings
for environment-based configuration loading and validation. Supports both
development and production configurations.
"""

import sys
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env files.
    
    This class handles all configuration for the server including:
    - Server configuration (host, port, debug mode)
    - Claude Code SDK configuration
    - Logging configuration
    - CORS configuration for local development
    """
    
    # Server Configuration
    host: str = Field(
        default="127.0.0.1",
        description="Host address to bind the server to"
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Port number to bind the server to"
    )
    debug: bool = Field(
        default=True,
        description="Enable debug mode"
    )
    reload: bool = Field(
        default=True,
        description="Enable auto-reload for development"
    )
    
    # Claude Code SDK Configuration
    claude_code_path: Optional[str] = Field(
        default=None,
        description="Path to Claude Code SDK executable"
    )
    claude_code_options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional options for Claude Code SDK"
    )
    claude_code_timeout: int = Field(
        default=300,
        ge=1,
        description="Timeout in seconds for Claude Code SDK requests"
    )
    claude_code_max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries for Claude Code SDK requests"
    )
    
    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    log_format: str = Field(
        default="text",
        description="Log format (json, text)"
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Path to log file (if None, logs to stdout)"
    )
    log_max_size: int = Field(
        default=10485760,  # 10MB
        ge=1024,
        description="Maximum log file size in bytes"
    )
    
    # CORS Configuration
    allow_origins: List[str] = Field(
        default=["*"],
        description="List of allowed origins for CORS (empty = no CORS)"
    )
    allow_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS requests"
    )
    allow_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="List of allowed HTTP methods for CORS"
    )
    allow_headers: List[str] = Field(
        default=["*"],
        description="List of allowed headers for CORS"
    )
    
    # API Configuration
    api_title: str = Field(
        default="Claude Code Anthropic API",
        description="API title for OpenAPI documentation"
    )
    api_description: str = Field(
        default="FastAPI server that provides Anthropic API compatibility for Claude Code SDK",
        description="API description for OpenAPI documentation"
    )
    api_version: str = Field(
        default="1.0.0",
        description="API version"
    )
    
    # Request/Response Configuration
    max_request_size: int = Field(
        default=10485760,  # 10MB
        ge=1024,
        description="Maximum request size in bytes"
    )
    request_timeout: int = Field(
        default=300,
        ge=1,
        description="Request timeout in seconds"
    )
    
    # Health Check Configuration
    health_check_enabled: bool = Field(
        default=True,
        description="Enable health check endpoint"
    )
    metrics_enabled: bool = Field(
        default=True,
        description="Enable metrics endpoint"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="",
        extra="ignore"
    )
        
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate that log_level is a valid logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper
    
    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate that log_format is either 'json' or 'text'."""
        valid_formats = ["json", "text"]
        v_lower = v.lower()
        if v_lower not in valid_formats:
            raise ValueError(f"log_format must be one of {valid_formats}")
        return v_lower
    
    @field_validator("allow_origins")
    @classmethod
    def validate_allow_origins(cls, v: List[str]) -> List[str]:
        """Validate CORS origins list."""
        # Empty list should raise error for development safety
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("CORS origins list cannot be empty")
        return v
    
    @field_validator("allow_methods")
    @classmethod
    def validate_allow_methods(cls, v: List[str]) -> List[str]:
        """Validate HTTP methods list."""
        valid_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"]
        for method in v:
            if method.upper() not in valid_methods:
                raise ValueError(f"Invalid HTTP method: {method}")
        return [method.upper() for method in v]
    
    def setup_loguru(self) -> None:
        """
        Configure loguru for beautiful, clean logging.
        """
        # Remove default handler
        logger.remove()
        
        if self.log_format == "json":
            # Simple JSON using loguru's built-in format
            logger.add(
                sys.stdout,
                format="{level.name} | {message} | {extra}",
                level=self.log_level,
                colorize=False
            )
        else:
            # Beautiful text format with colors and log levels
            def beautiful_format(record):
                level = record["level"].name
                message = record["message"]
                extra = record["extra"]
                
                # Color the log level
                level_colors = {
                    'DEBUG': '<dim>DEBUG</dim>',
                    'INFO': '<cyan>INFO</cyan>',
                    'WARNING': '<yellow>WARN</yellow>',
                    'ERROR': '<red>ERROR</red>',
                    'CRITICAL': '<magenta>CRITICAL</magenta>'
                }
                
                colored_level = level_colors.get(level, level)
                
                # Build context string with colors
                context_parts = []
                
                if 'duration_seconds' in extra and extra['duration_seconds'] is not None:
                    context_parts.append(f"<dim>{extra['duration_seconds']:.2f}s</dim>")
                
                if 'status_code' in extra and extra['status_code'] is not None:
                    status = extra['status_code']
                    if 200 <= status < 300:
                        context_parts.append(f"<green>{status}</green>")
                    elif 400 <= status < 500:
                        context_parts.append(f"<yellow>{status}</yellow>")
                    else:
                        context_parts.append(f"<red>{status}</red>")
                
                if 'method' in extra and 'path' in extra:
                    context_parts.append(f"<bold>{extra['method']}</bold> {extra['path']}")
                
                context_str = ""
                if context_parts:
                    context_str = f" <dim>({' · '.join(context_parts)})</dim>"
                
                return f"{colored_level:>8} {message}{context_str}\n"
            
            # Add console handler with the custom format
            logger.add(
                sys.stdout,
                format=beautiful_format,
                level=self.log_level,
                colorize=True
            )
        
        # Add file handler if specified
        if self.log_file:
            logger.add(
                self.log_file,
                format="{time:HH:mm:ss} | {level} | {message} | {extra}",
                level=self.log_level,
                rotation="10 MB",
                retention="7 days",
                compression="gz"
            )
    
    def get_cors_config(self) -> Dict[str, Any]:
        """
        Get CORS configuration dictionary.
        
        Returns:
            Dict containing CORS configuration for FastAPI
        """
        return {
            "allow_origins": self.allow_origins,
            "allow_credentials": self.allow_credentials,
            "allow_methods": self.allow_methods,
            "allow_headers": self.allow_headers
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration dictionary.
        
        Returns:
            Dict containing logging configuration
        """
        return {
            "level": self.log_level,
            "format": self.log_format,
            "file": self.log_file,
            "max_size": self.log_max_size
        }
    
    def get_claude_code_config(self) -> Dict[str, Any]:
        """
        Get Claude Code SDK configuration dictionary.
        
        Returns:
            Dict containing Claude Code SDK configuration
        """
        config = {
            "timeout": self.claude_code_timeout,
            "max_retries": self.claude_code_max_retries,
            **self.claude_code_options
        }
        
        if self.claude_code_path:
            config["path"] = self.claude_code_path
            
        return config


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    This function implements a singleton pattern to ensure that settings
    are loaded only once and reused throughout the application.
    
    Returns:
        Settings: The global settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """
    Reload settings from environment variables and .env files.
    
    This function forces a reload of the settings, useful for testing
    or when configuration changes need to be picked up.
    
    Returns:
        Settings: The newly loaded settings instance
    """
    global _settings
    _settings = Settings()
    return _settings


def configure_logging(settings: Optional[Settings] = None) -> None:
    """
    Configure loguru for beautiful logging.
    
    Args:
        settings: Settings instance to use for configuration.
                 If None, uses the global settings instance.
    """
    if settings is None:
        settings = get_settings()
    
    settings.setup_loguru()
    
    # Completely silence third-party loggers for clean output
    import logging

    # Set all third-party loggers to CRITICAL to silence them
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "fastapi", "httpx", "multipart"]:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)
        logging.getLogger(logger_name).disabled = True