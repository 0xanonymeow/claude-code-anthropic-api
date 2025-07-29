"""
Unit tests for the configuration management system.

Tests cover configuration loading, validation, environment variable handling,
and all configuration methods.
"""

import os
import tempfile
from typing import Any, Dict
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.core.config import Settings, configure_logging, get_settings, reload_settings


class TestSettings:
    """Test cases for the Settings class."""

    def test_default_settings(self) -> None:
        """Test that default settings are loaded correctly."""
        settings = Settings()

        # Server configuration defaults
        assert settings.host == "127.0.0.1"
        assert settings.port == 8000
        assert settings.debug is True
        assert settings.reload is True

        # Claude Code SDK defaults
        assert settings.claude_code_path is None
        assert settings.claude_code_options == {}
        assert settings.claude_code_timeout == 300
        assert settings.claude_code_max_retries == 3

        # Logging defaults
        assert settings.log_level == "INFO"
        assert settings.log_format == "json"
        assert settings.log_file is None
        assert settings.log_max_size == 10485760
        assert settings.log_backup_count == 5

        # CORS defaults
        assert settings.allow_origins == ["*"]
        assert settings.allow_credentials is True
        assert settings.allow_methods == ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        assert settings.allow_headers == ["*"]

        # API defaults
        assert settings.api_title == "Claude Code Anthropic API"
        assert "FastAPI server" in settings.api_description
        assert settings.api_version == "1.0.0"

        # Request/Response defaults
        assert settings.max_request_size == 10485760
        assert settings.request_timeout == 300

        # Health check defaults
        assert settings.health_check_enabled is True
        assert settings.metrics_enabled is True

    def test_environment_variable_override(self) -> None:
        """Test that environment variables override default values."""
        env_vars = {
            "HOST": "0.0.0.0",
            "PORT": "9000",
            "DEBUG": "false",
            "LOG_LEVEL": "DEBUG",
            "LOG_FORMAT": "text",
            "CLAUDE_CODE_TIMEOUT": "600",
            "CLAUDE_CODE_MAX_RETRIES": "5",
        }

        with patch.dict(os.environ, env_vars):
            settings = Settings()

            assert settings.host == "0.0.0.0"
            assert settings.port == 9000
            assert settings.debug is False
            assert settings.log_level == "DEBUG"
            assert settings.log_format == "text"
            assert settings.claude_code_timeout == 600
            assert settings.claude_code_max_retries == 5

    def test_port_validation(self) -> None:
        """Test port number validation."""
        # Valid port
        settings = Settings(port=8080)
        assert settings.port == 8080

        # Invalid ports
        with pytest.raises(ValidationError):
            Settings(port=0)

        with pytest.raises(ValidationError):
            Settings(port=65536)

        with pytest.raises(ValidationError):
            Settings(port=-1)

    def test_log_level_validation(self) -> None:
        """Test log level validation."""
        # Valid log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            settings = Settings(log_level=level)
            assert settings.log_level == level

        # Case insensitive
        settings = Settings(log_level="debug")
        assert settings.log_level == "DEBUG"

        # Invalid log level
        with pytest.raises(ValidationError):
            Settings(log_level="INVALID")

    def test_log_format_validation(self) -> None:
        """Test log format validation."""
        # Valid formats
        settings = Settings(log_format="json")
        assert settings.log_format == "json"

        settings = Settings(log_format="text")
        assert settings.log_format == "text"

        # Case insensitive
        settings = Settings(log_format="JSON")
        assert settings.log_format == "json"

        # Invalid format
        with pytest.raises(ValidationError):
            Settings(log_format="xml")

    def test_allow_origins_validation(self) -> None:
        """Test CORS origins validation."""
        # Valid origins
        settings = Settings(
            allow_origins=["http://localhost:3000", "https://example.com"]
        )
        assert settings.allow_origins == [
            "http://localhost:3000",
            "https://example.com",
        ]

        # Empty list should raise error
        with pytest.raises(ValidationError):
            Settings(allow_origins=[])

    def test_allow_methods_validation(self) -> None:
        """Test HTTP methods validation."""
        # Valid methods
        settings = Settings(allow_methods=["GET", "POST"])
        assert settings.allow_methods == ["GET", "POST"]

        # Case insensitive
        settings = Settings(allow_methods=["get", "post"])
        assert settings.allow_methods == ["GET", "POST"]

        # Invalid method
        with pytest.raises(ValidationError):
            Settings(allow_methods=["INVALID"])

    def test_timeout_validation(self) -> None:
        """Test timeout validation."""
        # Valid timeouts
        settings = Settings(claude_code_timeout=60, request_timeout=120)
        assert settings.claude_code_timeout == 60
        assert settings.request_timeout == 120

        # Invalid timeouts (must be >= 1)
        with pytest.raises(ValidationError):
            Settings(claude_code_timeout=0)

        with pytest.raises(ValidationError):
            Settings(request_timeout=-1)

    def test_size_validation(self) -> None:
        """Test size validation for log and request sizes."""
        # Valid sizes
        settings = Settings(log_max_size=1024, max_request_size=2048)
        assert settings.log_max_size == 1024
        assert settings.max_request_size == 2048

        # Invalid sizes (must be >= 1024)
        with pytest.raises(ValidationError):
            Settings(log_max_size=512)

        with pytest.raises(ValidationError):
            Settings(max_request_size=100)


class TestSettingsMethods:
    """Test cases for Settings methods."""

    def test_get_logging_config_json_format(self) -> None:
        """Test logging configuration generation for JSON format."""
        settings = Settings(log_level="DEBUG", log_format="json")
        config = settings.get_logging_config()

        assert config["version"] == 1
        assert config["disable_existing_loggers"] is False
        assert "json" in config["formatters"]
        assert "text" in config["formatters"]
        assert config["root"]["level"] == "DEBUG"
        assert "console" in config["handlers"]

        # Check JSON formatter
        json_formatter = config["formatters"]["json"]
        assert "timestamp" in json_formatter["format"]
        assert "level" in json_formatter["format"]
        assert "message" in json_formatter["format"]

    def test_get_logging_config_text_format(self) -> None:
        """Test logging configuration generation for text format."""
        settings = Settings(log_level="WARNING", log_format="text")
        config = settings.get_logging_config()

        assert config["root"]["level"] == "WARNING"

        # Check text formatter
        text_formatter = config["formatters"]["text"]
        assert "%(asctime)s" in text_formatter["format"]
        assert "%(levelname)s" in text_formatter["format"]
        assert "%(message)s" in text_formatter["format"]

    def test_get_logging_config_with_file(self) -> None:
        """Test logging configuration with file handler."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            log_file = tmp_file.name

        try:
            settings = Settings(
                log_file=log_file, log_max_size=1024, log_backup_count=3
            )
            config = settings.get_logging_config()

            assert "file" in config["handlers"]
            file_handler = config["handlers"]["file"]
            assert file_handler["class"] == "logging.handlers.RotatingFileHandler"
            assert file_handler["filename"] == log_file
            assert file_handler["maxBytes"] == 1024
            assert file_handler["backupCount"] == 3

            # Check that file handler is added to loggers
            assert "file" in config["root"]["handlers"]
            for logger_config in config["loggers"].values():
                assert "file" in logger_config["handlers"]
        finally:
            os.unlink(log_file)

    def test_get_cors_config(self) -> None:
        """Test CORS configuration generation."""
        settings = Settings(
            allow_origins=["http://localhost:3000"],
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

        cors_config = settings.get_cors_config()

        assert cors_config["allow_origins"] == ["http://localhost:3000"]
        assert cors_config["allow_credentials"] is False
        assert cors_config["allow_methods"] == ["GET", "POST"]
        assert cors_config["allow_headers"] == ["Content-Type"]

    def test_get_claude_code_config_minimal(self) -> None:
        """Test Claude Code SDK configuration with minimal settings."""
        settings = Settings()
        config = settings.get_claude_code_config()

        assert config["timeout"] == 300
        assert config["max_retries"] == 3
        assert "path" not in config

    def test_get_claude_code_config_full(self) -> None:
        """Test Claude Code SDK configuration with all settings."""
        settings = Settings(
            claude_code_path="/usr/local/bin/claude-code",
            claude_code_timeout=600,
            claude_code_max_retries=5,
            claude_code_options={"model": "claude-sonnet-4", "temperature": 0.7},
        )

        config = settings.get_claude_code_config()

        assert config["path"] == "/usr/local/bin/claude-code"
        assert config["timeout"] == 600
        assert config["max_retries"] == 5
        assert config["model"] == "claude-sonnet-4"
        assert config["temperature"] == 0.7


class TestGlobalSettings:
    """Test cases for global settings functions."""

    def test_get_settings_singleton(self) -> None:
        """Test that get_settings returns the same instance."""
        # Clear any existing settings
        import src.core.config

        src.core.config._settings = None

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_reload_settings(self) -> None:
        """Test that reload_settings creates a new instance."""
        # Clear any existing settings
        import src.core.config

        src.core.config._settings = None

        settings1 = get_settings()
        settings2 = reload_settings()

        # Should be different instances but same values
        assert settings1 is not settings2
        assert settings1.host == settings2.host
        assert settings1.port == settings2.port

    def test_reload_settings_with_env_change(self) -> None:
        """Test that reload_settings picks up environment changes."""
        # Clear any existing settings
        import src.core.config

        src.core.config._settings = None

        # Get initial settings
        settings1 = get_settings()
        original_port = settings1.port

        # Change environment and reload
        with patch.dict(os.environ, {"PORT": "9999"}):
            settings2 = reload_settings()
            assert settings2.port == 9999
            assert settings2.port != original_port


class TestConfigureLogging:
    """Test cases for logging configuration."""

    @patch("logging.config.dictConfig")
    @patch("logging.getLogger")
    def test_configure_logging_default(self, mock_get_logger, mock_dict_config) -> None:
        """Test logging configuration with default settings."""
        mock_logger = mock_get_logger.return_value

        configure_logging()

        # Should call dictConfig with logging configuration
        mock_dict_config.assert_called_once()
        config_arg = mock_dict_config.call_args[0][0]
        assert config_arg["version"] == 1
        assert "formatters" in config_arg
        assert "handlers" in config_arg

        # Should log configuration info
        mock_logger.info.assert_called()

    @patch("logging.config.dictConfig")
    @patch("logging.getLogger")
    def test_configure_logging_custom_settings(
        self, mock_get_logger, mock_dict_config
    ) -> None:
        """Test logging configuration with custom settings."""
        mock_logger = mock_get_logger.return_value

        settings = Settings(log_level="DEBUG", log_format="text")
        configure_logging(settings)

        mock_dict_config.assert_called_once()
        config_arg = mock_dict_config.call_args[0][0]
        assert config_arg["root"]["level"] == "DEBUG"

        # Check that info messages were logged
        assert mock_logger.info.call_count >= 2


class TestDotEnvFile:
    """Test cases for .env file loading."""

    def test_env_file_loading(self) -> None:
        """Test that .env file is loaded correctly."""
        env_content = """HOST=192.168.1.100
PORT=7000
DEBUG=false
LOG_LEVEL=ERROR
CLAUDE_CODE_TIMEOUT=900
"""

        # Create a temporary file and close it immediately to avoid Windows permission issues
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False
        ) as tmp_file:
            tmp_file.write(env_content)
            tmp_file_name = tmp_file.name

        try:
            # Create settings with custom env file
            settings = Settings(_env_file=tmp_file_name)

            assert settings.host == "192.168.1.100"
            assert settings.port == 7000
            assert settings.debug is False
            assert settings.log_level == "ERROR"
            assert settings.claude_code_timeout == 900
        finally:
            try:
                os.unlink(tmp_file_name)
            except (OSError, PermissionError):
                # Ignore cleanup errors on Windows
                pass


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear the global settings cache before each test."""
    import src.core.config

    src.core.config._settings = None
    yield
    src.core.config._settings = None
