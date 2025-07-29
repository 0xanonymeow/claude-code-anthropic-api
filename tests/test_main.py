"""
Tests for the main FastAPI application.

This module tests the main application setup, middleware, exception handlers,
and core endpoints to ensure proper functionality.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.main import app
from src.models.anthropic import AnthropicError, ErrorResponse, ErrorType


class TestMainApplication:
    """Test the main FastAPI application setup and configuration."""

    def test_app_creation(self):
        """Test that the FastAPI app is created successfully."""
        assert app is not None
        assert app.title == "Claude Code Anthropic API"
        assert app.version == "1.0.0"

    def test_cors_middleware_configured(self):
        """Test that CORS middleware is properly configured."""
        # Check that CORS middleware is in the middleware stack
        from starlette.middleware.cors import CORSMiddleware

        middleware_classes = [middleware.cls for middleware in app.user_middleware]
        assert CORSMiddleware in middleware_classes


class TestRootEndpoint:
    """Test the root endpoint functionality."""

    def test_root_endpoint(self):
        """Test the root endpoint returns correct information."""
        with TestClient(app) as client:
            response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert data["name"] == "Claude Code Anthropic API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
        assert "endpoints" in data
        assert data["endpoints"]["messages"] == "/v1/messages"
        assert data["endpoints"]["models"] == "/v1/models"
        assert data["endpoints"]["health"] == "/health"
        assert data["endpoints"]["metrics"] == "/metrics"


class TestHealthEndpoint:
    """Test the health check endpoint."""

    @patch("src.main.get_claude_client")
    def test_health_endpoint_healthy(self, mock_get_client):
        """Test health endpoint when Claude client is healthy."""
        # Mock Claude client health check
        mock_client = AsyncMock()
        mock_client.health_check.return_value = {
            "status": "healthy",
            "claude_code_sdk": "connected",
            "models_available": 5,
        }
        mock_get_client.return_value = mock_client

        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "1.0.0"
        assert "components" in data
        assert data["components"]["claude_code_sdk"]["status"] == "healthy"
        assert data["components"]["api_server"]["status"] == "healthy"

    @patch("src.main.get_claude_client")
    def test_health_endpoint_unhealthy(self, mock_get_client):
        """Test health endpoint when Claude client is unhealthy."""
        # Mock Claude client health check failure
        mock_client = AsyncMock()
        mock_client.health_check.return_value = {
            "status": "unhealthy",
            "error": "Connection failed",
        }
        mock_get_client.return_value = mock_client

        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 503
        data = response.json()

        assert data["status"] == "unhealthy"
        assert "components" in data
        assert data["components"]["claude_code_sdk"]["status"] == "unhealthy"

    @patch("src.main.get_claude_client")
    def test_health_endpoint_exception(self, mock_get_client):
        """Test health endpoint when an exception occurs."""
        # Mock Claude client to raise exception
        mock_client = AsyncMock()
        mock_client.health_check.side_effect = Exception("Connection error")
        mock_get_client.return_value = mock_client

        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 503
        data = response.json()

        assert data["status"] == "unhealthy"
        assert "error" in data
        assert "Connection error" in data["error"]


class TestMetricsEndpoint:
    """Test the metrics endpoint."""

    @patch("src.main.get_claude_client")
    def test_metrics_endpoint_success(self, mock_get_client):
        """Test metrics endpoint returns correct data."""
        # Mock Claude client health check
        mock_client = AsyncMock()
        mock_client.health_check.return_value = {
            "status": "healthy",
            "models_available": 5,
        }
        mock_get_client.return_value = mock_client

        with TestClient(app) as client:
            response = client.get("/metrics")

        assert response.status_code == 200
        data = response.json()

        assert "timestamp" in data
        assert data["version"] == "1.0.0"
        assert "claude_code_sdk" in data
        assert data["claude_code_sdk"]["status"] == "healthy"
        assert data["claude_code_sdk"]["models_available"] == 5
        assert "server" in data
        assert data["server"]["debug_mode"] is True

    @patch("src.main.get_claude_client")
    def test_metrics_endpoint_error(self, mock_get_client):
        """Test metrics endpoint when an error occurs."""
        # Mock Claude client to raise exception
        mock_client = AsyncMock()
        mock_client.health_check.side_effect = Exception("Metrics error")
        mock_get_client.return_value = mock_client

        with TestClient(app) as client:
            response = client.get("/metrics")

        assert response.status_code == 503
        data = response.json()

        assert "error" in data
        assert "Metrics error" in data["error"]
        assert data["claude_code_sdk"]["status"] == "unhealthy"


class TestExceptionHandlers:
    """Test the custom exception handlers."""

    def test_validation_error_handler(self):
        """Test that validation errors are handled properly."""
        with TestClient(app) as client:
            # Send invalid request to trigger validation error
            response = client.post(
                "/v1/messages",
                json={
                    "model": "invalid-model",
                    "messages": [],  # Empty messages should trigger validation error
                    "max_tokens": -1,  # Invalid max_tokens
                },
            )

        assert response.status_code == 422  # FastAPI validation errors use 422
        data = response.json()

        assert data["type"] == "error"
        assert data["error"]["type"] == "invalid_request_error"
        assert "validation failed" in data["error"]["message"].lower()

    def test_http_exception_handler(self):
        """Test that HTTP exceptions are handled properly."""
        with TestClient(app) as client:
            # Request non-existent endpoint
            response = client.get("/nonexistent")

        assert response.status_code == 404

    @patch("src.main.get_claude_client")
    def test_general_exception_handler(self, mock_get_client):
        """Test that unexpected exceptions are handled properly."""
        # Mock Claude client to raise unexpected exception
        mock_client = AsyncMock()
        mock_client.health_check.side_effect = RuntimeError("Unexpected error")
        mock_get_client.return_value = mock_client

        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 503
        data = response.json()

        assert data["status"] == "unhealthy"
        assert "error" in data


class TestMiddleware:
    """Test middleware functionality."""

    def test_logging_middleware_adds_process_time(self):
        """Test that logging middleware adds process time header."""
        with TestClient(app) as client:
            response = client.get("/")

        assert response.status_code == 200
        assert "X-Process-Time" in response.headers

        # Process time should be a valid float
        process_time = float(response.headers["X-Process-Time"])
        assert process_time >= 0


class TestApplicationLifespan:
    """Test application lifespan events."""

    @patch("src.main.get_claude_client")
    @patch("src.main.close_claude_client")
    def test_lifespan_startup_and_shutdown(self, mock_close_client, mock_get_client):
        """Test that lifespan events are handled properly."""
        # Mock Claude client
        mock_client = AsyncMock()
        mock_client.health_check.return_value = {"status": "healthy"}
        mock_get_client.return_value = mock_client

        # Test that the app can be created and used
        with TestClient(app) as client:
            response = client.get("/")
            assert response.status_code == 200

        # Verify that client initialization was attempted
        mock_get_client.assert_called()


class TestConfigurationIntegration:
    """Test integration with configuration system."""

    def test_app_uses_settings(self):
        """Test that the app uses configuration from settings."""
        # The app should be configured with settings from config
        assert app.title == "Claude Code Anthropic API"
        assert app.debug is True  # Default debug setting

    def test_health_endpoint_configuration_check(self):
        """Test that health endpoint checks configuration."""
        # This test verifies that the health endpoint respects the configuration
        # In the actual implementation, the endpoint checks settings.health_check_enabled
        with TestClient(app) as client:
            response = client.get("/health")

        # With default settings, health check should be enabled
        assert response.status_code in [
            200,
            503,
        ]  # Either healthy or unhealthy, but enabled

    def test_metrics_endpoint_configuration_check(self):
        """Test that metrics endpoint checks configuration."""
        # This test verifies that the metrics endpoint respects the configuration
        # In the actual implementation, the endpoint checks settings.metrics_enabled
        with TestClient(app) as client:
            response = client.get("/metrics")

        # With default settings, metrics should be enabled
        assert response.status_code in [
            200,
            503,
        ]  # Either healthy or unhealthy, but enabled
