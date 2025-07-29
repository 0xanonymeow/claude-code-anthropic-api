"""
Integration tests for the /v1/messages API endpoint.

This module contains comprehensive tests for the messages endpoint including
request validation, streaming and non-streaming responses, error handling,
and Anthropic API compatibility verification.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.messages import router
from src.core.claude_client import ClaudeClient
from src.models.anthropic import (
    AnthropicError,
    ContentBlock,
    ContentType,
    ErrorResponse,
    ErrorType,
    Message,
    MessageRequest,
    MessageResponse,
    MessageRole,
    Model,
    Usage,
)

# Create test app
app = FastAPI()
app.include_router(router)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_claude_client():
    """Create mock Claude client."""
    mock_client = AsyncMock(spec=ClaudeClient)
    
    # Mock available models
    mock_models = [
        Model(id="claude-3-5-sonnet-20241022", display_name="Claude 3.5 Sonnet"),
        Model(id="claude-3-haiku-20240307", display_name="Claude 3 Haiku"),
    ]
    mock_client.get_available_models.return_value = mock_models
    
    return mock_client


@pytest.fixture
def sample_message_request():
    """Create sample message request."""
    return {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [
            {"role": "user", "content": "Hello, Claude!"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }


@pytest.fixture
def sample_message_response():
    """Create sample message response."""
    return MessageResponse(
        id="msg_123456",
        content=[ContentBlock(type=ContentType.TEXT, text="Hello! How can I help you today?")],
        model="claude-3-5-sonnet-20241022",
        usage=Usage(input_tokens=10, output_tokens=12)
    )


class TestMessagesEndpoint:
    """Test cases for the /v1/messages endpoint."""
    
    @patch('src.api.messages.get_claude_client')
    def test_create_message_non_streaming_success(
        self, 
        mock_get_client, 
        client, 
        mock_claude_client, 
        sample_message_request, 
        sample_message_response
    ):
        """Test successful non-streaming message creation."""
        # Setup mocks
        mock_get_client.return_value = mock_claude_client
        mock_claude_client.create_message.return_value = sample_message_response
        
        # Make request
        response = client.post("/v1/messages", json=sample_message_request)
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["id"] == "msg_123456"
        assert response_data["type"] == "message"
        assert response_data["role"] == "assistant"
        assert response_data["model"] == "claude-3-5-sonnet-20241022"
        assert len(response_data["content"]) == 1
        assert response_data["content"][0]["type"] == "text"
        assert response_data["content"][0]["text"] == "Hello! How can I help you today?"
        assert response_data["usage"]["input_tokens"] == 10
        assert response_data["usage"]["output_tokens"] == 12
        
        # Verify Claude client was called correctly
        mock_claude_client.create_message.assert_called_once()
        call_args = mock_claude_client.create_message.call_args[0][0]
        assert call_args.model == "claude-3-5-sonnet-20241022"
        assert len(call_args.messages) == 1
        assert call_args.messages[0].role == MessageRole.USER
        assert call_args.max_tokens == 100
        assert call_args.temperature == 0.7
    
    @patch('src.api.messages.get_claude_client')
    def test_create_message_streaming_success(
        self, 
        mock_get_client, 
        client, 
        mock_claude_client, 
        sample_message_request
    ):
        """Test successful streaming message creation."""
        # Setup streaming request
        streaming_request = sample_message_request.copy()
        streaming_request["stream"] = True
        
        # Mock streaming response
        async def mock_stream():
            yield "event: message_start\ndata: {\"type\":\"message_start\"}\n\n"
            yield "event: content_block_delta\ndata: {\"type\":\"content_block_delta\"}\n\n"
            yield "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
        
        mock_get_client.return_value = mock_claude_client
        mock_claude_client.create_message_stream.return_value = mock_stream()
        
        # Make request
        response = client.post("/v1/messages", json=streaming_request)
        
        # Verify response
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream"
        
        # Verify streaming content
        content = response.text
        assert "event: message_start" in content
        assert "event: content_block_delta" in content
        assert "event: message_stop" in content
        
        # Verify Claude client was called correctly
        mock_claude_client.create_message_stream.assert_called_once()
    
    @patch('src.api.messages.get_claude_client')
    def test_create_message_unsupported_model(
        self, 
        mock_get_client, 
        client, 
        mock_claude_client, 
        sample_message_request
    ):
        """Test error handling for unsupported model."""
        # Setup request with unsupported model
        invalid_request = sample_message_request.copy()
        invalid_request["model"] = "unsupported-model"
        
        mock_get_client.return_value = mock_claude_client
        
        # Make request
        response = client.post("/v1/messages", json=invalid_request)
        
        # Verify error response
        assert response.status_code == 400
        response_data = response.json()
        
        assert response_data["detail"]["type"] == "error"
        assert response_data["detail"]["error"]["type"] == "invalid_request_error"
        assert "unsupported-model" in response_data["detail"]["error"]["message"]
        assert "Available models:" in response_data["detail"]["error"]["message"]
    
    def test_create_message_validation_errors(self, client):
        """Test request validation error handling."""
        test_cases = [
            # Missing required fields
            ({}, "Field required"),
            ({"model": "claude-3-5-sonnet-20241022"}, "Field required"),
            
            # Invalid field types
            ({
                "model": "claude-3-5-sonnet-20241022",
                "messages": "not a list",
                "max_tokens": 100
            }, "Input should be a valid list"),
            
            # Invalid field values
            ({
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 0  # Should be > 0
            }, "Input should be greater than 0"),
            
            # Invalid temperature range
            ({
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
                "temperature": 3.0  # Should be <= 2.0
            }, "Input should be less than or equal to 2"),
            
            # Empty messages list
            ({
                "model": "claude-3-5-sonnet-20241022",
                "messages": [],
                "max_tokens": 100
            }, "List should have at least 1 item"),
            
            # Invalid message role
            ({
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "invalid", "content": "Hello"}],
                "max_tokens": 100
            }, "Input should be 'user' or 'assistant'"),
        ]
        
        for invalid_request, expected_error_fragment in test_cases:
            response = client.post("/v1/messages", json=invalid_request)
            assert response.status_code == 422  # FastAPI validation error
            # Note: FastAPI returns 422 for validation errors, not our custom 400
    
    @patch('src.api.messages.get_claude_client')
    def test_create_message_claude_client_error(
        self, 
        mock_get_client, 
        client, 
        mock_claude_client, 
        sample_message_request
    ):
        """Test error handling when Claude client fails."""
        mock_get_client.return_value = mock_claude_client
        mock_claude_client.create_message.side_effect = Exception("Claude SDK connection failed")
        
        # Make request
        response = client.post("/v1/messages", json=sample_message_request)
        
        # Verify error response (connection errors map to 503)
        assert response.status_code == 503
        response_data = response.json()
        
        assert response_data["detail"]["type"] == "error"
        assert response_data["detail"]["error"]["type"] == "api_error"
        assert "Failed to connect to Claude Code SDK" in response_data["detail"]["error"]["message"]
    
    @patch('src.api.messages.get_claude_client')
    def test_create_message_connection_error(
        self, 
        mock_get_client, 
        client, 
        mock_claude_client, 
        sample_message_request
    ):
        """Test specific handling of connection errors."""
        mock_get_client.return_value = mock_claude_client
        mock_claude_client.create_message.side_effect = Exception("connection timeout")
        
        # Make request
        response = client.post("/v1/messages", json=sample_message_request)
        
        # Verify error response
        assert response.status_code == 503
        response_data = response.json()
        
        assert response_data["detail"]["type"] == "error"
        assert response_data["detail"]["error"]["type"] == "api_error"
        assert "Failed to connect to Claude Code SDK" in response_data["detail"]["error"]["message"]
    
    @patch('src.api.messages.get_claude_client')
    def test_create_message_not_found_error(
        self, 
        mock_get_client, 
        client, 
        mock_claude_client, 
        sample_message_request
    ):
        """Test specific handling of not found errors."""
        mock_get_client.return_value = mock_claude_client
        mock_claude_client.create_message.side_effect = Exception("model not found")
        
        # Make request
        response = client.post("/v1/messages", json=sample_message_request)
        
        # Verify error response
        assert response.status_code == 404
        response_data = response.json()
        
        assert response_data["detail"]["type"] == "error"
        assert response_data["detail"]["error"]["type"] == "not_found_error"
        assert "model not found" in response_data["detail"]["error"]["message"]
    
    def test_create_message_complex_content(self, client):
        """Test handling of complex message content with multiple blocks."""
        complex_request = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see in this image?"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 100
        }
        
        with patch('src.api.messages.get_claude_client') as mock_get_client:
            mock_claude_client = AsyncMock(spec=ClaudeClient)
            mock_models = [Model(id="claude-3-5-sonnet-20241022", display_name="Claude 3.5 Sonnet")]
            mock_claude_client.get_available_models.return_value = mock_models
            
            mock_response = MessageResponse(
                id="msg_complex",
                content=[ContentBlock(type=ContentType.TEXT, text="I can see a small test image.")],
                model="claude-3-5-sonnet-20241022",
                usage=Usage(input_tokens=20, output_tokens=8)
            )
            mock_claude_client.create_message.return_value = mock_response
            mock_get_client.return_value = mock_claude_client
            
            # Make request
            response = client.post("/v1/messages", json=complex_request)
            
            # Verify response
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["id"] == "msg_complex"
            assert response_data["content"][0]["text"] == "I can see a small test image."
    
    def test_create_message_multi_turn_conversation(self, client):
        """Test handling of multi-turn conversations."""
        multi_turn_request = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": "Hello, what's your name?"},
                {"role": "assistant", "content": "Hello! I'm Claude, an AI assistant."},
                {"role": "user", "content": "Nice to meet you, Claude!"}
            ],
            "max_tokens": 100
        }
        
        with patch('src.api.messages.get_claude_client') as mock_get_client:
            mock_claude_client = AsyncMock(spec=ClaudeClient)
            mock_models = [Model(id="claude-3-5-sonnet-20241022", display_name="Claude 3.5 Sonnet")]
            mock_claude_client.get_available_models.return_value = mock_models
            
            mock_response = MessageResponse(
                id="msg_multiturn",
                content=[ContentBlock(type=ContentType.TEXT, text="Nice to meet you too!")],
                model="claude-3-5-sonnet-20241022",
                usage=Usage(input_tokens=30, output_tokens=6)
            )
            mock_claude_client.create_message.return_value = mock_response
            mock_get_client.return_value = mock_claude_client
            
            # Make request
            response = client.post("/v1/messages", json=multi_turn_request)
            
            # Verify response
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["id"] == "msg_multiturn"
            
            # Verify the request was processed with all messages
            mock_claude_client.create_message.assert_called_once()
            call_args = mock_claude_client.create_message.call_args[0][0]
            assert len(call_args.messages) == 3
            assert call_args.messages[0].role == MessageRole.USER
            assert call_args.messages[1].role == MessageRole.ASSISTANT
            assert call_args.messages[2].role == MessageRole.USER


class TestMessagesHealthCheck:
    """Test cases for the messages health check endpoint."""
    
    @patch('src.api.messages.get_claude_client')
    def test_messages_health_check_healthy(self, mock_get_client, client):
        """Test health check when service is healthy."""
        mock_claude_client = AsyncMock(spec=ClaudeClient)
        mock_claude_client.health_check.return_value = {
            "status": "healthy",
            "claude_code_sdk": "connected"
        }
        mock_get_client.return_value = mock_claude_client
        
        # Make request
        response = client.get("/v1/messages/health")
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["status"] == "healthy"
        assert response_data["endpoint"] == "/v1/messages"
        assert response_data["claude_sdk"]["status"] == "healthy"
    
    @patch('src.api.messages.get_claude_client')
    def test_messages_health_check_unhealthy(self, mock_get_client, client):
        """Test health check when service is unhealthy."""
        mock_claude_client = AsyncMock(spec=ClaudeClient)
        mock_claude_client.health_check.side_effect = Exception("Connection failed")
        mock_get_client.return_value = mock_claude_client
        
        # Make request
        response = client.get("/v1/messages/health")
        
        # Verify response
        assert response.status_code == 200  # Health check endpoint doesn't return error status
        response_data = response.json()
        
        assert response_data["status"] == "unhealthy"
        assert response_data["endpoint"] == "/v1/messages"
        assert "Connection failed" in response_data["error"]


class TestStreamingIntegration:
    """Integration tests specifically for streaming functionality."""
    
    @patch('src.api.messages.get_claude_client')
    def test_streaming_response_format(self, mock_get_client, client):
        """Test that streaming responses follow SSE format correctly."""
        streaming_request = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "stream": True
        }
        
        # Mock streaming response with proper SSE format
        async def mock_stream():
            yield "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_123\"}}\n\n"
            yield "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0}\n\n"
            yield "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n"
            yield "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n"
            yield "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
        
        mock_claude_client = AsyncMock(spec=ClaudeClient)
        mock_models = [Model(id="claude-3-5-sonnet-20241022", display_name="Claude 3.5 Sonnet")]
        mock_claude_client.get_available_models.return_value = mock_models
        mock_claude_client.create_message_stream.return_value = mock_stream()
        mock_get_client.return_value = mock_claude_client
        
        # Make request
        response = client.post("/v1/messages", json=streaming_request)
        
        # Verify response headers
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream"
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["connection"] == "keep-alive"
        
        # Verify SSE format
        content = response.text
        lines = content.split('\n')
        
        # Check for proper SSE event structure
        assert any(line.startswith("event: message_start") for line in lines)
        assert any(line.startswith("event: content_block_start") for line in lines)
        assert any(line.startswith("event: content_block_delta") for line in lines)
        assert any(line.startswith("event: content_block_stop") for line in lines)
        assert any(line.startswith("event: message_stop") for line in lines)
        
        # Check for proper data lines
        assert any(line.startswith("data: ") for line in lines)
    
    @patch('src.api.messages.get_claude_client')
    def test_streaming_error_handling(self, mock_get_client, client):
        """Test error handling in streaming responses."""
        streaming_request = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "stream": True
        }
        
        mock_claude_client = AsyncMock(spec=ClaudeClient)
        mock_models = [Model(id="claude-3-5-sonnet-20241022", display_name="Claude 3.5 Sonnet")]
        mock_claude_client.get_available_models.return_value = mock_models
        mock_claude_client.create_message_stream.side_effect = Exception("Streaming failed")
        mock_get_client.return_value = mock_claude_client
        
        # Make request
        response = client.post("/v1/messages", json=streaming_request)
        
        # Verify error response
        assert response.status_code == 500
        response_data = response.json()
        
        assert response_data["detail"]["type"] == "error"
        assert response_data["detail"]["error"]["type"] == "api_error"
        assert "Failed to initialize streaming" in response_data["detail"]["error"]["message"]