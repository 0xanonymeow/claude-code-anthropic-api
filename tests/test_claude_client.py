"""
Unit tests for Claude Code SDK client wrapper.

This module contains comprehensive tests for the ClaudeClient class,
including mocked Claude Code SDK responses, request/response transformation,
error handling, and model management functionality.
"""

import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.core.claude_client import ClaudeClient, get_claude_client, close_claude_client
from src.core.config import Settings
from src.models.anthropic import (
    MessageRequest,
    MessageResponse,
    Message,
    ContentBlock,
    ContentType,
    MessageRole,
    StopReason,
    Usage,
    Model,
    ErrorResponse,
    ErrorType,
    AnthropicError,
)

# Import Claude SDK errors for testing
from claude_code_sdk import ClaudeSDKError
from claude_code_sdk._errors import (
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ProcessError,
)


class TestClaudeClient:
    """Test cases for ClaudeClient class."""
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            claude_code_timeout=30,
            claude_code_max_retries=2,
            claude_code_options={"test_option": "test_value"}
        )
    
    @pytest.fixture
    def claude_client(self, settings):
        """Create ClaudeClient instance for testing."""
        return ClaudeClient(settings)
    
    @pytest.fixture
    def sample_message_request(self):
        """Create sample MessageRequest for testing."""
        return MessageRequest(
            model="claude-sonnet-4-20250514",
            messages=[
                Message(role=MessageRole.USER, content="Hello, Claude!")
            ],
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            stop_sequences=["Human:", "Assistant:"],
            system="You are a helpful assistant."
        )
    
    @pytest.fixture
    def sample_claude_response(self):
        """Create sample Claude Code SDK response."""
        return {
            "content": [{"type": "text", "text": "Hello! How can I help you today?"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 15,
                "output_tokens": 12
            }
        }
    
    def test_init(self, settings):
        """Test ClaudeClient initialization."""
        client = ClaudeClient(settings)
        assert client.settings == settings
        assert client._sdk is None
        assert client._model_cache is None
        assert isinstance(client._model_mapping, dict)
        assert len(client._model_mapping) > 0
    
    def test_get_model_mapping(self, claude_client):
        """Test model mapping retrieval."""
        mapping = claude_client._get_model_mapping()
        
        assert isinstance(mapping, dict)
        assert "claude-sonnet-4-20250514" in mapping
        assert "claude-3-5-sonnet-20241022" in mapping
        assert "claude-3-haiku-20240307" in mapping
        
        # Verify mapping values
        assert mapping["claude-sonnet-4-20250514"] == "claude-sonnet-4-20250514"
    
    def test_map_anthropic_to_claude_model_valid(self, claude_client):
        """Test valid model mapping from Anthropic to Claude."""
        result = claude_client._map_anthropic_to_claude_model("claude-sonnet-4-20250514")
        assert result == "claude-sonnet-4-20250514"
    
    def test_map_anthropic_to_claude_model_invalid(self, claude_client):
        """Test invalid model mapping raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported model"):
            claude_client._map_anthropic_to_claude_model("invalid-model")
    
    def test_map_claude_to_anthropic_model(self, claude_client):
        """Test model mapping from Claude to Anthropic."""
        result = claude_client._map_claude_to_anthropic_model("claude-sonnet-4-20250514")
        assert result == "claude-sonnet-4-20250514"
        
        # Test unknown model returns as-is
        result = claude_client._map_claude_to_anthropic_model("unknown-model")
        assert result == "unknown-model"
    
    def test_translate_request_to_claude_basic(self, claude_client, sample_message_request):
        """Test basic request translation to Claude format."""
        result = claude_client._translate_request_to_claude(sample_message_request)
        
        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["max_tokens"] == 100
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["top_k"] == 40
        assert result["stop_sequences"] == ["Human:", "Assistant:"]
        assert result["system"] == "You are a helpful assistant."
        
        # Check messages format
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == [{"type": "text", "text": "Hello, Claude!"}]
    
    def test_translate_request_to_claude_minimal(self, claude_client):
        """Test minimal request translation."""
        request = MessageRequest(
            model="claude-3-haiku-20240307",
            messages=[Message(role=MessageRole.USER, content="Hi")],
            max_tokens=50
        )
        
        result = claude_client._translate_request_to_claude(request)
        
        assert result["model"] == "claude-3-haiku-20240307"
        assert result["max_tokens"] == 50
        assert len(result["messages"]) == 1
        
        # Optional parameters should not be present
        assert "temperature" not in result
        assert "top_p" not in result
        assert "top_k" not in result
        assert "stop_sequences" not in result
        assert "system" not in result
    
    def test_convert_content_to_claude_string(self, claude_client):
        """Test content conversion with string input."""
        result = claude_client._convert_content_to_claude("Hello world")
        assert result == "Hello world"
    
    def test_convert_content_to_claude_content_blocks(self, claude_client):
        """Test content conversion with ContentBlock list."""
        content_blocks = [
            ContentBlock(type=ContentType.TEXT, text="Hello"),
            ContentBlock(type=ContentType.TEXT, text="World")
        ]
        
        result = claude_client._convert_content_to_claude(content_blocks)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "Hello"}
        assert result[1] == {"type": "text", "text": "World"}
    
    def test_convert_messages_to_prompt(self, claude_client):
        """Test message conversion to prompt string."""
        messages = [
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi there!"),
            Message(role=MessageRole.USER, content="How are you?")
        ]
        
        result = claude_client._convert_messages_to_prompt(messages)
        
        expected = "Human: Hello\n\nAssistant: Hi there!\n\nHuman: How are you?"
        assert result == expected
    
    def test_convert_messages_to_prompt_content_blocks(self, claude_client):
        """Test message conversion with ContentBlock content."""
        messages = [
            Message(role=MessageRole.USER, content=[
                ContentBlock(type=ContentType.TEXT, text="Hello "),
                ContentBlock(type=ContentType.TEXT, text="world")
            ])
        ]
        
        result = claude_client._convert_messages_to_prompt(messages)
        
        assert result == "Human: Hello world"
    
    def test_estimate_tokens(self, claude_client):
        """Test token estimation."""
        # Test empty string
        assert claude_client._estimate_tokens("") == 1
        
        # Test short string
        assert claude_client._estimate_tokens("hello") == 2  # 5 chars / 4 = 1.25 -> 2
        
        # Test longer string
        assert claude_client._estimate_tokens("This is a longer test string") == 7  # 28 chars / 4 = 7
    
    def test_translate_response_from_claude(self, claude_client, sample_message_request, sample_claude_response):
        """Test response translation from Claude to Anthropic format."""
        result = claude_client._translate_response_from_claude(
            sample_claude_response, 
            sample_message_request
        )
        
        assert isinstance(result, MessageResponse)
        assert result.id.startswith("msg_")
        assert result.model == "claude-sonnet-4-20250514"
        assert result.stop_reason == StopReason.END_TURN
        assert result.stop_sequence is None
        
        # Check content
        assert len(result.content) == 1
        assert result.content[0].type == ContentType.TEXT
        assert result.content[0].text == "Hello! How can I help you today?"
        
        # Check usage
        assert result.usage.input_tokens == 15
        assert result.usage.output_tokens == 12
        assert result.usage.total_tokens == 27
    
    def test_translate_response_from_claude_string_content(self, claude_client, sample_message_request):
        """Test response translation with string content."""
        claude_response = {
            "content": "Simple string response",
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        
        result = claude_client._translate_response_from_claude(
            claude_response, 
            sample_message_request
        )
        
        assert len(result.content) == 1
        assert result.content[0].text == "Simple string response"
        assert result.stop_reason == StopReason.MAX_TOKENS
    
    def test_translate_response_from_claude_empty_content(self, claude_client, sample_message_request):
        """Test response translation with empty content."""
        claude_response = {
            "content": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 0}
        }
        
        result = claude_client._translate_response_from_claude(
            claude_response, 
            sample_message_request
        )
        
        # Should create minimal text block
        assert len(result.content) == 1
        assert result.content[0].text == " "
    
    def test_map_claude_error_to_anthropic_connection(self, claude_client):
        """Test error mapping for connection errors."""
        error = CLIConnectionError("Connection failed")
        result = claude_client._map_claude_error_to_anthropic(error)
        
        assert isinstance(result, ErrorResponse)
        assert result.error.type == ErrorType.API_ERROR
        assert "Failed to connect" in result.error.message
    
    def test_map_claude_error_to_anthropic_cli_not_found(self, claude_client):
        """Test error mapping for CLI not found errors."""
        error = CLINotFoundError("CLI not found")
        result = claude_client._map_claude_error_to_anthropic(error)
        
        assert result.error.type == ErrorType.NOT_FOUND_ERROR
        assert "Claude Code CLI not found" in result.error.message
    
    def test_map_claude_error_to_anthropic_json_decode(self, claude_client):
        """Test error mapping for JSON decode errors."""
        original_error = ValueError("Invalid JSON")
        error = CLIJSONDecodeError("invalid json line", original_error)
        result = claude_client._map_claude_error_to_anthropic(error)
        
        assert result.error.type == ErrorType.API_ERROR
        assert "Failed to decode response" in result.error.message
    
    def test_map_claude_error_to_anthropic_process_error(self, claude_client):
        """Test error mapping for process errors."""
        error = ProcessError("Process failed")
        result = claude_client._map_claude_error_to_anthropic(error)
        
        assert result.error.type == ErrorType.API_ERROR
        assert "Claude Code process error" in result.error.message
    
    def test_map_claude_error_to_anthropic_generic_sdk_error(self, claude_client):
        """Test error mapping for generic SDK errors."""
        error = ClaudeSDKError("Generic SDK error")
        result = claude_client._map_claude_error_to_anthropic(error)
        
        assert result.error.type == ErrorType.API_ERROR
        assert "Claude Code SDK error" in result.error.message
    

    
    def test_map_claude_error_to_anthropic_unexpected_error(self, claude_client):
        """Test error mapping for unexpected errors."""
        error = ValueError("Unexpected error")
        result = claude_client._map_claude_error_to_anthropic(error)
        
        assert result.error.type == ErrorType.API_ERROR
        assert "Unexpected error" in result.error.message
    
    @pytest.mark.asyncio
    async def test_get_sdk_initialization(self, claude_client):
        """Test SDK initialization."""
        with patch('src.core.claude_client.ClaudeSDKClient') as mock_sdk_class:
            mock_sdk = AsyncMock()
            mock_sdk_class.return_value = mock_sdk
            
            sdk = await claude_client._get_sdk()
            
            assert sdk == mock_sdk
            assert claude_client._sdk == mock_sdk
            mock_sdk_class.assert_called_once()
            mock_sdk.connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_sdk_reuse_existing(self, claude_client):
        """Test SDK instance reuse."""
        mock_sdk = AsyncMock()
        claude_client._sdk = mock_sdk
        
        with patch('src.core.claude_client.ClaudeSDKClient') as mock_sdk_class:
            sdk = await claude_client._get_sdk()
            
            assert sdk == mock_sdk
            mock_sdk_class.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_create_message_success(self, claude_client, sample_message_request, sample_claude_response):
        """Test successful message creation."""
        # Mock message object
        mock_message = MagicMock()
        mock_message.content = [MagicMock()]
        mock_message.content[0].text = "Hello! How can I help you today?"
        
        async def mock_receive_messages():
            yield mock_message
        
        mock_sdk = AsyncMock()
        mock_sdk.query = AsyncMock()
        mock_sdk.receive_messages = mock_receive_messages
        claude_client._sdk = mock_sdk
        
        result = await claude_client.create_message(sample_message_request)
        
        assert isinstance(result, MessageResponse)
        assert result.model == "claude-sonnet-4-20250514"
        assert len(result.content) == 1
        assert result.content[0].text == "Hello! How can I help you today?"
        
        # Verify SDK was called
        mock_sdk.query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_message_error(self, claude_client, sample_message_request):
        """Test message creation with error."""
        mock_sdk = AsyncMock()
        mock_sdk.query.side_effect = ProcessError("Process failed")
        claude_client._sdk = mock_sdk
        
        with pytest.raises(Exception) as exc_info:
            await claude_client.create_message(sample_message_request)
        
        assert "Claude Code process error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_create_message_stream_success(self, claude_client, sample_message_request):
        """Test successful streaming message creation."""
        # Mock message objects
        mock_message1 = MagicMock()
        mock_message1.content = [MagicMock()]
        mock_message1.content[0].text = "Hello"
        
        mock_message2 = MagicMock()
        mock_message2.content = " world"
        
        async def mock_receive_messages():
            yield mock_message1
            yield mock_message2
        
        mock_sdk = AsyncMock()
        mock_sdk.query = AsyncMock()
        mock_sdk.receive_messages = mock_receive_messages
        claude_client._sdk = mock_sdk
        
        # Set stream=True in request
        sample_message_request.stream = True
        
        events = []
        async for event in claude_client.create_message_stream(sample_message_request):
            events.append(event)
        
        assert len(events) >= 5  # At least message_start, content_block_start, deltas, content_block_stop, message_stop
        assert "event: message_start" in events[0]
        assert "event: content_block_start" in events[1]
        assert "event: content_block_delta" in events[2]
        assert "event: message_stop" in events[-1]
    
    @pytest.mark.asyncio
    async def test_create_message_stream_error(self, claude_client, sample_message_request):
        """Test streaming message creation with error."""
        mock_sdk = AsyncMock()
        mock_sdk.query.side_effect = CLIConnectionError("Connection failed")
        claude_client._sdk = mock_sdk
        
        sample_message_request.stream = True
        
        events = []
        async for event in claude_client.create_message_stream(sample_message_request):
            events.append(event)
        
        # Should receive initial events plus error event
        assert len(events) >= 1
        assert "event: error" in events[-1]  # Error should be the last event
        assert "Failed to connect" in events[-1]
    
    @pytest.mark.asyncio
    async def test_get_available_models_success(self, claude_client):
        """Test successful model retrieval."""
        models = await claude_client.get_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        
        # Check first model
        model = models[0]
        assert isinstance(model, Model)
        assert model.id in claude_client._model_mapping
        assert model.display_name
        
        # Test caching
        models2 = await claude_client.get_available_models()
        assert models == models2
        assert claude_client._model_cache == models
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, claude_client):
        """Test successful health check."""
        # Mock the get_available_models method
        claude_client._model_cache = [
            Model(id="claude-sonnet-4-20250514", display_name="Claude 3 Sonnet")
        ]
        
        with patch.object(claude_client, '_get_sdk') as mock_get_sdk:
            mock_sdk = AsyncMock()
            mock_get_sdk.return_value = mock_sdk
            
            result = await claude_client.health_check()
            
            assert result["status"] == "healthy"
            assert result["claude_code_sdk"] == "connected"
            assert result["models_available"] == 1
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, claude_client):
        """Test health check failure."""
        with patch.object(claude_client, '_get_sdk') as mock_get_sdk:
            mock_get_sdk.side_effect = Exception("Connection failed")
            
            result = await claude_client.health_check()
            
            assert result["status"] == "unhealthy"
            assert result["claude_code_sdk"] == "disconnected"
            assert "Connection failed" in result["error"]
    
    @pytest.mark.asyncio
    async def test_close(self, claude_client):
        """Test client cleanup."""
        mock_sdk = AsyncMock()
        mock_sdk.disconnect = AsyncMock()
        claude_client._sdk = mock_sdk
        claude_client._model_cache = [Model(id="test", display_name="Test")]
        
        await claude_client.close()
        
        mock_sdk.disconnect.assert_called_once()
        assert claude_client._sdk is None
        assert claude_client._model_cache is None
    
    @pytest.mark.asyncio
    async def test_close_disconnect_error(self, claude_client):
        """Test client cleanup when disconnect fails."""
        mock_sdk = AsyncMock()
        mock_sdk.disconnect.side_effect = Exception("Disconnect failed")
        claude_client._sdk = mock_sdk
        
        # Should not raise exception
        await claude_client.close()
        assert claude_client._sdk is None


class TestGlobalClientFunctions:
    """Test cases for global client management functions."""
    
    def test_get_claude_client_new_instance(self):
        """Test getting new client instance."""
        settings = Settings()
        client = get_claude_client(settings)
        
        assert isinstance(client, ClaudeClient)
        assert client.settings == settings
    
    def test_get_claude_client_reuse_instance(self):
        """Test client instance reuse."""
        settings = Settings()
        client1 = get_claude_client(settings)
        client2 = get_claude_client(settings)
        
        assert client1 is client2
    
    @pytest.mark.asyncio
    async def test_close_claude_client(self):
        """Test closing global client."""
        settings = Settings()
        client = get_claude_client(settings)
        
        with patch.object(client, 'close') as mock_close:
            await close_claude_client()
            mock_close.assert_called_once()
    
    def teardown_method(self):
        """Clean up after each test."""
        # Reset global client
        import src.core.claude_client
        src.core.claude_client._claude_client = None