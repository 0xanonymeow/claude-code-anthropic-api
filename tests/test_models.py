"""
Unit tests for Anthropic API data models.

Tests model validation, serialization, and edge cases to ensure
compatibility with Anthropic's API specification.
"""

from typing import List

import pytest
from pydantic import ValidationError

from src.models.anthropic import (
    AnthropicError,
    ContentBlock,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    ContentType,
    ErrorResponse,
    ErrorType,
    ImageSource,
    Message,
    MessageDeltaEvent,
    MessageRequest,
    MessageResponse,
    MessageRole,
    MessageStartEvent,
    MessageStopEvent,
    Model,
    ModelsResponse,
    PingEvent,
    StopReason,
    Usage,
)


class TestContentBlock:
    """Test ContentBlock model validation."""

    def test_text_content_block_valid(self):
        """Test valid text content block creation."""
        block = ContentBlock(type=ContentType.TEXT, text="Hello, world!")
        assert block.type == ContentType.TEXT
        assert block.text == "Hello, world!"
        assert block.source is None

    def test_image_content_block_valid(self):
        """Test valid image content block creation."""
        source = ImageSource(
            type="base64",
            media_type="image/jpeg",
            data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        block = ContentBlock(type=ContentType.IMAGE, source=source)
        assert block.type == ContentType.IMAGE
        assert block.source == source
        assert block.text is None

    def test_text_content_block_missing_text(self):
        """Test text content block without text field fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            ContentBlock(type=ContentType.TEXT)
        assert "Text content blocks must have 'text' field" in str(exc_info.value)

    def test_image_content_block_missing_source(self):
        """Test image content block without source field fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            ContentBlock(type=ContentType.IMAGE)
        assert "Image content blocks must have 'source' field" in str(exc_info.value)

    def test_text_content_block_with_source_invalid(self):
        """Test text content block with source field fails validation."""
        source = ImageSource(
            type="base64",
            media_type="image/jpeg", 
            data="test_data"
        )
        with pytest.raises(ValidationError) as exc_info:
            ContentBlock(type=ContentType.TEXT, text="Hello", source=source)
        assert "Text content blocks cannot have 'source' field" in str(exc_info.value)

    def test_image_content_block_with_text_invalid(self):
        """Test image content block with text field fails validation."""
        source = ImageSource(
            type="base64",
            media_type="image/jpeg",
            data="test_data"
        )
        with pytest.raises(ValidationError) as exc_info:
            ContentBlock(type=ContentType.IMAGE, text="Hello", source=source)
        assert "Image content blocks cannot have 'text' field" in str(exc_info.value)


class TestImageSource:
    """Test ImageSource model validation."""

    def test_valid_image_source(self):
        """Test valid image source creation."""
        source = ImageSource(
            type="base64",
            media_type="image/png",
            data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        assert source.type == "base64"
        assert source.media_type == "image/png"
        assert source.data is not None

    def test_invalid_media_type(self):
        """Test invalid media type fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            ImageSource(
                type="base64",
                media_type="image/bmp",  # Not supported
                data="test_data"
            )
        assert "Unsupported media type" in str(exc_info.value)

    def test_valid_media_types(self):
        """Test all valid media types are accepted."""
        valid_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        for media_type in valid_types:
            source = ImageSource(
                type="base64",
                media_type=media_type,
                data="test_data"
            )
            assert source.media_type == media_type


class TestMessage:
    """Test Message model validation."""

    def test_message_with_string_content(self):
        """Test message with string content is converted to ContentBlock."""
        message = Message(role=MessageRole.USER, content="Hello!")
        assert message.role == MessageRole.USER
        assert isinstance(message.content, list)
        assert len(message.content) == 1
        assert message.content[0].type == ContentType.TEXT
        assert message.content[0].text == "Hello!"

    def test_message_with_content_blocks(self):
        """Test message with ContentBlock list."""
        blocks = [
            ContentBlock(type=ContentType.TEXT, text="Hello!"),
            ContentBlock(type=ContentType.TEXT, text="How are you?")
        ]
        message = Message(role=MessageRole.ASSISTANT, content=blocks)
        assert message.role == MessageRole.ASSISTANT
        assert message.content == blocks


class TestUsage:
    """Test Usage model validation."""

    def test_valid_usage(self):
        """Test valid usage creation."""
        usage = Usage(input_tokens=10, output_tokens=20)
        assert usage.input_tokens == 10
        assert usage.output_tokens == 20
        assert usage.total_tokens == 30

    def test_negative_tokens_invalid(self):
        """Test negative token counts fail validation."""
        with pytest.raises(ValidationError):
            Usage(input_tokens=-1, output_tokens=20)
        
        with pytest.raises(ValidationError):
            Usage(input_tokens=10, output_tokens=-1)


class TestMessageRequest:
    """Test MessageRequest model validation."""

    def test_valid_message_request(self):
        """Test valid message request creation."""
        messages = [
            Message(role=MessageRole.USER, content="Hello!"),
            Message(role=MessageRole.ASSISTANT, content="Hi there!"),
            Message(role=MessageRole.USER, content="How are you?")
        ]
        request = MessageRequest(
            model="claude-sonnet-4-20250514",
            messages=messages,
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            stop_sequences=["Human:", "Assistant:"],
            stream=False,
            system="You are a helpful assistant."
        )
        
        assert request.model == "claude-sonnet-4-20250514"
        assert len(request.messages) == 3
        assert request.max_tokens == 100
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.top_k == 40
        assert request.stop_sequences == ["Human:", "Assistant:"]
        assert request.stream is False
        assert request.system == "You are a helpful assistant."

    def test_empty_messages_invalid(self):
        """Test empty messages list fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            MessageRequest(
                model="claude-sonnet-4-20250514",
                messages=[],
                max_tokens=100
            )
        assert "List should have at least 1 item" in str(exc_info.value)

    def test_first_message_not_user_invalid(self):
        """Test first message not from user fails validation."""
        messages = [
            Message(role=MessageRole.ASSISTANT, content="Hello!")
        ]
        with pytest.raises(ValidationError) as exc_info:
            MessageRequest(
                model="claude-sonnet-4-20250514",
                messages=messages,
                max_tokens=100
            )
        assert "First message must be from user" in str(exc_info.value)

    def test_consecutive_same_role_invalid(self):
        """Test consecutive messages from same role fail validation."""
        messages = [
            Message(role=MessageRole.USER, content="Hello!"),
            Message(role=MessageRole.USER, content="How are you?")  # Consecutive user messages
        ]
        with pytest.raises(ValidationError) as exc_info:
            MessageRequest(
                model="claude-sonnet-4-20250514",
                messages=messages,
                max_tokens=100
            )
        assert "Messages must alternate between user and assistant roles" in str(exc_info.value)

    def test_parameter_validation(self):
        """Test parameter range validation."""
        messages = [Message(role=MessageRole.USER, content="Hello!")]
        
        # Test max_tokens validation
        with pytest.raises(ValidationError):
            MessageRequest(model="test", messages=messages, max_tokens=0)
        
        with pytest.raises(ValidationError):
            MessageRequest(model="test", messages=messages, max_tokens=5000)
        
        # Test temperature validation
        with pytest.raises(ValidationError):
            MessageRequest(model="test", messages=messages, max_tokens=100, temperature=-0.1)
        
        with pytest.raises(ValidationError):
            MessageRequest(model="test", messages=messages, max_tokens=100, temperature=2.1)
        
        # Test top_p validation
        with pytest.raises(ValidationError):
            MessageRequest(model="test", messages=messages, max_tokens=100, top_p=-0.1)
        
        with pytest.raises(ValidationError):
            MessageRequest(model="test", messages=messages, max_tokens=100, top_p=1.1)

    def test_stop_sequences_validation(self):
        """Test stop sequences validation."""
        messages = [Message(role=MessageRole.USER, content="Hello!")]
        
        # Test too many stop sequences
        with pytest.raises(ValidationError):
            MessageRequest(
                model="test",
                messages=messages,
                max_tokens=100,
                stop_sequences=["a", "b", "c", "d", "e"]  # More than 4
            )
        
        # Test empty stop sequence
        with pytest.raises(ValidationError) as exc_info:
            MessageRequest(
                model="test",
                messages=messages,
                max_tokens=100,
                stop_sequences=[""]
            )
        assert "Stop sequences cannot be empty" in str(exc_info.value)


class TestMessageResponse:
    """Test MessageResponse model validation."""

    def test_valid_message_response(self):
        """Test valid message response creation."""
        content = [ContentBlock(type=ContentType.TEXT, text="Hello there!")]
        usage = Usage(input_tokens=10, output_tokens=15)
        
        response = MessageResponse(
            id="msg_123",
            content=content,
            model="claude-sonnet-4-20250514",
            stop_reason=StopReason.END_TURN,
            stop_sequence=None,
            usage=usage
        )
        
        assert response.id == "msg_123"
        assert response.type == "message"
        assert response.role == "assistant"
        assert response.content == content
        assert response.model == "claude-sonnet-4-20250514"
        assert response.stop_reason == StopReason.END_TURN
        assert response.usage == usage

    def test_empty_content_invalid(self):
        """Test empty content fails validation."""
        usage = Usage(input_tokens=10, output_tokens=15)
        
        with pytest.raises(ValidationError) as exc_info:
            MessageResponse(
                id="msg_123",
                content=[],
                model="claude-sonnet-4-20250514",
                usage=usage
            )
        assert "List should have at least 1 item" in str(exc_info.value)


class TestStreamingEvents:
    """Test streaming event models."""

    def test_message_start_event(self):
        """Test MessageStartEvent creation."""
        content = [ContentBlock(type=ContentType.TEXT, text="Hello!")]
        usage = Usage(input_tokens=10, output_tokens=0)
        message = MessageResponse(
            id="msg_123",
            content=content,
            model="claude-sonnet-4-20250514",
            usage=usage
        )
        
        event = MessageStartEvent(message=message)
        assert event.type == "message_start"
        assert event.message == message

    def test_content_block_start_event(self):
        """Test ContentBlockStartEvent creation."""
        block = ContentBlock(type=ContentType.TEXT, text="Hello!")
        event = ContentBlockStartEvent(index=0, content_block=block)
        assert event.type == "content_block_start"
        assert event.index == 0
        assert event.content_block == block

    def test_content_block_delta_event(self):
        """Test ContentBlockDeltaEvent creation."""
        event = ContentBlockDeltaEvent(index=0, delta={"text": "Hello"})
        assert event.type == "content_block_delta"
        assert event.index == 0
        assert event.delta == {"text": "Hello"}

    def test_content_block_stop_event(self):
        """Test ContentBlockStopEvent creation."""
        event = ContentBlockStopEvent(index=0)
        assert event.type == "content_block_stop"
        assert event.index == 0

    def test_message_delta_event(self):
        """Test MessageDeltaEvent creation."""
        usage = Usage(input_tokens=10, output_tokens=5)
        event = MessageDeltaEvent(delta={"stop_reason": "end_turn"}, usage=usage)
        assert event.type == "message_delta"
        assert event.delta == {"stop_reason": "end_turn"}
        assert event.usage == usage

    def test_message_stop_event(self):
        """Test MessageStopEvent creation."""
        event = MessageStopEvent()
        assert event.type == "message_stop"

    def test_ping_event(self):
        """Test PingEvent creation."""
        event = PingEvent()
        assert event.type == "ping"


class TestErrorModels:
    """Test error model validation."""

    def test_anthropic_error(self):
        """Test AnthropicError creation."""
        error = AnthropicError(
            type=ErrorType.INVALID_REQUEST_ERROR,
            message="Invalid request format"
        )
        assert error.type == ErrorType.INVALID_REQUEST_ERROR
        assert error.message == "Invalid request format"

    def test_error_response(self):
        """Test ErrorResponse creation."""
        error = AnthropicError(
            type=ErrorType.API_ERROR,
            message="Internal server error"
        )
        response = ErrorResponse(error=error)
        assert response.type == "error"
        assert response.error == error


class TestModelModels:
    """Test model listing models."""

    def test_model(self):
        """Test Model creation."""
        model = Model(
            id="claude-sonnet-4-20250514",
            display_name="Claude 3 Sonnet",
            created_at="2024-02-29T00:00:00Z"
        )
        assert model.id == "claude-sonnet-4-20250514"
        assert model.type == "model"
        assert model.display_name == "Claude 3 Sonnet"
        assert model.created_at == "2024-02-29T00:00:00Z"

    def test_models_response(self):
        """Test ModelsResponse creation."""
        models = [
            Model(id="claude-sonnet-4-20250514", display_name="Claude 3 Sonnet"),
            Model(id="claude-3-haiku-20240307", display_name="Claude 3 Haiku")
        ]
        response = ModelsResponse(
            data=models,
            has_more=False,
            first_id="claude-sonnet-4-20250514",
            last_id="claude-3-haiku-20240307"
        )
        assert response.data == models
        assert response.has_more is False
        assert response.first_id == "claude-sonnet-4-20250514"
        assert response.last_id == "claude-3-haiku-20240307"


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_message_request_serialization(self):
        """Test MessageRequest JSON serialization."""
        messages = [Message(role=MessageRole.USER, content="Hello!")]
        request = MessageRequest(
            model="claude-sonnet-4-20250514",
            messages=messages,
            max_tokens=100,
            temperature=0.7
        )
        
        # Test serialization
        json_data = request.model_dump()
        assert json_data["model"] == "claude-sonnet-4-20250514"
        assert json_data["max_tokens"] == 100
        assert json_data["temperature"] == 0.7
        
        # Test deserialization
        new_request = MessageRequest.model_validate(json_data)
        assert new_request.model == request.model
        assert new_request.max_tokens == request.max_tokens
        assert new_request.temperature == request.temperature

    def test_message_response_serialization(self):
        """Test MessageResponse JSON serialization."""
        content = [ContentBlock(type=ContentType.TEXT, text="Hello there!")]
        usage = Usage(input_tokens=10, output_tokens=15)
        response = MessageResponse(
            id="msg_123",
            content=content,
            model="claude-sonnet-4-20250514",
            usage=usage
        )
        
        # Test serialization
        json_data = response.model_dump()
        assert json_data["id"] == "msg_123"
        assert json_data["type"] == "message"
        assert json_data["role"] == "assistant"
        
        # Test deserialization
        new_response = MessageResponse.model_validate(json_data)
        assert new_response.id == response.id
        assert new_response.content[0].text == response.content[0].text