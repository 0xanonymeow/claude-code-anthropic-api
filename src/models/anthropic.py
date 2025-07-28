"""
Pydantic models for Anthropic API compatibility.

This module contains all the data models that match Anthropic's API specification
for the /v1/messages endpoint, including request/response models, content blocks,
and error handling models.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


class ContentType(str, Enum):
    """Content block types supported by the API."""
    TEXT = "text"
    IMAGE = "image"


class MessageRole(str, Enum):
    """Message roles in conversations."""
    USER = "user"
    ASSISTANT = "assistant"


class StopReason(str, Enum):
    """Possible reasons for stopping generation."""
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"


class ImageSource(BaseModel):
    """Image source information for image content blocks."""
    type: Literal["base64"] = "base64"
    media_type: str = Field(..., description="MIME type of the image")
    data: str = Field(..., description="Base64-encoded image data")

    @field_validator("media_type")
    @classmethod
    def validate_media_type(cls, v):
        """Validate that media type is a supported image format."""
        allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        if v not in allowed_types:
            raise ValueError(f"Unsupported media type: {v}. Must be one of {allowed_types}")
        return v


class ContentBlock(BaseModel):
    """Content block that can contain text or image data."""
    type: ContentType = Field(..., description="Type of content block")
    text: Optional[str] = Field(None, description="Text content (required for text blocks)")
    source: Optional[ImageSource] = Field(None, description="Image source (required for image blocks)")

    @model_validator(mode='after')
    def validate_content_block(self):
        """Ensure content block has appropriate fields for its type."""
        if self.type == ContentType.TEXT:
            if not self.text:
                raise ValueError("Text content blocks must have 'text' field")
            if self.source:
                raise ValueError("Text content blocks cannot have 'source' field")
        elif self.type == ContentType.IMAGE:
            if not self.source:
                raise ValueError("Image content blocks must have 'source' field")
            if self.text:
                raise ValueError("Image content blocks cannot have 'text' field")

        return self


class Message(BaseModel):
    """A message in a conversation."""
    role: MessageRole = Field(..., description="Role of the message sender")
    content: Union[str, List[ContentBlock]] = Field(..., description="Message content")

    @field_validator("content", mode='before')
    @classmethod
    def validate_content(cls, v):
        """Convert string content to ContentBlock list if needed."""
        if isinstance(v, str):
            return [ContentBlock(type=ContentType.TEXT, text=v)]
        return v


class Usage(BaseModel):
    """Token usage information."""
    input_tokens: int = Field(..., ge=0, description="Number of input tokens")
    output_tokens: int = Field(..., ge=0, description="Number of output tokens")

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens used."""
        return self.input_tokens + self.output_tokens


class MessageRequest(BaseModel):
    """Request model for the /v1/messages endpoint."""
    model: str = Field(..., description="Model identifier")
    messages: List[Message] = Field(..., min_length=1, description="List of messages in the conversation")
    max_tokens: int = Field(..., gt=0, le=4096, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: Optional[int] = Field(None, ge=1, description="Top-k sampling parameter")
    stop_sequences: Optional[List[str]] = Field(None, max_length=4, description="Stop sequences")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    system: Optional[Union[str, List[ContentBlock]]] = Field(None, description="System prompt as string or content blocks")
    
    @field_validator("system", mode='before')
    @classmethod
    def validate_system(cls, v):
        """Validate and normalize system parameter to handle both string and content block formats."""
        if v is None or v == "":
            return None
        
        # If it's already a string, return as-is
        if isinstance(v, str):
            return v
            
        # If it's a list of content blocks, convert to string
        if isinstance(v, list):
            text_parts = []
            for block in v:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif hasattr(block, 'text') and hasattr(block, 'type') and block.type == ContentType.TEXT:
                    text_parts.append(block.text)
            return "\n".join(text_parts) if text_parts else None
            
        # Invalid format
        raise ValueError("System parameter must be a string or array of content blocks")
        
    system_processed: Optional[str] = Field(None, exclude=True, description="Processed system prompt as string")

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        """Validate message sequence follows conversation rules."""
        if not v:
            raise ValueError("Messages list cannot be empty")
        
        # First message should be from user
        if v[0].role != MessageRole.USER:
            raise ValueError("First message must be from user")
        
        # Messages should alternate between user and assistant
        for i in range(1, len(v)):
            prev_role = v[i-1].role
            curr_role = v[i].role
            
            if prev_role == curr_role:
                raise ValueError(f"Messages must alternate between user and assistant roles. Found consecutive {curr_role} messages at position {i}")
        
        return v

    @field_validator("stop_sequences")
    @classmethod
    def validate_stop_sequences(cls, v):
        """Validate stop sequences."""
        if v is not None:
            for seq in v:
                if not seq or len(seq.strip()) == 0:
                    raise ValueError("Stop sequences cannot be empty or whitespace-only")
        return v


class MessageResponse(BaseModel):
    """Response model for the /v1/messages endpoint."""
    id: str = Field(..., description="Unique message identifier")
    type: Literal["message"] = Field("message", description="Response type")
    role: Literal["assistant"] = Field("assistant", description="Role of the response")
    content: List[ContentBlock] = Field(..., min_length=1, description="Response content blocks")
    model: str = Field(..., description="Model that generated the response")
    stop_reason: Optional[StopReason] = Field(None, description="Reason for stopping generation")
    stop_sequence: Optional[str] = Field(None, description="Stop sequence that triggered stopping")
    usage: Usage = Field(..., description="Token usage information")

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v):
        """Ensure response has content."""
        if not v:
            raise ValueError("Response content cannot be empty")
        return v


class StreamEvent(BaseModel):
    """Base class for streaming events."""
    type: str = Field(..., description="Event type")


class MessageStartEvent(StreamEvent):
    """Event sent at the start of a streaming response."""
    type: Literal["message_start"] = "message_start"
    message: MessageResponse = Field(..., description="Initial message data")


class ContentBlockStartEvent(StreamEvent):
    """Event sent when a content block starts."""
    type: Literal["content_block_start"] = "content_block_start"
    index: int = Field(..., ge=0, description="Index of the content block")
    content_block: ContentBlock = Field(..., description="Content block data")


class ContentBlockDeltaEvent(StreamEvent):
    """Event sent for incremental content updates."""
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int = Field(..., ge=0, description="Index of the content block")
    delta: Dict[str, Any] = Field(..., description="Incremental content update")


class ContentBlockStopEvent(StreamEvent):
    """Event sent when a content block ends."""
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int = Field(..., ge=0, description="Index of the content block")


class MessageDeltaEvent(StreamEvent):
    """Event sent for message-level updates."""
    type: Literal["message_delta"] = "message_delta"
    delta: Dict[str, Any] = Field(..., description="Message-level updates")
    usage: Usage = Field(..., description="Updated usage information")


class MessageStopEvent(StreamEvent):
    """Event sent when streaming ends."""
    type: Literal["message_stop"] = "message_stop"


class PingEvent(StreamEvent):
    """Ping event to keep connection alive."""
    type: Literal["ping"] = "ping"


class ErrorType(str, Enum):
    """Types of API errors."""
    INVALID_REQUEST_ERROR = "invalid_request_error"
    AUTHENTICATION_ERROR = "authentication_error"
    PERMISSION_ERROR = "permission_error"
    NOT_FOUND_ERROR = "not_found_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    API_ERROR = "api_error"
    OVERLOADED_ERROR = "overloaded_error"


class AnthropicError(BaseModel):
    """Error information following Anthropic's format."""
    type: ErrorType = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")


class ErrorResponse(BaseModel):
    """Error response following Anthropic's format."""
    type: Literal["error"] = Field("error", description="Response type")
    error: AnthropicError = Field(..., description="Error details")


class Model(BaseModel):
    """Model information for the /v1/models endpoint."""
    id: str = Field(..., description="Model identifier")
    type: Literal["model"] = Field("model", description="Object type")
    display_name: str = Field(..., description="Human-readable model name")
    created_at: Optional[str] = Field(None, description="Model creation timestamp")


class ModelsResponse(BaseModel):
    """Response for the /v1/models endpoint."""
    data: List[Model] = Field(..., description="List of available models")
    has_more: bool = Field(False, description="Whether there are more models")
    first_id: Optional[str] = Field(None, description="ID of the first model")
    last_id: Optional[str] = Field(None, description="ID of the last model")