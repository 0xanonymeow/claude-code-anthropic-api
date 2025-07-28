"""
Claude Code SDK client wrapper for Anthropic API compatibility.

This module provides the main interface to Claude Code SDK, handling request
translation from Anthropic format to Claude Code SDK format, response
transformation back to Anthropic format, and error mapping between the two systems.
"""

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from claude_code_sdk import ClaudeSDKClient, ClaudeSDKError
from claude_code_sdk._errors import (
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ProcessError,
)
from claude_code_sdk.types import ClaudeCodeOptions

from ..models.anthropic import (
    MessageRequest,
    MessageResponse,
    Message,
    ContentBlock,
    ContentType,
    MessageRole,
    StopReason,
    Usage,
    Model,
    AnthropicError,
    ErrorType,
    ErrorResponse,
    StreamEvent,
    MessageStartEvent,
    ContentBlockStartEvent,
    ContentBlockDeltaEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageStopEvent,
)
from .config import Settings
from ..utils.logging import get_logger, log_claude_sdk_error, log_function_call
from ..utils.error_handling import (
    create_service_unavailable_error,
    create_invalid_model_error
)


logger = get_logger(__name__)


class ClaudeClient:
    """
    Main interface to Claude Code SDK with Anthropic API compatibility.
    
    This class handles:
    - Request translation from Anthropic format to Claude Code SDK format
    - Response transformation from Claude Code SDK back to Anthropic format
    - Error mapping between Claude Code SDK errors and Anthropic API errors
    - Model management and availability checking
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the Claude client with configuration.
        
        Args:
            settings: Application settings containing Claude Code SDK configuration
        """
        self.settings = settings
        self._sdk: Optional[ClaudeSDKClient] = None
        self._model_cache: Optional[List[Model]] = None
        self._model_mapping = self._get_model_mapping()
        self._stream_lock = asyncio.Lock()  # Prevent concurrent stream access
        
    async def _get_sdk(self) -> ClaudeSDKClient:
        """
        Get or create Claude Code SDK instance with enhanced error handling.
        
        Returns:
            ClaudeSDKClient: Configured SDK instance
            
        Raises:
            Exception: If SDK initialization fails
        """
        if self._sdk is None:
            try:
                claude_config = self.settings.get_claude_code_config()
                
                logger.info("Initializing Claude Code SDK", {
                    'event': 'claude_sdk_init_start',
                    'config': {k: v for k, v in claude_config.items() if k != 'path'}  # Don't log sensitive paths
                })
                
                # Create ClaudeCodeOptions from config
                options = ClaudeCodeOptions()
                if "path" in claude_config:
                    options["path"] = claude_config["path"]
                
                self._sdk = ClaudeSDKClient(options=options)
                await self._sdk.connect()
                
                logger.info("Claude Code SDK initialized and connected", {
                    'event': 'claude_sdk_init_success',
                    'service': 'claude_code_sdk'
                })
                
            except Exception as e:
                logger.error("Failed to initialize Claude Code SDK", {
                    'event': 'claude_sdk_init_failed',
                    'error': str(e),
                    'error_type': e.__class__.__name__,
                    'service': 'claude_code_sdk'
                }, exc_info=True)
                raise
                
        return self._sdk
    
    def _get_model_mapping(self) -> Dict[str, str]:
        """
        Get mapping from Anthropic model names to Claude Code SDK model names.
        
        Returns:
            Dict mapping Anthropic model IDs to Claude Code SDK model IDs
        """
        return {
            "claude-sonnet-4-20250514": "claude-sonnet-4-20250514",
            "claude-opus-4-20250514": "claude-opus-4-20250514",
        }
    
    def _map_anthropic_to_claude_model(self, anthropic_model: str) -> str:
        """
        Map Anthropic model ID to Claude Code SDK model ID.
        
        Args:
            anthropic_model: Anthropic model identifier
            
        Returns:
            Claude Code SDK model identifier
            
        Raises:
            ValueError: If model is not supported
        """
        if anthropic_model not in self._model_mapping:
            raise ValueError(f"Unsupported model: {anthropic_model}")
        return self._model_mapping[anthropic_model]
    
    def _map_claude_to_anthropic_model(self, claude_model: str) -> str:
        """
        Map Claude Code SDK model ID to Anthropic model ID.
        
        Args:
            claude_model: Claude Code SDK model identifier
            
        Returns:
            Anthropic model identifier
        """
        # Reverse lookup in model mapping
        for anthropic_id, claude_id in self._model_mapping.items():
            if claude_id == claude_model:
                return anthropic_id
        return claude_model  # Return as-is if no mapping found
    
    def _translate_request_to_claude(self, request: MessageRequest) -> Dict[str, Any]:
        """
        Translate Anthropic MessageRequest to Claude Code SDK format.
        
        Args:
            request: Anthropic API request
            
        Returns:
            Dict containing Claude Code SDK request parameters
        """
        claude_model = self._map_anthropic_to_claude_model(request.model)
        
        # Convert messages to Claude format
        claude_messages = []
        for msg in request.messages:
            claude_msg = {
                "role": msg.role.value,
                "content": self._convert_content_to_claude(msg.content)
            }
            claude_messages.append(claude_msg)
        
        # Build Claude request
        claude_request = {
            "model": claude_model,
            "messages": claude_messages,
            "max_tokens": request.max_tokens,
        }
        
        # Add optional parameters
        if request.temperature is not None:
            claude_request["temperature"] = request.temperature
        if request.top_p is not None:
            claude_request["top_p"] = request.top_p
        if request.top_k is not None:
            claude_request["top_k"] = request.top_k
        if request.stop_sequences:
            claude_request["stop_sequences"] = request.stop_sequences
        if request.system:
            claude_request["system"] = request.system
        if request.stream is not None:
            claude_request["stream"] = request.stream
            
        return claude_request
    
    def _convert_content_to_claude(self, content: Any) -> Any:
        """
        Convert Anthropic content format to Claude Code SDK format.
        
        Args:
            content: Anthropic content (string or list of ContentBlocks)
            
        Returns:
            Claude Code SDK compatible content
        """
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            claude_content = []
            for block in content:
                if isinstance(block, ContentBlock):
                    if block.type == ContentType.TEXT:
                        claude_content.append({
                            "type": "text",
                            "text": block.text
                        })
                    elif block.type == ContentType.IMAGE and block.source:
                        claude_content.append({
                            "type": "image",
                            "source": {
                                "type": block.source.type,
                                "media_type": block.source.media_type,
                                "data": block.source.data
                            }
                        })
                else:
                    # Handle dict format
                    claude_content.append(block)
            return claude_content
        
        return content
    
    def _convert_messages_to_prompt(self, messages: List[Message]) -> str:
        """
        Convert Anthropic messages to a simple prompt string.
        
        Args:
            messages: List of Anthropic messages
            
        Returns:
            Simple prompt string for Claude Code SDK
        """
        prompt_parts = []
        
        for message in messages:
            role_prefix = "Human: " if message.role == MessageRole.USER else "Assistant: "
            
            if isinstance(message.content, str):
                prompt_parts.append(f"{role_prefix}{message.content}")
            elif isinstance(message.content, list):
                content_text = ""
                for block in message.content:
                    if isinstance(block, ContentBlock) and block.type == ContentType.TEXT:
                        content_text += block.text or ""
                    elif isinstance(block, dict) and block.get("type") == "text":
                        content_text += block.get("text", "")
                prompt_parts.append(f"{role_prefix}{content_text}")
        
        return "\n\n".join(prompt_parts)
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text (rough approximation).
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Rough approximation: ~4 characters per token
        if not text:
            return 1
        return max(1, (len(text) + 3) // 4)  # Round up division
    
    def _translate_response_from_claude(
        self, 
        claude_response: Dict[str, Any], 
        original_request: MessageRequest
    ) -> MessageResponse:
        """
        Translate Claude Code SDK response to Anthropic MessageResponse format.
        
        Args:
            claude_response: Response from Claude Code SDK
            original_request: Original Anthropic request for context
            
        Returns:
            Anthropic-compatible MessageResponse
        """
        # Generate unique message ID
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        
        # Convert content blocks
        content_blocks = []
        claude_content = claude_response.get("content", [])
        
        if isinstance(claude_content, str):
            content_blocks.append(ContentBlock(type=ContentType.TEXT, text=claude_content))
        elif isinstance(claude_content, list):
            for block in claude_content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        content_blocks.append(ContentBlock(
                            type=ContentType.TEXT,
                            text=block.get("text", "")
                        ))
                    # Add support for other content types as needed
                else:
                    # Handle string content in list
                    content_blocks.append(ContentBlock(type=ContentType.TEXT, text=str(block)))
        
        # Ensure we have at least one content block
        if not content_blocks:
            content_blocks.append(ContentBlock(type=ContentType.TEXT, text=" "))
        
        # Map stop reason
        stop_reason = None
        claude_stop_reason = claude_response.get("stop_reason")
        if claude_stop_reason == "end_turn":
            stop_reason = StopReason.END_TURN
        elif claude_stop_reason == "max_tokens":
            stop_reason = StopReason.MAX_TOKENS
        elif claude_stop_reason == "stop_sequence":
            stop_reason = StopReason.STOP_SEQUENCE
        
        # Extract usage information
        usage_data = claude_response.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0)
        )
        
        return MessageResponse(
            id=message_id,
            content=content_blocks,
            model=original_request.model,  # Return original Anthropic model ID
            stop_reason=stop_reason,
            stop_sequence=claude_response.get("stop_sequence"),
            usage=usage
        )
    
    def _map_claude_error_to_anthropic(self, error: Exception) -> ErrorResponse:
        """
        Map Claude Code SDK errors to Anthropic API error format.
        
        Args:
            error: Exception from Claude Code SDK
            
        Returns:
            Anthropic-compatible ErrorResponse
        """
        if isinstance(error, CLINotFoundError):
            anthropic_error = AnthropicError(
                type=ErrorType.NOT_FOUND_ERROR,
                message="Claude Code CLI not found"
            )
        elif isinstance(error, CLIConnectionError):
            anthropic_error = AnthropicError(
                type=ErrorType.API_ERROR,
                message="Failed to connect to Claude Code CLI"
            )
        elif isinstance(error, CLIJSONDecodeError):
            anthropic_error = AnthropicError(
                type=ErrorType.API_ERROR,
                message="Failed to decode response from Claude Code CLI"
            )
        elif isinstance(error, ProcessError):
            anthropic_error = AnthropicError(
                type=ErrorType.API_ERROR,
                message=f"Claude Code process error: {str(error)}"
            )
        elif isinstance(error, ClaudeSDKError):
            anthropic_error = AnthropicError(
                type=ErrorType.API_ERROR,
                message=f"Claude Code SDK error: {str(error)}"
            )
        else:
            anthropic_error = AnthropicError(
                type=ErrorType.API_ERROR,
                message=f"Unexpected error: {str(error)}"
            )
        
        return ErrorResponse(error=anthropic_error)
    
    async def create_message(self, request: MessageRequest) -> MessageResponse:
        """
        Create a message using Claude Code SDK (non-streaming) with enhanced error handling.
        
        Args:
            request: Anthropic API message request
            
        Returns:
            Anthropic-compatible message response
            
        Raises:
            Exception: If Claude Code SDK call fails
        """
        timer_id = logger.start_timer("create_message")
        
        try:
            # Validate model
            if request.model not in self._model_mapping:
                error_response = create_invalid_model_error(
                    request.model,
                    list(self._model_mapping.keys())
                )
                raise ValueError(error_response.error.message)
            
            logger.log_claude_sdk_request("create_message", request.model, {
                'max_tokens': request.max_tokens,
                'temperature': request.temperature,
                'message_count': len(request.messages)
            })
            
            # Convert messages to a simple prompt for now
            # In a real implementation, this would be more sophisticated
            prompt = self._convert_messages_to_prompt(request.messages)
            
            # Get SDK instance and make request
            sdk = await self._get_sdk()
            
            # Use query method to send the prompt
            await sdk.query(prompt)
            
            # Collect the full response with concurrency protection
            full_response = ""
            async with self._stream_lock:
                async for message in sdk.receive_messages():
                    if hasattr(message, 'content') and message.content:
                        if isinstance(message.content, list):
                            for block in message.content:
                                if hasattr(block, 'text'):
                                    full_response += block.text
                        elif isinstance(message.content, str):
                            full_response += message.content
                        break  # For non-streaming, we take the first complete response
            
            # Create a mock Claude response format
            claude_response = {
                "content": [{"type": "text", "text": full_response}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": self._estimate_tokens(prompt),
                    "output_tokens": self._estimate_tokens(full_response)
                }
            }
            
            # Translate response back to Anthropic format
            response = self._translate_response_from_claude(claude_response, request)
            
            # Log successful completion
            duration = logger.stop_timer(timer_id, "create_message", {
                'message_id': response.id,
                'model': request.model,
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens,
                'success': True
            })
            
            logger.log_claude_sdk_response("create_message", request.model, duration, True, {
                'message_id': response.id,
                'tokens_used': response.usage.total_tokens
            })
            
            return response
            
        except Exception as e:
            # Log error with context
            duration = logger.stop_timer(timer_id, "create_message", {
                'success': False,
                'error': str(e),
                'error_type': e.__class__.__name__,
                'model': request.model
            })
            
            logger.log_claude_sdk_response("create_message", request.model, duration, False, {
                'error': str(e),
                'error_type': e.__class__.__name__
            })
            
            log_claude_sdk_error(logger, "create_message", e, {
                'model': request.model,
                'request_id': getattr(request, 'id', 'unknown')
            })
            
            error_response = self._map_claude_error_to_anthropic(e)
            raise Exception(error_response.error.message) from e
    
    async def create_message_stream(
        self, 
        request: MessageRequest
    ) -> AsyncGenerator[str, None]:
        """
        Create a streaming message using Claude Code SDK with enhanced error handling.
        
        Args:
            request: Anthropic API message request with stream=True
            
        Yields:
            Server-Sent Events formatted strings
            
        Raises:
            Exception: If Claude Code SDK call fails
        """
        timer_id = logger.start_timer("create_message_stream")
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        
        try:
            # Validate model
            if request.model not in self._model_mapping:
                error_response = create_invalid_model_error(
                    request.model,
                    list(self._model_mapping.keys())
                )
                raise ValueError(error_response.error.message)
            
            logger.log_claude_sdk_request("create_message_stream", request.model, {
                'max_tokens': request.max_tokens,
                'temperature': request.temperature,
                'message_count': len(request.messages),
                'message_id': message_id,
                'streaming': True
            })
            
            # Convert messages to a simple prompt
            prompt = self._convert_messages_to_prompt(request.messages)
            
            # Get SDK instance and make streaming request
            sdk = await self._get_sdk()
            
            current_content = ""
            input_tokens = self._estimate_tokens(prompt)
            output_tokens = 0
            
            # Send message start event
            initial_response = MessageResponse(
                id=message_id,
                content=[ContentBlock(type=ContentType.TEXT, text=" ")],
                model=request.model,
                usage=Usage(input_tokens=input_tokens, output_tokens=0)
            )
            event = MessageStartEvent(message=initial_response)
            yield f"event: message_start\ndata: {event.model_dump_json()}\n\n"
            
            logger.log_streaming_event("message_start", message_id, {
                'model': request.model,
                'input_tokens': input_tokens
            })
            
            # Send content block start event
            content_block = ContentBlock(type=ContentType.TEXT, text=" ")
            event = ContentBlockStartEvent(index=0, content_block=content_block)
            yield f"event: content_block_start\ndata: {event.model_dump_json()}\n\n"
            
            logger.log_streaming_event("content_block_start", message_id)
            
            # Start the query
            await sdk.query(prompt)
            
            # Stream the response with concurrency protection
            async with self._stream_lock:
                async for message in sdk.receive_messages():
                    if hasattr(message, 'content') and message.content:
                        if isinstance(message.content, list):
                            for block in message.content:
                                if hasattr(block, 'text') and block.text:
                                    # Send content delta event
                                    event = ContentBlockDeltaEvent(
                                        index=0,
                                        delta={"type": "text_delta", "text": block.text}
                                    )
                                    yield f"event: content_block_delta\ndata: {event.model_dump_json()}\n\n"
                                    current_content += block.text
                                    output_tokens += self._estimate_tokens(block.text)
                                    
                                    logger.log_streaming_event("content_block_delta", message_id, {
                                        'text_length': len(block.text),
                                        'total_output_tokens': output_tokens
                                    })
                                    
                        elif isinstance(message.content, str):
                            # Send content delta event
                            event = ContentBlockDeltaEvent(
                                index=0,
                                delta={"type": "text_delta", "text": message.content}
                            )
                            yield f"event: content_block_delta\ndata: {event.model_dump_json()}\n\n"
                            current_content += message.content
                            output_tokens += self._estimate_tokens(message.content)
                            
                            logger.log_streaming_event("content_block_delta", message_id, {
                                'text_length': len(message.content),
                                'total_output_tokens': output_tokens
                            })
                        break  # End of response
            
            # Send content block stop event
            event = ContentBlockStopEvent(index=0)
            yield f"event: content_block_stop\ndata: {event.model_dump_json()}\n\n"
            
            logger.log_streaming_event("content_block_stop", message_id)
            
            # Send message delta event with final usage
            usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)
            event = MessageDeltaEvent(
                delta={"stop_reason": "end_turn"},
                usage=usage
            )
            yield f"event: message_delta\ndata: {event.model_dump_json()}\n\n"
            
            logger.log_streaming_event("message_delta", message_id, {
                'final_usage': usage.model_dump()
            })
            
            # Send message stop event
            event = MessageStopEvent()
            yield f"event: message_stop\ndata: {event.model_dump_json()}\n\n"
            
            # Log successful completion
            duration = logger.stop_timer(timer_id, "create_message_stream", {
                'message_id': message_id,
                'model': request.model,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_content_length': len(current_content),
                'success': True
            })
            
            logger.log_claude_sdk_response("create_message_stream", request.model, duration, True, {
                'message_id': message_id,
                'tokens_used': input_tokens + output_tokens,
                'streaming': True
            })
            
            logger.log_streaming_event("message_stop", message_id, {
                'final_tokens': input_tokens + output_tokens,
                'duration_seconds': duration
            })
            
        except Exception as e:
            # Log error with context
            duration = logger.stop_timer(timer_id, "create_message_stream", {
                'success': False,
                'error': str(e),
                'error_type': e.__class__.__name__,
                'model': request.model,
                'message_id': message_id
            })
            
            logger.log_claude_sdk_response("create_message_stream", request.model, duration, False, {
                'error': str(e),
                'error_type': e.__class__.__name__,
                'message_id': message_id,
                'streaming': True
            })
            
            log_claude_sdk_error(logger, "create_message_stream", e, {
                'model': request.model,
                'message_id': message_id,
                'streaming': True
            })
            
            # Send error event in stream
            error_response = self._map_claude_error_to_anthropic(e)
            error_event = {
                "type": "error",
                "error": error_response.error.model_dump()
            }
            yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
            
            logger.log_streaming_event("error", message_id, {
                'error': str(e),
                'error_type': e.__class__.__name__
            })
    
    async def get_available_models(self) -> List[Model]:
        """
        Get list of available models from Claude Code SDK.
        
        Returns:
            List of available models in Anthropic format
        """
        if self._model_cache is not None:
            return self._model_cache
        
        try:
            logger.info("Fetching available models from Claude Code SDK")
            
            # For now, return the models we know are supported
            # In a real implementation, this would query the Claude Code SDK
            models = []
            for anthropic_id, claude_id in self._model_mapping.items():
                model = Model(
                    id=anthropic_id,
                    display_name=anthropic_id.replace("-", " ").title(),
                    created_at=None  # Claude Code SDK might not provide this
                )
                models.append(model)
            
            self._model_cache = models
            logger.info(f"Found {len(models)} available models")
            return models
            
        except Exception as e:
            logger.error(f"Error fetching models: {str(e)}")
            # Return empty list on error, or raise exception based on requirements
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Claude Code SDK connection.
        
        Returns:
            Dict containing health status information
        """
        try:
            sdk = await self._get_sdk()
            # Try to get models as a basic health check
            await self.get_available_models()
            
            return {
                "status": "healthy",
                "claude_code_sdk": "connected",
                "models_available": len(self._model_cache or [])
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "claude_code_sdk": "disconnected",
                "error": str(e)
            }
    
    async def close(self) -> None:
        """
        Close the Claude Code SDK connection and cleanup resources.
        """
        if self._sdk:
            try:
                # Disconnect from Claude Code SDK
                await self._sdk.disconnect()
                logger.info("Claude Code SDK connection closed")
            except Exception as e:
                logger.error(f"Error closing Claude Code SDK: {str(e)}")
            finally:
                self._sdk = None
                self._model_cache = None


# Global client instance
_claude_client: Optional[ClaudeClient] = None


def get_claude_client(settings: Optional[Settings] = None) -> ClaudeClient:
    """
    Get the global Claude client instance.
    
    Args:
        settings: Settings to use for client initialization
        
    Returns:
        ClaudeClient: The global client instance
    """
    global _claude_client
    if _claude_client is None:
        if settings is None:
            from .config import get_settings
            settings = get_settings()
        _claude_client = ClaudeClient(settings)
    return _claude_client


async def close_claude_client() -> None:
    """
    Close the global Claude client instance.
    """
    global _claude_client
    if _claude_client:
        await _claude_client.close()
        _claude_client = None