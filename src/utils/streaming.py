"""
Streaming utilities for Server-Sent Events (SSE) support.

This module provides utilities for formatting and managing streaming responses
that match Anthropic's streaming specification, including proper SSE formatting,
connection management, and error handling.
"""

import json
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional, Union
from contextlib import asynccontextmanager

from ..models.anthropic import (
    StreamEvent,
    MessageStartEvent,
    ContentBlockStartEvent,
    ContentBlockDeltaEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageStopEvent,
    PingEvent,
    ErrorResponse,
    AnthropicError,
    ErrorType
)


class SSEFormatter:
    """Utility class for formatting Server-Sent Events."""
    
    @staticmethod
    def format_event(event_type: str, data: Union[str, Dict[str, Any]], event_id: Optional[str] = None) -> str:
        """
        Format data as a Server-Sent Event.
        
        Args:
            event_type: The event type (e.g., 'message_start', 'content_block_delta')
            data: The event data (string or dict that will be JSON-encoded)
            event_id: Optional event ID
            
        Returns:
            Formatted SSE string
        """
        lines = []
        
        if event_id:
            lines.append(f"id: {event_id}")
            
        lines.append(f"event: {event_type}")
        
        if isinstance(data, dict):
            data_str = json.dumps(data, separators=(',', ':'))
        else:
            data_str = str(data)
            
        lines.append(f"data: {data_str}")
        lines.append("")  # Empty line to terminate the event
        
        return "\n".join(lines)
    
    @staticmethod
    def format_stream_event(event: StreamEvent, event_id: Optional[str] = None) -> str:
        """
        Format a StreamEvent as SSE.
        
        Args:
            event: The stream event to format
            event_id: Optional event ID
            
        Returns:
            Formatted SSE string
        """
        return SSEFormatter.format_event(
            event_type=event.type,
            data=event.model_dump(),
            event_id=event_id
        )
    
    @staticmethod
    def format_error_event(error: Union[Exception, str, ErrorResponse], event_id: Optional[str] = None) -> str:
        """
        Format an error as an SSE error event.
        
        Args:
            error: The error to format (Exception, string, or ErrorResponse)
            event_id: Optional event ID
            
        Returns:
            Formatted SSE error event
        """
        if isinstance(error, ErrorResponse):
            error_data = error.model_dump()
        elif isinstance(error, Exception):
            error_data = ErrorResponse(
                error=AnthropicError(
                    type=ErrorType.API_ERROR,
                    message=str(error)
                )
            ).model_dump()
        else:
            error_data = ErrorResponse(
                error=AnthropicError(
                    type=ErrorType.API_ERROR,
                    message=str(error)
                )
            ).model_dump()
        
        return SSEFormatter.format_event(
            event_type="error",
            data=error_data,
            event_id=event_id
        )
    
    @staticmethod
    def format_done_event(event_id: Optional[str] = None) -> str:
        """
        Format a stream termination event.
        
        Args:
            event_id: Optional event ID
            
        Returns:
            Formatted SSE done event
        """
        return SSEFormatter.format_event(
            event_type="done",
            data="[DONE]",
            event_id=event_id
        )


class StreamManager:
    """Manages streaming connections and event generation."""
    
    def __init__(self, ping_interval: float = 30.0):
        """
        Initialize the stream manager.
        
        Args:
            ping_interval: Interval in seconds between ping events to keep connection alive
        """
        self.ping_interval = ping_interval
        self._event_counter = 0
    
    def _get_next_event_id(self) -> str:
        """Generate the next event ID."""
        self._event_counter += 1
        return str(self._event_counter)
    
    async def stream_events(
        self,
        event_generator: AsyncGenerator[StreamEvent, None],
        include_ping: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Convert a stream of events to SSE-formatted strings.
        
        Args:
            event_generator: Async generator yielding StreamEvent objects
            include_ping: Whether to include periodic ping events
            
        Yields:
            SSE-formatted event strings
        """
        ping_task = None
        
        try:
            if include_ping:
                ping_task = asyncio.create_task(self._ping_generator())
            
            async for event in event_generator:
                event_id = self._get_next_event_id()
                sse_data = SSEFormatter.format_stream_event(event, event_id)
                yield sse_data
                
        except Exception as e:
            # Send error event
            error_id = self._get_next_event_id()
            error_sse = SSEFormatter.format_error_event(e, error_id)
            yield error_sse
            
        finally:
            if ping_task:
                ping_task.cancel()
                try:
                    await ping_task
                except asyncio.CancelledError:
                    pass
            
            # Send termination event
            done_id = self._get_next_event_id()
            done_sse = SSEFormatter.format_done_event(done_id)
            yield done_sse
    
    async def _ping_generator(self) -> None:
        """Generate periodic ping events to keep connection alive."""
        while True:
            await asyncio.sleep(self.ping_interval)
            # This would be yielded by the main generator if we had a way to inject it
            # For now, we'll handle pings differently in the actual implementation
    
    @asynccontextmanager
    async def managed_stream(self, event_generator: AsyncGenerator[StreamEvent, None]):
        """
        Context manager for handling streaming with proper cleanup.
        
        Args:
            event_generator: The event generator to manage
            
        Yields:
            SSE-formatted event generator
        """
        try:
            yield self.stream_events(event_generator, include_ping=False)
        except Exception as e:
            # Log error or handle cleanup
            raise
        finally:
            # Ensure generator is properly closed
            if hasattr(event_generator, 'aclose'):
                await event_generator.aclose()


class ClaudeSDKStreamAdapter:
    """Adapter for converting Claude Code SDK streaming responses to Anthropic format."""
    
    def __init__(self):
        """Initialize the adapter."""
        self.content_block_index = 0
        self.message_id: Optional[str] = None
    
    async def adapt_claude_stream(
        self,
        claude_stream: AsyncGenerator[Dict[str, Any], None],
        message_id: str,
        model: str
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Convert Claude Code SDK streaming response to Anthropic streaming events.
        
        Args:
            claude_stream: Claude Code SDK streaming response
            message_id: Message ID for the response
            model: Model identifier
            
        Yields:
            StreamEvent objects in Anthropic format
        """
        self.message_id = message_id
        self.content_block_index = 0
        
        # Send message start event
        from ..models.anthropic import MessageResponse, ContentBlock, ContentType, Usage
        
        # Create initial response with placeholder content that will be updated
        initial_content = ContentBlock(type=ContentType.TEXT, text=" ")
        initial_response = MessageResponse(
            id=message_id,
            content=[initial_content],
            model=model,
            usage=Usage(input_tokens=0, output_tokens=0)
        )
        
        yield MessageStartEvent(message=initial_response)
        
        # Process Claude SDK stream
        content_started = False
        accumulated_text = ""
        
        try:
            async for chunk in claude_stream:
                if self._is_content_start(chunk):
                    if not content_started:
                        # Start first content block
                        content_block = ContentBlock(
                            type=ContentType.TEXT,
                            text=" "
                        )
                        yield ContentBlockStartEvent(
                            index=self.content_block_index,
                            content_block=content_block
                        )
                        content_started = True
                
                elif self._is_content_delta(chunk):
                    # Send content delta
                    text_delta = self._extract_text_delta(chunk)
                    if text_delta:
                        accumulated_text += text_delta
                        yield ContentBlockDeltaEvent(
                            index=self.content_block_index,
                            delta={"type": "text_delta", "text": text_delta}
                        )
                
                elif self._is_content_stop(chunk):
                    # End content block
                    if content_started:
                        yield ContentBlockStopEvent(index=self.content_block_index)
                
                elif self._is_message_delta(chunk):
                    # Send message delta with usage info
                    usage_info = self._extract_usage(chunk)
                    if usage_info:
                        yield MessageDeltaEvent(
                            delta={"stop_reason": self._extract_stop_reason(chunk)},
                            usage=usage_info
                        )
        
        except Exception as e:
            # Let the error propagate to be handled by StreamManager
            raise
        
        finally:
            # Send message stop event
            yield MessageStopEvent()
    
    def _is_content_start(self, chunk: Dict[str, Any]) -> bool:
        """Check if chunk indicates content start."""
        return chunk.get("type") == "content_block_start"
    
    def _is_content_delta(self, chunk: Dict[str, Any]) -> bool:
        """Check if chunk is a content delta."""
        return chunk.get("type") == "content_block_delta"
    
    def _is_content_stop(self, chunk: Dict[str, Any]) -> bool:
        """Check if chunk indicates content stop."""
        return chunk.get("type") == "content_block_stop"
    
    def _is_message_delta(self, chunk: Dict[str, Any]) -> bool:
        """Check if chunk is a message delta."""
        return chunk.get("type") == "message_delta"
    
    def _extract_text_delta(self, chunk: Dict[str, Any]) -> Optional[str]:
        """Extract text delta from chunk."""
        delta = chunk.get("delta", {})
        return delta.get("text")
    
    def _extract_usage(self, chunk: Dict[str, Any]) -> Optional['Usage']:
        """Extract usage information from chunk."""
        from ..models.anthropic import Usage
        
        usage_data = chunk.get("usage")
        if usage_data:
            return Usage(
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0)
            )
        return None
    
    def _extract_stop_reason(self, chunk: Dict[str, Any]) -> Optional[str]:
        """Extract stop reason from chunk."""
        delta = chunk.get("delta", {})
        return delta.get("stop_reason")


# Convenience functions for common streaming operations

async def create_sse_stream(
    event_generator: AsyncGenerator[StreamEvent, None],
    ping_interval: float = 30.0
) -> AsyncGenerator[str, None]:
    """
    Create an SSE stream from a StreamEvent generator.
    
    Args:
        event_generator: Generator yielding StreamEvent objects
        ping_interval: Interval for ping events
        
    Yields:
        SSE-formatted strings
    """
    manager = StreamManager(ping_interval=ping_interval)
    async for sse_data in manager.stream_events(event_generator):
        yield sse_data


async def adapt_claude_to_sse(
    claude_stream: AsyncGenerator[Dict[str, Any], None],
    message_id: str,
    model: str,
    ping_interval: float = 30.0
) -> AsyncGenerator[str, None]:
    """
    Convert Claude Code SDK stream directly to SSE format.
    
    Args:
        claude_stream: Claude Code SDK streaming response
        message_id: Message ID for the response
        model: Model identifier
        ping_interval: Interval for ping events
        
    Yields:
        SSE-formatted strings
    """
    adapter = ClaudeSDKStreamAdapter()
    event_stream = adapter.adapt_claude_stream(claude_stream, message_id, model)
    
    async for sse_data in create_sse_stream(event_stream, ping_interval):
        yield sse_data