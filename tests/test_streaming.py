"""
Unit tests for streaming utilities and Server-Sent Events support.

Tests cover SSE formatting, stream management, Claude SDK adaptation,
error handling, and connection management functionality.
"""

import asyncio
import json
from typing import Any, AsyncGenerator, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.anthropic import (
    AnthropicError,
    ContentBlock,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    ContentType,
    ErrorResponse,
    ErrorType,
    MessageDeltaEvent,
    MessageResponse,
    MessageStartEvent,
    MessageStopEvent,
    PingEvent,
    StreamEvent,
    Usage,
)
from src.utils.streaming import (
    ClaudeSDKStreamAdapter,
    SSEFormatter,
    StreamManager,
    adapt_claude_to_sse,
    create_sse_stream,
)


class TestSSEFormatter:
    """Test cases for SSE formatting functionality."""

    def test_format_event_basic(self):
        """Test basic event formatting."""
        result = SSEFormatter.format_event("test_event", {"key": "value"})

        expected = 'event: test_event\ndata: {"key":"value"}\n'
        assert result == expected

    def test_format_event_with_id(self):
        """Test event formatting with ID."""
        result = SSEFormatter.format_event(
            "test_event", {"key": "value"}, event_id="123"
        )

        expected = 'id: 123\nevent: test_event\ndata: {"key":"value"}\n'
        assert result == expected

    def test_format_event_string_data(self):
        """Test event formatting with string data."""
        result = SSEFormatter.format_event("test_event", "simple string")

        expected = "event: test_event\ndata: simple string\n"
        assert result == expected

    def test_format_stream_event(self):
        """Test formatting of StreamEvent objects."""
        event = MessageStopEvent()
        result = SSEFormatter.format_stream_event(event, event_id="456")

        expected = 'id: 456\nevent: message_stop\ndata: {"type":"message_stop"}\n'
        assert result == expected

    def test_format_error_event_from_exception(self):
        """Test error event formatting from Exception."""
        error = ValueError("Test error message")
        result = SSEFormatter.format_error_event(error)

        assert "event: error" in result
        assert "Test error message" in result
        assert "api_error" in result

    def test_format_error_event_from_string(self):
        """Test error event formatting from string."""
        result = SSEFormatter.format_error_event("Custom error message")

        assert "event: error" in result
        assert "Custom error message" in result
        assert "api_error" in result

    def test_format_error_event_from_error_response(self):
        """Test error event formatting from ErrorResponse."""
        error_response = ErrorResponse(
            error=AnthropicError(
                type=ErrorType.RATE_LIMIT_ERROR, message="Rate limit exceeded"
            )
        )
        result = SSEFormatter.format_error_event(error_response)

        assert "event: error" in result
        assert "Rate limit exceeded" in result
        assert "rate_limit_error" in result

    def test_format_done_event(self):
        """Test done event formatting."""
        result = SSEFormatter.format_done_event(event_id="789")

        expected = "id: 789\nevent: done\ndata: [DONE]\n"
        assert result == expected


class TestStreamManager:
    """Test cases for stream management functionality."""

    @pytest.fixture
    def stream_manager(self):
        """Create a StreamManager instance for testing."""
        return StreamManager(ping_interval=1.0)

    async def test_stream_events_basic(self, stream_manager):
        """Test basic event streaming."""

        async def mock_event_generator():
            yield MessageStopEvent()

        events = []
        async for sse_data in stream_manager.stream_events(mock_event_generator()):
            events.append(sse_data)

        assert len(events) == 2  # One event + done event
        assert "message_stop" in events[0]
        assert "[DONE]" in events[1]

    async def test_stream_events_with_error(self, stream_manager):
        """Test event streaming with error handling."""

        async def error_generator():
            yield MessageStopEvent()
            raise ValueError("Test error")

        events = []
        async for sse_data in stream_manager.stream_events(error_generator()):
            events.append(sse_data)

        assert len(events) == 3  # One event + error event + done event
        assert "message_stop" in events[0]
        assert "error" in events[1]
        assert "Test error" in events[1]
        assert "[DONE]" in events[2]

    async def test_managed_stream_context(self, stream_manager):
        """Test managed stream context manager."""

        async def mock_event_generator():
            yield MessageStopEvent()

        events = []
        async with stream_manager.managed_stream(mock_event_generator()) as stream:
            async for sse_data in stream:
                events.append(sse_data)

        assert len(events) == 2  # One event + done event
        assert "message_stop" in events[0]

    def test_event_id_generation(self, stream_manager):
        """Test event ID generation."""
        id1 = stream_manager._get_next_event_id()
        id2 = stream_manager._get_next_event_id()

        assert id1 == "1"
        assert id2 == "2"
        assert id1 != id2


class TestClaudeSDKStreamAdapter:
    """Test cases for Claude SDK stream adaptation."""

    @pytest.fixture
    def adapter(self):
        """Create a ClaudeSDKStreamAdapter instance for testing."""
        return ClaudeSDKStreamAdapter()

    async def test_adapt_claude_stream_basic(self, adapter):
        """Test basic Claude stream adaptation."""

        async def mock_claude_stream():
            yield {"type": "content_block_start", "index": 0}
            yield {"type": "content_block_delta", "delta": {"text": "Hello"}}
            yield {"type": "content_block_delta", "delta": {"text": " world"}}
            yield {"type": "content_block_stop", "index": 0}
            yield {
                "type": "message_delta",
                "usage": {"input_tokens": 5, "output_tokens": 2},
            }

        events = []
        async for event in adapter.adapt_claude_stream(
            mock_claude_stream(), message_id="test_123", model="claude-sonnet-4"
        ):
            events.append(event)

        # Should have: message_start, content_block_start, 2 deltas, content_block_stop, message_delta, message_stop
        assert len(events) == 7
        assert isinstance(events[0], MessageStartEvent)
        assert isinstance(events[1], ContentBlockStartEvent)
        assert isinstance(events[2], ContentBlockDeltaEvent)
        assert isinstance(events[3], ContentBlockDeltaEvent)
        assert isinstance(events[4], ContentBlockStopEvent)
        assert isinstance(events[5], MessageDeltaEvent)
        assert isinstance(events[6], MessageStopEvent)

    async def test_adapt_claude_stream_with_error(self, adapter):
        """Test Claude stream adaptation with error."""

        async def error_claude_stream():
            yield {"type": "content_block_start", "index": 0}
            raise RuntimeError("Claude SDK error")

        with pytest.raises(RuntimeError, match="Claude SDK error"):
            events = []
            async for event in adapter.adapt_claude_stream(
                error_claude_stream(), message_id="test_123", model="claude-sonnet-4"
            ):
                events.append(event)

    def test_chunk_type_detection(self, adapter):
        """Test chunk type detection methods."""
        assert adapter._is_content_start({"type": "content_block_start"})
        assert not adapter._is_content_start({"type": "other"})

        assert adapter._is_content_delta({"type": "content_block_delta"})
        assert not adapter._is_content_delta({"type": "other"})

        assert adapter._is_content_stop({"type": "content_block_stop"})
        assert not adapter._is_content_stop({"type": "other"})

        assert adapter._is_message_delta({"type": "message_delta"})
        assert not adapter._is_message_delta({"type": "other"})

    def test_extract_text_delta(self, adapter):
        """Test text delta extraction."""
        chunk = {"delta": {"text": "Hello world"}}
        result = adapter._extract_text_delta(chunk)
        assert result == "Hello world"

        chunk_no_text = {"delta": {"other": "value"}}
        result = adapter._extract_text_delta(chunk_no_text)
        assert result is None

    def test_extract_usage(self, adapter):
        """Test usage information extraction."""
        chunk = {"usage": {"input_tokens": 10, "output_tokens": 5}}
        result = adapter._extract_usage(chunk)

        assert result is not None
        assert result.input_tokens == 10
        assert result.output_tokens == 5

        chunk_no_usage = {"other": "value"}
        result = adapter._extract_usage(chunk_no_usage)
        assert result is None

    def test_extract_stop_reason(self, adapter):
        """Test stop reason extraction."""
        chunk = {"delta": {"stop_reason": "max_tokens"}}
        result = adapter._extract_stop_reason(chunk)
        assert result == "max_tokens"

        chunk_no_stop = {"delta": {"other": "value"}}
        result = adapter._extract_stop_reason(chunk_no_stop)
        assert result is None


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    async def test_create_sse_stream(self):
        """Test create_sse_stream convenience function."""

        async def mock_event_generator():
            yield MessageStopEvent()

        events = []
        async for sse_data in create_sse_stream(
            mock_event_generator(), ping_interval=1.0
        ):
            events.append(sse_data)

        assert len(events) == 2  # One event + done event
        assert "message_stop" in events[0]
        assert "[DONE]" in events[1]

    async def test_adapt_claude_to_sse(self):
        """Test adapt_claude_to_sse convenience function."""

        async def mock_claude_stream():
            yield {"type": "content_block_start", "index": 0}
            yield {"type": "content_block_delta", "delta": {"text": "Hello"}}
            yield {"type": "content_block_stop", "index": 0}

        events = []
        async for sse_data in adapt_claude_to_sse(
            mock_claude_stream(),
            message_id="test_123",
            model="claude-sonnet-4",
            ping_interval=1.0,
        ):
            events.append(sse_data)

        # Should have multiple events including message_start, content events, message_stop, and done
        assert len(events) >= 4
        assert any("message_start" in event for event in events)
        assert any("content_block_start" in event for event in events)
        assert any("[DONE]" in event for event in events)


class TestStreamingIntegration:
    """Integration tests for streaming functionality."""

    async def test_full_streaming_pipeline(self):
        """Test the complete streaming pipeline from Claude SDK to SSE."""

        # Mock Claude SDK response
        async def mock_claude_stream():
            yield {"type": "content_block_start", "index": 0}
            yield {"type": "content_block_delta", "delta": {"text": "Hello"}}
            yield {"type": "content_block_delta", "delta": {"text": " there!"}}
            yield {"type": "content_block_stop", "index": 0}
            yield {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 3, "output_tokens": 3},
            }

        # Process through the full pipeline
        sse_events = []
        async for sse_data in adapt_claude_to_sse(
            mock_claude_stream(),
            message_id="msg_test_123",
            model="claude-sonnet-4-20250514",
        ):
            sse_events.append(sse_data)

        # Verify we got all expected events
        event_types = []
        for sse_data in sse_events:
            lines = sse_data.strip().split("\n")
            for line in lines:
                if line.startswith("event: "):
                    event_types.append(line.split(": ", 1)[1])

        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        assert "message_stop" in event_types
        assert "done" in event_types

    async def test_error_handling_in_pipeline(self):
        """Test error handling throughout the streaming pipeline."""

        async def error_claude_stream():
            yield {"type": "content_block_start", "index": 0}
            yield {"type": "content_block_delta", "delta": {"text": "Hello"}}
            raise ConnectionError("Connection lost")

        sse_events = []
        async for sse_data in adapt_claude_to_sse(
            error_claude_stream(),
            message_id="msg_error_test",
            model="claude-sonnet-4-20250514",
        ):
            sse_events.append(sse_data)

        # Should have normal events, then error event, then done event
        assert len(sse_events) >= 3

        # Check that we got an error event
        error_found = False
        for sse_data in sse_events:
            if "event: error" in sse_data and "Connection lost" in sse_data:
                error_found = True
                break

        assert error_found, "Error event should be present in stream"

        # Check that stream terminates with done event
        assert "[DONE]" in sse_events[-1]


@pytest.mark.asyncio
class TestAsyncBehavior:
    """Test async behavior and concurrency aspects."""

    async def test_concurrent_streams(self):
        """Test handling multiple concurrent streams."""

        async def mock_claude_stream(stream_id: str):
            for i in range(3):
                yield {
                    "type": "content_block_delta",
                    "delta": {"text": f"Stream {stream_id} chunk {i}"},
                }

        # Create multiple concurrent streams
        tasks = []
        for i in range(3):
            task = asyncio.create_task(
                self._collect_stream_events(
                    adapt_claude_to_sse(
                        mock_claude_stream(str(i)),
                        message_id=f"msg_{i}",
                        model="claude-sonnet-4",
                    )
                )
            )
            tasks.append(task)

        # Wait for all streams to complete
        results = await asyncio.gather(*tasks)

        # Verify all streams completed successfully
        assert len(results) == 3
        for result in results:
            assert len(result) > 0
            assert any("[DONE]" in event for event in result)

    async def _collect_stream_events(self, stream):
        """Helper to collect all events from a stream."""
        events = []
        async for event in stream:
            events.append(event)
        return events

    async def test_stream_cancellation(self):
        """Test proper cleanup when stream is cancelled."""

        async def long_claude_stream():
            for i in range(100):  # Long stream
                yield {"type": "content_block_delta", "delta": {"text": f"chunk {i}"}}
                await asyncio.sleep(0.01)  # Small delay

        events = []
        stream = adapt_claude_to_sse(
            long_claude_stream(), message_id="msg_cancel_test", model="claude-sonnet-4"
        )

        # Collect a few events then cancel
        async for sse_data in stream:
            events.append(sse_data)
            if len(events) >= 3:
                break

        # Should have collected some events
        assert len(events) >= 3
        # Stream should handle cancellation gracefully (no exception raised)
