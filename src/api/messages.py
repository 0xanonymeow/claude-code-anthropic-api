"""
Messages API endpoint implementation.

This module implements the /v1/messages endpoint that provides Anthropic API
compatibility for chat completions, supporting both streaming and non-streaming
responses with proper request validation and error handling.
"""

import logging
from typing import Union

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from ..core.claude_client import ClaudeClient, get_claude_client
from ..models.anthropic import (
    AnthropicError,
    ErrorResponse,
    ErrorType,
    MessageRequest,
    MessageResponse,
)
from ..utils.streaming import create_sse_stream

logger = logging.getLogger(__name__)

# Create router for messages endpoints
router = APIRouter(prefix="/v1", tags=["messages"])


async def get_client() -> ClaudeClient:
    """Dependency to get Claude client instance."""
    return get_claude_client()


def create_error_response(error_type: ErrorType, message: str, status_code: int = 400) -> HTTPException:
    """
    Create an HTTPException with Anthropic-compatible error format.
    
    Args:
        error_type: The type of error
        message: Error message
        status_code: HTTP status code
        
    Returns:
        HTTPException with proper error format
    """
    error_response = ErrorResponse(
        error=AnthropicError(type=error_type, message=message)
    )
    return HTTPException(
        status_code=status_code,
        detail=error_response.model_dump()
    )


@router.post("/messages", response_model=None)
async def create_message(
    request: MessageRequest,
    claude_client: ClaudeClient = Depends(get_client)
):
    """
    Create a message using Claude Code SDK.
    
    This endpoint supports both streaming and non-streaming responses based on
    the 'stream' parameter in the request. It validates the request using
    Pydantic models and returns responses in Anthropic's exact format.
    
    Args:
        request: The message request containing model, messages, and parameters
        claude_client: Claude client instance for processing requests
        
    Returns:
        MessageResponse for non-streaming requests or StreamingResponse for streaming
        
    Raises:
        HTTPException: For validation errors, model errors, or processing failures
    """
    try:
        logger.info(f"Received message request for model: {request.model}, streaming: {request.stream}")
        
        # Validate model is supported
        available_models = await claude_client.get_available_models()
        supported_model_ids = [model.id for model in available_models]
        
        if request.model not in supported_model_ids:
            raise create_error_response(
                ErrorType.INVALID_REQUEST_ERROR,
                f"Model '{request.model}' is not supported. Available models: {', '.join(supported_model_ids)}",
                status_code=400
            )
        
        # Handle streaming vs non-streaming requests
        if request.stream:
            logger.info("Processing streaming request")
            return await _handle_streaming_request(request, claude_client)
        else:
            logger.info("Processing non-streaming request")
            return await _handle_non_streaming_request(request, claude_client)
            
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except ValidationError as e:
        logger.error(f"Request validation error: {str(e)}")
        raise create_error_response(
            ErrorType.INVALID_REQUEST_ERROR,
            f"Request validation failed: {str(e)}",
            status_code=400
        )
    except Exception as e:
        logger.error(f"Unexpected error processing message request: {str(e)}")
        raise create_error_response(
            ErrorType.API_ERROR,
            f"Internal server error: {str(e)}",
            status_code=500
        )


async def _handle_non_streaming_request(
    request: MessageRequest,
    claude_client: ClaudeClient
) -> MessageResponse:
    """
    Handle non-streaming message request.
    
    Args:
        request: The message request
        claude_client: Claude client instance
        
    Returns:
        MessageResponse with the complete response
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        response = await claude_client.create_message(request)
        logger.info(f"Non-streaming message completed: {response.id}")
        return response
        
    except Exception as e:
        logger.error(f"Error in non-streaming request: {str(e)}")
        # Map Claude client errors to appropriate HTTP errors
        if "not found" in str(e).lower():
            raise create_error_response(
                ErrorType.NOT_FOUND_ERROR,
                str(e),
                status_code=404
            )
        elif "connection" in str(e).lower():
            raise create_error_response(
                ErrorType.API_ERROR,
                "Failed to connect to Claude Code SDK",
                status_code=503
            )
        else:
            raise create_error_response(
                ErrorType.API_ERROR,
                str(e),
                status_code=500
            )


async def _handle_streaming_request(
    request: MessageRequest,
    claude_client: ClaudeClient
) -> StreamingResponse:
    """
    Handle streaming message request.
    
    Args:
        request: The message request with stream=True
        claude_client: Claude client instance
        
    Returns:
        StreamingResponse with Server-Sent Events
        
    Raises:
        HTTPException: If streaming setup fails
    """
    try:
        # Get the streaming generator from Claude client
        stream_generator = claude_client.create_message_stream(request)
        
        # Set up proper SSE headers
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
        
        logger.info("Starting streaming response")
        return StreamingResponse(
            stream_generator,
            media_type="text/event-stream",
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"Error setting up streaming request: {str(e)}")
        raise create_error_response(
            ErrorType.API_ERROR,
            f"Failed to initialize streaming: {str(e)}",
            status_code=500
        )


# Note: Validation error handling is done by FastAPI automatically
# Custom validation error handling would need to be implemented at the app level


# Health check endpoint for the messages API
@router.get("/messages/health")
async def messages_health_check(claude_client: ClaudeClient = Depends(get_client)):
    """
    Health check endpoint for the messages API.
    
    Args:
        claude_client: Claude client instance
        
    Returns:
        Health status information
    """
    try:
        health_status = await claude_client.health_check()
        return {
            "status": "healthy",
            "endpoint": "/v1/messages",
            "claude_sdk": health_status
        }
    except Exception as e:
        logger.error(f"Messages health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "endpoint": "/v1/messages",
            "error": str(e)
        }