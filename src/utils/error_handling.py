"""
Centralized error handling utilities for the Claude Code Anthropic API server.

This module provides comprehensive error handling with Anthropic-compatible error responses,
error mapping utilities, HTTP status code mapping, and structured error logging.
"""

import logging
import traceback
from typing import Any, Dict, Optional, Type, Union
from enum import Enum

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from ..models.anthropic import (
    ErrorResponse,
    AnthropicError,
    ErrorType
)


logger = logging.getLogger(__name__)


class HTTPStatusCode(int, Enum):
    """HTTP status codes for error responses."""
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    REQUEST_TIMEOUT = 408
    CONFLICT = 409
    PAYLOAD_TOO_LARGE = 413
    UNPROCESSABLE_ENTITY = 422
    TOO_MANY_REQUESTS = 429
    INTERNAL_SERVER_ERROR = 500
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504


class ErrorCategory(str, Enum):
    """Categories of errors for better organization."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RESOURCE = "resource"
    RATE_LIMIT = "rate_limit"
    EXTERNAL_SERVICE = "external_service"
    INTERNAL = "internal"
    NETWORK = "network"


class ErrorMapper:
    """Maps different types of errors to Anthropic-compatible error responses."""
    
    # Mapping of exception types to error types and HTTP status codes
    ERROR_MAPPINGS = {
        # Validation errors
        ValidationError: (ErrorType.INVALID_REQUEST_ERROR, HTTPStatusCode.BAD_REQUEST, ErrorCategory.VALIDATION),
        ValueError: (ErrorType.INVALID_REQUEST_ERROR, HTTPStatusCode.BAD_REQUEST, ErrorCategory.VALIDATION),
        TypeError: (ErrorType.INVALID_REQUEST_ERROR, HTTPStatusCode.BAD_REQUEST, ErrorCategory.VALIDATION),
        
        # HTTP errors
        HTTPException: (None, None, ErrorCategory.INTERNAL),  # Special handling
        
        # Network/connection errors
        ConnectionError: (ErrorType.API_ERROR, HTTPStatusCode.SERVICE_UNAVAILABLE, ErrorCategory.NETWORK),
        TimeoutError: (ErrorType.API_ERROR, HTTPStatusCode.GATEWAY_TIMEOUT, ErrorCategory.NETWORK),
        
        # Resource errors
        FileNotFoundError: (ErrorType.NOT_FOUND_ERROR, HTTPStatusCode.NOT_FOUND, ErrorCategory.RESOURCE),
        PermissionError: (ErrorType.PERMISSION_ERROR, HTTPStatusCode.FORBIDDEN, ErrorCategory.AUTHORIZATION),
        
        # Generic errors
        Exception: (ErrorType.API_ERROR, HTTPStatusCode.INTERNAL_SERVER_ERROR, ErrorCategory.INTERNAL),
    }
    
    # Claude Code SDK specific error mappings
    CLAUDE_SDK_ERROR_MAPPINGS = {
        "CLINotFoundError": (ErrorType.API_ERROR, HTTPStatusCode.SERVICE_UNAVAILABLE, "Claude Code CLI not found"),
        "CLIConnectionError": (ErrorType.API_ERROR, HTTPStatusCode.SERVICE_UNAVAILABLE, "Failed to connect to Claude Code CLI"),
        "CLIJSONDecodeError": (ErrorType.API_ERROR, HTTPStatusCode.BAD_GATEWAY, "Invalid response from Claude Code CLI"),
        "ProcessError": (ErrorType.API_ERROR, HTTPStatusCode.BAD_GATEWAY, "Claude Code process error"),
        "ClaudeSDKError": (ErrorType.API_ERROR, HTTPStatusCode.BAD_GATEWAY, "Claude Code SDK error"),
    }
    
    @classmethod
    def map_exception_to_error_response(
        cls,
        exception: Exception,
        request: Optional[Request] = None,
        include_details: bool = True
    ) -> tuple[ErrorResponse, int]:
        """
        Map an exception to an Anthropic-compatible error response.
        
        Args:
            exception: The exception to map
            request: Optional FastAPI request object for context
            include_details: Whether to include detailed error information
            
        Returns:
            Tuple of (ErrorResponse, HTTP status code)
        """
        # Handle HTTPException specially
        if isinstance(exception, HTTPException):
            return cls._handle_http_exception(exception, include_details)
        
        # Handle Claude Code SDK errors by class name
        exception_class_name = exception.__class__.__name__
        if exception_class_name in cls.CLAUDE_SDK_ERROR_MAPPINGS:
            error_type, status_code, default_message = cls.CLAUDE_SDK_ERROR_MAPPINGS[exception_class_name]
            message = str(exception) if str(exception) else default_message
            
            error_response = ErrorResponse(
                error=AnthropicError(
                    type=error_type,
                    message=message
                )
            )
            return error_response, status_code.value
        
        # Handle other exceptions using type mapping
        exception_type = type(exception)
        error_type, status_code, category = cls._get_error_mapping(exception_type)
        
        # Create error message
        message = cls._create_error_message(exception, category, include_details)
        
        # Log the error with appropriate level
        cls._log_error(exception, request, category, status_code)
        
        error_response = ErrorResponse(
            error=AnthropicError(
                type=error_type,
                message=message
            )
        )
        
        return error_response, status_code.value
    
    @classmethod
    def _handle_http_exception(cls, exception: HTTPException, include_details: bool) -> tuple[ErrorResponse, int]:
        """Handle HTTPException specifically."""
        # Map HTTP status codes to Anthropic error types
        status_to_error_type = {
            400: ErrorType.INVALID_REQUEST_ERROR,
            401: ErrorType.AUTHENTICATION_ERROR,
            403: ErrorType.PERMISSION_ERROR,
            404: ErrorType.NOT_FOUND_ERROR,
            429: ErrorType.RATE_LIMIT_ERROR,
            503: ErrorType.OVERLOADED_ERROR,
        }
        
        error_type = status_to_error_type.get(exception.status_code, ErrorType.API_ERROR)
        
        # If detail is already in Anthropic format, return as-is
        if isinstance(exception.detail, dict) and "error" in exception.detail:
            return ErrorResponse(**exception.detail), exception.status_code
        
        error_response = ErrorResponse(
            error=AnthropicError(
                type=error_type,
                message=str(exception.detail)
            )
        )
        
        return error_response, exception.status_code
    
    @classmethod
    def _get_error_mapping(cls, exception_type: Type[Exception]) -> tuple[ErrorType, HTTPStatusCode, ErrorCategory]:
        """Get error mapping for exception type."""
        # Check direct mapping first
        if exception_type in cls.ERROR_MAPPINGS:
            return cls.ERROR_MAPPINGS[exception_type]
        
        # Check parent classes
        for mapped_type, mapping in cls.ERROR_MAPPINGS.items():
            if issubclass(exception_type, mapped_type):
                return mapping
        
        # Default mapping
        return ErrorType.API_ERROR, HTTPStatusCode.INTERNAL_SERVER_ERROR, ErrorCategory.INTERNAL
    
    @classmethod
    def _create_error_message(
        cls,
        exception: Exception,
        category: ErrorCategory,
        include_details: bool
    ) -> str:
        """Create user-friendly error message."""
        base_message = str(exception) if str(exception) else "An error occurred"
        
        if not include_details:
            # Return generic messages for production
            generic_messages = {
                ErrorCategory.VALIDATION: "Invalid request parameters",
                ErrorCategory.AUTHENTICATION: "Authentication required",
                ErrorCategory.AUTHORIZATION: "Access denied",
                ErrorCategory.RESOURCE: "Resource not found",
                ErrorCategory.RATE_LIMIT: "Rate limit exceeded",
                ErrorCategory.EXTERNAL_SERVICE: "External service unavailable",
                ErrorCategory.NETWORK: "Network error occurred",
                ErrorCategory.INTERNAL: "Internal server error",
            }
            return generic_messages.get(category, "An error occurred")
        
        return base_message
    
    @classmethod
    def _log_error(
        cls,
        exception: Exception,
        request: Optional[Request],
        category: ErrorCategory,
        status_code: HTTPStatusCode
    ) -> None:
        """Log error with appropriate level and context."""
        # Determine log level based on error category and status code
        if status_code.value >= 500:
            log_level = logging.ERROR
        elif status_code.value >= 400:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO
        
        # Create log context
        context = {
            "exception_type": exception.__class__.__name__,
            "category": category.value,
            "status_code": status_code.value,
            "error_message": str(exception)
        }
        
        if request:
            context.update({
                "method": request.method,
                "url": str(request.url),
                "client_host": request.client.host if request.client else "unknown"
            })
        
        # Log with stack trace for server errors
        if status_code.value >= 500:
            logger.log(
                log_level,
                f"Error {status_code.value}: {exception.__class__.__name__} - {str(exception)}",
                extra=context,
                exc_info=True
            )
        else:
            logger.log(
                log_level,
                f"Error {status_code.value}: {exception.__class__.__name__} - {str(exception)}",
                extra=context
            )


class ErrorHandler:
    """Centralized error handler for the application."""
    
    def __init__(self, debug: bool = False):
        """
        Initialize error handler.
        
        Args:
            debug: Whether to include detailed error information
        """
        self.debug = debug
        self.mapper = ErrorMapper()
    
    async def handle_error(
        self,
        exception: Exception,
        request: Optional[Request] = None
    ) -> JSONResponse:
        """
        Handle an error and return appropriate JSON response.
        
        Args:
            exception: The exception to handle
            request: Optional FastAPI request object
            
        Returns:
            JSONResponse with error details
        """
        error_response, status_code = self.mapper.map_exception_to_error_response(
            exception,
            request,
            include_details=self.debug
        )
        
        return JSONResponse(
            status_code=status_code,
            content=error_response.model_dump()
        )
    
    def create_validation_error_response(
        self,
        validation_errors: list,
        request: Optional[Request] = None
    ) -> JSONResponse:
        """
        Create error response for validation errors.
        
        Args:
            validation_errors: List of validation errors from Pydantic
            request: Optional FastAPI request object
            
        Returns:
            JSONResponse with validation error details
        """
        # Extract validation error details
        error_details = []
        for error in validation_errors:
            field_path = " -> ".join(str(loc) for loc in error["loc"])
            error_details.append(f"{field_path}: {error['msg']}")
        
        error_message = f"Request validation failed: {'; '.join(error_details)}"
        
        if request:
            logger.warning(
                f"Validation error for {request.method} {request.url.path}: {error_message}",
                extra={
                    "method": request.method,
                    "url": str(request.url),
                    "client_host": request.client.host if request.client else "unknown",
                    "validation_errors": validation_errors
                }
            )
        
        error_response = ErrorResponse(
            error=AnthropicError(
                type=ErrorType.INVALID_REQUEST_ERROR,
                message=error_message
            )
        )
        
        return JSONResponse(
            status_code=HTTPStatusCode.BAD_REQUEST.value,
            content=error_response.model_dump()
        )


# Utility functions for common error scenarios

def create_not_found_error(resource: str, identifier: str = "") -> ErrorResponse:
    """Create a not found error response."""
    message = f"{resource} not found"
    if identifier:
        message += f": {identifier}"
    
    return ErrorResponse(
        error=AnthropicError(
            type=ErrorType.NOT_FOUND_ERROR,
            message=message
        )
    )


def create_rate_limit_error(retry_after: Optional[int] = None) -> ErrorResponse:
    """Create a rate limit error response."""
    message = "Rate limit exceeded"
    if retry_after:
        message += f". Retry after {retry_after} seconds"
    
    return ErrorResponse(
        error=AnthropicError(
            type=ErrorType.RATE_LIMIT_ERROR,
            message=message
        )
    )


def create_service_unavailable_error(service: str = "Claude Code SDK") -> ErrorResponse:
    """Create a service unavailable error response."""
    return ErrorResponse(
        error=AnthropicError(
            type=ErrorType.API_ERROR,
            message=f"{service} is currently unavailable"
        )
    )


def create_invalid_model_error(model: str, available_models: list = None) -> ErrorResponse:
    """Create an invalid model error response."""
    message = f"Invalid model: {model}"
    if available_models:
        message += f". Available models: {', '.join(available_models)}"
    
    return ErrorResponse(
        error=AnthropicError(
            type=ErrorType.INVALID_REQUEST_ERROR,
            message=message
        )
    )