"""
Tests for error handling utilities.

This module tests the centralized error handling system including error mapping,
HTTP status code mapping, and Anthropic-compatible error response generation.
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.utils.error_handling import (
    ErrorMapper,
    ErrorHandler,
    HTTPStatusCode,
    ErrorCategory,
    create_not_found_error,
    create_rate_limit_error,
    create_service_unavailable_error,
    create_invalid_model_error
)
from src.models.anthropic import ErrorType, ErrorResponse, AnthropicError


class TestErrorMapper:
    """Test the ErrorMapper class."""
    
    def test_map_validation_error(self):
        """Test mapping of ValidationError to Anthropic format."""
        error = ValueError("Invalid parameter value")
        error_response, status_code = ErrorMapper.map_exception_to_error_response(error)
        
        assert isinstance(error_response, ErrorResponse)
        assert error_response.error.type == ErrorType.INVALID_REQUEST_ERROR
        assert "Invalid parameter value" in error_response.error.message
        assert status_code == HTTPStatusCode.BAD_REQUEST.value
    
    def test_map_http_exception(self):
        """Test mapping of HTTPException to Anthropic format."""
        error = HTTPException(status_code=404, detail="Resource not found")
        error_response, status_code = ErrorMapper.map_exception_to_error_response(error)
        
        assert isinstance(error_response, ErrorResponse)
        assert error_response.error.type == ErrorType.NOT_FOUND_ERROR
        assert error_response.error.message == "Resource not found"
        assert status_code == 404
    
    def test_map_http_exception_with_anthropic_detail(self):
        """Test HTTPException with already formatted Anthropic detail."""
        anthropic_detail = {
            "error": {
                "type": "rate_limit_error",
                "message": "Rate limit exceeded"
            }
        }
        error = HTTPException(status_code=429, detail=anthropic_detail)
        error_response, status_code = ErrorMapper.map_exception_to_error_response(error)
        
        assert isinstance(error_response, ErrorResponse)
        assert status_code == 429
    
    def test_map_claude_sdk_error(self):
        """Test mapping of Claude SDK specific errors."""
        # Create a mock error class with the right name
        class CLIConnectionError(Exception):
            pass
        
        error = CLIConnectionError("Connection failed")
        
        error_response, status_code = ErrorMapper.map_exception_to_error_response(error)
        
        assert isinstance(error_response, ErrorResponse)
        assert error_response.error.type == ErrorType.API_ERROR
        assert "Connection failed" in error_response.error.message
        assert status_code == HTTPStatusCode.SERVICE_UNAVAILABLE.value
    
    def test_map_generic_exception(self):
        """Test mapping of generic exceptions."""
        error = Exception("Something went wrong")
        error_response, status_code = ErrorMapper.map_exception_to_error_response(error)
        
        assert isinstance(error_response, ErrorResponse)
        assert error_response.error.type == ErrorType.API_ERROR
        assert "Something went wrong" in error_response.error.message
        assert status_code == HTTPStatusCode.INTERNAL_SERVER_ERROR.value
    
    def test_map_exception_without_details(self):
        """Test mapping exceptions without detailed error information."""
        error = ValueError("Sensitive internal error")
        error_response, status_code = ErrorMapper.map_exception_to_error_response(
            error, include_details=False
        )
        
        assert isinstance(error_response, ErrorResponse)
        assert error_response.error.message == "Invalid request parameters"
        assert status_code == HTTPStatusCode.BAD_REQUEST.value
    
    @patch('src.utils.error_handling.logger')
    def test_error_logging(self, mock_logger):
        """Test that errors are properly logged."""
        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock()
        request.url.path = "/v1/messages"
        request.client = Mock()
        request.client.host = "127.0.0.1"
        
        error = ValueError("Test error")
        ErrorMapper.map_exception_to_error_response(error, request)
        
        # Verify logging was called
        mock_logger.log.assert_called()


class TestErrorHandler:
    """Test the ErrorHandler class."""
    
    @pytest.fixture
    def error_handler(self):
        """Create an ErrorHandler instance for testing."""
        return ErrorHandler(debug=True)
    
    @pytest.fixture
    def mock_request(self):
        """Create a mock request for testing."""
        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock()
        request.url.path = "/v1/messages"
        request.client = Mock()
        request.client.host = "127.0.0.1"
        return request
    
    async def test_handle_error(self, error_handler, mock_request):
        """Test error handling with JSONResponse creation."""
        error = ValueError("Test error")
        response = await error_handler.handle_error(error, mock_request)
        
        assert isinstance(response, JSONResponse)
        assert response.status_code == HTTPStatusCode.BAD_REQUEST.value
    
    def test_create_validation_error_response(self, error_handler, mock_request):
        """Test creation of validation error responses."""
        validation_errors = [
            {"loc": ["field1"], "msg": "Field is required"},
            {"loc": ["field2", "subfield"], "msg": "Invalid value"}
        ]
        
        response = error_handler.create_validation_error_response(
            validation_errors, mock_request
        )
        
        assert isinstance(response, JSONResponse)
        assert response.status_code == HTTPStatusCode.BAD_REQUEST.value
        
        # Check response content
        content = response.body.decode()
        assert "field1: Field is required" in content
        assert "field2 -> subfield: Invalid value" in content
    
    def test_debug_mode_error_details(self):
        """Test that debug mode includes detailed error information."""
        debug_handler = ErrorHandler(debug=True)
        production_handler = ErrorHandler(debug=False)
        
        error = Exception("Detailed internal error")
        
        debug_response, _ = debug_handler.mapper.map_exception_to_error_response(
            error, include_details=True
        )
        production_response, _ = production_handler.mapper.map_exception_to_error_response(
            error, include_details=False
        )
        
        assert "Detailed internal error" in debug_response.error.message
        assert debug_response.error.message != production_response.error.message


class TestUtilityFunctions:
    """Test utility functions for creating common error responses."""
    
    def test_create_not_found_error(self):
        """Test creation of not found error responses."""
        error_response = create_not_found_error("Model", "claude-sonnet-4")
        
        assert isinstance(error_response, ErrorResponse)
        assert error_response.error.type == ErrorType.NOT_FOUND_ERROR
        assert "Model not found: claude-sonnet-4" in error_response.error.message
    
    def test_create_rate_limit_error(self):
        """Test creation of rate limit error responses."""
        error_response = create_rate_limit_error(retry_after=60)
        
        assert isinstance(error_response, ErrorResponse)
        assert error_response.error.type == ErrorType.RATE_LIMIT_ERROR
        assert "Rate limit exceeded" in error_response.error.message
        assert "Retry after 60 seconds" in error_response.error.message
    
    def test_create_service_unavailable_error(self):
        """Test creation of service unavailable error responses."""
        error_response = create_service_unavailable_error("Test Service")
        
        assert isinstance(error_response, ErrorResponse)
        assert error_response.error.type == ErrorType.API_ERROR
        assert "Test Service is currently unavailable" in error_response.error.message
    
    def test_create_invalid_model_error(self):
        """Test creation of invalid model error responses."""
        available_models = ["claude-sonnet-4", "claude-haiku-4"]
        error_response = create_invalid_model_error("invalid-model", available_models)
        
        assert isinstance(error_response, ErrorResponse)
        assert error_response.error.type == ErrorType.INVALID_REQUEST_ERROR
        assert "Invalid model: invalid-model" in error_response.error.message
        assert "claude-sonnet-4" in error_response.error.message
        assert "claude-haiku-4" in error_response.error.message


class TestErrorCategories:
    """Test error categorization and mapping."""
    
    def test_error_categories_enum(self):
        """Test that all error categories are properly defined."""
        categories = [
            ErrorCategory.VALIDATION,
            ErrorCategory.AUTHENTICATION,
            ErrorCategory.AUTHORIZATION,
            ErrorCategory.RESOURCE,
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.EXTERNAL_SERVICE,
            ErrorCategory.INTERNAL,
            ErrorCategory.NETWORK
        ]
        
        assert len(categories) == 8
        assert all(isinstance(cat, ErrorCategory) for cat in categories)
    
    def test_http_status_codes_enum(self):
        """Test that HTTP status codes are properly defined."""
        status_codes = [
            HTTPStatusCode.BAD_REQUEST,
            HTTPStatusCode.UNAUTHORIZED,
            HTTPStatusCode.FORBIDDEN,
            HTTPStatusCode.NOT_FOUND,
            HTTPStatusCode.TOO_MANY_REQUESTS,
            HTTPStatusCode.INTERNAL_SERVER_ERROR,
            HTTPStatusCode.SERVICE_UNAVAILABLE
        ]
        
        assert all(isinstance(code, HTTPStatusCode) for code in status_codes)
        assert HTTPStatusCode.BAD_REQUEST.value == 400
        assert HTTPStatusCode.INTERNAL_SERVER_ERROR.value == 500


class TestErrorMappingIntegration:
    """Test integration between different error handling components."""
    
    @pytest.fixture
    def error_handler(self):
        """Create an ErrorHandler for integration testing."""
        return ErrorHandler(debug=True)
    
    async def test_end_to_end_error_handling(self, error_handler):
        """Test complete error handling flow from exception to JSON response."""
        # Create a mock request
        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock()
        request.url.path = "/v1/messages"
        request.client = Mock()
        request.client.host = "127.0.0.1"
        
        # Test different types of errors
        test_cases = [
            (ValueError("Invalid input"), HTTPStatusCode.BAD_REQUEST),
            (ConnectionError("Network error"), HTTPStatusCode.SERVICE_UNAVAILABLE),
            (FileNotFoundError("File missing"), HTTPStatusCode.NOT_FOUND),
            (PermissionError("Access denied"), HTTPStatusCode.FORBIDDEN),
            (Exception("Generic error"), HTTPStatusCode.INTERNAL_SERVER_ERROR)
        ]
        
        for error, expected_status in test_cases:
            response = await error_handler.handle_error(error, request)
            
            assert isinstance(response, JSONResponse)
            assert response.status_code == expected_status.value
            
            # Verify response structure
            content = response.body.decode()
            assert "error" in content
            assert "type" in content
            assert "message" in content
    
    def test_error_response_serialization(self):
        """Test that error responses can be properly serialized."""
        error_response = ErrorResponse(
            error=AnthropicError(
                type=ErrorType.INVALID_REQUEST_ERROR,
                message="Test error message"
            )
        )
        
        # Test model_dump
        data = error_response.model_dump()
        assert data["type"] == "error"
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["message"] == "Test error message"
        
        # Test JSON serialization
        json_str = error_response.model_dump_json()
        assert "invalid_request_error" in json_str
        assert "Test error message" in json_str


@pytest.mark.asyncio
class TestAsyncErrorHandling:
    """Test asynchronous error handling scenarios."""
    
    async def test_async_error_handler(self):
        """Test error handler with async operations."""
        handler = ErrorHandler(debug=True)
        
        # Simulate an async error scenario
        async def failing_operation():
            raise ValueError("Async operation failed")
        
        try:
            await failing_operation()
        except Exception as e:
            response = await handler.handle_error(e)
            assert isinstance(response, JSONResponse)
            assert response.status_code == HTTPStatusCode.BAD_REQUEST.value
    
    async def test_concurrent_error_handling(self):
        """Test error handling under concurrent conditions."""
        import asyncio
        
        handler = ErrorHandler(debug=True)
        
        async def create_error(error_msg):
            error = ValueError(error_msg)
            return await handler.handle_error(error)
        
        # Create multiple concurrent error handling tasks
        tasks = [
            create_error(f"Error {i}")
            for i in range(10)
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # Verify all responses are valid
        assert len(responses) == 10
        for response in responses:
            assert isinstance(response, JSONResponse)
            assert response.status_code == HTTPStatusCode.BAD_REQUEST.value