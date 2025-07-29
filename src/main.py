"""
Main application for Claude Code Anthropic API compatibility server.

This module creates an HTTP server that provides Anthropic API compatibility,
enabling existing Anthropic API client code to work with Claude Code SDK.
Includes middleware, exception handlers, and endpoints for seamless integration.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .api.messages import router as messages_router
from .api.models import router as models_router
from .core.claude_client import close_claude_client, get_claude_client
from .core.config import configure_logging, get_settings
from .models.anthropic import AnthropicError, ErrorResponse, ErrorType
from .utils.error_handling import ErrorHandler, ErrorMapper
from .utils.loguru_utils import (
    LoguruLogger,
    RequestContextMiddleware,
    log_health_check,
    log_request_validation_error,
    request_id_var,
)

# Configure logging before creating the app
settings = get_settings()
configure_logging(settings)
logger = LoguruLogger("main")

# Initialize error handler
error_handler = ErrorHandler(debug=settings.debug)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    
    Handles Claude Code SDK initialization on startup and cleanup on shutdown.
    """
    # Startup  
    logger.info(f"Starting Claude Code API server v{settings.api_version}")
    
    try:
        # Initialize Claude client
        claude_client = get_claude_client(settings)
        logger.info("Claude Code SDK initialized")
        
        # Perform initial health check
        health_status = await claude_client.health_check()
        
        if health_status["status"] != "healthy":
            logger.warning("Claude SDK health check failed - continuing startup")
        else:
            logger.info("Server ready")
            
    except Exception as e:
        logger.error("Failed to initialize Claude Code SDK - continuing startup",
            event='claude_sdk_init_failed',
            error=str(e),
            error_type=e.__class__.__name__
        )
        # Continue startup even if Claude SDK fails - let health checks handle it
    
    yield
    
    # Shutdown
    try:
        await close_claude_client()
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))


# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    debug=settings.debug,
    lifespan=lifespan
)


# Add path prefix middleware (before other middleware)
from .middleware.path_prefix import PathPrefixMiddleware

app.add_middleware(PathPrefixMiddleware)

# Add request context middleware for tracking
app.add_middleware(RequestContextMiddleware)

# Add CORS middleware only if origins are configured
cors_config = settings.get_cors_config()
if cors_config["allow_origins"]:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config["allow_origins"],
        allow_credentials=cors_config["allow_credentials"],
        allow_methods=cors_config["allow_methods"],
        allow_headers=cors_config["allow_headers"],
    )


# Request/Response logging middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """
    Enhanced middleware for logging HTTP requests and responses.
    
    Logs request details, response status, processing time, and performance metrics
    for debugging and monitoring purposes.
    """
    start_time = time.time()
    
    # Process request
    try:
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Only log errors and slow requests in production
        if response.status_code >= 400:
            logger.log_api_response(request, response, process_time)
        elif process_time > 1.0:  # Log slow requests (>1s)
            logger.warning("Slow request detected",
                method=request.method,
                path=request.url.path,
                duration_seconds=process_time,
                status_code=response.status_code
            )
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        
        logger.error(
            f"Request failed: {request.method} {request.url.path}",
            method=request.method,
            path=request.url.path,
            duration_seconds=process_time,
            error=str(e),
            error_type=e.__class__.__name__
        )
        raise


# Enhanced exception handlers using centralized error handling
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle Pydantic validation errors with enhanced logging and error format.
    
    Args:
        request: The FastAPI request object
        exc: The validation error exception
        
    Returns:
        JSONResponse with Anthropic-compatible error format
    """
    # Log validation error with structured logging
    log_request_validation_error(logger, request, exc.errors())
    
    # Use centralized error handler
    return error_handler.create_validation_error_response(exc.errors(), request)


@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    """
    Handle general Pydantic validation errors with enhanced error handling.
    
    Args:
        request: The FastAPI request object
        exc: The validation error exception
        
    Returns:
        JSONResponse with Anthropic-compatible error format
    """
    return await error_handler.handle_error(exc, request)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle HTTP exceptions with enhanced error handling and logging.
    
    Args:
        request: The FastAPI request object
        exc: The HTTP exception
        
    Returns:
        JSONResponse with error details
    """
    return await error_handler.handle_error(exc, request)


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle unexpected exceptions with comprehensive error handling and logging.
    
    Args:
        request: The FastAPI request object
        exc: The exception
        
    Returns:
        JSONResponse with error details
    """
    return await error_handler.handle_error(exc, request)


# Include API routers
app.include_router(messages_router)
app.include_router(models_router)


# Health check endpoint with enhanced logging
@app.get("/health")
async def health_check():
    """
    Enhanced health check endpoint with comprehensive logging and monitoring.
    
    Returns:
        Dict containing overall health status and component details
    """
    if not settings.health_check_enabled:
        logger.warning("Health check endpoint accessed but disabled")
        return JSONResponse(
            status_code=404,
            content={"detail": "Health check endpoint is disabled"}
        )
    
    try:
        # Check Claude Code SDK health
        claude_client = get_claude_client()
        claude_health = await claude_client.health_check()
        
        # Determine overall health status
        overall_status = "healthy" if claude_health["status"] == "healthy" else "unhealthy"
        status_code = 200 if overall_status == "healthy" else 503
        
        health_data = {
            "status": overall_status,
            "version": settings.api_version,
            "components": {
                "claude_code_sdk": claude_health
            }
        }
        
        return JSONResponse(
            status_code=status_code,
            content=health_data
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        
        health_data = {
            "status": "unhealthy",
            "version": settings.api_version,
            "error": str(e)
        }
        
        return JSONResponse(
            status_code=503,
            content=health_data
        )


# Enhanced metrics endpoint
@app.get("/metrics")
async def metrics():
    """
    Enhanced metrics endpoint with comprehensive monitoring data.
    
    Returns:
        Dict containing detailed application metrics
    """
    if not settings.metrics_enabled:
        logger.warning("Metrics endpoint accessed but disabled")
        return JSONResponse(
            status_code=404,
            content={"detail": "Metrics endpoint is disabled"}
        )
    
    try:
        # Get basic metrics
        claude_client = get_claude_client()
        claude_health = await claude_client.health_check()
        
        metrics_data = {
            "version": settings.api_version,
            "claude_code_sdk": {
                "status": claude_health["status"]
            }
        }
        
        return JSONResponse(content=metrics_data)
        
    except Exception as e:
        logger.error("Metrics collection failed", error=str(e))
        
        metrics_data = {
            "version": settings.api_version,
            "error": str(e),
            "claude_code_sdk": {
                "status": "unhealthy"
            }
        }
        
        return JSONResponse(
            status_code=503,
            content=metrics_data
        )


# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint providing basic API information.
    
    Returns:
        Dict containing API information and available endpoints
    """
    return {
        "name": settings.api_title,
        "description": settings.api_description,
        "version": settings.api_version,
        "endpoints": {
            "messages": "/v1/messages",
            "models": "/v1/models",
            "health": "/health",
            "metrics": "/metrics",
            "docs": "/docs"
        },
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn

    # Run the server with suppressed uvicorn logs
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="critical",  # Suppress uvicorn logs
        access_log=False      # Disable access logs
    )