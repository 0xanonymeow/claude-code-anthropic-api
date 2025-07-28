"""
Modern logging utilities using loguru for beautiful, clean logs.

Provides enhanced logging with structured context, request tracking,
performance metrics, and clean formatting.
"""

import time
import uuid
from contextvars import ContextVar
from functools import wraps
from typing import Any, Dict, Optional, Union

from fastapi import Request, Response
from loguru import logger


# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)


def get_logger(name: str = "app"):
    """
    Get loguru logger with context binding.
    
    Args:
        name: Logger name (for identification)
        
    Returns:
        Loguru logger instance
    """
    return logger.bind(component=name)


def format_context(**kwargs) -> str:
    """Format context for beautiful text logs."""
    if not kwargs:
        return ""
    
    parts = []
    
    # Duration first (most important)
    if 'duration_seconds' in kwargs and kwargs['duration_seconds'] is not None:
        parts.append(f"{kwargs['duration_seconds']:.2f}s")
    
    # Status code with color hints
    if 'status_code' in kwargs and kwargs['status_code'] is not None:
        parts.append(str(kwargs['status_code']))
    
    # HTTP method and path
    if 'method' in kwargs and 'path' in kwargs:
        parts.append(f"{kwargs['method']} {kwargs['path']}")
    
    # Request ID (short)
    if 'request_id' in kwargs and kwargs['request_id']:
        parts.append(f"({kwargs['request_id'][:8]})")
    
    if parts:
        return f" <dim>({' · '.join(parts)})</dim>"
    return ""


def get_icon(level: str) -> str:
    """No icons - clean text only."""
    return ""


class LoguruLogger:
    """Enhanced logger wrapper for loguru with structured logging."""
    
    def __init__(self, name: str = "app"):
        """Initialize logger with name."""
        self.name = name
        self._start_times: Dict[str, float] = {}
    
    def _get_context(self) -> Dict[str, Any]:
        """Get current logging context."""
        context = {}
        
        request_id = request_id_var.get()
        if request_id:
            context['request_id'] = request_id
        
        user_id = user_id_var.get()
        if user_id:
            context['user_id'] = user_id
        
        return context
    
    def _log(self, level: str, message: str, **kwargs):
        """Internal log method with context."""
        context = self._get_context()
        context.update(kwargs)
        
        logger.bind(**context).log(level, message)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._log("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._log("CRITICAL", message, **kwargs)
    
    def start_timer(self, operation: str) -> str:
        """Start a timer for an operation."""
        timer_id = f"{operation}_{uuid.uuid4().hex[:8]}"
        self._start_times[timer_id] = time.time()
        
        self.debug(f"Started {operation}", operation=operation, timer_id=timer_id)
        return timer_id
    
    def stop_timer(self, timer_id: str, operation: str, **kwargs) -> float:
        """Stop a timer and log the duration."""
        if timer_id not in self._start_times:
            self.warning(f"Timer not found: {timer_id}")
            return 0.0
        
        duration = time.time() - self._start_times.pop(timer_id)
        
        self.info(
            f"Completed {operation}",
            operation=operation,
            timer_id=timer_id,
            duration_seconds=duration,
            **kwargs
        )
        
        return duration
    
    def log_api_request(self, request: Request, **kwargs) -> None:
        """Log API request."""
        self.info(
            f"Request {request.method} {request.url.path}",
            method=request.method,
            path=request.url.path,
            client_host=request.client.host if request.client else 'unknown',
            **kwargs
        )
    
    def log_api_response(self, request: Request, response: Response, duration: float, **kwargs) -> None:
        """Log API response."""
        level = "ERROR" if response.status_code >= 500 else "WARNING" if response.status_code >= 400 else "INFO"
        
        self._log(
            level,
            f"Response {response.status_code}",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_seconds=duration,
            **kwargs
        )
    
    def log_claude_sdk_request(self, operation: str, model: str, **kwargs) -> None:
        """Log Claude SDK request."""
        self.info(f"Claude SDK: {operation}", operation=operation, model=model, **kwargs)
    
    def log_claude_sdk_response(self, operation: str, model: str, duration: float, success: bool = True, **kwargs) -> None:
        """Log Claude SDK response."""
        level = "INFO" if success else "ERROR"
        status = "completed" if success else "failed"
        
        self._log(
            level,
            f"Claude SDK {operation} {status}",
            operation=operation,
            model=model,
            duration_seconds=duration,
            success=success,
            **kwargs
        )
    
    def log_performance_metric(self, metric_name: str, value: Union[int, float], unit: str = "", **kwargs) -> None:
        """Log performance metric."""
        self.info(
            f"Metric: {metric_name} = {value} {unit}",
            metric_name=metric_name,
            metric_value=value,
            metric_unit=unit,
            **kwargs
        )


# Utility functions for common logging scenarios

def log_request_validation_error(log: LoguruLogger, request: Request, validation_errors: list) -> None:
    """Log request validation error."""
    log.warning(
        f"Validation failed for {request.method} {request.url.path}",
        method=request.method,
        path=request.url.path,
        error_count=len(validation_errors)
    )


def log_claude_sdk_error(log: LoguruLogger, operation: str, error: Exception, **kwargs) -> None:
    """Log Claude SDK error."""
    log.error(
        f"Claude SDK error in {operation}: {str(error)}",
        operation=operation,
        error_type=error.__class__.__name__,
        error_message=str(error),
        **kwargs
    )


def log_health_check(log: LoguruLogger, component: str, status: str, **kwargs) -> None:
    """Log health check result."""
    level = "INFO" if status == 'healthy' else "WARNING"
    log._log(level, f"Health check: {component} is {status}", component=component, status=status, **kwargs)


def log_function_call(log: Optional[LoguruLogger] = None):
    """Decorator to log function calls with timing."""
    def decorator(func):
        nonlocal log
        if log is None:
            log = LoguruLogger(func.__module__)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            timer_id = log.start_timer(func.__name__)
            try:
                result = await func(*args, **kwargs)
                log.stop_timer(timer_id, func.__name__, success=True)
                return result
            except Exception as e:
                log.stop_timer(timer_id, func.__name__, success=False, error=str(e))
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            timer_id = log.start_timer(func.__name__)
            try:
                result = func(*args, **kwargs)
                log.stop_timer(timer_id, func.__name__, success=True)
                return result
            except Exception as e:
                log.stop_timer(timer_id, func.__name__, success=False, error=str(e))
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class RequestContextMiddleware:
    """Middleware for managing request context."""
    
    def __init__(self, app):
        """Initialize middleware."""
        self.app = app
    
    async def __call__(self, scope, receive, send):
        """Process request with context."""
        if scope["type"] == "http":
            # Generate request ID
            request_id = str(uuid.uuid4())
            request_id_var.set(request_id)
            
            # Add request ID to headers
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    headers.append([b"x-request-id", request_id.encode()])
                    message["headers"] = headers
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)