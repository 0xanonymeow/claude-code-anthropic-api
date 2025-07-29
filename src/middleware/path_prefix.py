"""Path prefix handling middleware for API endpoints"""

import logging
from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class PathPrefixMiddleware(BaseHTTPMiddleware):
    """Middleware to handle requests with or without /v1 prefix"""
    
    def __init__(self, app, endpoints_requiring_prefix: list = None):
        super().__init__(app)
        self.endpoints_requiring_prefix = endpoints_requiring_prefix or ["/messages", "/models"]
    
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable]):
        # Get the original path
        original_path = request.url.path
        
        # Check if path needs /v1 prefix
        if any(original_path.startswith(endpoint) for endpoint in self.endpoints_requiring_prefix):
            # Modify the request scope to use the new path with /v1 prefix
            new_path = f"/v1{original_path}"
            logger.info(f"Auto-prefixing {original_path} to {new_path}")
            
            request.scope["path"] = new_path
            request.scope["raw_path"] = new_path.encode()
        
        # Add custom 404 handler with helpful suggestions
        response = await call_next(request)
        
        # If we get a 404 and the path doesn't start with /v1, provide helpful suggestions
        if response.status_code == 404 and not original_path.startswith("/v1"):
            suggestions = []
            if original_path.startswith("/messages"):
                suggestions.append("Try /v1/messages instead")
            elif original_path.startswith("/models"):
                suggestions.append("Try /v1/models instead")
            elif original_path == "/":
                suggestions.extend(["Available endpoints: /v1/messages, /v1/models, /health"])
            
            if suggestions:
                message = f"Endpoint '{original_path}' not found. " + " | ".join(suggestions)
                
                error_response = {
                    "type": "error",
                    "error": {
                        "type": "not_found_error",
                        "message": message
                    }
                }
                return JSONResponse(
                    status_code=404,
                    content=error_response
                )
        
        return response