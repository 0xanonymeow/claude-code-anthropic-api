"""Path prefix handling middleware for API endpoints"""

import logging
from typing import List, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)


class PathPrefixMiddleware:
    """Middleware to handle requests with or without /v1 prefix"""

    def __init__(self, app: ASGIApp, endpoints_requiring_prefix: Optional[List[str]] = None):
        self.app = app
        self.endpoints_requiring_prefix = endpoints_requiring_prefix or [
            "/messages",
            "/models",
        ]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Get the original path
        original_path = scope["path"]

        # Check if path needs /v1 prefix
        modified_scope = dict(scope)
        if any(
            original_path.startswith(endpoint)
            for endpoint in self.endpoints_requiring_prefix
        ):
            # Modify the scope to use the new path with /v1 prefix
            new_path = f"/v1{original_path}"
            logger.info(f"Auto-prefixing {original_path} to {new_path}")

            modified_scope["path"] = new_path
            modified_scope["raw_path"] = new_path.encode()

        # Custom response handler for 404s
        response_started = False

        async def send_wrapper(message) -> None:
            nonlocal response_started

            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]

                # If we get a 404, provide Anthropic-compatible error response
                if status_code == 404:
                    suggestions = []
                    if original_path.startswith(
                        "/messages"
                    ) and not original_path.startswith("/v1/messages"):
                        suggestions.append("Try /v1/messages instead")
                    elif original_path.startswith(
                        "/models"
                    ) and not original_path.startswith("/v1/models"):
                        suggestions.append("Try /v1/models instead")
                    elif original_path == "/":
                        suggestions.extend(
                            ["Available endpoints: /v1/messages, /v1/models, /health"]
                        )

                    if suggestions:
                        message_text = (
                            f"Endpoint '{original_path}' not found. "
                            + " | ".join(suggestions)
                        )
                    else:
                        message_text = f"Endpoint '{original_path}' not found"

                    error_response = {
                        "type": "error",
                        "error": {"type": "not_found_error", "message": message_text},
                    }

                    # Create custom 404 response
                    response = JSONResponse(status_code=404, content=error_response)
                    await response(scope, receive, send)
                    return

            await send(message)

        await self.app(modified_scope, receive, send_wrapper)
