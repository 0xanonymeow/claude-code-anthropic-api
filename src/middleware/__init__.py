"""Middleware modules for the FastAPI application"""

from .path_prefix import PathPrefixMiddleware

__all__ = ["PathPrefixMiddleware"]