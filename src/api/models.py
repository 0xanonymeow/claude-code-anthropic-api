"""
Models API endpoint for Anthropic API compatibility.

This module implements the /v1/models endpoint that returns information about
available Claude Code models in Anthropic-compatible format.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..core.claude_client import ClaudeClient, get_claude_client
from ..models.anthropic import Model, ModelsResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models(
    claude_client: ClaudeClient = Depends(get_claude_client),
) -> ModelsResponse:
    """
    List available models.

    Returns information about available Claude Code models in Anthropic-compatible format.
    This endpoint provides model IDs, display names, and capabilities that can
    be used with the /v1/messages endpoint.

    Args:
        claude_client: Claude Code SDK client instance

    Returns:
        ModelsResponse: List of available models with metadata

    Raises:
        HTTPException: If unable to retrieve model information
    """
    try:
        logger.info("Retrieving available models")

        # Get models from Claude Code SDK
        models = await claude_client.get_available_models()

        # Create response with pagination metadata
        response = ModelsResponse(
            data=models,
            has_more=False,  # We return all models at once
            first_id=models[0].id if models else None,
            last_id=models[-1].id if models else None,
        )

        logger.info(f"Successfully retrieved {len(models)} models")
        return response

    except Exception as e:
        logger.error(f"Error retrieving models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"Failed to retrieve models: {str(e)}",
                },
            },
        )


@router.get("/v1/models/{model_id}", response_model=Model)
async def get_model(
    model_id: str,
    claude_client: ClaudeClient = Depends(get_claude_client),
) -> Model:
    """
    Retrieve information about a specific model.

    Args:
        model_id: The ID of the model to retrieve
        claude_client: Claude Code SDK client instance

    Returns:
        Model: Information about the requested model

    Raises:
        HTTPException: If model is not found or error occurs
    """
    try:
        logger.info(f"Retrieving model information for: {model_id}")

        # Get all available models
        models = await claude_client.get_available_models()

        # Find the requested model
        for model in models:
            if model.id == model_id:
                logger.info(f"Found model: {model_id}")
                return model

        # Model not found
        logger.warning(f"Model not found: {model_id}")
        raise HTTPException(
            status_code=404,
            detail={
                "type": "error",
                "error": {
                    "type": "not_found_error",
                    "message": f"Model '{model_id}' not found",
                },
            },
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error retrieving model {model_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"Failed to retrieve model information: {str(e)}",
                },
            },
        )
