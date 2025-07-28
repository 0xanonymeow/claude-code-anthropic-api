"""
Unit tests for the /v1/models API endpoint.

Tests the models API endpoint functionality including model listing,
individual model retrieval, error handling, and Claude Code SDK integration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.api.models import router, list_models, get_model
from src.models.anthropic import Model, ModelsResponse
from src.core.claude_client import ClaudeClient, get_claude_client


class TestModelsAPI:
    """Test the models API endpoints."""

    @pytest.fixture
    def mock_claude_client(self):
        """Create a mock Claude client for testing."""
        client = AsyncMock(spec=ClaudeClient)
        return client

    @pytest.fixture
    def sample_models(self):
        """Create sample models for testing."""
        return [
            Model(
                id="claude-3-5-sonnet-20241022",
                display_name="Claude 3.5 Sonnet",
                created_at="2024-10-22T00:00:00Z"
            ),
            Model(
                id="claude-3-5-haiku-20241022",
                display_name="Claude 3.5 Haiku",
                created_at="2024-10-22T00:00:00Z"
            ),
            Model(
                id="claude-3-opus-20240229",
                display_name="Claude 3 Opus",
                created_at="2024-02-29T00:00:00Z"
            ),
            Model(
                id="claude-sonnet-4-20250514",
                display_name="Claude 3 Sonnet",
                created_at="2024-02-29T00:00:00Z"
            ),
            Model(
                id="claude-3-haiku-20240307",
                display_name="Claude 3 Haiku",
                created_at="2024-03-07T00:00:00Z"
            )
        ]

    @pytest.mark.asyncio
    async def test_list_models_success(self, mock_claude_client, sample_models):
        """Test successful model listing."""
        # Setup mock
        mock_claude_client.get_available_models.return_value = sample_models
        
        # Call the endpoint
        response = await list_models(claude_client=mock_claude_client)
        
        # Verify response
        assert isinstance(response, ModelsResponse)
        assert len(response.data) == 5
        assert response.data == sample_models
        assert response.has_more is False
        assert response.first_id == "claude-3-5-sonnet-20241022"
        assert response.last_id == "claude-3-haiku-20240307"
        
        # Verify client was called
        mock_claude_client.get_available_models.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_models_empty_list(self, mock_claude_client):
        """Test model listing with empty model list."""
        # Setup mock to return empty list
        mock_claude_client.get_available_models.return_value = []
        
        # Call the endpoint
        response = await list_models(claude_client=mock_claude_client)
        
        # Verify response
        assert isinstance(response, ModelsResponse)
        assert len(response.data) == 0
        assert response.has_more is False
        assert response.first_id is None
        assert response.last_id is None

    @pytest.mark.asyncio
    async def test_list_models_single_model(self, mock_claude_client):
        """Test model listing with single model."""
        single_model = [
            Model(
                id="claude-sonnet-4-20250514",
                display_name="Claude 3 Sonnet",
                created_at="2024-02-29T00:00:00Z"
            )
        ]
        
        # Setup mock
        mock_claude_client.get_available_models.return_value = single_model
        
        # Call the endpoint
        response = await list_models(claude_client=mock_claude_client)
        
        # Verify response
        assert isinstance(response, ModelsResponse)
        assert len(response.data) == 1
        assert response.data == single_model
        assert response.has_more is False
        assert response.first_id == "claude-sonnet-4-20250514"
        assert response.last_id == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_list_models_claude_client_error(self, mock_claude_client):
        """Test model listing when Claude client raises an error."""
        # Setup mock to raise exception
        mock_claude_client.get_available_models.side_effect = Exception("Claude Code SDK connection failed")
        
        # Call the endpoint and expect HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await list_models(claude_client=mock_claude_client)
        
        # Verify exception details
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail["type"] == "error"
        assert exc_info.value.detail["error"]["type"] == "api_error"
        assert "Failed to retrieve models" in exc_info.value.detail["error"]["message"]
        assert "Claude Code SDK connection failed" in exc_info.value.detail["error"]["message"]

    @pytest.mark.asyncio
    async def test_get_model_success(self, mock_claude_client, sample_models):
        """Test successful individual model retrieval."""
        # Setup mock
        mock_claude_client.get_available_models.return_value = sample_models
        
        # Call the endpoint
        model_id = "claude-sonnet-4-20250514"
        response = await get_model(model_id=model_id, claude_client=mock_claude_client)
        
        # Verify response
        assert isinstance(response, Model)
        assert response.id == model_id
        assert response.display_name == "Claude 3 Sonnet"
        assert response.created_at == "2024-02-29T00:00:00Z"
        
        # Verify client was called
        mock_claude_client.get_available_models.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_model_not_found(self, mock_claude_client, sample_models):
        """Test individual model retrieval for non-existent model."""
        # Setup mock
        mock_claude_client.get_available_models.return_value = sample_models
        
        # Call the endpoint with non-existent model ID
        model_id = "non-existent-model"
        with pytest.raises(HTTPException) as exc_info:
            await get_model(model_id=model_id, claude_client=mock_claude_client)
        
        # Verify exception details
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail["type"] == "error"
        assert exc_info.value.detail["error"]["type"] == "not_found_error"
        assert f"Model '{model_id}' not found" in exc_info.value.detail["error"]["message"]

    @pytest.mark.asyncio
    async def test_get_model_empty_list(self, mock_claude_client):
        """Test individual model retrieval when no models are available."""
        # Setup mock to return empty list
        mock_claude_client.get_available_models.return_value = []
        
        # Call the endpoint
        model_id = "claude-sonnet-4-20250514"
        with pytest.raises(HTTPException) as exc_info:
            await get_model(model_id=model_id, claude_client=mock_claude_client)
        
        # Verify exception details
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail["error"]["type"] == "not_found_error"

    @pytest.mark.asyncio
    async def test_get_model_claude_client_error(self, mock_claude_client):
        """Test individual model retrieval when Claude client raises an error."""
        # Setup mock to raise exception
        mock_claude_client.get_available_models.side_effect = Exception("Connection timeout")
        
        # Call the endpoint and expect HTTPException
        model_id = "claude-sonnet-4-20250514"
        with pytest.raises(HTTPException) as exc_info:
            await get_model(model_id=model_id, claude_client=mock_claude_client)
        
        # Verify exception details
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail["type"] == "error"
        assert exc_info.value.detail["error"]["type"] == "api_error"
        assert "Failed to retrieve model information" in exc_info.value.detail["error"]["message"]

    @pytest.mark.asyncio
    async def test_get_model_case_sensitivity(self, mock_claude_client, sample_models):
        """Test that model retrieval is case-sensitive."""
        # Setup mock
        mock_claude_client.get_available_models.return_value = sample_models
        
        # Call the endpoint with different case
        model_id = "CLAUDE-3-SONNET-20240229"  # Uppercase
        with pytest.raises(HTTPException) as exc_info:
            await get_model(model_id=model_id, claude_client=mock_claude_client)
        
        # Verify it's treated as not found
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail["error"]["type"] == "not_found_error"

    def test_models_response_structure(self, sample_models):
        """Test ModelsResponse structure and serialization."""
        response = ModelsResponse(
            data=sample_models,
            has_more=False,
            first_id=sample_models[0].id,
            last_id=sample_models[-1].id
        )
        
        # Test serialization
        json_data = response.model_dump()
        assert "data" in json_data
        assert "has_more" in json_data
        assert "first_id" in json_data
        assert "last_id" in json_data
        assert len(json_data["data"]) == 5
        assert json_data["has_more"] is False
        
        # Test that each model has required fields
        for model_data in json_data["data"]:
            assert "id" in model_data
            assert "type" in model_data
            assert "display_name" in model_data
            assert model_data["type"] == "model"

    def test_model_structure(self):
        """Test individual Model structure and serialization."""
        model = Model(
            id="claude-sonnet-4-20250514",
            display_name="Claude 3 Sonnet",
            created_at="2024-02-29T00:00:00Z"
        )
        
        # Test serialization
        json_data = model.model_dump()
        assert json_data["id"] == "claude-sonnet-4-20250514"
        assert json_data["type"] == "model"
        assert json_data["display_name"] == "Claude 3 Sonnet"
        assert json_data["created_at"] == "2024-02-29T00:00:00Z"

    def test_model_without_created_at(self):
        """Test Model creation without created_at field."""
        model = Model(
            id="claude-sonnet-4-20250514",
            display_name="Claude 3 Sonnet"
        )
        
        # Test serialization
        json_data = model.model_dump()
        assert json_data["id"] == "claude-sonnet-4-20250514"
        assert json_data["type"] == "model"
        assert json_data["display_name"] == "Claude 3 Sonnet"
        assert json_data["created_at"] is None


class TestModelsAPIIntegration:
    """Integration tests for the models API with FastAPI."""

    @pytest.fixture
    def app(self):
        """Create FastAPI app with models router for testing."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_list_models_endpoint_integration(self, app, client):
        """Test /v1/models endpoint integration."""
        # Setup mock client
        mock_claude_client = AsyncMock()
        mock_models = [
            Model(id="claude-sonnet-4-20250514", display_name="Claude 3 Sonnet"),
            Model(id="claude-3-haiku-20240307", display_name="Claude 3 Haiku")
        ]
        mock_claude_client.get_available_models.return_value = mock_models
        
        # Override the dependency
        def get_mock_client():
            return mock_claude_client
        
        app.dependency_overrides[get_claude_client] = get_mock_client
        
        try:
            # Make request
            response = client.get("/v1/models")
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert "has_more" in data
            assert "first_id" in data
            assert "last_id" in data
            assert len(data["data"]) == 2
            assert data["has_more"] is False
        finally:
            # Clean up
            app.dependency_overrides.clear()

    def test_get_model_endpoint_integration(self, app, client):
        """Test /v1/models/{model_id} endpoint integration."""
        # Setup mock client
        mock_claude_client = AsyncMock()
        mock_models = [
            Model(id="claude-sonnet-4-20250514", display_name="Claude 3 Sonnet"),
            Model(id="claude-3-haiku-20240307", display_name="Claude 3 Haiku")
        ]
        mock_claude_client.get_available_models.return_value = mock_models
        
        # Override the dependency
        def get_mock_client():
            return mock_claude_client
        
        app.dependency_overrides[get_claude_client] = get_mock_client
        
        try:
            # Make request
            response = client.get("/v1/models/claude-sonnet-4-20250514")
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "claude-sonnet-4-20250514"
            assert data["type"] == "model"
            assert data["display_name"] == "Claude 3 Sonnet"
        finally:
            # Clean up
            app.dependency_overrides.clear()

    def test_get_model_not_found_integration(self, app, client):
        """Test /v1/models/{model_id} endpoint with non-existent model."""
        # Setup mock client
        mock_claude_client = AsyncMock()
        mock_claude_client.get_available_models.return_value = []
        
        # Override the dependency
        def get_mock_client():
            return mock_claude_client
        
        app.dependency_overrides[get_claude_client] = get_mock_client
        
        try:
            # Make request
            response = client.get("/v1/models/non-existent-model")
            
            # Verify response
            assert response.status_code == 404
            data = response.json()
            assert data["detail"]["type"] == "error"
            assert data["detail"]["error"]["type"] == "not_found_error"
            assert "not found" in data["detail"]["error"]["message"]
        finally:
            # Clean up
            app.dependency_overrides.clear()

    def test_list_models_error_integration(self, app, client):
        """Test /v1/models endpoint with Claude client error."""
        # Setup mock client to raise exception
        mock_claude_client = AsyncMock()
        mock_claude_client.get_available_models.side_effect = Exception("SDK Error")
        
        # Override the dependency
        def get_mock_client():
            return mock_claude_client
        
        app.dependency_overrides[get_claude_client] = get_mock_client
        
        try:
            # Make request
            response = client.get("/v1/models")
            
            # Verify response
            assert response.status_code == 500
            data = response.json()
            assert data["detail"]["type"] == "error"
            assert data["detail"]["error"]["type"] == "api_error"
            assert "Failed to retrieve models" in data["detail"]["error"]["message"]
        finally:
            # Clean up
            app.dependency_overrides.clear()


class TestModelMapping:
    """Test model ID mapping functionality."""

    @pytest.fixture
    def mock_claude_client(self):
        """Create a mock Claude client for testing."""
        client = AsyncMock(spec=ClaudeClient)
        return client

    @pytest.mark.asyncio
    async def test_model_id_consistency(self, mock_claude_client):
        """Test that model IDs are consistent between listing and individual retrieval."""
        sample_models = [
            Model(id="claude-sonnet-4-20250514", display_name="Claude 3 Sonnet"),
            Model(id="claude-3-haiku-20240307", display_name="Claude 3 Haiku")
        ]
        
        # Setup mock
        mock_claude_client.get_available_models.return_value = sample_models
        
        # Get models list
        models_response = await list_models(claude_client=mock_claude_client)
        
        # Test individual retrieval for each model
        for model in models_response.data:
            individual_model = await get_model(model_id=model.id, claude_client=mock_claude_client)
            assert individual_model.id == model.id
            assert individual_model.display_name == model.display_name
            assert individual_model.created_at == model.created_at

    def test_anthropic_model_format(self):
        """Test that models follow Anthropic's expected format."""
        model = Model(
            id="claude-sonnet-4-20250514",
            display_name="Claude 3 Sonnet",
            created_at="2024-02-29T00:00:00Z"
        )
        
        # Verify required fields
        assert model.id is not None
        assert model.type == "model"
        assert model.display_name is not None
        
        # Verify ID format (should match Anthropic's naming convention)
        assert model.id.startswith("claude-")
        assert "-" in model.id  # Should contain hyphens
        
        # Verify serialization matches expected format
        json_data = model.model_dump()
        expected_fields = {"id", "type", "display_name", "created_at"}
        assert set(json_data.keys()) == expected_fields