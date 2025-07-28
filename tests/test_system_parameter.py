"""Tests for system parameter validation in MessageRequest"""

import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_system_parameter_as_string():
    """Test system parameter with valid string value"""
    response = client.post("/v1/messages", json={
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
        "system": "You are a helpful assistant."
    })
    # Should not fail validation (status should not be 400 due to system param)
    assert response.status_code != 422
    if response.status_code == 400:
        error_data = response.json()
        # Make sure it's not a system parameter validation error
        assert "system" not in error_data.get("error", {}).get("message", "").lower()


def test_system_parameter_as_null():
    """Test system parameter with null value"""
    response = client.post("/v1/messages", json={
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
        "system": None
    })
    # Should not fail validation
    assert response.status_code != 422


def test_system_parameter_missing():
    """Test request without system parameter"""
    response = client.post("/v1/messages", json={
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100
    })
    # Should not fail validation
    assert response.status_code != 422


def test_system_parameter_empty_string():
    """Test system parameter with empty string"""
    response = client.post("/v1/messages", json={
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
        "system": ""
    })
    # Should not fail validation
    assert response.status_code != 422


def test_system_parameter_invalid_type():
    """Test system parameter with invalid type (should fail)"""
    response = client.post("/v1/messages", json={
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
        "system": 123  # Invalid type
    })
    # Should fail validation
    assert response.status_code == 422
    error_data = response.json()
    assert "system" in str(error_data).lower()


def test_system_parameter_long_string():
    """Test system parameter with long string"""
    long_system = "You are a helpful assistant. " * 100
    response = client.post("/v1/messages", json={
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
        "system": long_system
    })
    # Should not fail validation due to system parameter
    assert response.status_code != 422