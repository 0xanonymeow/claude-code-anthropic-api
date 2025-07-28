#!/usr/bin/env python3
"""
Basic usage example for claude-code-anthropic-api

This example demonstrates non-streaming chat completions using the Anthropic
API compatibility server. Uses standard Anthropic API format with Claude Code
SDK as the backend.
"""

import json
import requests
from typing import Dict, Any


def create_message(
    messages: list,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1000,
    temperature: float = 0.7,
    base_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """
    Send a message to the claude-code-anthropic-api server.
    
    Args:
        messages: List of message objects with 'role' and 'content'
        model: Model identifier to use
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 to 1.0)
        base_url: Base URL of the API server
        
    Returns:
        Dictionary containing the API response
    """
    url = f"{base_url}/v1/messages"
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        raise


def main():
    """Main example function demonstrating various API usage patterns."""
    
    print("Claude Code Anthropic API - Basic Usage Examples")
    print("=" * 50)
    
    # Example 1: Simple single message
    print("\n1. Simple single message:")
    try:
        messages = [
            {"role": "user", "content": "Hello! Can you explain what you are?"}
        ]
        
        response = create_message(messages)
        
        print(f"Response ID: {response['id']}")
        print(f"Model: {response['model']}")
        print(f"Content: {response['content'][0]['text']}")
        print(f"Usage: {response['usage']}")
        
    except Exception as e:
        print(f"Error in example 1: {e}")
    
    # Example 2: Multi-turn conversation
    print("\n2. Multi-turn conversation:")
    try:
        messages = [
            {"role": "user", "content": "What's the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "What's the population of that city?"}
        ]
        
        response = create_message(messages, temperature=0.3)
        
        print(f"Assistant: {response['content'][0]['text']}")
        print(f"Stop reason: {response['stop_reason']}")
        
    except Exception as e:
        print(f"Error in example 2: {e}")
    
    # Example 3: Using system prompt
    print("\n3. Using system prompt:")
    try:
        messages = [
            {"role": "user", "content": "Explain quantum computing"}
        ]
        
        response = create_message(
            messages,
            max_tokens=500,
            temperature=0.5
        )
        
        print(f"Response: {response['content'][0]['text'][:200]}...")
        print(f"Tokens used: Input={response['usage']['input_tokens']}, Output={response['usage']['output_tokens']}")
        
    except Exception as e:
        print(f"Error in example 3: {e}")
    
    # Example 4: Error handling
    print("\n4. Error handling example:")
    try:
        # This should cause a validation error
        invalid_messages = [
            {"role": "invalid_role", "content": "This should fail"}
        ]
        
        response = create_message(invalid_messages)
        print("Unexpected success!")
        
    except requests.exceptions.HTTPError as e:
        print(f"Expected error caught: {e.response.status_code}")
        error_data = e.response.json()
        print(f"Error type: {error_data.get('error', {}).get('type')}")
        print(f"Error message: {error_data.get('error', {}).get('message')}")
    except Exception as e:
        print(f"Other error: {e}")


def check_server_health(base_url: str = "http://localhost:8000"):
    """Check if the API server is running and healthy."""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        response.raise_for_status()
        print(f"✓ Server is healthy: {response.json()}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"✗ Server health check failed: {e}")
        print("Make sure the server is running with: uvicorn src.main:app --reload")
        return False


def list_available_models(base_url: str = "http://localhost:8000"):
    """List available models from the API."""
    try:
        response = requests.get(f"{base_url}/v1/models")
        response.raise_for_status()
        models_data = response.json()
        
        print("\nAvailable models:")
        for model in models_data.get('data', []):
            print(f"  - {model['id']}: {model.get('name', 'N/A')}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error listing models: {e}")


if __name__ == "__main__":
    # Check server health first
    if check_server_health():
        list_available_models()
        main()
    else:
        print("\nPlease start the server first:")
        print("  uvicorn src.main:app --reload")