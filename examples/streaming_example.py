#!/usr/bin/env python3
"""
Streaming usage example for claude-code-anthropic-api

This example demonstrates streaming chat completions using the Anthropic API
compatibility server. Uses Server-Sent Events (SSE) to receive real-time 
response updates from Claude Code SDK.
"""

import json
import requests
from typing import Iterator, Dict, Any
import time


def create_message_stream(
    messages: list,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1000,
    temperature: float = 0.7,
    base_url: str = "http://localhost:8000"
) -> Iterator[Dict[str, Any]]:
    """
    Send a streaming message to the claude-code-anthropic-api server.
    
    Args:
        messages: List of message objects with 'role' and 'content'
        model: Model identifier to use
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 to 1.0)
        base_url: Base URL of the API server
        
    Yields:
        Dictionary containing each streaming event
    """
    url = f"{base_url}/v1/messages"
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    
    try:
        with requests.post(url, json=payload, headers=headers, stream=True) as response:
            response.raise_for_status()
            
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    # Parse Server-Sent Events format
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        
                        if data_str.strip() == "[DONE]":
                            break
                            
                        try:
                            event_data = json.loads(data_str)
                            yield event_data
                        except json.JSONDecodeError:
                            print(f"Warning: Could not parse JSON: {data_str}")
                            continue
                            
    except requests.exceptions.RequestException as e:
        print(f"Error making streaming request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        raise


def print_streaming_response(events: Iterator[Dict[str, Any]]):
    """
    Print streaming events in a user-friendly format.
    
    Args:
        events: Iterator of streaming event dictionaries
    """
    full_response = ""
    start_time = time.time()
    token_count = 0
    
    print("Assistant: ", end="", flush=True)
    
    try:
        for event in events:
            if event.get("type") == "content_block_delta":
                # Extract text content from the delta
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    print(text, end="", flush=True)
                    full_response += text
                    token_count += 1
                    
            elif event.get("type") == "message_stop":
                print("\n")
                break
                
            elif event.get("type") == "error":
                print(f"\nError: {event.get('error', {}).get('message', 'Unknown error')}")
                break
                
    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n--- Streaming Stats ---")
    print(f"Duration: {duration:.2f}s")
    print(f"Approximate tokens: {token_count}")
    if duration > 0:
        print(f"Tokens per second: {token_count/duration:.1f}")


def main():
    """Main example function demonstrating streaming API usage."""
    
    print("Claude Code Anthropic API - Streaming Examples")
    print("=" * 50)
    
    # Example 1: Simple streaming conversation
    print("\n1. Simple streaming response:")
    try:
        messages = [
            {"role": "user", "content": "Write a short story about a robot learning to paint."}
        ]
        
        events = create_message_stream(messages, temperature=0.8)
        print_streaming_response(events)
        
    except Exception as e:
        print(f"Error in streaming example 1: {e}")
    
    # Example 2: Interactive conversation simulation
    print("\n2. Multi-turn streaming conversation:")
    try:
        conversation = [
            {"role": "user", "content": "What are the benefits of renewable energy?"}
        ]
        
        print("User: What are the benefits of renewable energy?")
        events = create_message_stream(conversation, temperature=0.3, max_tokens=300)
        print_streaming_response(events)
        
        # Simulate follow-up question
        print("\nUser: Which renewable energy source is most efficient?")
        conversation.extend([
            {"role": "assistant", "content": "[Previous response would be here]"},
            {"role": "user", "content": "Which renewable energy source is most efficient?"}
        ])
        
        events = create_message_stream(conversation, temperature=0.3, max_tokens=200)
        print_streaming_response(events)
        
    except Exception as e:
        print(f"Error in streaming example 2: {e}")
    
    # Example 3: Creative writing with higher temperature
    print("\n3. Creative writing with streaming:")
    try:
        messages = [
            {"role": "user", "content": "Write a haiku about artificial intelligence."}
        ]
        
        events = create_message_stream(messages, temperature=0.9, max_tokens=100)
        print_streaming_response(events)
        
    except Exception as e:
        print(f"Error in streaming example 3: {e}")


def demonstrate_raw_sse_parsing():
    """Demonstrate raw Server-Sent Events parsing for educational purposes."""
    print("\n4. Raw SSE parsing demonstration:")
    
    messages = [
        {"role": "user", "content": "Count from 1 to 5."}
    ]
    
    try:
        events = create_message_stream(messages, max_tokens=50)
        
        print("Raw streaming events:")
        for i, event in enumerate(events):
            print(f"Event {i+1}: {json.dumps(event, indent=2)}")
            if i >= 5:  # Limit output for demo
                print("... (truncated)")
                break
                
    except Exception as e:
        print(f"Error in raw SSE demo: {e}")


def compare_streaming_vs_non_streaming():
    """Compare streaming vs non-streaming response times."""
    print("\n5. Streaming vs Non-streaming comparison:")
    
    messages = [
        {"role": "user", "content": "Explain the concept of machine learning in simple terms."}
    ]
    
    # Non-streaming request
    print("Non-streaming request...")
    start_time = time.time()
    try:
        import sys
        sys.path.append('examples')
        from basic_usage import create_message
        
        response = create_message(messages, max_tokens=200)
        non_streaming_time = time.time() - start_time
        print(f"Non-streaming completed in {non_streaming_time:.2f}s")
        print(f"Response length: {len(response['content'][0]['text'])} characters")
        
    except Exception as e:
        print(f"Non-streaming error: {e}")
        non_streaming_time = None
    
    # Streaming request
    print("\nStreaming request...")
    start_time = time.time()
    try:
        events = create_message_stream(messages, max_tokens=200)
        
        char_count = 0
        first_token_time = None
        
        for event in events:
            if event.get("type") == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                    char_count += len(delta.get("text", ""))
            elif event.get("type") == "message_stop":
                break
        
        total_streaming_time = time.time() - start_time
        
        print(f"Streaming completed in {total_streaming_time:.2f}s")
        print(f"First token received in {first_token_time:.2f}s" if first_token_time else "No tokens received")
        print(f"Response length: {char_count} characters")
        
        if non_streaming_time:
            print(f"Time to first token advantage: {non_streaming_time - first_token_time:.2f}s")
            
    except Exception as e:
        print(f"Streaming error: {e}")


if __name__ == "__main__":
    # Check server health first
    import sys
    sys.path.append('examples')
    from basic_usage import check_server_health
    
    if check_server_health():
        main()
        demonstrate_raw_sse_parsing()
        compare_streaming_vs_non_streaming()
    else:
        print("\nPlease start the server first:")
        print("  uvicorn src.main:app --reload")