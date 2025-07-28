#!/bin/bash

# curl_examples.sh
# Comprehensive curl examples for claude-code-anthropic-api
# 
# This script demonstrates how to interact with the Anthropic API compatibility
# server using curl commands. Uses standard Anthropic API format with
# Claude Code SDK as the backend.
# Make sure the server is running: python -m src.main

set -e  # Exit on any error

# Configuration
BASE_URL="http://localhost:8000"
CONTENT_TYPE="Content-Type: application/json"

echo "Claude Code Anthropic API - curl Examples"
echo "========================================="

# Function to check if server is running
check_server() {
    echo "Checking server health..."
    if curl -s -f "$BASE_URL/health" > /dev/null; then
        echo "✓ Server is running"
        echo
    else
        echo "✗ Server is not running or not responding"
        echo "Please start the server with: uvicorn src.main:app --reload"
        exit 1
    fi
}

# Function to pretty print JSON responses
pretty_print() {
    if command -v jq &> /dev/null; then
        jq '.'
    else
        cat
    fi
}

# Check server health
check_server

# Example 1: Health check
echo "1. Health Check"
echo "==============="
echo "Command:"
echo "curl -X GET $BASE_URL/health"
echo
echo "Response:"
curl -s -X GET "$BASE_URL/health" | pretty_print
echo
echo

# Example 2: List available models
echo "2. List Available Models"
echo "========================"
echo "Command:"
echo "curl -X GET $BASE_URL/v1/models"
echo
echo "Response:"
curl -s -X GET "$BASE_URL/v1/models" | pretty_print
echo
echo

# Example 3: Simple non-streaming message
echo "3. Simple Non-Streaming Message"
echo "================================"
cat > /tmp/simple_message.json << 'EOF'
{
  "model": "claude-sonnet-4-20250514",
  "messages": [
    {
      "role": "user",
      "content": "Hello! Can you tell me a joke?"
    }
  ],
  "max_tokens": 150,
  "temperature": 0.7,
  "stream": false
}
EOF

echo "Command:"
echo "curl -X POST $BASE_URL/v1/messages \\"
echo "  -H \"$CONTENT_TYPE\" \\"
echo "  -d @/tmp/simple_message.json"
echo
echo "Response:"
curl -s -X POST "$BASE_URL/v1/messages" \
  -H "$CONTENT_TYPE" \
  -d @/tmp/simple_message.json | pretty_print
echo
echo

# Example 4: Multi-turn conversation
echo "4. Multi-Turn Conversation"
echo "=========================="
cat > /tmp/conversation.json << 'EOF'
{
  "model": "claude-sonnet-4-20250514",
  "messages": [
    {
      "role": "user",
      "content": "What's the capital of Japan?"
    },
    {
      "role": "assistant",
      "content": "The capital of Japan is Tokyo."
    },
    {
      "role": "user",
      "content": "What's the population of that city?"
    }
  ],
  "max_tokens": 200,
  "temperature": 0.3
}
EOF

echo "Command:"
echo "curl -X POST $BASE_URL/v1/messages \\"
echo "  -H \"$CONTENT_TYPE\" \\"
echo "  -d @/tmp/conversation.json"
echo
echo "Response:"
curl -s -X POST "$BASE_URL/v1/messages" \
  -H "$CONTENT_TYPE" \
  -d @/tmp/conversation.json | pretty_print
echo
echo

# Example 5: Streaming response
echo "5. Streaming Response"
echo "====================="
cat > /tmp/streaming_message.json << 'EOF'
{
  "model": "claude-sonnet-4-20250514",
  "messages": [
    {
      "role": "user",
      "content": "Write a short poem about programming."
    }
  ],
  "max_tokens": 200,
  "temperature": 0.8,
  "stream": true
}
EOF

echo "Command:"
echo "curl -X POST $BASE_URL/v1/messages \\"
echo "  -H \"$CONTENT_TYPE\" \\"
echo "  -H \"Accept: text/event-stream\" \\"
echo "  -d @/tmp/streaming_message.json"
echo
echo "Response (Server-Sent Events):"
curl -s -X POST "$BASE_URL/v1/messages" \
  -H "$CONTENT_TYPE" \
  -H "Accept: text/event-stream" \
  -d @/tmp/streaming_message.json
echo
echo

# Example 6: Using system prompt
echo "6. Using System Prompt"
echo "======================"
cat > /tmp/system_prompt.json << 'EOF'
{
  "model": "claude-sonnet-4-20250514",
  "messages": [
    {
      "role": "user",
      "content": "Explain quantum computing"
    }
  ],
  "system": "You are a helpful physics teacher. Explain complex topics in simple terms suitable for high school students.",
  "max_tokens": 300,
  "temperature": 0.5
}
EOF

echo "Command:"
echo "curl -X POST $BASE_URL/v1/messages \\"
echo "  -H \"$CONTENT_TYPE\" \\"
echo "  -d @/tmp/system_prompt.json"
echo
echo "Response:"
curl -s -X POST "$BASE_URL/v1/messages" \
  -H "$CONTENT_TYPE" \
  -d @/tmp/system_prompt.json | pretty_print
echo
echo

# Example 7: Advanced parameters
echo "7. Advanced Parameters"
echo "======================"
cat > /tmp/advanced_params.json << 'EOF'
{
  "model": "claude-sonnet-4-20250514",
  "messages": [
    {
      "role": "user",
      "content": "Generate a list of creative writing prompts. Stop when you reach 5 prompts."
    }
  ],
  "max_tokens": 400,
  "temperature": 0.9,
  "top_p": 0.95,
  "top_k": 40,
  "stop_sequences": ["6."]
}
EOF

echo "Command:"
echo "curl -X POST $BASE_URL/v1/messages \\"
echo "  -H \"$CONTENT_TYPE\" \\"
echo "  -d @/tmp/advanced_params.json"
echo
echo "Response:"
curl -s -X POST "$BASE_URL/v1/messages" \
  -H "$CONTENT_TYPE" \
  -d @/tmp/advanced_params.json | pretty_print
echo
echo

# Example 8: Error handling - Invalid model
echo "8. Error Handling - Invalid Model"
echo "=================================="
cat > /tmp/invalid_model.json << 'EOF'
{
  "model": "invalid-model-name",
  "messages": [
    {
      "role": "user",
      "content": "This should fail"
    }
  ],
  "max_tokens": 100
}
EOF

echo "Command:"
echo "curl -X POST $BASE_URL/v1/messages \\"
echo "  -H \"$CONTENT_TYPE\" \\"
echo "  -d @/tmp/invalid_model.json"
echo
echo "Response (Error):"
curl -s -X POST "$BASE_URL/v1/messages" \
  -H "$CONTENT_TYPE" \
  -d @/tmp/invalid_model.json | pretty_print
echo
echo

# Example 9: Error handling - Invalid role
echo "9. Error Handling - Invalid Role"
echo "================================="
cat > /tmp/invalid_role.json << 'EOF'
{
  "model": "claude-sonnet-4-20250514",
  "messages": [
    {
      "role": "invalid_role",
      "content": "This should also fail"
    }
  ],
  "max_tokens": 100
}
EOF

echo "Command:"
echo "curl -X POST $BASE_URL/v1/messages \\"
echo "  -H \"$CONTENT_TYPE\" \\"
echo "  -d @/tmp/invalid_role.json"
echo
echo "Response (Error):"
curl -s -X POST "$BASE_URL/v1/messages" \
  -H "$CONTENT_TYPE" \
  -d @/tmp/invalid_role.json | pretty_print
echo
echo

# Example 10: Metrics endpoint (if available)
echo "10. Metrics Endpoint"
echo "===================="
echo "Command:"
echo "curl -X GET $BASE_URL/metrics"
echo
echo "Response:"
if curl -s -f "$BASE_URL/metrics" > /dev/null 2>&1; then
    curl -s -X GET "$BASE_URL/metrics"
else
    echo "Metrics endpoint not available or not implemented yet"
fi
echo
echo

# Cleanup temporary files
rm -f /tmp/simple_message.json
rm -f /tmp/conversation.json
rm -f /tmp/streaming_message.json
rm -f /tmp/system_prompt.json
rm -f /tmp/advanced_params.json
rm -f /tmp/invalid_model.json
rm -f /tmp/invalid_role.json

echo "All curl examples completed!"
echo
echo "Tips:"
echo "- Install 'jq' for better JSON formatting: apt-get install jq (Ubuntu) or brew install jq (macOS)"
echo "- Use -v flag with curl for verbose output including headers"
echo "- Use -i flag with curl to include response headers"
echo "- For streaming responses, you can pipe to 'head -n 20' to limit output"
echo
echo "Example with verbose output:"
echo "curl -v -X POST $BASE_URL/v1/messages -H \"$CONTENT_TYPE\" -d '{\"model\":\"claude-sonnet-4-20250514\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":50}'"