# Claude Code Anthropic API

> ⚠️ **EARLY DEVELOPMENT WARNING** ⚠️  
> This project is in **early development stage**. Expect bugs, incomplete features, and many untested use cases and scenarios. **Use at your own risk and with caution.** Not recommended for production use without thorough testing in your specific environment.

## What is this?

This is a **FastAPI-based local server** that provides Anthropic's `/v1/messages` API interface while using Claude Code SDK as the backend. It acts as a bridge between Anthropic's familiar REST API format and Claude Code's local tooling capabilities.

**Purpose:**
- Enables existing Anthropic API client code to work with Claude Code SDK locally
- Provides API compatibility for applications already built on Anthropic's `/v1/messages` endpoint
- Useful for local development, testing, and scenarios where you want API-style access to Claude Code

**What it's NOT:**
- Not an official Anthropic product
- Not a replacement for Anthropic's production API
- Not guaranteed to be 100% API-compatible in all edge cases
- Not suitable for production workloads without extensive testing

## Current Status: Early Development (v0.1.0)

**Known Limitations:**
- Limited testing across different use cases
- Potential edge cases not handled
- Error handling may not cover all scenarios
- Performance characteristics unknown under load
- Streaming implementation may have issues
- Model compatibility not fully validated

**Use Cases This Might Work For:**
- Local development and testing
- Prototyping applications that need Anthropic API compatibility
- Bridging existing codebases to use Claude Code SDK
- Educational purposes and API experimentation

**Proceed With Caution If:**
- You need production-grade reliability
- You're handling sensitive or critical data
- You require guaranteed API compatibility
- You need enterprise-level support

## Quick Start

### Prerequisites

- Python 3.11+
- Claude Code CLI installed and configured
- Understanding that this is experimental software

### Installation

1. **Install Claude Code CLI first:**
   ```bash
   # Install Claude Code CLI
   pip install claude-code-sdk>=0.0.17
   
   # Configure with your API key
   claude auth login
   ```

2. **Clone and install this project:**
   ```bash
   git clone <repository-url>
   cd claude-code-anthropic-api
   pip install -e .
   ```

3. **Start the server:**
   ```bash
   python -m src.main
   ```

Server starts on `http://0.0.0.0:8000` by default.

### Basic Usage

```python
import requests

# Test the server
response = requests.post("http://localhost:8000/v1/messages", json={
    "model": "claude-sonnet-4-20250514", 
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
})

print(response.json())
```

## API Endpoints

### POST /v1/messages
Anthropic-compatible message completion endpoint.

**Basic request:**
```json
{
  "model": "claude-sonnet-4-20250514",
  "messages": [{"role": "user", "content": "Hello!"}],
  "max_tokens": 100,
  "stream": false
}
```

**Streaming request:**
```json
{
  "model": "claude-sonnet-4-20250514", 
  "messages": [{"role": "user", "content": "Tell me a story"}],
  "max_tokens": 200,
  "stream": true
}
```

### GET /v1/models
List available models from Claude Code SDK.

### GET /health
Basic health check endpoint.

## Configuration

Set environment variables or create `.env`:

```bash
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false
LOG_LEVEL=WARNING

# Claude Code SDK
CLAUDE_CODE_PATH=/path/to/claude-code
CLAUDE_CODE_TIMEOUT=300

# Security (production defaults)
ALLOW_ORIGINS=[]  # No CORS by default
ALLOW_CREDENTIALS=false
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests (when available)
pytest

# Format code
black src/ tests/
isort src/ tests/

# Type check
mypy src/
```

## Testing & Validation

**Before using this in any important context:**

1. Test with your specific use cases
2. Validate API responses match expectations  
3. Test error handling scenarios
4. Verify streaming works for your client
5. Load test if needed
6. Have rollback plans

## Known Issues & Limitations

- **Incomplete API coverage**: Not all Anthropic API features implemented
- **Error handling**: May not handle all edge cases gracefully
- **Performance**: Not optimized for high throughput
- **Streaming**: Potential issues with different clients
- **Model support**: Limited to what Claude Code SDK supports
- **Authentication**: No auth implemented (local use only)

## Contributing

This project needs help! Areas that need work:

- Comprehensive testing across use cases
- Better error handling and edge cases
- Performance optimization
- More complete API compatibility
- Documentation improvements
- Bug fixes and stability improvements

Please test thoroughly and report issues.

## Support & Disclaimer

**Support:** This is an experimental project. Support is best-effort only.

**Disclaimer:** Use at your own risk. No warranties provided. Not affiliated with Anthropic. Test thoroughly before any important use.

**License:** MIT (see LICENSE file)

---

**Remember:** This is experimental software in early development. Always have backup plans and test extensively before relying on it for anything important.