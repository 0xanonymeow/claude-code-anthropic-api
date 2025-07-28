# Makefile for claude-code-anthropic-api

# Variables
PYTHON = python3
VENV = venv
VENV_BIN = $(VENV)/bin
PIP = $(VENV_BIN)/pip
PYTHON_VENV = $(VENV_BIN)/python
UVICORN = $(VENV_BIN)/uvicorn
PYTEST = $(VENV_BIN)/pytest
BLACK = $(VENV_BIN)/black
ISORT = $(VENV_BIN)/isort
MYPY = $(VENV_BIN)/mypy
PRE_COMMIT = $(VENV_BIN)/pre-commit

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  setup     - Create virtual environment and install dependencies"
	@echo "  install   - Install dependencies in existing virtual environment"
	@echo "  start     - Start the FastAPI server"
	@echo "  test      - Run tests"
	@echo "  lint      - Run linting (black, isort, mypy)"
	@echo "  format    - Format code with black and isort"
	@echo "  clean     - Remove virtual environment and cache files"
	@echo "  dev       - Install development dependencies"
	@echo "  pre-commit - Install pre-commit hooks"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run - Run Docker container"
	@echo "  docker-compose-up - Start with docker-compose (development)"
	@echo "  docker-compose-prod - Start with docker-compose (production)"

# Setup target - creates venv and installs requirements
.PHONY: setup
setup: $(VENV)/pyvenv.cfg
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -e .
	@echo "Setup complete! Virtual environment created and dependencies installed."

# Create virtual environment
$(VENV)/pyvenv.cfg:
	@echo "Creating virtual environment..."
	@if ! $(PYTHON) -m venv $(VENV) 2>/dev/null; then \
		echo "Installing python3-venv package..."; \
		if command -v apt >/dev/null 2>&1; then \
			PYTHON_VERSION=$$($(PYTHON) --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2); \
			sudo apt update && sudo apt install -y python$$PYTHON_VERSION-venv; \
		elif command -v yum >/dev/null 2>&1; then \
			sudo yum install -y python3-venv; \
		elif command -v dnf >/dev/null 2>&1; then \
			sudo dnf install -y python3-venv; \
		else \
			echo "ERROR: Could not install python3-venv automatically."; \
			echo "Please install it manually for your system."; \
			exit 1; \
		fi; \
		echo "Retrying virtual environment creation..."; \
		$(PYTHON) -m venv $(VENV); \
	fi
	@echo "Virtual environment created successfully."

# Install dependencies in existing venv
.PHONY: install
install:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	$(PIP) install --upgrade pip
	$(PIP) install -e .

# Install development dependencies
.PHONY: dev
dev: setup
	$(PIP) install -e ".[dev]"

# Start the FastAPI server
.PHONY: start
start:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	$(UVICORN) src.main:app --host 0.0.0.0 --port 8000 --reload

# Run tests
.PHONY: test
test:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	$(PYTEST) tests/ -v --cov=src --cov-report=html --cov-report=xml

# Run linting
.PHONY: lint
lint:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	$(BLACK) --check src/ tests/
	$(ISORT) --check-only src/ tests/
	$(MYPY) src/

# Format code
.PHONY: format
format:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	$(BLACK) src/ tests/
	$(ISORT) src/ tests/

# Install pre-commit hooks
.PHONY: pre-commit
pre-commit: dev
	$(PRE_COMMIT) install

# Clean up
.PHONY: clean
clean:
	rm -rf $(VENV)
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
	rm -rf .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleanup complete!"

# Docker targets
.PHONY: docker-build
docker-build:
	docker build -t claude-code-anthropic-api .

.PHONY: docker-run
docker-run:
	docker run -p 8000:8000 claude-code-anthropic-api

.PHONY: docker-compose-up
docker-compose-up:
	docker-compose up --build

.PHONY: docker-compose-down
docker-compose-down:
	docker-compose down

.PHONY: docker-compose-prod
docker-compose-prod:
	docker-compose -f docker-compose.prod.yml up --build

.PHONY: docker-compose-prod-down
docker-compose-prod-down:
	docker-compose -f docker-compose.prod.yml down