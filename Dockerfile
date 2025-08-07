# Multi-stage build for Sophie Reflex Orchestrator
FROM python:3.11-slim as base

# Set metadata
LABEL maintainer="Sophie Reflex Orchestrator Team"
LABEL description="Modular swarm-based orchestration system"
LABEL version="2.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user and directories
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p /app /app/memory /app/logs /app/configs && \
    chown -R app:app /app

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Development stage for testing
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest==7.4.3 \
    pytest-asyncio==0.21.1 \
    pytest-cov==4.1.0 \
    black==23.11.0 \
    flake8==6.1.0 \
    mypy==1.7.1

# Copy application code
COPY --chown=app:app . .

# Create necessary directories
RUN mkdir -p /app/memory /app/logs /app/ui/static /app/ui/templates && \
    chown -R app:app /app

# Production stage
FROM base as production

# Copy application code
COPY --chown=app:app . .

# Create necessary directories
RUN mkdir -p /app/memory /app/logs /app/ui/static /app/ui/templates && \
    chown -R app:app /app

# Pre-download models for faster startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Create optimized startup script
COPY <<'EOF' /app/start.sh
#!/bin/bash
set -e

# Function to wait for service
wait_for_service() {
    local service_name=$1
    local service_url=$2
    local max_attempts=30
    local attempt=1
    
    echo "Waiting for $service_name to be ready..."
    while [ $attempt -le $max_attempts ]; do
        if curl -f "$service_url" &>/dev/null; then
            echo "$service_name is ready!"
            return 0
        fi
        echo "$service_name is unavailable - sleeping (attempt $attempt/$max_attempts)"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "ERROR: $service_name failed to start after $max_attempts attempts"
    exit 1
}

# Wait for dependencies
wait_for_service "ChromaDB" "http://chromadb:8000/api/v1/heartbeat"
wait_for_service "Redis" "redis://redis:6379"

# Create default templates if they don't exist
python -c "
import sys
sys.path.append('/app')
try:
    from ui.webhook_server import create_default_templates
    create_default_templates()
except ImportError:
    print('Webhook server not available, skipping template creation')
"

# Start the application with optimized settings
echo "Starting Sophie Reflex Orchestrator (Modular Architecture)..."
if [ "$1" = "server" ]; then
    exec python -m uvicorn ui.webhook_server:app --host 0.0.0.0 --port 8001 --workers 1
elif [ "$1" = "orchestrator" ]; then
    exec python main.py --interactive
elif [ "$1" = "test" ]; then
    exec python -m pytest tests/ -v
else
    exec python main.py "$@"
fi
EOF

# Make startup script executable
RUN chmod +x /app/start.sh

# Switch to app user
USER app

# Expose ports
EXPOSE 8000 8001

# Health check with better error handling
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8001/api/health || \
    curl -f http://localhost:8000/health || exit 1

# Set entrypoint
ENTRYPOINT ["/app/start.sh"]

# Default command
CMD ["--help"] 