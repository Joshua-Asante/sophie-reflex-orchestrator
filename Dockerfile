# Use Python 3.10 slim image
FROM python:3.10-slim

# Set metadata
LABEL maintainer="Sophie Reflex Orchestrator Team"
LABEL description="Minimal but powerful swarm-based orchestration system"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user and directories
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p /app /app/memory /app/logs /app/configs && \
    chown -R app:app /app

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=app:app . .

# Create necessary directories
RUN mkdir -p /app/memory /app/logs /app/ui/static /app/ui/templates && \
    chown -R app:app /app

# Download sentence transformers model (this can take time)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Create startup script
COPY <<'EOF' /app/start.sh
#!/bin/bash
set -e

# Wait for dependencies to be ready
echo "Waiting for ChromaDB to be ready..."
until curl -f http://chromadb:8000/api/v1/heartbeat &>/dev/null; do
    echo "ChromaDB is unavailable - sleeping"
    sleep 2
done

echo "ChromaDB is ready!"

echo "Waiting for Redis to be ready..."
until redis-cli -h redis ping &>/dev/null; do
    echo "Redis is unavailable - sleeping"
    sleep 2
done

echo "Redis is ready!"

# Create default templates if they don't exist
python -c "
import sys
sys.path.append('/app')
from ui.webhook_server import create_default_templates
create_default_templates()
"

# Start the application
echo "Starting Sophie Reflex Orchestrator..."
if [ "$1" = "server" ]; then
    exec python -m uvicorn ui.webhook_server:app --host 0.0.0.0 --port 8001
elif [ "$1" = "orchestrator" ]; then
    exec python main.py --interactive
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

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8001/api/health || exit 1

# Set entrypoint
ENTRYPOINT ["/app/start.sh"]

# Default command
CMD ["--help"]