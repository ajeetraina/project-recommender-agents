# syntax=docker/dockerfile:1
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 app

# Set up Python environment
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Development stage
FROM base as development

# Install development dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements-dev.txt

# Copy source code
COPY --chown=app:app src/ ./src/
COPY --chown=app:app tests/ ./tests/

USER app

# Development server with hot reload
CMD ["streamlit", "run", "src/main.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--browser.gatherUsageStats", "false"]

# Runtime stage for UI application
FROM base as runtime

# Install production dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=app:app src/ ./src/

# Create necessary directories
RUN mkdir -p /app/workspace && chown app:app /app/workspace

USER app

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/health || exit 1

# Start Streamlit application
CMD ["streamlit", "run", "src/main.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--browser.gatherUsageStats", "false", \
     "--server.headless", "true"]

# Agent runtime stage for background services
FROM base as agent-runtime

# Install production dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy agent-specific code
COPY --chown=app:app src/ ./src/

USER app

# Health check for agent services
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src.agents.health_check; src.agents.health_check.check()" || exit 1

# Start agent orchestrator
CMD ["python", "-m", "src.agents.orchestrator"]
