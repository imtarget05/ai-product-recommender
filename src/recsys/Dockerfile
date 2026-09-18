# syntax=docker/dockerfile:1.0.0
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

FROM python:3.11-slim AS test

WORKDIR /app

# Install test dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tests/ ./tests/

RUN python -m pytest tests/ -q

FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime dependencies only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy built application from builder stage
COPY --from=builder /app ./app

# Copy source code
COPY src/ ./src/

# Copy config and scripts
COPY scripts/ ./scripts/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY pytest.ini .

# Run test gate at build time - seed data and run pytest
RUN python -m scripts.generate_seed_data && python -m pytest tests/ -q \
    && rm -rf data && mkdir -p data

# Remove test files from the runtime filesystem
RUN rm -rf tests .pytest_cache

# Run as unprivileged user
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose default port
EXPOSE 8000

# Liveness probe against the FastAPI health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/api/v1/health" || exit 1

# Run Alembic migrations then start Uvicorn (supporting Railway's dynamic $PORT)
CMD ["sh", "-c", "alembic upgrade head && uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]