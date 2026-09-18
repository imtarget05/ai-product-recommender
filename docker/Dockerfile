# RecSys-AI FastAPI Serving Engine
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (curl retained for container HEALTHCHECK)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and config
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY pytest.ini .

# ---- Test gate at build time ----
# Seed the demo dataset (SQLite fallback, fully offline) and run the full
# pytest suite. A failing test fails the image build.
COPY tests/ ./tests/
RUN python -m scripts.generate_seed_data && python -m pytest tests/ -q \
    && rm -rf data && mkdir -p data

# Remove test files from the runtime filesystem (earlier layers still contain them).
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
