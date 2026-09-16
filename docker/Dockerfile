# Dockerfile for RecSys-AI FastAPI Serving Engine
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and scripts
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY pytest.ini .

# Expose FastAPI port
EXPOSE 8000

# Default entrypoint starts FastAPI with Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
