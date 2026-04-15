# Multi-stage Dockerfile for the Autonomous SRE FastAPI application.
# This image runs the API server only (no GPU / vLLM dependencies).
# The LLM workload runs separately on RunPod Serverless.

# ---------------------------------------------------------------------------
# Stage 1: dependency builder
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
# Build venv at the same path it will live in the runtime stage so shebangs are valid
RUN python -m venv /app/venv && \
    /app/venv/bin/pip install --no-cache-dir -r requirements.txt


# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/venv/bin:$PATH"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /app/venv /app/venv

# Copy application source
COPY app/ ./app/
COPY prompts/ ./prompts/
COPY data/ ./data/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/docs')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
