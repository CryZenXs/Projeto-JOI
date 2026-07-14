# ═══════════════════════════════════════════════════════════════════════════
# Projeto JOI - Multi-stage Dockerfile
# ═══════════════════════════════════════════════════════════════════════════
#
# Build:  docker build -t projeto-joi:latest -f docker/Dockerfile .
# Run:    docker run --rm -p 8000:8000 --env-file .env projeto-joi:latest
#
# Multi-stage reduces final image size from ~1.2GB to ~250MB by separating
# the build environment (with compilers, dev tools) from the runtime.
# ═══════════════════════════════════════════════════════════════════════════

# ─── Stage 1: Builder ───────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Set up build environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.8.5

# Install build dependencies (removed after build in final stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create venv in /opt/venv (reusable across stages)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies (cached layer if requirements unchanged)
WORKDIR /build
COPY pyproject.toml ./
# Install only runtime deps (not dev) for smaller image
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

# ─── Stage 2: Runtime ───────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Metadata
LABEL org.opencontainers.image.title="Projeto JOI" \
      org.opencontainers.image.description="Entidade digital companheira inspirada em Blade Runner 2049" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.licenses="MIT"

# Runtime environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    ENVIRONMENT=production

# Install only runtime libs (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1000 joi \
    && useradd --system --uid 1000 --gid joi --create-home --home-dir /home/joi --shell /bin/bash joi

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
WORKDIR /app
COPY --chown=joi:joi . .

# Create data directories with correct permissions
RUN mkdir -p /app/data/chroma && chown -R joi:joi /app

# Switch to non-root user for security
USER joi

# Expose FastAPI port
EXPOSE 8000

# Health check (every 30s, fail after 3 retries = 90s)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Use tini for proper signal handling (SIGTERM forwarding)
ENTRYPOINT ["/usr/bin/tini", "--"]

# Run uvicorn with production settings
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--no-access-log", \
     "--log-level", "warning"]
