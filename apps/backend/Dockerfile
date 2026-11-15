# ── Stage 1: Builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy project definition AND source code (setuptools needs src/ for egg_info)
COPY pyproject.toml .
COPY src/ src/

# Install all dependencies into /install prefix
RUN pip install --no-cache-dir --prefix=/install .[dev]

# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1001 nexusai && \
    useradd --uid 1001 --gid nexusai --shell /bin/bash --create-home nexusai

WORKDIR /home/nexusai/app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY --chown=nexusai:nexusai src/ src/
COPY --chown=nexusai:nexusai scripts/ scripts/

RUN mkdir -p /home/nexusai/app/.data/uploads && \
    chown -R nexusai:nexusai /home/nexusai/app

# Switch to non-root user
USER nexusai

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/home/nexusai/app/src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3)"

CMD ["uvicorn", "nexusai.main:app", "--host", "0.0.0.0", "--port", "8000"]
