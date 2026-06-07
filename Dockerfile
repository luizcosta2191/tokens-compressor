# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Install wheels in an isolated stage so the final image carries no build tools.
FROM python:3.13.5-slim AS builder

WORKDIR /build

# Install only what pip needs to compile wheels (some tiktoken deps need gcc)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Build wheels into a local directory; no network needed in the next stage
RUN pip install --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.13.5-slim AS runtime

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Install only curl for the health-check (no build tools needed)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Pull pre-built wheels from the builder stage (no compiler required here)
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* \
    && rm -rf /wheels

# Copy application source
COPY src/ ./src/

# Drop privileges
USER appuser

EXPOSE 8501

# Fail fast if the health endpoint is unreachable
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl --silent --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Install wheels in an isolated stage so the final image carries no build tools.
FROM python:3.13.5-slim AS builder

WORKDIR /build

# Install only what pip needs to compile wheels (some tiktoken deps need gcc)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Build wheels into a local directory; no network needed in the next stage
RUN pip install --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.13.5-slim AS runtime

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Install only curl for the health-check (no build tools needed)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Pull pre-built wheels from the builder stage (no compiler required here)
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* \
    && rm -rf /wheels

# Copy application source
COPY src/ ./src/

# Drop privileges
USER appuser

EXPOSE 8501

# Fail fast if the health endpoint is unreachable
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl --silent --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
