# ============================================================
# Stage 1: Build dependencies (for speed + caching)
# ============================================================
FROM python:3.12-slim-bookworm AS builder

ARG CPU_ONLY=true
ARG LOCAL_LLM=false
ARG OLLAMA_MODELS=""
ARG OVERRRIDE_INDEXING="false"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies for building Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libpq-dev \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies (GPU or CPU)
RUN pip install --upgrade pip && \
    if [ "$CPU_ONLY" = "true" ]; then \
      echo "📦 Installing CPU-only dependencies..."; \
      pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu; \
    else \
      echo "🎮 Installing full dependencies (with GPU support)..."; \
      pip install --no-cache-dir -r requirements.txt; \
    fi

# ============================================================
# Stage 2: Runtime image
# ============================================================
FROM python:3.12-slim-bookworm

# Build-time args propagated as env vars
ARG LOCAL_LLM=false
ARG CPU_ONLY=true
ARG OLLAMA_MODELS=""
ARG OVERRRIDE_INDEXING="false"

ENV LOCAL_LLM=${LOCAL_LLM}
ENV CPU_ONLY=${CPU_ONLY}
ENV OLLAMA_MODELS=${OLLAMA_MODELS}
ENV OVERRRIDE_INDEXING=${OVERRRIDE_INDEXING}

# Core Python settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# PostgreSQL env (consistent across docker-compose + k8s)
ENV POSTGRES_HOST=rag-postgres
ENV POSTGRES_PORT=5432
ENV POSTGRES_USER=rag_user

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
        ca-certificates \
        # optional: handy for debugging inside container
        postgresql-client && \
    rm -rf /var/lib/apt/lists/*

# Make sure Ollama directory exists
RUN mkdir -p /root/.ollama

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project files
COPY . .

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY scripts/init_db.py /app/init_db.py
RUN chmod +x /app/init_db.py

EXPOSE 8000

# Optionally install Ollama locally
RUN if [ "$LOCAL_LLM" = "true" ]; then \
      echo "🚀 Installing Ollama..."; \
      curl -fsSL https://ollama.com/install.sh | bash; \
    else \
      echo "🧪 Skipping Ollama installation"; \
    fi

ENTRYPOINT ["/entrypoint.sh"]