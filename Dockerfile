# Use official Python image
FROM python:3.12-slim-bookworm

# Allow lightweight CI builds
ARG CI_MODE=false
ENV CI_MODE=${CI_MODE}

# Allow lightweight CI builds
ARG CPU_ONLY=false
ENV CPU_ONLY=${CPU_ONLY}

# Allow pre-installation of specific ollama models
ARG OLLAMA_MODELS=""
ENV OLLAMA_MODELS=${OLLAMA_MODELS}

# Allow for pruning indexed file in RAG DB (re-start from scratch)
ARG OVERRRIDE_INDEXING="false"
ENV OVERRRIDE_INDEXING=${OVERRRIDE_INDEXING}

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y curl git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Make sure Ollama service directory exists
RUN mkdir -p /root/.ollama

# Set work directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
COPY requirements.cpu.txt .

RUN pip install --upgrade pip
RUN if [ "$CPU_ONLY" = "true" ]; then \
      echo "📦 Installing CPU-only dependencies..."; \
      pip install -r requirements.cpu.txt; \
    else \
      echo "🎮 Installing full dependencies (with GPU support)..."; \
      pip install -r requirements.txt; \
    fi

# Install Ollama only for non-CI builds
RUN if [ "$CI_MODE" != "true" ]; then \
      echo "🚀 Installing Ollama..."; \
      curl -fsSL https://ollama.com/install.sh | bash; \
    else \
      echo "🧪 Skipping Ollama installation in CI mode"; \
    fi

# Copy the whole project
COPY . .

COPY docker/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

# Expose FastAPI port
EXPOSE 8000

# Command to download arg models
ENTRYPOINT ["/entrypoint.sh"]
