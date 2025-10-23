# Use official Python image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y curl git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | bash

# Make sure Ollama service directory exists
RUN mkdir -p /root/.ollama

# ---------------------------------
# Allow passing models as build args
# Example usage:
# docker build --build-arg OLLAMA_MODELS="phi3.5:3.8b-mini-instruct-q4_K_M llama3.2:3b-instruct-q4_K_M" -t rag-app .
# ---------------------------------
ARG OLLAMA_MODELS=""
ENV OLLAMA_MODELS=$OLLAMA_MODELS

ARG OVERRRIDE_INDEXING=""
ENV OVERRRIDE_INDEXING=$OVERRRIDE_INDEXING

RUN pip install nltk

# Download NLTK data during build (not at runtime)
RUN python -c "import nltk; \
    nltk.download('punkt', download_dir='/usr/local/share/nltk_data'); \
    nltk.download('punkt_tab', download_dir='/usr/local/share/nltk_data'); \
    nltk.download('stopwords', download_dir='/usr/local/share/nltk_data'); \
    nltk.download('wordnet', download_dir='/usr/local/share/nltk_data')"

# Set work directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the whole project
COPY . .

COPY docker/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

# Expose FastAPI port
EXPOSE 8000

# Command to download arg models
ENTRYPOINT ["/entrypoint.sh"]