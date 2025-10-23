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

ENV OLLAMA_MODELS=$OLLAMA_MODELS

ENV OVERRRIDE_INDEXING=$OVERRRIDE_INDEXING

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