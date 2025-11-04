# ------------------------------------------------------------
# Makefile for running RAG app with Docker Compose
# Supports custom config_rag.yaml
# ------------------------------------------------------------

# Default config file path
CONFIG ?= ./config/rag_config.yml

# Default docker-compose file
COMPOSE_FILE := docker-compose.yml

COMPOSE_FILE_GPU := docker-compose.gpu.yml

# ------------------------------------------------------------
# Targets
# ------------------------------------------------------------

.PHONY: build up down logs help

help:
	@echo "Available targets:"
	@echo "  build            Build the Docker image(s)"
	@echo "  up               Start the services with Docker Compose using CONFIG=$(CONFIG)"
	@echo "  down             Stop and remove containers"
	@echo "  logs             Follow logs of all services"
	@echo "  indexer          Run only the indexer container"
	@echo "  app              Run only the main RAG app"

# Build the images
build:
	docker compose -f $(COMPOSE_FILE) build

# Start all services with custom config
up:
	RAG_CONFIG_PATH=$(CONFIG) docker compose -f $(COMPOSE_FILE) up
	
# Start all services with custom config
up-gpu:
	RAG_CONFIG_PATH=$(CONFIG) docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_FILE_GPU) up

# Stop all services
down:
	docker compose -f $(COMPOSE_FILE) down

# Follow logs
logs:
	docker compose -f $(COMPOSE_FILE) logs -f

# Run only the indexer
indexer:
	RAG_CONFIG_PATH=$(CONFIG) docker compose -f $(COMPOSE_FILE) up rag-indexer rag-qdrant rag-cache
	
# Run only the indexer
indexer-gpu:
	RAG_CONFIG_PATH=$(CONFIG) docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_FILE_GPU) up rag-indexer rag-qdrant rag-cache

# Run only the main app
app:
	RAG_CONFIG_PATH=$(CONFIG) docker compose -f $(COMPOSE_FILE) up rag-qdrant rag-cache rag-postgres rag-db-init rag-app
	
app-gpu:
	RAG_CONFIG_PATH=$(CONFIG) docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_FILE_GPU) up rag-qdrant rag-cache rag-postgres rag-db-init rag-app