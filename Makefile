# ------------------------------------------------------------
# Makefile for running RAG app with Docker Compose
# Supports custom config_rag.yaml
# ------------------------------------------------------------

# Default config file path
CONFIG ?= ./config/rag_config.yml

# Default docker-compose file
COMPOSE_FILE := docker-compose.yml

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

# Stop all services
down:
	docker compose -f $(COMPOSE_FILE) down

# Follow logs
logs:
	docker compose -f $(COMPOSE_FILE) logs -f

# Run only the indexer
indexer:
	RAG_CONFIG_PATH=$(CONFIG) docker compose -f $(COMPOSE_FILE) up -d rag-indexer

# Run only the main app
app:
	RAG_CONFIG_PATH=$(CONFIG) docker compose -f $(COMPOSE_FILE) up -d rag-app