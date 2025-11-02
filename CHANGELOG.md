# Changelog

All notable changes to this project will be documented in this file.

## [1.3.0] - 2025-11-02
### Added
- Full Docker Compose orchestration with separate GPU (docker-compose.gpu.yml) and CPU (docker-compose.yml) configurations for flexible deployment.​

- GitHub Actions CI/CD workflows including automated Docker image builds (docker.yml), Python unit tests (unit-tests.yml), and CodeQL security scanning (codeql.yml).​

- Production-ready Makefile with commands for building, running, and managing Docker services (build, up, up-gpu, down, logs, indexer, app).​

- Redis caching system (RAGCacheHelper) for query results, LLM responses, and embedding storage with configurable TTL and automatic invalidation.​

- Multi-backend LLM support through adapter pattern with implementations for Ollama (OllamaModelAdapter), Gemini API (GeminiModelAdapter), and Hugging Face Transformers (TransformersModelAdapter).​

- FastAPI-based REST API (api/rag_api.py) with endpoints for asking questions (/v1/rag/ask), semantic search (/v1/rag/search), model listing (/v1/rag/models), and system stats (/v1/rag/stats).​

- Centralized configuration loader (config/config_loader.py) for YAML-based RAG configuration with environment variable expansion and validation.​

- LLM configuration system (llm/config.py) defining available models across Ollama, Gemini, and Transformers backends with detailed specifications.​

- Advanced search capabilities (search/advanced_search.py) with hybrid semantic + keyword search, query preprocessing, and result aggregation.​

- Comprehensive logging configuration (config/logging_config.py) with structured logging across all modules.​

- Secret management system (config/secrets_manager.py) supporting GCP Secret Manager, AWS Secrets Manager, Azure Key Vault, and Docker secrets.​

- Enhanced README with detailed installation guide, Docker Compose usage, Makefile commands, configuration examples, and architecture diagrams.​

- CHANGELOG.md documenting all releases from v1.0.1 to v1.2.0.​

- .dockerignore and .gitignore files for optimized Docker builds and clean repository management.​

### Changed / Improved
- Complete LLM architecture refactor replacing monolithic implementation with clean adapter pattern supporting multiple backends (llm/base_adapter.py, llm/bridge.py).​

- Enhanced Qdrant integration with improved utility functions (indexer/qdrant_utils.py) including stats retrieval, health checks, and batch operations.​

- -Improved hybrid search indexing (indexer/hybrid_index.py) with persistent TF-IDF model storage, better keyword extraction, and optimized memory usage.​

- Refactored text processing (data/text_utils.py) with smarter chunking respecting sentence boundaries, table handling, and metadata preservation.​

- Reorganized settings module (config/settings.py) with expanded configuration options for Redis, Qdrant, LLM backends, and API settings.​

- Enhanced Confluence indexer (indexer/confluence_indexer.py) with concurrent page processing, better error handling, and progress tracking.​

- Improved embedding generation with normalization and batch processing optimizations.​

- Better context enrichment with adjacent chunk fetching for more coherent LLM responses.​

### Fixed
- Memory management in keyword search to prevent RAM exhaustion on large document collections.​

- Container health checks and service dependencies in Docker Compose ensuring proper startup order.​

- API response formatting cleaning escaped newlines and improving JSON serialization.​

- Qdrant connection reliability with retry logic and better error handling.​

- Cache key consistency using normalized Unicode strings and consistent float formatting to prevent cache misses.​

### Removed
- Legacy API modules (api/confluence.py) replaced by enhanced Confluence indexer.​

- Outdated text processor (data/text_processor.py) replaced by improved text_utils module.​

- Old vector store implementation (data/vector_store.py) replaced by enhanced qdrant_utils.​

- Monolithic main scripts replaced by modular main_api.py and main_indexor.py entry points.​

## [1.2.0] - 2025-10-24
### Added
- Full Docker support for deployment and local development, including Dockerfile, docker-compose.yml, and entrypoint script.
- GitHub Actions workflow for automated Docker image builds and CI integration.
- Redis caching helper for faster query responses and reduced recomputation.
- Config loader and centralized YAML-based configuration management.
- New Makefile commands for building, testing, and container operations.
- Comprehensive README updates detailing setup, configuration, and Docker usage.
### Changed / Improved
- Refactored LLM bridge and adapter modules for cleaner API integration and easier extension of model backends.
- Reorganized configuration and environment handling to improve maintainability and flexibility.
- Improved RAG API logic for more efficient query processing and retrieval.
- Enhanced text indexer modules for better Qdrant integration and processing of larger documents.
### Fixed
- Minor dependency and import cleanup across modules.
- Removed redundant configuration entries and outdated code paths.

## [1.1.0] - 2025-10-07
### Added

- Support for fetching all chunks of a page (k=-1) and merging their text for full-page context.
- Centralized secret manager integration with GCP, AWS, and Azure support.
- Direct connection to Qdrant for querying documents (primarily for testing purposes).
- Metadata field added to track the source of each document (Confluence, Notion, etc.).
- Configuration parameters for LLM predictions: max_tokens and temperature for both generation and query refinement.

### Changed / Improved

- Keyword search refactored to prevent excessive RAM usage and potential crashes on large datasets.
- Chunking system enhanced to smartly handle tables: conversion to Markdown, preservation of headers across overlapping chunks, and consistent formatting.
- Context prompt refined to instruct the LLM to avoid guessing meanings of unknown acronyms.
- Adjacent chunk fetching improved and merged so that each search result contains combined text from related chunks, while preserving the original number of results.

### Fixed

- Fixes in the hybrid search system to correctly integrate semantic and keyword scores after text merging.

## [1.0.1] - 2025-09-26
### Added
- Check to ensure Qdrant Docker container is running before attempting to start it.
- Threading support in the Confluence indexer for faster page processing.
- Configuration support for indexing multiple root Confluence pages.

### Changed / Improved
- Text extraction now detects and cleans tables in Confluence pages.
- Chunking system updated to associate tables with their corresponding text chunks.
- Tables are stored alongside chunks in the vector database as structured metadata.
- Tables are included as context when querying with the LLM, improving answer relevance.
