# Changelog

All notable changes to this project will be documented in this file.

## [1.3.0] - 2025-11-02

### Added
- **Universal Indexing System:** Introduced the `UniversalIndexer` and abstract `DataSourceAdapter` pattern for easy integration of new documentation sources (e.g., Notion, PDF).
- **Asynchronous Confluence API:** Switched page fetching to `asyncio` and `aiohttp` for significantly faster, non-blocking I/O during indexing.
- **Dedicated RAG API (FastAPI):** Added `api/rag_api.py` with health checks and a structured API layer, separating the indexing and query services.
- **Comprehensive CI/CD:** Introduced GitHub Actions for **CodeQL security scanning**, **automated CPU/GPU Docker builds**, and **Python unit testing**.
- **Centralized Config Loader:** Implemented `config_loader.py` for structured YAML configuration loading with robust validation and environment variable expansion.
- New Makefile targets (`up-gpu`, `indexer-gpu`, `app-gpu`) for easier **GPU-enabled Docker Compose** launches.

### Changed / Improved
- **Modular Architecture:** Refactored monolithic files (`main.py`, `search/indexer.py`) into distinct, decoupled services (`main_indexer.py`, `main_rag.py`) for improved maintainability and clear service boundaries.
- **Hybrid Search Logic:** Simplified the hybrid search score calculation logic in `HybridSearch.search` for cleaner integration of semantic and keyword scores.
- Updated `LocalLLMBridge` and associated settings for better integration with configurable LLM backends.

### Fixed
- Removed redundant and deprecated monolithic files: `api/confluence.py`, `main.py`, and `search/indexer.py`.
- 
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
