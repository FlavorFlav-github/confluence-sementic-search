# Changelog

All notable changes to this project will be documented in this file.

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
