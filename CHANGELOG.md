# Changelog

All notable changes to this project will be documented in this file.

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
