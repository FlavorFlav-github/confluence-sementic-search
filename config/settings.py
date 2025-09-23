"""
Configuration settings for the Confluence Search Engine.
This module handles sensitive information and model parameters.
"""
import os
from typing import Final

# -------------------------
# Confluence API Configuration
# -------------------------
# Fetch sensitive data from environment variables for security.
CONFLUENCE_BASE_URL: Final[str] = os.getenv("CONFLUENCE_BASE_URL", "https://confluence.sage.com/rest/api")
CONFLUENCE_API_TOKEN: Final[str] = os.getenv("CONFLUENCE_API_TOKEN", "your-api-token-here") # Use a default for dev
CONFLUENCE_ROOT_PAGE_ID: Final[int] = int(os.getenv("CONFLUENCE_ROOT_PAGE_ID", "417798815"))
SPACE_KEY: Final[str] = os.getenv("SPACE_KEY", "FRCIELESP")

# -------------------------
# Qdrant Vector DB Configuration
# -------------------------
QDRANT_URL: Final[str] = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME: Final[str] = os.getenv("COLLECTION_NAME", "confluence_pages_sxp_enhanced_v3")
EMBEDDING_SIZE: Final[int] = 768

# -------------------------
# Embedding and NLP Configuration
# -------------------------
SENTENCE_TRANSFORMER: Final[str] = 'all-mpnet-base-v2'

# Advanced text processing
CHUNK_SIZE_LIMIT: Final[int] = 600
CHUNK_OVERLAP: Final[int] = 150
MIN_CHUNK_SIZE: Final[int] = 50

# Search configuration
DEFAULT_TOP_K: Final[int] = 10
RERANK_TOP_K: Final[int] = 20
HYBRID_ALPHA: Final[float] = 0.7