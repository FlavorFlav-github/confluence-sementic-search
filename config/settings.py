import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# -------------------------
# 1. Enhanced Config
# -------------------------
CONFLUENCE_BASE_URL = "https://confluence.sage.com/rest/api"
CONFLUENCE_API_TOKEN = "CONFLUENCE_API_TOKEN"
CONFLUENCE_ROOT_PAGE_ID = 417798815
SPACE_KEY = "FRCIELESP"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "confluence_pages_sxp_enhanced_v3"

# Enhanced embedding configuration
SENTENCE_TRANSFORMER = 'all-mpnet-base-v2'
EMBEDDING_SIZE = 768

# Advanced text processing configuration
CHUNK_SIZE_LIMIT = 600  # Slightly larger chunks for better context
CHUNK_OVERLAP = 150  # More overlap for better continuity
MIN_CHUNK_SIZE = 50  # Minimum viable chunk size
ENRICH_WITH_NEIGHBORS = 1

# Search configuration
DEFAULT_TOP_K = 10
RERANK_TOP_K = 20
HYBRID_ALPHA = 0.7  # Weight for semantic vs keyword search (0.7 = 70% semantic, 30% keyword)

LLM_BACKEND_TYPE="ollama"
LLM_MODEL="phi3.5_q8_0"