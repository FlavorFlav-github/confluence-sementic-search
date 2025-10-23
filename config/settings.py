import os

from config.secrets_manager import get_secret

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = "python"

# -------------------------
# 1. Enhanced Config
# -------------------------
QDRANT_BASE_URL = get_secret("QDRANT_BASE_URL")
QDRANT_PORT = os.getenv("QDRANT_PORT")
QDRANT_URL = f"{QDRANT_BASE_URL}:{QDRANT_PORT}"

COLLECTION_NAME = get_secret("QDRANT_COLLECTION_NAME")

PATH_CONFIG_RAG = get_secret("PATH_CONFIG_RAG", "/app/config/rag_config.yml")

REDIS_HOST = get_secret("REDIS_HOST", "localhost")
REDIS_PORT = get_secret("REDIS_PORT", "6379")
REDIS_CACHE_TTL_DAYS = min(int(get_secret("REDIS_CACHE_TTL_DAYS", "5")), 10)

# Enhanced embedding configuration
SENTENCE_TRANSFORMER = 'all-mpnet-base-v2'
EMBEDDING_SIZE = 768

INDEXING_MAX_CONCURRENT = 30
INDEXING_BATCH_SIZE = 500

# Advanced text processing configuration
CHUNK_SIZE_LIMIT = 750  # Slightly larger chunks for better context
CHUNK_OVERLAP = 150  # More overlap for better continuity
MIN_CHUNK_SIZE = 100  # Minimum viable chunk size
ENRICH_WITH_NEIGHBORS = 2

# Search configuration
DEFAULT_TOP_K = 15
RERANK_TOP_K = 5
SOURCE_THRESHOLD = 0.4
HYBRID_ALPHA = 0.7  # Weight for semantic vs keyword search (0.7 = 70% semantic, 30% keyword)

LLM_BACKEND_TYPE_GENERATION="ollama"
LLM_MODEL_GENERATION="phi3.5_q8_0"

LLM_BACKEND_TYPE_REFINEMENT="gemini"
LLM_MODEL_REFINE="flash"

LLM_MAX_TOKEN_GENERATION = 1000
LLM_TEMP_GENERATION = 0.2

LLM_MAX_TOKEN_REFINEMENT = 500
LLM_TEMP_REFINEMENT = 0.2

# TODO : Set up configuration for front server
API_ALLOWED_ORIGINS = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:3000",  # React dev server (if using CRA)
]