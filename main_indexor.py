"""
Main application file for the Confluence search engine.
Initializes all components and runs the indexing and search pipelines.
"""
import time

from sentence_transformers import SentenceTransformer

from cache.redis_cache_helper import RAGCacheHelper
# Import custom modules
from config.logging_config import logger
from config.settings import (CHUNK_SIZE_LIMIT, MIN_CHUNK_SIZE, \
    CHUNK_OVERLAP, EMBEDDING_SIZE, \
    SENTENCE_TRANSFORMER, COLLECTION_NAME, \
    INDEXING_BATCH_SIZE, INDEXING_MAX_CONCURRENT, QDRANT_URL,OVERRIDE_INDEXING,
                             REDIS_HOST, REDIS_CACHE_TTL_DAYS, REDIS_PORT)

from indexer.universal_indexer import UniversalIndexer
from indexer.hybrid_index import HybridSearchIndex
from indexer.qdrant_utils import get_qdrant_client
from indexer.text_processor import EnhancedTextProcessor
from config.config_loader import load_rag_config

# Configuration flag: Set to True to completely wipe and rebuild the index (Qdrant collection and TF-IDF model)
RESET_INDEXOR = OVERRIDE_INDEXING
INTERVAL_SECONDS = 6 * 60 * 60  # 6 hours

def main():
    """
    Main execution function. Initializes the vector database connection, 
    loads/initializes models, and starts the indexing process.
    """
    logger.info(f"Starting Confluence Indexing Pipeline with reset indexor as {RESET_INDEXOR}...")

    print("🔧 Loading RAG configuration...")
    sources = load_rag_config()

    # 1. Initialize and connect to Qdrant (starts container if needed)
    try:
        qdrant = get_qdrant_client(QDRANT_URL)
    except Exception as ex:
        logger.error(f"Failed to initialize Qdrant. Exiting. Error: {ex}")
        return

    # 2. Initialize Hybrid Search Index (Keyword/TF-IDF)
    hybrid_search_index = HybridSearchIndex()
    # Attempt to load an existing TF-IDF model from disk
    hybrid_search_index.load_tfidf()
    
    # 3. Initialize Embedding Model (SentenceTransformer)
    logger.info(f"Loading Sentence Transformer model: {SENTENCE_TRANSFORMER}")
    embed_model = SentenceTransformer(SENTENCE_TRANSFORMER)

    # 4. Initialize Text Processor with chunking configurations
    text_processor = EnhancedTextProcessor(CHUNK_SIZE_LIMIT, MIN_CHUNK_SIZE, CHUNK_OVERLAP)

    # 5. Initialize cache REDIS
    cache = RAGCacheHelper(
        host=REDIS_HOST,
        port=REDIS_PORT,
        ttl_days=REDIS_CACHE_TTL_DAYS
    )
    cache.check_and_start_redis()
    if cache.health_check():
        logger.info("✅ Redis cache initialized and healthy")
    else:
        logger.warning("⚠️ Redis health check failed, disabling cache")

    for src in sources:
        print(f"📚 Initializing source: {src['name']} ({src['type']})")
        data_source = src['type']
        root_ids = src['root_ids']
        logger.info(f"Indexing {data_source} content from root ID: {root_ids}")
        logger.info(f"Collection: {COLLECTION_NAME} | Reset: {RESET_INDEXOR}")

        data_source_options = {"base_url": src['base_url'],
                               "api_token": src['api_token'] }

        # 5. Initialize and run the Confluence Indexer
        confluence_indexer = UniversalIndexer(
            qdrant_client=qdrant,
            embedding_model=embed_model,
            text_processor=text_processor,
            hybrid_index=hybrid_search_index,
            data_source_name=data_source,
            data_source_options=data_source_options,
            root_page_ids=root_ids,
            collection_name=COLLECTION_NAME,
            embedding_size=EMBEDDING_SIZE,
            redis_client=cache
        )

        # Start the indexing process (recursive fetching, chunking, embedding, and upserting)
        confluence_indexer.index_pages(reset=RESET_INDEXOR, batch_size=INDEXING_BATCH_SIZE, max_concurrent=INDEXING_MAX_CONCURRENT)

        logger.info("Confluence Indexing Pipeline completed successfully.")


if __name__ == "__main__":
    print("🕒 Indexer service started, syncing every 6 hours")
    while True:
        try:
            main()  # run your full sync routine
        except Exception as e:
            print(f"❌ Error during indexing: {e}")
        print(f"✅ Sleeping for {INTERVAL_SECONDS / 3600} hours...")
        time.sleep(INTERVAL_SECONDS)
