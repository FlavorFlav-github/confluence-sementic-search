"""
Main application file for the Confluence search engine.
Initializes all components and runs the indexing and search pipelines.
"""
from sentence_transformers import SentenceTransformer

# Import custom modules
from config.logging_config import logger
from config.settings import CHUNK_SIZE_LIMIT, MIN_CHUNK_SIZE, \
    CHUNK_OVERLAP, COLLECTION_NAME, EMBEDDING_SIZE, \
    SENTENCE_TRANSFORMER, DATA_SOURCE_NAME, DATA_SOURCE_BASE_URL, DATA_SOURCE_API_TOKEN, DATA_SOURCE_ROOT_PAGE_ID, \
    INDEXING_BATCH_SIZE, INDEXING_MAX_CONCURRENT
from indexer.universal_indexer import UniversalIndexer
from indexer.hybrid_index import HybridSearchIndex
from indexer.qdrant_utils import check_and_start_qdrant
from indexer.text_processor import EnhancedTextProcessor

# Configuration flag: Set to True to completely wipe and rebuild the index (Qdrant collection and TF-IDF model)
RESET_INDEXOR = True

def main():
    """
    Main execution function. Initializes the vector database connection, 
    loads/initializes models, and starts the indexing process.
    """
    logger.info("Starting Confluence Indexing Pipeline...")
    
    # 1. Initialize and connect to Qdrant (starts container if needed)
    try:
        qdrant = check_and_start_qdrant()
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant. Exiting. Error: {e}")
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
    
    logger.info(f"Indexing {DATA_SOURCE_NAME} content from root ID: {DATA_SOURCE_ROOT_PAGE_ID}")
    logger.info(f"Collection: {COLLECTION_NAME} | Reset: {RESET_INDEXOR}")

    data_source_options = {"base_url": DATA_SOURCE_BASE_URL,
                           "api_token": DATA_SOURCE_API_TOKEN }

    # 5. Initialize and run the Confluence Indexer
    confluence_indexer = UniversalIndexer(
        qdrant_client=qdrant, 
        embedding_model=embed_model, 
        text_processor=text_processor, 
        hybrid_index=hybrid_search_index,
        data_source_name=DATA_SOURCE_NAME,
        data_source_options=data_source_options,
        root_page_ids=DATA_SOURCE_ROOT_PAGE_ID,
        collection_name=COLLECTION_NAME, 
        embedding_size=EMBEDDING_SIZE
    )
    
    # Start the indexing process (recursive fetching, chunking, embedding, and upserting)
    confluence_indexer.index_pages(reset=RESET_INDEXOR, batch_size=INDEXING_BATCH_SIZE, max_concurrent=INDEXING_MAX_CONCURRENT)
    
    logger.info("Confluence Indexing Pipeline completed successfully.")


if __name__ == "__main__":
    main()
