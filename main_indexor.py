"""
Main application file for the Confluence search engine.
Initializes all components and runs the indexing and search pipelines.
"""
from sentence_transformers import SentenceTransformer

from config.logging_config import logger
from config.settings import CONFLUENCE_ROOT_PAGE_ID, CHUNK_SIZE_LIMIT, MIN_CHUNK_SIZE, \
    CHUNK_OVERLAP, COLLECTION_NAME, SPACE_KEY, EMBEDDING_SIZE, CONFLUENCE_BASE_URL, CONFLUENCE_API_TOKEN, \
    SENTENCE_TRANSFORMER
from indexer.confluence_indexer import ConfluenceIndexer
from indexer.hybrid_index import HybridSearchIndex
from indexer.qdrant_utils import check_and_start_qdrant
from indexer.text_processor import EnhancedTextProcessor

RESET_INDEXOR = True

def main():
    qdrant = check_and_start_qdrant()
    hybrid_search_index = HybridSearchIndex()
    hybrid_search_index.load_tfidf()
    embed_model = SentenceTransformer(SENTENCE_TRANSFORMER)

    logger.info(f"Indexing Confluence from root {CONFLUENCE_ROOT_PAGE_ID}")
    # Initialize text processor
    text_processor = EnhancedTextProcessor(CHUNK_SIZE_LIMIT, MIN_CHUNK_SIZE, CHUNK_OVERLAP)
    confluence_indexer = ConfluenceIndexer(qdrant, embed_model, text_processor, hybrid_search_index, CONFLUENCE_ROOT_PAGE_ID, COLLECTION_NAME, SPACE_KEY, EMBEDDING_SIZE, CONFLUENCE_BASE_URL, CONFLUENCE_API_TOKEN)
    confluence_indexer.index_pages(reset=RESET_INDEXOR)

if __name__ == "__main__":
    main()