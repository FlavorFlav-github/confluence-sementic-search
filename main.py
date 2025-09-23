"""
Main application file for the Confluence search engine.
Initializes all components and runs the indexing and search pipelines.
"""
import logging
import time
from sentence_transformers import SentenceTransformer

# Configure root logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import modules from your project
from config import settings
from api.confluence import ConfluenceClient
from data.text_processor import EnhancedTextProcessor
from data.vector_store import QdrantManager
from search.hybrid_search import HybridSearch
from search.indexer import index_confluence_pages


def main():
    """
    Initializes and runs the Confluence search engine.
    """
    logger.info("Initializing search engine components...")

    # Initialize external clients and models
    qdrant_manager = QdrantManager(settings.QDRANT_URL)
    confluence_client = ConfluenceClient(settings.CONFLUENCE_BASE_URL, settings.CONFLUENCE_API_TOKEN)
    text_processor = EnhancedTextProcessor()

    try:
        embedding_model = SentenceTransformer(settings.SENTENCE_TRANSFORMER)
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        return

    hybrid_search_system = HybridSearch(qdrant_manager.get_client(), embedding_model)

    # --- Indexing Phase ---
    logger.info(f"Starting indexing from root page {settings.CONFLUENCE_ROOT_PAGE_ID}")
    index_confluence_pages(
        qdrant_manager=qdrant_manager,
        confluence_client=confluence_client,
        text_processor=text_processor,
        embedding_model=embedding_model,
        hybrid_search_index=hybrid_search_system,
        root_page_id=settings.CONFLUENCE_ROOT_PAGE_ID,
        reset=True
    )

    # --- Search Phase ---
    queries = [
        "What are the personae for SBCP in SXP",
        "user experience design patterns",
        "API documentation",
        "security requirements"
    ]

    for query in queries:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"TESTING QUERY: '{query}'")
        logger.info(f"{'=' * 80}")

        print("\n🔍 HYBRID SEARCH RESULTS:")
        hybrid_results = hybrid_search_system.hybrid_search(query, top_k=5)
        hybrid_search_system.explain_results(hybrid_results, query)

        print("\n🧠 SEMANTIC-ONLY SEARCH RESULTS:")
        semantic_results = hybrid_search_system.semantic_search(query, top_k=3)
        hybrid_search_system.explain_results(semantic_results, query)

        print("\n" + "─" * 80)


if __name__ == "__main__":
    main()