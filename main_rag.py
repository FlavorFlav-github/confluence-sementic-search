"""
Main application file for the Confluence search engine.
Initializes all components and runs the indexing and search pipelines.
"""
import os

# Import custom modules
from config.logging_config import logger
from config.settings import (LLM_MODEL_REFINE, LLM_MODEL_GENERATION,
                             LLM_BACKEND_TYPE_GENERATION, LLM_BACKEND_TYPE_REFINEMENT,
                             DEFAULT_TOP_K, RERANK_TOP_K, SOURCE_THRESHOLD,
                             REDIS_HOST, REDIS_PORT, REDIS_CACHE_TTL_DAYS, QDRANT_URL)
from indexer.hybrid_index import HybridSearchIndex
from indexer.qdrant_utils import get_qdrant_client
from llm.bridge import LocalLLMBridge
from llm.config import LLMConfig
from search.advanced_search import AdvancedSearch


def main():
    """
    Main execution function. Initializes the Qdrant database, the hybrid search system,
    and sets up the Local RAG (Retrieval-Augmented Generation) system for interactive Q&A.
    """
    logger.info("Starting Local RAG System Initialization...")

    # 1. Initialize and connect to Qdrant (vector database)
    try:
        qdrant = get_qdrant_client(QDRANT_URL)
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant. Exiting. Error: {e}")
        return

    # 2. Initialize Hybrid Search Index (Keyword/TF-IDF)
    hybrid_search_index = HybridSearchIndex()
    # Load the pre-trained TF-IDF model if it exists
    hybrid_search_index.load_tfidf()
    
    # 3. Initialize Advanced Search System (combines vector and keyword search)
    search_system = AdvancedSearch(qdrant, hybrid_search_index)

    # Print recommendations for the LLM setup (based on LLMConfig logic)
    LLMConfig.print_recommendations()

    print("🚀 CREATING LOCAL RAG SYSTEM...")
    
    # 4. Initialize the Local RAG Bridge
    # The bridge connects the search system (retrieval) with the LLM (generation)
    rag_system = LocalLLMBridge(
        search_system=search_system,
        generation_model_key=LLM_MODEL_GENERATION,
        refinement_model_key=LLM_MODEL_REFINE,
        generation_model_backend_type=LLM_BACKEND_TYPE_GENERATION,
        refinement_model_backend_type=LLM_BACKEND_TYPE_REFINEMENT,
        redis_host=REDIS_HOST,
        redis_port=REDIS_PORT,
        redis_cache_ttl_days=REDIS_CACHE_TTL_DAYS
    )
    
    # 5. Setup the LLM model (e.g., download model weights, initialize framework)
    try:
        rag_system.setup_model()
    except Exception as e:
        logger.error(f"Failed to set up the LLM model. Exiting. Error: {e}")
        # The original code checks for 'None', but if setup_model raises an exception, 
        # it's better to exit cleanly here.
        # If the original implementation of setup_model sets rag_system to None on failure, 
        # the check below handles it.
        pass 
    
    # Check if model setup was successful (assuming it sets an internal flag or handles failure)
    # The original check:
    if rag_system is None:
        print("No local RAG system created")
        return

    print("\n🧪 TESTING LOCAL RAG SYSTEM:")
    print("=" * 50)

    # 6. Interactive Q&A loop
    print("\n🎯 Ready for interactive Q&A!")
    while True:
        question = input("\n❓ Your question (or 'quit'): ").strip()
        
        # Exit condition
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Exiting RAG system. Goodbye!")
            break
            
        if question:
            try:
                # Perform RAG query
                result = rag_system.ask(question, top_k=DEFAULT_TOP_K, final_top_k=RERANK_TOP_K, score_threshold=SOURCE_THRESHOLD)
                
                # Print LLM Answer
                print(f"\n🤖 {result.get('answer', 'Sorry, I could not generate an answer.')}")

                # Print Sources (retrieved documents)
                print(f"\n📚 Sources:")
                sources = result.get('sources', [])
                if sources:
                    for source in sources:
                        # Display source title and search score
                        print(f"  • ({source.get('source', 'N/A')}) {source.get('title', 'N/A')} (Score: {source.get('score', 0.0):.3f}) - {source.get('link', None)}")
                else:
                    print("  • No relevant sources found.")
                    
            except Exception as e:
                logger.error(f"Error during RAG query: {e}")
                print("\n❌ An error occurred while processing your question.")


if __name__ == "__main__":
    main()
