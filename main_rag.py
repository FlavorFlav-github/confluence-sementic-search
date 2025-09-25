"""
Main application file for the Confluence search engine.
Initializes all components and runs the indexing and search pipelines.
"""
from config.logging_config import logger
from config.settings import LLM_MODEL, LLM_BACKEND_TYPE
from indexer.hybrid_index import HybridSearchIndex
from indexer.qdrant_utils import check_and_start_qdrant
from llm.bridge import LocalLLMBridge
from llm.config import LLMConfig
from search.advanced_search import AdvancedSearch


def main():
    qdrant = check_and_start_qdrant()
    hybrid_search_index = HybridSearchIndex()
    hybrid_search_index.load_tfidf()
    search_system = AdvancedSearch(qdrant, hybrid_search_index)

    LLMConfig.print_recommendations()

    print("🚀 CREATING LOCAL RAG SYSTEM...")
    rag_system = LocalLLMBridge(
        search_system=search_system,
        model_key=LLM_MODEL,
        backend_type=LLM_BACKEND_TYPE
    )

    rag_system.setup_model()

    if rag_system is None:
        print("No local RAG system created")
        return

    print("\n🧪 TESTING LOCAL RAG SYSTEM:")
    print("=" * 50)

    # Interactive mode
    print("\n🎯 Ready for interactive Q&A!")
    while True:
        question = input("\n❓ Your question (or 'quit'): ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if question:
            result = rag_system.ask(question)
            print(f"\n🤖 {result['answer']}")

            print(f"\n📚 Sources:")
            for source in result['sources']:
                print(f"  • {source['title']} (Score: {source['score']:.3f})")


if __name__ == "__main__":
    main()