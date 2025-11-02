# File: bridge.txt
from config.logging_config import logger
from typing import Dict, Any, List, Optional

from llm.base_adapter import LLMAdapter
from llm.config import LLMConfig  # Assuming this contains RECOMMENDED_MODELS
from llm.gemini_adapter import GeminiModelAdapter
from llm.ollama_adapter import OllamaModelAdapter
from llm.transformers_adapter import TransformerModelAdapter
from config.settings import (LLM_MAX_TOKEN_GENERATION, LLM_TEMP_GENERATION,
                             LLM_MAX_TOKEN_REFINEMENT, LLM_TEMP_REFINEMENT,
                             ENRICH_WITH_NEIGHBORS, COLLECTION_NAME)
from cache.redis_cache_helper import RAGCacheHelper

# Assuming this constant is defined in config.settings
# from config.settings import ENRICH_WITH_NEIGHBORS
ENRICH_WITH_NEIGHBORS = ENRICH_WITH_NEIGHBORS  # Placeholder for demonstration
MAX_TOKEN_GENERATION = LLM_MAX_TOKEN_GENERATION
TEMP_GENERATION = LLM_TEMP_GENERATION
MAX_TOKEN_REFINEMENT = LLM_MAX_TOKEN_REFINEMENT
TEMP_REFINEMENT = LLM_TEMP_REFINEMENT

class LocalLLMBridge:
    """
    Manages local Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) operations.

    This class orchestrates the RAG process, using two separate LLM adapters:
    one for fast query refinement and one for accurate answer generation.
    """

    # A class-level dictionary to store available adapters.
    AVAILABLE_ADAPTERS = {
        "ollama": OllamaModelAdapter,
        "gemini": GeminiModelAdapter,
        "transformers": TransformerModelAdapter
    }

    def __init__(self, search_system: Any, generation_model_key: str, refinement_model_key: str,
                 generation_model_backend_type: str = "ollama", refinement_model_backend_type: str = "ollama", enable_cache=True,
                 redis_host='localhost', redis_port=6379, redis_cache_ttl_days=30):
        """
        Initializes the bridge with separate adapters for generation and refinement.

        Args:
            search_system (Any): The external search/vector-db system instance.
            generation_model_key (str): Key for the main LLM (e.g., 'llama3').
            refinement_model_key (str): Key for the smaller, faster LLM (e.g., 'phi3.5_q4_K_M').
            generation_model_backend_type (str): The type of LLM backend to use for generation (e.g., 'ollama').
            refinement_model_backend_type (str): The type of LLM backend to use for refinement(e.g., 'ollama').

        Raises:
            ValueError: If the `backend_type` is not supported or model keys are not found.
        """
        self.search = search_system
        self.generation_model_backend_type = generation_model_backend_type
        self.refinement_model_backend_type = refinement_model_backend_type
        self.generation_model_key = generation_model_key
        self.refinement_model_key = refinement_model_key
        self.collection_name = COLLECTION_NAME

        # Initialize cache
        self.cache_enabled = enable_cache
        self.cache: Optional[RAGCacheHelper] = None

        if self.cache_enabled:

            self.cache = RAGCacheHelper(
                host=redis_host,
                port=redis_port,
                ttl_days=redis_cache_ttl_days
            )
            self.cache.check_and_start_redis()
            if self.cache.health_check():
                logger.info("✅ Redis cache initialized and healthy")
            else:
                logger.warning("⚠️ Redis health check failed, disabling cache")
                self.cache_enabled = False



        # --- Validate and Instantiate two distinct adapters ---
        AdapterClassGeneration = self.AVAILABLE_ADAPTERS.get(generation_model_backend_type)
        if not AdapterClassGeneration:
            raise ValueError(
                f"Unknown backend type: {generation_model_backend_type}. Must be one of: {list(self.AVAILABLE_ADAPTERS.keys())}")

        AdapterClassRefine = self.AVAILABLE_ADAPTERS.get(refinement_model_backend_type)
        if not AdapterClassRefine:
            raise ValueError(
                f"Unknown backend type: {refinement_model_backend_type}. Must be one of: {list(self.AVAILABLE_ADAPTERS.keys())}")

        # 1. Get model names from config
        if generation_model_key not in LLMConfig.AVAILABLE_MODELS:
            raise ValueError(f"Generator model key '{generation_model_key}' not found for backend '{generation_model_backend_type}'.")

        if refinement_model_key not in LLMConfig.AVAILABLE_MODELS:
            raise ValueError(f"Refiner model key '{refinement_model_key}' not found for backend '{refinement_model_backend_type}'.")

        gen_model_name = LLMConfig.AVAILABLE_MODELS[generation_model_key]["name"]
        ref_model_name = LLMConfig.AVAILABLE_MODELS[refinement_model_key]["name"]

        # 2. Instantiate Adapters
        # The adapter class is expected to implement LLMAdapter
        self.generator: LLMAdapter = AdapterClassGeneration(search_system, gen_model_name)
        self.refiner: LLMAdapter = AdapterClassRefine(search_system, ref_model_name)

        # Use the generator's name for final output tracking
        self.model_name = gen_model_name

        # Get the system prompt from the generator (main model)
        self.system_prompt = self.generator.system_prompt

        self.is_ready = False

        print(f"⚙️ Bridge created. Generator: {gen_model_name}, Refiner: {ref_model_name}")

    def setup_model(self) -> bool:
        """
        Sets up both the Generator and Refiner models.
        """
        print(f"Starting setup for Generator ({self.generator.model_name})...")
        gen_success = self.generator.setup()
        print(f"Starting setup for Refiner ({self.refiner.model_name})...")
        ref_success = self.refiner.setup()

        # Only consider the bridge ready if *both* models are successfully set up
        self.is_ready = gen_success and ref_success
        return self.is_ready

    def _update_collection_timestamp(self, timestamp: Optional[float] = None) -> None:
        """
        Update the collection's last update timestamp in Redis.
        Call this after indexing new documents or updating the collection.

        Args:
            timestamp: Unix timestamp (if None, uses current time)
        """
        if self.cache_enabled and self.cache:
            self.cache.set_collection_update_time(self.collection_name, timestamp)
            logger.info(f"📝 Collection '{self.collection_name}' timestamp updated")

    def invalidate_cache(self) -> int:
        """
        Manually invalidate all cached entries for the current collection.
        Useful after major updates.

        Returns:
            Number of cache entries deleted
        """
        if self.cache_enabled and self.cache:
            count = self.cache.invalidate_collection_cache(self.collection_name)
            self._update_collection_timestamp()
            return count
        return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for the current collection."""
        if self.cache_enabled and self.cache:
            return self.cache.get_cache_stats(self.collection_name)
        return {'cache_enabled': False}

    def _check_cached_answer(self, use_cache: bool, question: str, top_k: int, final_top_k: int, score_threshold: float):
        if self.cache_enabled and use_cache and self.cache:
            # Get collection's last update time
            collection_update_time = self.cache.get_collection_update_time(self.collection_name)

            # Try to get cached answer
            cached_answer = self.cache.get_cached_answer(
                question=question,
                collection_name=self.collection_name,
                collection_last_update=collection_update_time,
                top_k=top_k,
                final_top_k=final_top_k,
                score_threshold=score_threshold
            )

            if cached_answer:
                logger.info(f"✅ Returning cached answer for: '{question[:50]}...'")
                # Add cache indicator to response
                cached_answer['from_cache'] = True
                return cached_answer
        return None

    def _save_cached_answer(self, use_cache: bool, question: str, result: Dict[str, Any], top_k: int, final_top_k: int, score_threshold: float):
        if self.cache_enabled and use_cache and self.cache:
            try:
                self.cache.cache_answer(
                    question=question,
                    answer=result,
                    collection_name=self.collection_name,
                    top_k=top_k,
                    final_top_k=final_top_k,
                    score_threshold=score_threshold
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to cache answer: {e}")

    def _refine_query(self, question: str, max_token: int, temp: float):
        try:
            refine_prompt = f"Rewrite '\"{question}\"' into 3 concise, alternative search queries. Return them as a bullet list, without explanation."

            # --- Call the Refiner's generation method ---
            refined_output = self.refiner.ask(refine_prompt, max_token, temp)

            refined_queries = [q.strip("-• ").strip() for q in refined_output.splitlines() if q.strip()]
            refined_queries.insert(0, question)
            logger.info(f"🔍 Refined queries ({self.refiner.model_name}): {refined_queries}")
        except Exception as e:
            logger.error(f"Query refinement failed with refiner model: {e}")
            refined_queries = [question]
        return refined_queries

    def _get_search_results(self, question: str, use_hybrid: bool, refined_queries: List[str], top_k: int, final_top_k: int, score_threshold: float):
        if use_hybrid:
            search_results = self.search.hybrid_search(refined_queries, top_k=top_k, final_top_k=final_top_k,
                                                       score_threashold=score_threshold)
        else:
            search_results = self.search.semantic_search(refined_queries, top_k=top_k, final_top_k=final_top_k,
                                                         score_threashold=score_threshold)
        if not search_results:
            return False, {'question': question, 'answer': "I couldn't find any relevant information.", 'sources': [],
                    'model_used': self.model_name}

        if ENRICH_WITH_NEIGHBORS > -2:
            search_results = self.search.merge_adjacent_chunks_qdrant(search_results, k=ENRICH_WITH_NEIGHBORS)
        return True, search_results

    def ask(self, question: str, top_k: int = 10, final_top_k: int = 3, score_threshold: float = 0, use_cache: bool = True, use_hybrid: bool = True) -> Dict:
        """
        Performs the full RAG process using the Refiner for query expansion and 
        the Generator for the final answer.
        """
        if not self.generator.is_ready or not self.refiner.is_ready:
            raise RuntimeError("One or both LLM models are not set up or ready.")

        # --- CACHE CHECK ---
        cached_answer = self._check_cached_answer(use_cache, question, top_k, final_top_k, score_threshold)
        if cached_answer:
            return cached_answer

        # Step 0: Refine query using the *Refiner* LLM
        refined_queries = self._refine_query(question, MAX_TOKEN_REFINEMENT, TEMP_REFINEMENT)

        # Step 1: Perform semantic search using the refined queries
        result_found, search_results = self._get_search_results(question, use_hybrid, refined_queries, top_k, final_top_k, score_threshold)
        if not result_found:
            return search_results

        # Step 2: Format Context for the LLM
        context_pieces = []
        for i, result in enumerate(search_results, 1):
            context_pieces.append(
                {
                    'text': f"[({result.source}) Source {i} - Page title : {result.title}]\nPage link : {result.link}\nPage extract : {result.text.strip()}\n",
                    'title': result.title,
                    'link': result.link,
                    'score': result.score,
                    'source': result.source}
            )

        context = "\n\n".join([piece['text'] for piece in context_pieces])

        # Combine the system prompt, context, and question for the final LLM prompt
        final_prompt = f"{self.system_prompt}\n\nDOCUMENTATION:\n{context}\n\nQUESTION: {question}"

        # Step 3: Generate the Final Answer using the *Generator* LLM
        try:
            print(f"🤖 Generating answer with local LLM ({self.generator.model_name})...")
            # --- Call the Generator's generation method ---
            answer = self.generator.ask(final_prompt, MAX_TOKEN_GENERATION, TEMP_GENERATION)

        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
            # logger.error(f"LLM generation error with generator model: {e}")

        result = {
            'question': question,
            'answer': answer,
            'sources': [
                {
                    'source': piece['source'],
                    'title': piece['title'],
                    'link': piece['link'],
                    'score': piece['score']
                } for piece in context_pieces
            ],
            'model_used': self.model_name,
            'from_cache': False
        }

        # Step 4: Cache the answer
        self._save_cached_answer(use_cache, question, result, top_k, final_top_k, score_threshold)

        # Step 5: Format and Return Results
        return result