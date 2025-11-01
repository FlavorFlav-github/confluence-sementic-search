import pytest
from unittest.mock import Mock, MagicMock, patch, call
from typing import Any, Dict, List

# Import the class to test
from llm.bridge import LocalLLMBridge


@pytest.fixture
def mock_search_system():
    """Mock search system with all required methods."""
    search = Mock()
    search.semantic_search = Mock(return_value=[])
    search.hybrid_search = Mock(return_value=[])
    search.merge_adjacent_chunks_qdrant = Mock(side_effect=lambda x, k: x)
    return search


@pytest.fixture
def mock_search_result():
    """Create a mock search result object."""
    result = Mock()
    result.source = "test_source.pdf"
    result.title = "Test Document"
    result.link = "https://example.com/test"
    result.text = "This is test content"
    result.score = 0.95
    return result


@pytest.fixture
def mock_llm_config():
    """Mock LLMConfig with available models."""
    config = {
        "llama3": {"name": "llama3:latest"},
        "phi3.5_q4_K_M": {"name": "phi3.5:latest"},
        "gemini-pro": {"name": "gemini-1.5-pro"}
    }
    return config


@pytest.fixture
def mock_adapters():
    """Mock adapter classes."""
    mock_gen_adapter = Mock()
    mock_gen_adapter.return_value.model_name = "llama3:latest"
    mock_gen_adapter.return_value.system_prompt = "You are a helpful assistant."
    mock_gen_adapter.return_value.is_ready = True
    mock_gen_adapter.return_value.setup = Mock(return_value=True)
    mock_gen_adapter.return_value.ask = Mock(return_value="Generated answer")

    mock_ref_adapter = Mock()
    mock_ref_adapter.return_value.model_name = "phi3.5:latest"
    mock_ref_adapter.return_value.system_prompt = "You are a query refiner."
    mock_ref_adapter.return_value.is_ready = True
    mock_ref_adapter.return_value.setup = Mock(return_value=True)
    mock_ref_adapter.return_value.ask = Mock(return_value="- Query 1\n- Query 2\n- Query 3")

    return mock_gen_adapter, mock_ref_adapter


@pytest.fixture
def mock_cache():
    """Mock RAGCacheHelper."""
    cache = Mock()
    cache.check_and_start_redis = Mock()
    cache.health_check = Mock(return_value=True)
    cache.get_collection_update_time = Mock(return_value=None)
    cache.get_cached_answer = Mock(return_value=None)
    cache.cache_answer = Mock()
    cache.set_collection_update_time = Mock()
    cache.invalidate_collection_cache = Mock(return_value=5)
    cache.get_cache_stats = Mock(return_value={'hits': 10, 'misses': 5})
    return cache


class TestLocalLLMBridgeInitialization:
    """Test bridge initialization scenarios."""

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_init_successful_with_cache(self, mock_config, mock_cache,
                                        mock_search_system, mock_llm_config,
                                        mock_adapters, mock_search_result):
        """Test successful initialization with cache enabled."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, mock_ref_adapter = mock_adapters
        mock_cache.return_value.get_cached_answer.return_value = None
        mock_cache.return_value.get_collection_update_time.return_value = None

        # Make hybrid search return a valid search result object
        mock_search_system.hybrid_search.return_value = [mock_search_result]

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=True
            )

            result = bridge.ask("What is Python?", use_cache=True)
            assert bridge.cache_enabled is True
            assert result['from_cache'] is False
            bridge.search.hybrid_search.assert_called_once()
            bridge.cache.get_cached_answer.assert_called_once()
            bridge.cache.cache_answer.assert_called_once()

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_ask_cache_disabled_no_caching(self, mock_config, mock_cache_class,
                                           mock_search_system, mock_llm_config,
                                           mock_adapters, mock_search_result):
        """Test that ask() doesn't cache when cache is disabled."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, mock_ref_adapter = mock_adapters

        mock_search_system.hybrid_search = Mock(return_value=[mock_search_result])
        mock_search_system.merge_adjacent_chunks_qdrant = Mock(
            return_value=[mock_search_result]
        )

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=False
            )

            result = bridge.ask("What is Python?", use_cache=True)

            assert result['from_cache'] is False
            # Cache should not be used

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_ask_with_custom_parameters(self, mock_config, mock_cache_class,
                                        mock_search_system, mock_llm_config,
                                        mock_adapters, mock_cache, mock_search_result):
        """Test ask() with custom parameters."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, mock_ref_adapter = mock_adapters
        mock_cache_class.return_value = mock_cache

        mock_search_system.hybrid_search = Mock(return_value=[mock_search_result])
        mock_search_system.merge_adjacent_chunks_qdrant = Mock(
            return_value=[mock_search_result]
        )

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=False
            )

            result = bridge.ask(
                "What is Python?",
                top_k=20,
                final_top_k=5,
                score_threshold=0.7,
                use_hybrid=True
            )

            # Verify search was called with correct parameters
            mock_search_system.hybrid_search.assert_called_once()
            call_args = mock_search_system.hybrid_search.call_args
            assert call_args[1]['top_k'] == 20
            assert call_args[1]['final_top_k'] == 5
            assert call_args[1]['score_threashold'] == 0.7

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_ask_multiple_search_results(self, mock_config, mock_cache_class,
                                         mock_search_system, mock_llm_config,
                                         mock_adapters, mock_cache):
        """Test ask() with multiple search results."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, mock_ref_adapter = mock_adapters
        mock_cache_class.return_value = mock_cache

        # Create multiple mock results
        result1 = Mock()
        result1.source = "source1.pdf"
        result1.title = "Doc 1"
        result1.link = "https://example.com/1"
        result1.text = "Content 1"
        result1.score = 0.95

        result2 = Mock()
        result2.source = "source2.pdf"
        result2.title = "Doc 2"
        result2.link = "https://example.com/2"
        result2.text = "Content 2"
        result2.score = 0.85

        result3 = Mock()
        result3.source = "source3.pdf"
        result3.title = "Doc 3"
        result3.link = "https://example.com/3"
        result3.text = "Content 3"
        result3.score = 0.75

        mock_search_system.hybrid_search = Mock(return_value=[result1, result2, result3])
        mock_search_system.merge_adjacent_chunks_qdrant = Mock(
            return_value=[result1, result2, result3]
        )

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=False
            )

            result = bridge.ask("What is Python?")

            assert len(result['sources']) == 3
            assert result['sources'][0]['title'] == "Doc 1"
            assert result['sources'][1]['title'] == "Doc 2"
            assert result['sources'][2]['title'] == "Doc 3"
            assert result['sources'][0]['score'] == 0.95

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_ask_cache_fails_gracefully(self, mock_config, mock_cache_class,
                                        mock_search_system, mock_llm_config,
                                        mock_adapters, mock_cache, mock_search_result):
        """Test that ask() continues even if caching fails."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, mock_ref_adapter = mock_adapters

        # Configure cache to fail on cache_answer
        mock_cache.cache_answer = Mock(side_effect=Exception("Cache error"))
        mock_cache_class.return_value = mock_cache

        mock_search_system.hybrid_search = Mock(return_value=[mock_search_result])
        mock_search_system.merge_adjacent_chunks_qdrant = Mock(
            return_value=[mock_search_result]
        )

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=True
            )

            # Should not raise exception
            result = bridge.ask("What is Python?", use_cache=True)

            assert result['answer'] == "Generated answer"

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    @patch('llm.bridge.ENRICH_WITH_NEIGHBORS', -3)
    def test_ask_skips_enrichment_when_disabled(self, mock_config, mock_cache_class,
                                                mock_search_system, mock_llm_config,
                                                mock_adapters, mock_cache,
                                                mock_search_result):
        """Test that enrichment is skipped when ENRICH_WITH_NEIGHBORS is -3."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, mock_ref_adapter = mock_adapters
        mock_cache_class.return_value = mock_cache

        mock_search_system.hybrid_search = Mock(return_value=[mock_search_result])

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=False
            )

            result = bridge.ask("What is Python?")

            # merge_adjacent_chunks_qdrant should not be called
            mock_search_system.merge_adjacent_chunks_qdrant.assert_not_called()


class TestLocalLLMBridgeIntegration:
    """Integration-style tests for complete workflows."""

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_full_workflow_with_mixed_backends(self, mock_config, mock_cache_class,
                                               mock_search_system, mock_llm_config):
        """Test initialization with different backends for generation and refinement."""
        # Extend config with gemini
        extended_config = mock_llm_config.copy()
        extended_config["gemini-pro"] = {"name": "gemini-1.5-pro"}
        mock_config.AVAILABLE_MODELS = extended_config

        mock_ollama_adapter = Mock()
        mock_ollama_adapter.return_value.model_name = "phi3.5:latest"
        mock_ollama_adapter.return_value.system_prompt = "Refiner prompt"
        mock_ollama_adapter.return_value.is_ready = True
        mock_ollama_adapter.return_value.setup = Mock(return_value=True)
        mock_ollama_adapter.return_value.ask = Mock(return_value="- Query 1\n- Query 2")

        mock_gemini_adapter = Mock()
        mock_gemini_adapter.return_value.model_name = "gemini-1.5-pro"
        mock_gemini_adapter.return_value.system_prompt = "Generator prompt"
        mock_gemini_adapter.return_value.is_ready = True
        mock_gemini_adapter.return_value.setup = Mock(return_value=True)
        mock_gemini_adapter.return_value.ask = Mock(return_value="Gemini answer")

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_ollama_adapter,
            'gemini': mock_gemini_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="gemini-pro",
                refinement_model_key="phi3.5_q4_K_M",
                generation_model_backend_type="gemini",
                refinement_model_backend_type="ollama",
                enable_cache=False
            )

            assert bridge.generation_model_backend_type == "gemini"
            assert bridge.refinement_model_backend_type == "ollama"
            assert bridge.model_name == "gemini-1.5-pro"

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_context_formatting(self, mock_config, mock_cache_class,
                                mock_search_system, mock_llm_config,
                                mock_adapters, mock_cache):
        """Test that context is properly formatted for the LLM."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, mock_ref_adapter = mock_adapters
        mock_cache_class.return_value = mock_cache

        # Create mock result
        result = Mock()
        result.source = "test.pdf"
        result.title = "Test Title"
        result.link = "https://test.com"
        result.text = "Test content here"
        result.score = 0.9

        mock_search_system.hybrid_search = Mock(return_value=[result])
        mock_search_system.merge_adjacent_chunks_qdrant = Mock(return_value=[result])

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=False
            )

            bridge.ask("What is Python?")

            # Check that generator was called
            assert mock_gen_adapter.return_value.ask.called
            # Get the prompt that was passed
            call_args = mock_gen_adapter.return_value.ask.call_args
            prompt = call_args[0][0]

            # Verify prompt contains expected elements
            assert "DOCUMENTATION:" in prompt
            assert "QUESTION:" in prompt
            assert "Test Title" in prompt
            assert "https://test.com" in prompt
            assert "Test content here" in prompt

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_refined_queries_format(self, mock_config, mock_cache_class,
                                    mock_search_system, mock_llm_config,
                                    mock_adapters, mock_cache, mock_search_result):
        """Test that refined queries are properly parsed and used."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, mock_ref_adapter = mock_adapters

        # Configure refiner to return formatted queries
        mock_ref_adapter.return_value.ask = Mock(
            return_value="- What is Python programming\n- Python language basics\n• Python overview"
        )

        mock_cache_class.return_value = mock_cache
        mock_search_system.hybrid_search = Mock(return_value=[mock_search_result])
        mock_search_system.merge_adjacent_chunks_qdrant = Mock(
            return_value=[mock_search_result]
        )

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=False
            )

            bridge.ask("What is Python?")

            # Check that search was called
            assert mock_search_system.hybrid_search.called
            call_args = mock_search_system.hybrid_search.call_args
            queries = call_args[0][0]

            # Original question should be first
            assert queries[0] == "What is Python?"
            # Should have refined queries
            assert len(queries) > 1


class TestLocalLLMBridgeEdgeCases:
    """Test edge cases and error scenarios."""

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_empty_question(self, mock_config, mock_cache_class,
                            mock_search_system, mock_llm_config,
                            mock_adapters, mock_cache):
        """Test asking an empty question."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, mock_ref_adapter = mock_adapters
        mock_cache_class.return_value = mock_cache

        mock_search_system.hybrid_search = Mock(return_value=[])

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=False
            )

            result = bridge.ask("")

            assert "couldn't find any relevant information" in result['answer']

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_very_long_question(self, mock_config, mock_cache_class,
                                mock_search_system, mock_llm_config,
                                mock_adapters, mock_cache, mock_search_result):
        """Test asking a very long question."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, mock_ref_adapter = mock_adapters
        mock_cache_class.return_value = mock_cache

        long_question = "What is Python? " * 100

        mock_search_system.hybrid_search = Mock(return_value=[mock_search_result])
        mock_search_system.merge_adjacent_chunks_qdrant = Mock(
            return_value=[mock_search_result]
        )

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=False
            )

            result = bridge.ask(long_question)

            assert result['question'] == long_question
            assert result['answer'] == "Generated answer"

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_special_characters_in_question(self, mock_config, mock_cache_class,
                                            mock_search_system, mock_llm_config,
                                            mock_adapters, mock_cache,
                                            mock_search_result):
        """Test question with special characters."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, mock_ref_adapter = mock_adapters
        mock_cache_class.return_value = mock_cache

        special_question = "What's <Python> & how does it handle \"strings\"?"

        mock_search_system.hybrid_search = Mock(return_value=[mock_search_result])
        mock_search_system.merge_adjacent_chunks_qdrant = Mock(
            return_value=[mock_search_result]
        )

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=False
            )

            result = bridge.ask(special_question)

            assert result['question'] == special_question
            assert result['answer'] == "Generated answer"


class TestLocalLLMBridgeSetup:
    """Test model setup functionality."""

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_setup_model_both_succeed(self, mock_config, mock_cache_class,
                                      mock_search_system, mock_llm_config,
                                      mock_adapters):
        """Test setup when both models succeed."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, mock_ref_adapter = mock_adapters

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
            'gemini': mock_ref_adapter
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=False
            )

            result = bridge.setup_model()

            assert result is True
            assert bridge.is_ready is True
            bridge.generator.setup.call_count == 2

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_setup_model_generator_fails(self, mock_config, mock_cache_class,
                                         mock_search_system, mock_llm_config,
                                         mock_adapters):
        """Test setup when generator fails."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, mock_ref_adapter = mock_adapters

        # Configure generator to fail
        mock_gen_adapter.return_value.setup = Mock(return_value=False)

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=False
            )

            result = bridge.setup_model()

            assert result is False
            assert bridge.is_ready is False

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_setup_model_refiner_fails(self, mock_config, mock_cache_class,
                                       mock_search_system, mock_llm_config,
                                       mock_adapters):
        """Test setup when refiner fails."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, mock_ref_adapter = mock_adapters

        # Configure refiner to fail
        mock_ref_adapter.return_value.setup = Mock(return_value=False)

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_ref_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=False
            )

            result = bridge.setup_model()

            assert result is False
            assert bridge.is_ready is False


class TestLocalLLMBridgeCacheOperations:
    """Test cache-related operations."""

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_invalidate_cache_enabled(self, mock_config, mock_cache_class,
                                      mock_search_system, mock_llm_config,
                                      mock_adapters, mock_cache):
        """Test cache invalidation when cache is enabled."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, _ = mock_adapters
        mock_cache_class.return_value = mock_cache

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=True
            )

            result = bridge.invalidate_cache()

            assert result == 5
            mock_cache.invalidate_collection_cache.assert_called_once()
            mock_cache.set_collection_update_time.assert_called_once()

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_invalidate_cache_disabled(self, mock_config, mock_cache_class,
                                       mock_search_system, mock_llm_config,
                                       mock_adapters):
        """Test cache invalidation when cache is disabled."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, _ = mock_adapters

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=False
            )

            result = bridge.invalidate_cache()

            assert result == 0

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_get_cache_stats_enabled(self, mock_config, mock_cache_class,
                                     mock_search_system, mock_llm_config,
                                     mock_adapters, mock_cache):
        """Test getting cache stats when cache is enabled."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, _ = mock_adapters
        mock_cache_class.return_value = mock_cache

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=True
            )

            stats = bridge.get_cache_stats()

            assert stats == {'hits': 10, 'misses': 5}
            mock_cache.get_cache_stats.assert_called_once()

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_get_cache_stats_disabled(self, mock_config, mock_cache_class,
                                      mock_search_system, mock_llm_config,
                                      mock_adapters):
        """Test getting cache stats when cache is disabled."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, _ = mock_adapters

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=False
            )

            stats = bridge.get_cache_stats()

            assert stats == {'cache_enabled': False}

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_update_collection_timestamp(self, mock_config, mock_cache_class,
                                         mock_search_system, mock_llm_config,
                                         mock_adapters, mock_cache):
        """Test updating collection timestamp."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, _ = mock_adapters
        mock_cache_class.return_value = mock_cache

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=True
            )

            bridge._update_collection_timestamp(123456.0)

            mock_cache.set_collection_update_time.assert_called_once_with(
                bridge.collection_name, 123456.0
            )


class TestLocalLLMBridgeAsk:
    """Test the main ask() method."""

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_ask_models_not_ready(self, mock_config, mock_cache_class,
                                  mock_search_system, mock_llm_config,
                                  mock_adapters):
        """Test ask() when models are not ready."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, mock_ref_adapter = mock_adapters

        # Set models as not ready
        mock_gen_adapter.return_value.is_ready = False
        mock_ref_adapter.return_value.is_ready = False

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=False
            )

            with pytest.raises(RuntimeError, match="One or both LLM models are not set up"):
                bridge.ask("What is Python?")

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_ask_with_cache_hit(self, mock_config, mock_cache_class,
                                mock_search_system, mock_llm_config,
                                mock_adapters, mock_cache):
        """Test ask() with cache hit."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, mock_ref_adapter = mock_adapters

        cached_response = {
            'question': 'What is Python?',
            'answer': 'Cached answer',
            'sources': [],
            'model_used': 'llama3:latest'
        }
        mock_cache.get_cached_answer = Mock(return_value=cached_response)
        mock_cache_class.return_value = mock_cache

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=True
            )

            result = bridge.ask("What is Python?", use_cache=True)

            assert result['from_cache'] is True
            assert result['answer'] == 'Cached answer'
            mock_cache.get_cached_answer.assert_called_once()
            # Search should not be called on cache hit
            mock_search_system.semantic_search.assert_not_called()

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_ask_no_search_results(self, mock_config, mock_cache_class,
                                   mock_search_system, mock_llm_config,
                                   mock_adapters, mock_cache):
        """Test ask() when no search results are found."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, mock_ref_adapter = mock_adapters
        mock_cache_class.return_value = mock_cache

        # No search results
        mock_search_system.hybrid_search = Mock(return_value=[])

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=False
            )

            result = bridge.ask("What is Python?")

            assert "couldn't find any relevant information" in result['answer']
            assert result['sources'] == []

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_ask_successful_with_hybrid_search(self, mock_config, mock_cache_class,
                                               mock_search_system, mock_llm_config,
                                               mock_adapters, mock_cache,
                                               mock_search_result):
        """Test successful ask() with hybrid search."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, mock_ref_adapter = mock_adapters
        mock_cache_class.return_value = mock_cache

        # Configure search to return results
        mock_search_system.hybrid_search = Mock(return_value=[mock_search_result])
        mock_search_system.merge_adjacent_chunks_qdrant = Mock(
            return_value=[mock_search_result]
        )

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=False
            )

            result = bridge.ask("What is Python?", use_hybrid=True)

            assert result['question'] == "What is Python?"
            assert result['answer'] == "Generated answer"
            assert len(result['sources']) == 1
            assert result['sources'][0]['title'] == "Test Document"
            assert result['from_cache'] is False
            mock_search_system.hybrid_search.assert_called_once()

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_ask_successful_with_semantic_search(self, mock_config, mock_cache_class,
                                                 mock_search_system, mock_llm_config,
                                                 mock_adapters, mock_cache,
                                                 mock_search_result):
        """Test successful ask() with semantic search."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, mock_ref_adapter = mock_adapters
        mock_cache_class.return_value = mock_cache

        # Configure search to return results
        mock_search_system.semantic_search = Mock(return_value=[mock_search_result])
        mock_search_system.merge_adjacent_chunks_qdrant = Mock(
            return_value=[mock_search_result]
        )

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=False
            )

            result = bridge.ask("What is Python?", use_hybrid=False)

            assert result['question'] == "What is Python?"
            assert result['answer'] == "Generated answer"
            mock_search_system.semantic_search.assert_called_once()
            mock_search_system.hybrid_search.assert_not_called()

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_ask_refinement_fails(self, mock_config, mock_cache_class,
                                  mock_search_system, mock_llm_config,
                                  mock_adapters, mock_cache, mock_search_result):
        """Test ask() when query refinement fails."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, mock_ref_adapter = mock_adapters
        mock_cache_class.return_value = mock_cache

        # Configure refiner to raise exception
        mock_ref_adapter.return_value.ask = Mock(side_effect=Exception("Refiner error"))

        mock_search_system.hybrid_search = Mock(return_value=[mock_search_result])
        mock_search_system.merge_adjacent_chunks_qdrant = Mock(
            return_value=[mock_search_result]
        )

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=False
            )

            result = bridge.ask("What is Python?")

            # Should fall back to original question
            assert result['answer'] == "Generated answer"
            # Search should still be called with original question
            mock_search_system.hybrid_search.assert_called_once()

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_ask_generation_fails(self, mock_config, mock_cache_class,
                                  mock_search_system, mock_llm_config,
                                  mock_adapters, mock_cache, mock_search_result):
        """Test ask() when answer generation fails."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, mock_ref_adapter = mock_adapters
        mock_cache_class.return_value = mock_cache

        # Configure generator to raise exception
        mock_gen_adapter.return_value.ask = Mock(
            side_effect=Exception("Generation error")
        )

        mock_search_system.hybrid_search = Mock(return_value=[mock_search_result])
        mock_search_system.merge_adjacent_chunks_qdrant = Mock(
            return_value=[mock_search_result]
        )

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=False
            )

            result = bridge.ask("What is Python?")

            assert "Error generating answer" in result['answer']

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_ask_caches_answer(self, mock_config, mock_cache_class,
                               mock_search_system, mock_llm_config,
                               mock_adapters, mock_cache, mock_search_result):
        """Test that ask() caches the answer."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, mock_ref_adapter = mock_adapters
        mock_cache_class.return_value = mock_cache

        mock_search_system.hybrid_search = Mock(return_value=[mock_search_result])
        mock_search_system.merge_adjacent_chunks_qdrant = Mock(
            return_value=[mock_search_result]
        )

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=True,
                redis_host='localhost',
                redis_port=6379
            )

            assert bridge.generation_model_key == "llama3"
            assert bridge.refinement_model_key == "phi3.5_q4_K_M"
            assert bridge.cache_enabled is True
            assert bridge.is_ready is False
            mock_cache_class.assert_called_once()

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_init_without_cache(self, mock_config, mock_cache_class,
                                mock_search_system, mock_llm_config, mock_adapters):
        """Test initialization with cache disabled."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, mock_ref_adapter = mock_adapters

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=False
            )

            assert bridge.cache_enabled is False
            assert bridge.cache is None
            mock_cache_class.assert_not_called()

    @patch('llm.bridge.LLMConfig')
    def test_init_invalid_backend_type_generation(self, mock_config,
                                                  mock_search_system,
                                                  mock_llm_config):
        """Test initialization with invalid generation backend type."""
        mock_config.AVAILABLE_MODELS = mock_llm_config

        with pytest.raises(ValueError, match="Unknown backend type: invalid_backend"):
            LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                generation_model_backend_type="invalid_backend",
                enable_cache=False
            )

    @patch('llm.bridge.LLMConfig')
    def test_init_invalid_backend_type_refinement(self, mock_config,
                                                  mock_search_system,
                                                  mock_llm_config, mock_adapters):
        """Test initialization with invalid refinement backend type."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, _ = mock_adapters

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            with pytest.raises(ValueError, match="Unknown backend type: invalid_backend"):
                LocalLLMBridge(
                    search_system=mock_search_system,
                    generation_model_key="llama3",
                    refinement_model_key="phi3.5_q4_K_M",
                    refinement_model_backend_type="invalid_backend",
                    enable_cache=False
                )

    @patch('llm.bridge.LLMConfig')
    def test_init_invalid_generation_model_key(self, mock_config,
                                               mock_search_system,
                                               mock_llm_config, mock_adapters):
        """Test initialization with invalid generation model key."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, _ = mock_adapters

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            with pytest.raises(ValueError, match="Generator model key 'invalid_key' not found"):
                LocalLLMBridge(
                    search_system=mock_search_system,
                    generation_model_key="invalid_key",
                    refinement_model_key="phi3.5_q4_K_M",
                    enable_cache=False
                )

    @patch('llm.bridge.LLMConfig')
    def test_init_invalid_refinement_model_key(self, mock_config,
                                               mock_search_system,
                                               mock_llm_config, mock_adapters):
        """Test initialization with invalid refinement model key."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, _ = mock_adapters

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            with pytest.raises(ValueError, match="Refiner model key 'invalid_key' not found"):
                LocalLLMBridge(
                    search_system=mock_search_system,
                    generation_model_key="llama3",
                    refinement_model_key="invalid_key",
                    enable_cache=False
                )

    @patch('llm.bridge.RAGCacheHelper')
    @patch('llm.bridge.LLMConfig')
    def test_init_cache_health_check_fails(self, mock_config, mock_cache_class,
                                           mock_search_system, mock_llm_config,
                                           mock_adapters):
        """Test initialization when cache health check fails."""
        mock_config.AVAILABLE_MODELS = mock_llm_config
        mock_gen_adapter, mock_ref_adapter = mock_adapters

        # Configure cache to fail health check
        mock_cache_instance = Mock()
        mock_cache_instance.check_and_start_redis = Mock()
        mock_cache_instance.health_check = Mock(return_value=False)
        mock_cache_class.return_value = mock_cache_instance

        with patch.dict('llm.bridge.LocalLLMBridge.AVAILABLE_ADAPTERS', {
            'ollama': mock_gen_adapter,
        }):
            bridge = LocalLLMBridge(
                search_system=mock_search_system,
                generation_model_key="llama3",
                refinement_model_key="phi3.5_q4_K_M",
                enable_cache=True
            )

            assert bridge.cache_enabled is False

# Pytest configuration and helper functions
@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks between tests."""
    yield