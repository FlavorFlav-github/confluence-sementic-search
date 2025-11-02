import pytest
from unittest.mock import Mock, patch
from collections import namedtuple

from search.advanced_search import AdvancedSearch
from search.models import SearchResult

# Mock Qdrant point structure
MockQdrantPoint = namedtuple('MockQdrantPoint', ['id', 'score', 'payload'])


class TestAdvancedSearch:
    """Comprehensive unit tests for AdvancedSearch class."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Creates a mock Qdrant client."""
        return Mock()

    @pytest.fixture
    def mock_hybrid_search_index(self):
        """Creates a mock hybrid search index."""
        mock_index = Mock()
        mock_index.is_fitted = True
        return mock_index

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Creates a mock SentenceTransformer."""
        return Mock()

    @pytest.fixture
    def advanced_search(self, mock_qdrant_client, mock_hybrid_search_index, mock_sentence_transformer):
        """Creates an AdvancedSearch instance with mocked dependencies."""
        with patch('search.advanced_search.SentenceTransformer', return_value=mock_sentence_transformer):
            search = AdvancedSearch(mock_qdrant_client, mock_hybrid_search_index)
            search.embed_model = mock_sentence_transformer
        return search

    @pytest.fixture
    def sample_payload(self):
        """Returns a sample payload structure."""
        return {
            'page_id': 'page_123',
            'title': 'Test Document',
            'source': 'test_source',
            'text': 'This is test content for the document.',
            'position': 0,
            'link': 'http://example.com/page',
            'last_updated': '2024-01-01',
            'chunk_id': 'page_123_0',
            'hierarchy': ['Section 1', 'Subsection 1.1']
        }

    # ==================== Test preprocess_query ====================

    def test_preprocess_query_removes_special_characters(self, advanced_search):
        """Test that special characters are removed from query."""
        query = "test@query#with$special%chars!"
        result = advanced_search.preprocess_query(query)
        assert result == "test query with special chars"

    def test_preprocess_query_preserves_hyphens_and_periods(self, advanced_search):
        """Test that hyphens and periods are preserved."""
        query = "test-query with.periods"
        result = advanced_search.preprocess_query(query)
        assert result == "test-query with.periods"

    def test_preprocess_query_normalizes_whitespace(self, advanced_search):
        """Test that multiple spaces are normalized to single space."""
        query = "test   query    with     spaces"
        result = advanced_search.preprocess_query(query)
        assert result == "test query with spaces"

    def test_preprocess_query_strips_leading_trailing_spaces(self, advanced_search):
        """Test that leading and trailing spaces are removed."""
        query = "   test query   "
        result = advanced_search.preprocess_query(query)
        assert result == "test query"

    def test_preprocess_query_empty_string(self, advanced_search):
        """Test preprocessing of empty string."""
        query = ""
        result = advanced_search.preprocess_query(query)
        assert result == ""

    # ==================== Test semantic_search ====================

    @patch('search.advanced_search.common.embed_text')
    def test_semantic_search_single_query(self, mock_embed, advanced_search, mock_qdrant_client, sample_payload):
        """Test semantic search with a single query."""
        # Setup mocks
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        mock_qdrant_client.search.return_value = [
            MockQdrantPoint(id='1', score=0.95, payload=sample_payload)
        ]

        # Execute
        results = advanced_search.semantic_search(queries=["test query"], top_k=5, final_top_k=3)

        # Verify
        assert len(results) == 1
        assert results[0].page_id == 'page_123'
        assert results[0].score > 0
        assert results[0].semantic_score == 0.95
        assert results[0].keyword_score == 0.0
        mock_qdrant_client.search.assert_called_once()

    @patch('search.advanced_search.common.embed_text')
    def test_semantic_search_multiple_queries(self, mock_embed, advanced_search, mock_qdrant_client, sample_payload):
        """Test semantic search with multiple queries."""
        # Setup mocks
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        mock_qdrant_client.search.return_value = [
            MockQdrantPoint(id='1', score=0.95, payload=sample_payload)
        ]

        # Execute
        results = advanced_search.semantic_search(
            queries=["query1", "query2"],
            top_k=5,
            final_top_k=3
        )

        # Verify
        assert len(results) >= 1
        assert mock_qdrant_client.search.call_count == 2

    @patch('search.advanced_search.common.embed_text')
    def test_semantic_search_with_page_id_filter(self, mock_embed, advanced_search, mock_qdrant_client, sample_payload):
        """Test semantic search with page_id filter."""
        # Setup mocks
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        mock_qdrant_client.search.return_value = [
            MockQdrantPoint(id='1', score=0.95, payload=sample_payload)
        ]

        # Execute
        filters = {'page_ids': ['page_123', 'page_456']}
        _ = advanced_search.semantic_search(
            queries=["test"],
            filters=filters,
            final_top_k=3
        )

        # Verify search was called with filter
        call_args = mock_qdrant_client.search.call_args
        assert call_args[1]['query_filter'] is not None

    @patch('search.advanced_search.common.embed_text')
    def test_semantic_search_with_space_key_filter(self, mock_embed, advanced_search, mock_qdrant_client,
                                                   sample_payload):
        """Test semantic search with space_key filter."""
        # Setup mocks
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        mock_qdrant_client.search.return_value = [
            MockQdrantPoint(id='1', score=0.95, payload=sample_payload)
        ]

        # Execute
        filters = {'space_key': ['my_space']}
        _ = advanced_search.semantic_search(
            queries=["test"],
            filters=filters,
            final_top_k=3
        )

        # Verify
        call_args = mock_qdrant_client.search.call_args
        assert call_args[1]['query_filter'] is not None

    @patch('search.advanced_search.common.embed_text')
    def test_semantic_search_with_min_text_length_filter(self, mock_embed, advanced_search, mock_qdrant_client,
                                                         sample_payload):
        """Test semantic search with min_text_length filter."""
        # Setup mocks
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        mock_qdrant_client.search.return_value = [
            MockQdrantPoint(id='1', score=0.95, payload=sample_payload)
        ]

        # Execute
        filters = {'min_text_length': 100}
        _ = advanced_search.semantic_search(
            queries=["test"],
            filters=filters,
            final_top_k=3
        )

        # Verify
        call_args = mock_qdrant_client.search.call_args
        assert call_args[1]['query_filter'] is not None

    @patch('search.advanced_search.common.embed_text')
    def test_semantic_search_score_threshold(self, mock_embed, advanced_search, mock_qdrant_client, sample_payload):
        """Test semantic search filters results by score threshold."""
        # Setup mocks
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        payload_low = sample_payload.copy()
        payload_high = sample_payload.copy()
        payload_high['page_id'] = 'page_456'

        mock_qdrant_client.search.return_value = [
            MockQdrantPoint(id='1', score=0.3, payload=payload_low),
            MockQdrantPoint(id='2', score=0.8, payload=payload_high)
        ]

        # Execute with threshold
        results = advanced_search.semantic_search(
            queries=["test"],
            score_threashold=0.5,
            final_top_k=3
        )

        # Verify only high score result is included
        assert len(results) == 1
        assert results[0].page_id == 'page_456'

    @patch('search.advanced_search.common.embed_text')
    def test_semantic_search_aggregation_by_page(self, mock_embed, advanced_search, mock_qdrant_client, sample_payload):
        """Test that results from same page are aggregated correctly."""
        # Setup mocks - multiple chunks from same page
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        payload1 = sample_payload.copy()
        payload1['chunk_id'] = 'page_123_0'
        payload2 = sample_payload.copy()
        payload2['chunk_id'] = 'page_123_1'

        mock_qdrant_client.search.return_value = [
            MockQdrantPoint(id='1', score=0.9, payload=payload1),
            MockQdrantPoint(id='2', score=0.8, payload=payload2)
        ]

        # Execute
        results = advanced_search.semantic_search(queries=["test"], final_top_k=3)

        # Verify aggregation - should return 1 result for the page
        assert len(results) == 1
        # Score should be boosted by occurrence
        assert results[0].score > 0.9

    # ==================== Test fetch_adjacent_chunks ====================

    def test_fetch_adjacent_chunks_k_equals_zero(self, advanced_search):
        """Test that k=0 returns empty list."""
        result = SearchResult(
            page_id='page_123',
            title='Test',
            source='test',
            text='test',
            score=0.9,
            semantic_score=0.9,
            keyword_score=0.0,
            position=0,
            link='link',
            last_updated='2024-01-01',
            chunk_id='page_123_5',
            page_hierarchy=[]
        )

        adjacent = advanced_search.fetch_adjacent_chunks(result, k=0)
        assert adjacent == []

    def test_fetch_adjacent_chunks_k_equals_one(self, advanced_search, mock_qdrant_client, sample_payload):
        """Test fetching 1 chunk before and after."""
        result = SearchResult(
            page_id='page_123',
            title='Test',
            source='test',
            text='test',
            score=0.9,
            semantic_score=0.9,
            keyword_score=0.0,
            position=0,
            link='link',
            last_updated='2024-01-01',
            chunk_id='page_123_5',
            page_hierarchy=[]
        )

        # Mock scroll responses
        payload_before = sample_payload.copy()
        payload_before['chunk_id'] = 'page_123_4'
        payload_after = sample_payload.copy()
        payload_after['chunk_id'] = 'page_123_6'

        mock_qdrant_client.scroll.side_effect = [
            ([MockQdrantPoint(id='1', score=0.8, payload=payload_before)], None),
            ([MockQdrantPoint(id='2', score=0.8, payload=payload_after)], None)
        ]

        adjacent = advanced_search.fetch_adjacent_chunks(result, k=1)

        assert len(adjacent) == 2
        assert mock_qdrant_client.scroll.call_count == 2

    def test_fetch_adjacent_chunks_k_equals_minus_one(self, advanced_search, mock_qdrant_client, sample_payload):
        """Test fetching all chunks for a page (k=-1)."""
        result = SearchResult(
            page_id='page_123',
            title='Test',
            source='test',
            text='test',
            score=0.9,
            semantic_score=0.9,
            keyword_score=0.0,
            position=0,
            link='link',
            last_updated='2024-01-01',
            chunk_id='page_123_5',
            page_hierarchy=[]
        )

        # Mock scroll response with multiple chunks
        payloads = []
        for i in range(10):
            if i != 5:  # Exclude the original chunk
                p = sample_payload.copy()
                p['chunk_id'] = f'page_123_{i}'
                payloads.append(MockQdrantPoint(id=str(i), score=0.8, payload=p))

        mock_qdrant_client.scroll.return_value = (payloads, None)

        adjacent = advanced_search.fetch_adjacent_chunks(result, k=-1)

        assert len(adjacent) == 9  # 10 total - 1 original
        mock_qdrant_client.scroll.assert_called_once()

    def test_fetch_adjacent_chunks_invalid_chunk_id_format(self, advanced_search):
        """Test handling of invalid chunk_id format."""
        result = SearchResult(
            page_id='page_123',
            title='Test',
            source='test',
            text='test',
            score=0.9,
            semantic_score=0.9,
            keyword_score=0.0,
            position=0,
            link='link',
            last_updated='2024-01-01',
            chunk_id='invalid_chunk_id',  # No underscore and number
            page_hierarchy=[]
        )

        adjacent = advanced_search.fetch_adjacent_chunks(result, k=1)
        assert adjacent == []

    # ==================== Test merge_adjacent_chunks_qdrant ====================

    def test_merge_adjacent_chunks_qdrant_k_equals_one(self, advanced_search, mock_qdrant_client, sample_payload):
        """Test merging adjacent chunks with k=1."""
        result = SearchResult(
            page_id='page_123',
            title='Test',
            source='test',
            text='Main chunk text',
            score=0.9,
            semantic_score=0.9,
            keyword_score=0.0,
            position=0,
            link='link',
            last_updated='2024-01-01',
            chunk_id='page_123_5',
            page_hierarchy=[]
        )

        # Mock adjacent chunks
        payload_before = sample_payload.copy()
        payload_before['chunk_id'] = 'page_123_4'
        payload_before['text'] = 'Before chunk text'
        payload_after = sample_payload.copy()
        payload_after['chunk_id'] = 'page_123_6'
        payload_after['text'] = 'After chunk text'

        mock_qdrant_client.scroll.side_effect = [
            ([MockQdrantPoint(id='1', score=0.8, payload=payload_before)], None),
            ([MockQdrantPoint(id='2', score=0.8, payload=payload_after)], None)
        ]

        merged = advanced_search.merge_adjacent_chunks_qdrant([result], k=1)

        assert len(merged) == 1
        assert 'Before chunk text' in merged[0].text
        assert 'Main chunk text' in merged[0].text
        assert 'After chunk text' in merged[0].text
        # Check order
        assert merged[0].text.index('Before') < merged[0].text.index('Main') < merged[0].text.index('After')

    def test_merge_adjacent_chunks_qdrant_preserves_metadata(self, advanced_search, mock_qdrant_client):
        """Test that merging preserves original metadata."""
        result = SearchResult(
            page_id='page_123',
            title='Test Title',
            source='test_source',
            text='Main text',
            score=0.95,
            semantic_score=0.9,
            keyword_score=0.05,
            position=5,
            link='http://test.com',
            last_updated='2024-01-01',
            chunk_id='page_123_5',
            page_hierarchy=['Section 1']
        )

        mock_qdrant_client.scroll.return_value = ([], None)

        merged = advanced_search.merge_adjacent_chunks_qdrant([result], k=1)

        assert merged[0].page_id == 'page_123'
        assert merged[0].title == 'Test Title'
        assert merged[0].score == 0.95
        assert merged[0].semantic_score == 0.9
        assert merged[0].keyword_score == 0.05

    def test_merge_adjacent_chunks_qdrant_multiple_results(self, advanced_search, mock_qdrant_client):
        """Test merging for multiple search results."""
        results = [
            SearchResult(
                page_id='page_123',
                title='Test1',
                source='test',
                text='Text1',
                score=0.9,
                semantic_score=0.9,
                keyword_score=0.0,
                position=0,
                link='link1',
                last_updated='2024-01-01',
                chunk_id='page_123_5',
                page_hierarchy=[]
            ),
            SearchResult(
                page_id='page_456',
                title='Test2',
                source='test',
                text='Text2',
                score=0.8,
                semantic_score=0.8,
                keyword_score=0.0,
                position=0,
                link='link2',
                last_updated='2024-01-01',
                chunk_id='page_456_3',
                page_hierarchy=[]
            )
        ]

        mock_qdrant_client.scroll.return_value = ([], None)

        merged = advanced_search.merge_adjacent_chunks_qdrant(results, k=1)

        assert len(merged) == 2

    # ==================== Test hybrid_search ====================

    @patch('search.advanced_search.common.embed_text')
    def test_hybrid_search_semantic_only(self, mock_embed, advanced_search, mock_qdrant_client,
                                         mock_hybrid_search_index, sample_payload):
        """Test hybrid search when keyword index is not fitted."""
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        mock_qdrant_client.search.return_value = [
            MockQdrantPoint(id='1', score=0.95, payload=sample_payload)
        ]
        mock_hybrid_search_index.is_fitted = False

        results = advanced_search.hybrid_search(queries=["test query"], final_top_k=3)

        assert len(results) >= 1
        assert results[0].keyword_score == 0.0

    @patch('search.advanced_search.common.embed_text')
    def test_hybrid_search_with_keyword_results(self, mock_embed, advanced_search, mock_qdrant_client,
                                                mock_hybrid_search_index, sample_payload):
        """Test hybrid search combining semantic and keyword scores."""
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        mock_qdrant_client.search.return_value = [
            MockQdrantPoint(id='1', score=0.9, payload=sample_payload)
        ]

        # Mock keyword search results
        mock_hybrid_search_index.is_fitted = True
        mock_hybrid_search_index.keyword_search.return_value = [
            ('page_123_0', 0.8)
        ]

        results = advanced_search.hybrid_search(queries=["test"], final_top_k=3, alpha=0.5)

        assert len(results) >= 1
        # Should have both semantic and keyword scores
        assert results[0].semantic_score > 0
        assert results[0].keyword_score > 0

    @patch('search.advanced_search.common.embed_text')
    def test_hybrid_search_alpha_weighting(self, mock_embed, advanced_search, mock_qdrant_client,
                                           mock_hybrid_search_index, sample_payload):
        """Test that alpha parameter correctly weights semantic vs keyword scores."""
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        mock_qdrant_client.search.return_value = [
            MockQdrantPoint(id='1', score=0.9, payload=sample_payload)
        ]

        mock_hybrid_search_index.is_fitted = True
        mock_hybrid_search_index.keyword_search.return_value = [
            ('page_123_0', 0.6)
        ]

        # Test with alpha=0.7 (70% semantic, 30% keyword)
        results = advanced_search.hybrid_search(queries=["test"], final_top_k=3, alpha=0.7)

        expected_score = 0.7 * 0.9 + 0.3 * 0.6
        assert abs(results[0].score - expected_score) < 0.01

    @patch('search.advanced_search.common.embed_text')
    def test_hybrid_search_multiple_queries(self, mock_embed, advanced_search, mock_qdrant_client,
                                            mock_hybrid_search_index, sample_payload):
        """Test hybrid search with multiple queries."""
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        mock_qdrant_client.search.return_value = [
            MockQdrantPoint(id='1', score=0.9, payload=sample_payload)
        ]

        mock_hybrid_search_index.is_fitted = True
        mock_hybrid_search_index.keyword_search.return_value = [('page_123_0', 0.8)]

        _ = advanced_search.hybrid_search(queries=["query1", "query2"], final_top_k=3)

        assert mock_hybrid_search_index.keyword_search.call_count == 2

    @patch('search.advanced_search.common.embed_text')
    def test_hybrid_search_score_threshold(self, mock_embed, advanced_search, mock_qdrant_client,
                                           mock_hybrid_search_index, sample_payload):
        """Test hybrid search with score threshold."""
        mock_embed.return_value = [[0.1, 0.2, 0.3]]

        payload_low = sample_payload.copy()
        payload_high = sample_payload.copy()
        payload_high['page_id'] = 'page_456'
        payload_high['chunk_id'] = 'page_456_0'

        mock_qdrant_client.search.return_value = [
            MockQdrantPoint(id='1', score=0.3, payload=payload_low),
            MockQdrantPoint(id='2', score=0.9, payload=payload_high)
        ]

        mock_hybrid_search_index.is_fitted = False

        results = advanced_search.hybrid_search(
            queries=["test"],
            score_threashold=0.5,
            final_top_k=5
        )

        # Only high scoring result should pass threshold
        assert all(r.semantic_score > 0.5 for r in results)


    # ==================== Test explain_results ====================

    def test_explain_results_prints_correctly(self, advanced_search, capsys):
        """Test that explain_results prints formatted output."""
        results = [
            SearchResult(
                page_id='page_123',
                title='Test Document',
                source='test_source',
                text='This is a test document content',
                score=0.95,
                semantic_score=0.9,
                keyword_score=0.05,
                position=0,
                link='http://example.com',
                last_updated='2024-01-01',
                chunk_id='page_123_0',
                page_hierarchy=['Section 1', 'Subsection 1.1']
            )
        ]

        advanced_search.explain_results(results, "test query")

        captured = capsys.readouterr()
        assert "SEARCH RESULTS FOR: 'test query'" in captured.out
        assert "Test Document" in captured.out
        assert "0.9500" in captured.out
        assert "http://example.com" in captured.out

    def test_explain_results_handles_empty_hierarchy(self, advanced_search, capsys):
        """Test explain_results handles empty page_hierarchy."""
        results = [
            SearchResult(
                page_id='page_123',
                title='Test',
                source='test',
                text='content',
                score=0.9,
                semantic_score=0.9,
                keyword_score=0.0,
                position=0,
                link='link',
                last_updated='2024-01-01',
                chunk_id='page_123_0',
                page_hierarchy=[]
            )
        ]

        advanced_search.explain_results(results, "test")

        captured = capsys.readouterr()
        assert "N/A" in captured.out

    def test_explain_results_truncates_long_text(self, advanced_search, capsys):
        """Test that explain_results truncates long text snippets."""
        long_text = "a" * 500
        results = [
            SearchResult(
                page_id='page_123',
                title='Test',
                source='test',
                text=long_text,
                score=0.9,
                semantic_score=0.9,
                keyword_score=0.0,
                position=0,
                link='link',
                last_updated='2024-01-01',
                chunk_id='page_123_0',
                page_hierarchy=[]
            )
        ]

        advanced_search.explain_results(results, "test")

        captured = capsys.readouterr()
        # Should show truncated text with ellipsis
        assert "..." in captured.out
        assert len(captured.out) < len(long_text) + 500  # Much shorter than full text

    # ==================== Integration-style Tests ====================

    @patch('search.advanced_search.common.embed_text')
    def test_full_search_workflow(self, mock_embed, advanced_search, mock_qdrant_client,
                                  mock_hybrid_search_index, sample_payload):
        """Test a complete search workflow from query to results."""
        # Setup
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        mock_qdrant_client.search.return_value = [
            MockQdrantPoint(id='1', score=0.95, payload=sample_payload)
        ]
        mock_hybrid_search_index.is_fitted = True
        mock_hybrid_search_index.keyword_search.return_value = [('page_123_0', 0.8)]

        # Execute hybrid search
        results = advanced_search.hybrid_search(
            queries=["test query"],
            top_k=10,
            final_top_k=3,
            alpha=0.7
        )

        # Verify
        assert len(results) >= 1
        assert results[0].page_id == 'page_123'
        assert results[0].semantic_score > 0
        assert results[0].keyword_score > 0

        # Merge adjacent chunks
        payload_adjacent = sample_payload.copy()
        payload_adjacent['chunk_id'] = 'page_123_1'
        payload_adjacent['text'] = 'Adjacent content'
        mock_qdrant_client.scroll.return_value = (
            [MockQdrantPoint(id='2', score=0.9, payload=payload_adjacent)],
            None
        )

        merged_results = advanced_search.merge_adjacent_chunks_qdrant(results, k=1)

        assert len(merged_results) == len(results)
        assert len(merged_results[0].text) > len(results[0].text)

    # ==================== Edge Cases and Error Handling ====================

    @patch('search.advanced_search.common.embed_text')
    def test_semantic_search_empty_results(self, mock_embed, advanced_search, mock_qdrant_client):
        """Test semantic search when no results are found."""
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        mock_qdrant_client.search.return_value = []

        results = advanced_search.semantic_search(queries=["test"], final_top_k=3)

        assert results == []

    @patch('search.advanced_search.common.embed_text')
    def test_semantic_search_with_none_hierarchy(self, mock_embed, advanced_search,
                                                 mock_qdrant_client, sample_payload):
        """Test handling of missing hierarchy field in payload."""
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        payload = sample_payload.copy()
        del payload['hierarchy']

        mock_qdrant_client.search.return_value = [
            MockQdrantPoint(id='1', score=0.9, payload=payload)
        ]

        results = advanced_search.semantic_search(queries=["test"], final_top_k=3)

        assert len(results) == 1
        assert results[0].page_hierarchy == []

    def test_fetch_adjacent_chunks_no_results_found(self, advanced_search, mock_qdrant_client):
        """Test fetch_adjacent_chunks when chunks don't exist."""
        result = SearchResult(
            page_id='page_123',
            title='Test',
            source='test',
            text='test',
            score=0.9,
            semantic_score=0.9,
            keyword_score=0.0,
            position=0,
            link='link',
            last_updated='2024-01-01',
            chunk_id='page_123_5',
            page_hierarchy=[]
        )

        # Mock empty results
        mock_qdrant_client.scroll.return_value = ([], None)

        adjacent = advanced_search.fetch_adjacent_chunks(result, k=1)

        assert adjacent == []

    @patch('search.advanced_search.common.embed_text')
    def test_hybrid_search_keyword_index_not_fitted(self, mock_embed, advanced_search,
                                                    mock_qdrant_client, mock_hybrid_search_index,
                                                    sample_payload):
        """Test hybrid search gracefully handles unfitted keyword index."""
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        mock_qdrant_client.search.return_value = [
            MockQdrantPoint(id='1', score=0.9, payload=sample_payload)
        ]

        # Keyword index not fitted
        mock_hybrid_search_index.is_fitted = False

        results = advanced_search.hybrid_search(queries=["test"], final_top_k=3)

        # Should still return results from semantic search
        assert len(results) >= 1
        # Should not call keyword_search
        mock_hybrid_search_index.keyword_search.assert_not_called()

    @patch('search.advanced_search.common.embed_text')
    def test_semantic_search_all_filters_combined(self, mock_embed, advanced_search,
                                                  mock_qdrant_client, sample_payload):
        """Test semantic search with all filter types combined."""
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        mock_qdrant_client.search.return_value = [
            MockQdrantPoint(id='1', score=0.9, payload=sample_payload)
        ]

        filters = {
            'page_ids': ['page_123'],
            'space_key': ['my_space'],
            'min_text_length': 50
        }

        results = advanced_search.semantic_search(
            queries=["test"],
            filters=filters,
            final_top_k=3
        )

        # Verify that search was called with a filter
        call_args = mock_qdrant_client.search.call_args
        assert call_args[1]['query_filter'] is not None
        assert len(results) >= 0

    def test_merge_adjacent_chunks_empty_list(self, advanced_search):
        """Test merge_adjacent_chunks with empty input."""
        merged = advanced_search.merge_adjacent_chunks_qdrant([], k=1)
        assert merged == []

    @patch('search.advanced_search.common.embed_text')
    def test_hybrid_search_max_keyword_score_selection(self, mock_embed, advanced_search,
                                                       mock_qdrant_client, mock_hybrid_search_index,
                                                       sample_payload):
        """Test that hybrid search keeps the maximum keyword score when multiple queries hit same chunk."""
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        mock_qdrant_client.search.return_value = [
            MockQdrantPoint(id='1', score=0.9, payload=sample_payload)
        ]

        mock_hybrid_search_index.is_fitted = True
        # Return same chunk_id with different scores for different queries
        mock_hybrid_search_index.keyword_search.side_effect = [
            [('page_123_0', 0.6)],
            [('page_123_0', 0.8)]
        ]

        results = advanced_search.hybrid_search(
            queries=["query1", "query2"],
            final_top_k=3,
            alpha=0.5
        )

        # Should keep the maximum keyword score (0.8)
        assert results[0].keyword_score == 0.8

    def test_fetch_adjacent_chunks_boundary_conditions(self, advanced_search,
                                                       mock_qdrant_client, sample_payload):
        """Test fetch_adjacent_chunks at chunk boundaries (chunk_0)."""
        result = SearchResult(
            page_id='page_123',
            title='Test',
            source='test',
            text='test',
            score=0.9,
            semantic_score=0.9,
            keyword_score=0.0,
            position=0,
            link='link',
            last_updated='2024-01-01',
            chunk_id='page_123_0',  # First chunk
            page_hierarchy=[]
        )

        # Mock only forward chunks exist
        payload_after = sample_payload.copy()
        payload_after['chunk_id'] = 'page_123_1'

        mock_qdrant_client.scroll.side_effect = [
            ([], None),  # page_123_-1 doesn't exist
            ([MockQdrantPoint(id='1', score=0.8, payload=payload_after)], None)
        ]

        adjacent = advanced_search.fetch_adjacent_chunks(result, k=1)

        # Should only get the forward chunk
        assert len(adjacent) == 1
        assert adjacent[0].chunk_id == 'page_123_1'

    @patch('search.advanced_search.common.embed_text')
    def test_semantic_search_duplicate_page_aggregation(self, mock_embed, advanced_search,
                                                        mock_qdrant_client, sample_payload):
        """Test that semantic search correctly aggregates multiple chunks from same page."""
        mock_embed.return_value = [[0.1, 0.2, 0.3]]

        # Create 3 chunks from the same page
        payloads = []
        for i in range(3):
            p = sample_payload.copy()
            p['chunk_id'] = f'page_123_{i}'
            payloads.append(MockQdrantPoint(id=str(i), score=0.8 + i * 0.05, payload=p))

        mock_qdrant_client.search.return_value = payloads

        results = advanced_search.semantic_search(queries=["test"], final_top_k=5)

        # Should return 1 aggregated result for the page
        assert len(results) == 1
        assert results[0].page_id == 'page_123'
        # Score should be boosted by occurrence (0.1 * 3 chunks = +0.3)
        assert results[0].score > max(p.score for p in payloads)

    @patch('search.advanced_search.common.embed_text')
    def test_hybrid_search_empty_keyword_results(self, mock_embed, advanced_search,
                                                 mock_qdrant_client, mock_hybrid_search_index,
                                                 sample_payload):
        """Test hybrid search when keyword search returns no results."""
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        mock_qdrant_client.search.return_value = [
            MockQdrantPoint(id='1', score=0.9, payload=sample_payload)
        ]

        mock_hybrid_search_index.is_fitted = True
        mock_hybrid_search_index.keyword_search.return_value = []

        results = advanced_search.hybrid_search(queries=["test"], final_top_k=3)

        # Should still return semantic results
        assert len(results) >= 1
        assert results[0].keyword_score == 0.0

    def test_merge_adjacent_chunks_sorting_order(self, advanced_search, mock_qdrant_client, sample_payload):
        """Test that merged chunks are sorted in correct order."""
        result = SearchResult(
            page_id='page_123',
            title='Test',
            source='test',
            text='Chunk 10',
            score=0.9,
            semantic_score=0.9,
            keyword_score=0.0,
            position=0,
            link='link',
            last_updated='2024-01-01',
            chunk_id='page_123_10',
            page_hierarchy=[]
        )

        # Mock adjacent chunks (8, 9, 11, 12) - k=2 means 2 before and 2 after
        payloads = []
        for i in [8, 9, 11, 12]:
            p = sample_payload.copy()
            p['chunk_id'] = f'page_123_{i}'
            p['text'] = f'Chunk {i}'
            payloads.append(MockQdrantPoint(id=str(i), score=0.8, payload=p))

        mock_qdrant_client.scroll.side_effect = [
            ([payloads[0]], None),  # chunk 8
            ([payloads[1]], None),  # chunk 9
            ([payloads[2]], None),  # chunk 11
            ([payloads[3]], None),  # chunk 12
        ]

        merged = advanced_search.merge_adjacent_chunks_qdrant([result], k=2)

        # Verify chunks are in correct numeric order
        assert 'Chunk 8' in merged[0].text
        assert 'Chunk 9' in merged[0].text
        assert 'Chunk 10' in merged[0].text
        assert 'Chunk 11' in merged[0].text
        assert 'Chunk 12' in merged[0].text

        # Verify order
        text = merged[0].text
        assert text.index('Chunk 8') < text.index('Chunk 9') < text.index('Chunk 10') < text.index(
            'Chunk 11') < text.index('Chunk 12')

    @patch('search.advanced_search.common.embed_text')
    def test_semantic_search_respects_top_k_and_final_top_k(self, mock_embed, advanced_search,
                                                            mock_qdrant_client, sample_payload):
        """Test that semantic search correctly limits results based on top_k and final_top_k."""
        mock_embed.return_value = [[0.1, 0.2, 0.3]]

        # Create 10 results from different pages
        payloads = []
        for i in range(10):
            p = sample_payload.copy()
            p['page_id'] = f'page_{i}'
            p['chunk_id'] = f'page_{i}_0'
            payloads.append(MockQdrantPoint(id=str(i), score=0.9 - i * 0.05, payload=p))

        mock_qdrant_client.search.return_value = payloads

        results = advanced_search.semantic_search(
            queries=["test"],
            top_k=10,
            final_top_k=3
        )

        # Should only return final_top_k results
        assert len(results) == 3

    @patch('search.advanced_search.common.embed_text')
    def test_hybrid_search_sorting_by_combined_score(self, mock_embed, advanced_search,
                                                     mock_qdrant_client, mock_hybrid_search_index,
                                                     sample_payload):
        """Test that hybrid search sorts results by combined score."""
        mock_embed.return_value = [[0.1, 0.2, 0.3]]

        # Create multiple results with different scores
        payloads = []
        for i in range(3):
            p = sample_payload.copy()
            p['page_id'] = f'page_{i}'
            p['chunk_id'] = f'page_{i}_0'
            payloads.append(MockQdrantPoint(id=str(i), score=0.5 + i * 0.1, payload=p))

        mock_qdrant_client.search.return_value = payloads

        mock_hybrid_search_index.is_fitted = True
        # Give different keyword scores
        mock_hybrid_search_index.keyword_search.return_value = [
            ('page_0_0', 0.9),  # High keyword, low semantic
            ('page_1_0', 0.5),
            ('page_2_0', 0.1),  # Low keyword, high semantic
        ]

        results = advanced_search.hybrid_search(
            queries=["test"],
            final_top_k=5,
            alpha=0.5
        )

        # Results should be sorted by combined score
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_preprocess_query_unicode_characters(self, advanced_search):
        """Test preprocessing with unicode and special characters."""
        query = "test café résumé naïve"
        result = advanced_search.preprocess_query(query)
        # Should preserve word characters including unicode
        assert "café" in result or "caf" in result
        assert len(result) > 0

    @patch('search.advanced_search.common.embed_text')
    def test_semantic_search_with_empty_query_list(self, mock_embed, advanced_search, mock_qdrant_client):
        """Test semantic search with empty query list."""
        results = advanced_search.semantic_search(queries=[], final_top_k=3)
        assert results == []
        mock_qdrant_client.search.assert_not_called()

    def test_fetch_adjacent_chunks_with_large_k(self, advanced_search, mock_qdrant_client, sample_payload):
        """Test fetch_adjacent_chunks with large k value."""
        result = SearchResult(
            page_id='page_123',
            title='Test',
            source='test',
            text='test',
            score=0.9,
            semantic_score=0.9,
            keyword_score=0.0,
            position=0,
            link='link',
            last_updated='2024-01-01',
            chunk_id='page_123_50',
            page_hierarchy=[]
        )

        # Mock empty results for all adjacent chunks
        mock_qdrant_client.scroll.return_value = ([], None)

        _ = advanced_search.fetch_adjacent_chunks(result, k=10)

        # Should request 20 chunks (10 before + 10 after)
        assert mock_qdrant_client.scroll.call_count == 20

    @patch('search.advanced_search.common.embed_text')
    def test_semantic_search_preserves_all_payload_fields(self, mock_embed, advanced_search,
                                                          mock_qdrant_client, sample_payload):
        """Test that all payload fields are correctly transferred to SearchResult."""
        mock_embed.return_value = [[0.1, 0.2, 0.3]]

        # Add all possible fields
        complete_payload = sample_payload.copy()
        complete_payload['hierarchy'] = ['Level1', 'Level2', 'Level3']

        mock_qdrant_client.search.return_value = [
            MockQdrantPoint(id='1', score=0.95, payload=complete_payload)
        ]

        results = advanced_search.semantic_search(queries=["test"], final_top_k=3)

        result = results[0]
        assert result.page_id == complete_payload['page_id']
        assert result.title == complete_payload['title']
        assert result.source == complete_payload['source']
        assert result.text == complete_payload['text']
        assert result.position == complete_payload['position']
        assert result.link == complete_payload['link']
        assert result.last_updated == complete_payload['last_updated']
        assert result.chunk_id == complete_payload['chunk_id']
        assert result.page_hierarchy == complete_payload['hierarchy']


# ==================== Parametrized Tests ====================

class TestAdvancedSearchParametrized:
    """Parametrized tests for various scenarios."""

    @pytest.mark.parametrize("query,expected", [
        ("simple query", "simple query"),
        ("query!@#$%^&*()with_special", "query with_special"),
        ("multiple   spaces", "multiple spaces"),
        ("  leading_trailing  ", "leading_trailing"),
        ("keep-hyphens", "keep-hyphens"),
        ("keep.periods", "keep.periods"),
    ])
    def test_preprocess_query_variations(self, query, expected):
        """Test various query preprocessing scenarios."""
        with patch('search.advanced_search.SentenceTransformer'):
            search = AdvancedSearch(Mock(), Mock())
            result = search.preprocess_query(query)
            assert result == expected

    @pytest.mark.parametrize("alpha,sem_score,kw_score", [
        (0.0, 0.9, 0.8),  # 100% keyword
        (0.5, 0.9, 0.8),  # 50/50 blend
        (1.0, 0.9, 0.8),  # 100% semantic
        (0.3, 0.7, 0.6),  # 30% semantic, 70% keyword
    ])
    @patch('search.advanced_search.common.embed_text')
    def test_hybrid_search_alpha_values(self, mock_embed, alpha, sem_score, kw_score):
        """Test hybrid search with different alpha values."""
        mock_qdrant = Mock()
        mock_hybrid_index = Mock()
        mock_hybrid_index.is_fitted = True

        with patch('search.advanced_search.SentenceTransformer'):
            search = AdvancedSearch(mock_qdrant, mock_hybrid_index)

        mock_embed.return_value = [[0.1, 0.2, 0.3]]

        payload = {
            'page_id': 'page_1',
            'title': 'Test',
            'source': 'test',
            'text': 'content',
            'position': 0,
            'link': 'link',
            'last_updated': '2024-01-01',
            'chunk_id': 'page_1_0',
            'hierarchy': []
        }

        mock_qdrant.search.return_value = [
            MockQdrantPoint(id='1', score=sem_score, payload=payload)
        ]
        mock_hybrid_index.keyword_search.return_value = [('page_1_0', kw_score)]

        results = search.hybrid_search(queries=["test"], final_top_k=1, alpha=alpha)

        expected_score = alpha * sem_score + (1 - alpha) * kw_score
        assert abs(results[0].score - expected_score) < 0.01

    @pytest.mark.parametrize("k_value", [0, 1, 2, 5, -1])
    def test_fetch_adjacent_chunks_k_values(self, k_value):
        """Test fetch_adjacent_chunks with various k values."""
        mock_qdrant = Mock()
        mock_hybrid_index = Mock()

        with patch('search.advanced_search.SentenceTransformer'):
            search = AdvancedSearch(mock_qdrant, mock_hybrid_index)

        result = SearchResult(
            page_id='page_123',
            title='Test',
            source='test',
            text='test',
            score=0.9,
            semantic_score=0.9,
            keyword_score=0.0,
            position=0,
            link='link',
            last_updated='2024-01-01',
            chunk_id='page_123_10',
            page_hierarchy=[]
        )

        mock_qdrant.scroll.return_value = ([], None)

        adjacent = search.fetch_adjacent_chunks(result, k=k_value)

        if k_value == 0:
            assert adjacent == []
            mock_qdrant.scroll.assert_not_called()
        elif k_value == -1:
            # Should fetch all chunks with single call
            assert mock_qdrant.scroll.call_count == 1
        else:
            # Should fetch 2*k chunks
            assert mock_qdrant.scroll.call_count == 2 * k_value