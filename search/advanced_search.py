import re
from collections import defaultdict
from typing import List, Optional, Dict

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range
from sentence_transformers import SentenceTransformer

from config.logging_config import logger
from config.settings import COLLECTION_NAME, RERANK_TOP_K, DEFAULT_TOP_K, HYBRID_ALPHA, SENTENCE_TRANSFORMER
from indexer.hybrid_index import HybridSearchIndex
from search.models import SearchResult
from indexer import common

class AdvancedSearch:
    """Advanced search with multiple search modes and ranking."""

    def __init__(self, qdrant_client: QdrantClient, hybrid_search_index: HybridSearchIndex):
        self.qdrant = qdrant_client
        self.hybrid_search_index = hybrid_search_index
        logger.info("Initializing embedding model and NLP components...")
        self.embed_model = SentenceTransformer(SENTENCE_TRANSFORMER)

    def preprocess_query(self, query: str) -> str:
        """Clean and optimize query for better results."""
        # Remove special characters but keep important ones
        query = re.sub(r'[^\w\s\-\.]', ' ', query)
        # Normalize whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        return query

    def semantic_search(self, queries: List[str], top_k: int = DEFAULT_TOP_K,
                        filters: Optional[Dict] = None, final_top_k: int = 3) -> List[SearchResult]:
        """
        Enhanced semantic search:
        - Accepts a list of queries
        - Runs semantic search for each
        - Aggregates results by page_id
        - Reranks based on combined score + frequency
        - Returns top `final_top_k` pages
        """
        aggregated_results = defaultdict(list)

        # Pre-build filters
        search_filter = None
        if filters:
            conditions = []
            if 'page_ids' in filters:
                conditions.append(
                    FieldCondition(key="page_id", match=MatchValue(value=filters['page_ids']))
                )
            if 'space_key' in filters:
                conditions.append(
                    FieldCondition(key="space_key", match=MatchValue(value=filters['space_key']))
                )
            if 'min_text_length' in filters:
                conditions.append(
                    FieldCondition(key="text_length", range=Range(gte=filters['min_text_length']))
                )

            if conditions:
                search_filter = Filter(must=conditions)

        # Process each query
        for query in queries:
            query = self.preprocess_query(query)
            query_vector = common.embed_text(self.embed_model,[query])[0]

            results = self.qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=top_k,
                with_payload=True
            )

            for r in results:
                sr = SearchResult(
                    page_id=r.payload['page_id'],
                    title=r.payload['title'],
                    text=r.payload['text'],
                    score=r.score,
                    semantic_score=r.score,
                    keyword_score=0.0,
                    position=r.payload['position'],
                    link=r.payload['link'],
                    last_updated=r.payload['last_updated'],
                    chunk_id=r.payload['chunk_id'],
                    page_hierarchy=r.payload.get('hierarchy', [])
                )
                aggregated_results[sr.page_id].append(sr)

        # Merge + rerank
        merged_results = []
        for page_id, results in aggregated_results.items():
            occurrence = len(results)
            avg_score = sum(r.score for r in results) / occurrence
            max_score = max(r.score for r in results)

            # Weighting: occurrences + score
            combined_score = avg_score + (0.1 * occurrence) + (0.05 * max_score)

            # Take first result as representative (can merge text if needed)
            base = results[0]
            merged_results.append(
                SearchResult(
                    page_id=page_id,
                    title=base.title,
                    text=base.text,
                    score=combined_score,
                    semantic_score=avg_score,
                    keyword_score=0.0,
                    position=base.position,
                    link=base.link,
                    last_updated=base.last_updated,
                    chunk_id=base.chunk_id,
                    page_hierarchy=base.page_hierarchy,
                )
            )

        # Sort by combined_score
        merged_results.sort(key=lambda x: x.score, reverse=True)

        return merged_results[:final_top_k]

    def fetch_adjacent_chunks(self, result, k=1):
        """
        Given a SearchResult, fetch its adjacent chunks (before and after)
        from the same page_id in Qdrant.

        Args:
            result: SearchResult object
            k: how many chunks before/after to include (default 1)

        Returns:
            List[SearchResult]
        """
        if k == 0:
            return []
        page_id = result.page_id
        try:
            base_chunk_id = int(result.chunk_id)
        except ValueError:
            # If chunk_id is not numeric, skip
            return []

        target_chunk_ids = [str(base_chunk_id - i) for i in range(1, k + 1)] + \
                           [str(base_chunk_id + i) for i in range(1, k + 1)]

        adjacent_results = []
        for target_id in target_chunk_ids:
            response = self.qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter={
                    "must": [
                        {"key": "page_id", "match": {"value": page_id}},
                        {"key": "chunk_id", "match": {"value": target_id}}
                    ]
                },
                limit=1,
                with_payload=True
            )
            points, _ = response
            if points:
                h = points[0]
                adjacent_results.append(SearchResult(
                    page_id=h.payload['page_id'],
                    title=h.payload['title'],
                    text=h.payload['text'],
                    score=result.score * 0.9,  # lower weight than semantic hit
                    semantic_score=result.score,
                    keyword_score=0.0,
                    position=h.payload['position'],
                    link=h.payload['link'],
                    last_updated=h.payload['last_updated'],
                    chunk_id=h.payload['chunk_id'],
                    page_hierarchy=h.payload.get('hierarchy', [])
                ))

        return adjacent_results

    def hybrid_search(self, query: str, top_k: int = DEFAULT_TOP_K,
                      alpha: float = HYBRID_ALPHA) -> List[SearchResult]:
        """Hybrid search combining semantic and keyword approaches."""
        # Get semantic results
        semantic_results = self.semantic_search(query, top_k=RERANK_TOP_K)

        # Get keyword results
        keyword_results = []
        if self.hybrid_search_index.is_fitted:
            kw_scores = self.hybrid_search_index.keyword_search(query, top_k=RERANK_TOP_K)

            for chunk_id, kw_score in kw_scores:
                # Find corresponding semantic result
                for sem_result in semantic_results:
                    if sem_result.chunk_id == chunk_id:
                        keyword_results.append((sem_result, kw_score))
                        break

        # Combine and rerank results
        combined_results = {}

        # Add semantic results
        for result in semantic_results:
            combined_results[result.chunk_id] = SearchResult(
                page_id=result.page_id,
                title=result.title,
                text=result.text,
                score=alpha * result.semantic_score,
                semantic_score=result.semantic_score,
                keyword_score=0.0,
                position=result.position,
                link=result.link,
                last_updated=result.last_updated,
                chunk_id=result.chunk_id,
                page_hierarchy=result.page_hierarchy
            )

        # Add keyword scores
        for result, kw_score in keyword_results:
            if result.chunk_id in combined_results:
                combined_results[result.chunk_id].keyword_score = kw_score
                combined_results[result.chunk_id].score = (
                    alpha * result.semantic_score + (1 - alpha) * kw_score
                )

        # Sort by combined score and return top results
        final_results = sorted(combined_results.values(), key=lambda x: x.score, reverse=True)
        return final_results[:top_k]

    def search_by_page_title(self, title_query: str, top_k: int = 5) -> List[SearchResult]:
        """Search specifically within pages matching title criteria."""
        # First find pages with matching titles
        all_results = self.qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=None,
            limit=1000,  # Adjust based on your corpus size
            with_payload=["title", "page_id"]
        )[0]

        matching_page_ids = []
        title_lower = title_query.lower()
        for point in all_results:
            if title_lower in point.payload['title'].lower():
                matching_page_ids.append(point.payload['page_id'])

        if not matching_page_ids:
            return []

        # Search within matching pages
        return self.semantic_search("", top_k=top_k, filters={'page_ids': matching_page_ids})

    def explain_results(self, results: List[SearchResult], query: str) -> None:
        """Provide detailed explanation of search results."""
        print(f"\n{'=' * 60}")
        print(f"SEARCH RESULTS FOR: '{query}'")
        print(f"{'=' * 60}")

        for i, result in enumerate(results, 1):
            print(
                f"\n[{i}] Score: {result.score:.4f} (Semantic: {result.semantic_score:.4f}, Keyword: {result.keyword_score:.4f})")
            print(f"📄 Page: {result.title}")
            print(f"🔗 Link: {result.link}")
            print(f"📍 Hierarchy: {' > '.join(result.page_hierarchy) if result.page_hierarchy else 'N/A'}")
            print(f"📝 Snippet: {result.text[:200]}...")
            print(f"🕒 Last Updated: {result.last_updated}")
            print(f"📊 Chunk: {result.chunk_id} (Position: {result.position})")
