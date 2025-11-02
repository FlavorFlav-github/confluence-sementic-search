import re
from collections import defaultdict
from typing import List, Optional, Dict

from qdrant_client import QdrantClient
# Import necessary Qdrant models for defining filters
from qdrant_client.http.models import Filter, FieldCondition, Range, MatchAny
from sentence_transformers import SentenceTransformer

from config.logging_config import logger
# Import application-wide configuration constants
from config.settings import COLLECTION_NAME, HYBRID_ALPHA, SENTENCE_TRANSFORMER
from indexer.hybrid_index import HybridSearchIndex
from search.models import SearchResult
from indexer import common


class AdvancedSearch:
    """
    Provides advanced search capabilities over a Qdrant vector database.

    Features include enhanced semantic search (using query expansion and result aggregation),
    hybrid search (combining vector and keyword scores), and context enrichment.
    """

    def __init__(self, qdrant_client: QdrantClient, hybrid_search_index: HybridSearchIndex):
        """
        Initializes the AdvancedSearch system.

        Args:
            qdrant_client (QdrantClient): An initialized client for the Qdrant vector database.
            hybrid_search_index (HybridSearchIndex): An instance for performing keyword (lexical) search.
        """
        self.qdrant = qdrant_client
        self.hybrid_search_index = hybrid_search_index
        logger.info("Initializing embedding model and NLP components...")
        # Load the SentenceTransformer model used for converting queries into vectors (embeddings)
        self.embed_model = SentenceTransformer(SENTENCE_TRANSFORMER)

    @staticmethod
    def preprocess_query(query: str) -> str:
        """
        Cleans and optimizes the raw user query for better search results.

        This typically involves removing unwanted characters and normalizing whitespace.

        Args:
            query (str): The raw user query string.

        Returns:
            str: The cleaned and normalized query string.
        """
        # Remove special characters but keep alphanumeric, whitespace, hyphen, and period
        query = re.sub(r'[^\w\s\-\.]', ' ', query)
        # Normalize multiple spaces to a single space and strip leading/trailing whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        return query

    def semantic_search(self, queries: List[str], top_k: int = 10,
                        filters: Optional[Dict] = None, final_top_k: int = 3, score_threashold: float = 0) -> List[SearchResult]:
        """
        Performs enhanced semantic search using one or more queries.

        The results from all queries are aggregated and then re-ranked based on
        score, relevance, and document frequency (hits per page_id).

        Args:
            queries (List[str]): A list of query strings (e.g., original query + refined alternatives).
            top_k (int): The number of initial documents to retrieve per query before aggregation.
            filters (Optional[Dict]): Dictionary of criteria to filter Qdrant points (e.g., page_ids, space_key).
            final_top_k (int): The final number of unique SearchResult objects to return after re-ranking.
            score_threashold (float): Minimum score required for a source to be considered as relevant. (default: 0 = All sources accepted)

        Returns:
            List[SearchResult]: A list of aggregated and re-ranked SearchResult objects.
        """
        # Dictionary to store results, keyed by page_id, allowing aggregation
        aggregated_results = defaultdict(list)

        # --- Filter Construction ---
        search_filter = None
        if filters:
            conditions = []
            # Filter by a list of page IDs
            if 'page_ids' in filters:
                conditions.append(
                    FieldCondition(key="page_id", match=MatchAny(any=filters['page_ids']))
                )
            # Filter by a specific space key
            if 'space_key' in filters:
                conditions.append(
                    FieldCondition(key="space_key", match=MatchAny(any=filters['space_key']))
                )
            # Filter by minimum text length (e.g., to exclude very short chunks)
            if 'min_text_length' in filters:
                conditions.append(
                    FieldCondition(key="text_length", range=Range(gte=filters['min_text_length']))
                )

            if conditions:
                # Combine all conditions with an AND (must) logic
                search_filter = Filter(must=conditions)

        # --- Query Processing and Vector Search ---
        for query in queries:
            query = self.preprocess_query(query)
            # Embed the processed query string into a vector using the loaded model
            query_vector = common.embed_text(self.embed_model, [query])[0]

            # Execute the vector search in Qdrant
            results = self.qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=top_k,
                with_payload=True  # Retrieve the document metadata
            )

            # Convert Qdrant results to SearchResult objects and aggregate
            for r in results:
                if r.score > score_threashold:
                    sr = SearchResult(
                        page_id=r.payload['page_id'],
                        title=r.payload['title'],
                        source=r.payload['source'],
                        text=r.payload['text'],
                        score=r.score,
                        semantic_score=r.score,
                        keyword_score=0.0,  # Semantic search only, keyword score is zero
                        position=r.payload['position'],
                        link=r.payload['link'],
                        last_updated=r.payload['last_updated'],
                        chunk_id=r.payload['chunk_id'],
                        page_hierarchy=r.payload.get('hierarchy', [])
                    )
                    # Store the result under its page_id
                    aggregated_results[sr.page_id].append(sr)

        # --- Result Merging and Reranking ---
        merged_results = []
        for page_id, results in aggregated_results.items():
            occurrence = len(results)
            # Calculate the average and maximum semantic score across all chunks on this page
            avg_score = sum(r.score for r in results) / occurrence
            max_score = max(r.score for r in results)

            # Define a custom combined score for re-ranking pages
            # Weights the average score heavily, and boosts based on occurrence count and max score
            combined_score = avg_score + (0.1 * occurrence) + (0.05 * max_score)

            # Use the first result as the representative for the merged page result
            base = results[0]
            merged_results.append(
                SearchResult(
                    page_id=page_id,
                    title=base.title,
                    source=base.source,
                    text=base.text,
                    score=combined_score,  # Use the custom combined score for ranking
                    semantic_score=avg_score,
                    keyword_score=0.0,
                    position=base.position,
                    link=base.link,
                    last_updated=base.last_updated,
                    chunk_id=base.chunk_id,
                    page_hierarchy=base.page_hierarchy,
                )
            )

        # Sort all aggregated results by the calculated combined score
        merged_results.sort(key=lambda x: x.score, reverse=True)

        # Return the final top N results
        return merged_results[:final_top_k]

    def fetch_adjacent_chunks(self, result: SearchResult, k: int = 1) -> List[SearchResult]:
        """
        Fetches chunks that are positionally adjacent to a given search result from the same document (page_id).

        If k=-1, all chunks for the page are fetched.

        Args:
            result (SearchResult): The core search result object (the 'hit' chunk).
            k (int): The number of chunks to fetch before and after the hit chunk (default is 1).
                     If k=-1, fetch all chunks for the page.

        Returns:
            List[SearchResult]: A list of adjacent SearchResult objects.
        """
        if k == 0:
            return []

        page_id = result.page_id

        # Parse chunk_id correctly
        try:
            base_id, idx = result.chunk_id.rsplit("_", 1)
            idx = int(idx)
        except ValueError:
            logger.warning(f"Chunk ID '{result.chunk_id}' is not in expected format. Skipping adjacent fetch.")
            return []

        adjacent_results = []

        if k == -1:
            # Fetch all chunks for this page
            response = self.qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter={
                    "must": [
                        {"key": "page_id", "match": {"value": page_id}}
                    ]
                },
                limit=10000,  # or some large number to cover all chunks
                with_payload=True
            )
            points, _ = response
            for h in points:
                if h.payload['chunk_id'] != result.chunk_id:  # exclude the original chunk
                    adjacent_results.append(SearchResult(
                        page_id=h.payload['page_id'],
                        title=h.payload['title'],
                        source=h.payload.get('source'),
                        text=h.payload['text'],
                        score=result.score * 0.9,
                        semantic_score=h.payload.get("semantic_score", result.semantic_score),
                        keyword_score=0.0,
                        position=h.payload['position'],
                        link=h.payload['link'],
                        last_updated=h.payload['last_updated'],
                        chunk_id=h.payload['chunk_id'],
                        page_hierarchy=h.payload.get('hierarchy', [])
                    ))
        else:
            # Generate target chunk IDs for k before and after
            target_chunk_ids = [f"{base_id}_{idx - i}" for i in range(1, k + 1)] + \
                               [f"{base_id}_{idx + i}" for i in range(1, k + 1)]
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
                        source=h.payload.get('source'),
                        text=h.payload['text'],
                        score=result.score * 0.9,
                        semantic_score=h.payload.get("semantic_score", result.semantic_score),
                        keyword_score=0.0,
                        position=h.payload['position'],
                        link=h.payload['link'],
                        last_updated=h.payload['last_updated'],
                        chunk_id=h.payload['chunk_id'],
                        page_hierarchy=h.payload.get('hierarchy', [])
                    ))

        return adjacent_results

    def merge_adjacent_chunks_qdrant(
            self,
            search_results: List['SearchResult'],
            k: int = 1
    ) -> List['SearchResult']:
        """
        For each SearchResult, fetches up to `k` adjacent chunks before and after
        (using Qdrant) and merges their text in the correct order.

        If k=-1, all chunks of the page are fetched and merged.

        Args:
            search_results: List of SearchResult objects from the hybrid search.
            k: Number of adjacent chunks to fetch before and after each hit. -1 = entire page.

        Returns:
            List[SearchResult]: New SearchResult list with merged text (same count as input).
        """
        merged_results = []

        for result in search_results:
            # Fetch context from Qdrant
            adjacent_chunks = self.fetch_adjacent_chunks(result, k=k)

            # Include the main chunk
            all_related = adjacent_chunks + [result]

            # Sort by chunk numeric suffix
            def chunk_sort_key(r: SearchResult):
                try:
                    base, idx = r.chunk_id.rsplit("_", 1)
                    return int(idx)
                except ValueError:
                    return r.chunk_id  # fallback

            all_related.sort(key=chunk_sort_key)

            # Merge their text content
            merged_text = "\n\n".join(r.text for r in all_related if r.text)

            # Build new SearchResult preserving original metadata
            merged = result.__class__(
                page_id=result.page_id,
                title=result.title,
                source=result.source,
                text=merged_text,
                score=result.score,  # keep original combined score
                semantic_score=result.semantic_score,
                keyword_score=result.keyword_score,
                position=result.position,
                link=result.link,
                last_updated=result.last_updated,
                chunk_id=result.chunk_id,
                page_hierarchy=result.page_hierarchy
            )

            merged_results.append(merged)

        return merged_results

    def hybrid_search(
            self,
            queries: List[str],
            top_k: int = 10,
            final_top_k: int = 3,
            alpha: float = HYBRID_ALPHA,
            score_threashold: float = 0
    ) -> List[SearchResult]:
        """
        Hybrid search supporting multiple queries (semantic + keyword).
        Combines and reranks results from semantic and lexical searches.

        Args:
            queries (List[str]): List of query strings.
            top_k (int): Number of results per query to fetch.
            final_top_k (int): Final number of results to return after merging.
            alpha (float): Weight for semantic vs. keyword blending.
            score_threashold (float): Optional threshold for filtering weak matches.

        Returns:
            List[SearchResult]: Top-ranked results merged across all queries.
        """

        # --- 1. Semantic Search ---
        semantic_results = self.semantic_search(
            queries=queries,
            top_k=top_k,
            final_top_k=final_top_k + 3,
            score_threashold=score_threashold
        )

        # --- 2. Keyword Search ---
        keyword_results = []
        if self.hybrid_search_index.is_fitted:
            for query in queries:
                kw_scores = self.hybrid_search_index.keyword_search(query, top_k=top_k)
                for chunk_id, kw_score in kw_scores:
                    keyword_results.append((query, chunk_id, kw_score))

        # --- 3. Combine Semantic + Keyword Results ---
        combined_results = {}

        # Initialize combined results with semantic scores
        for result in semantic_results:
            combined_results[result.chunk_id] = SearchResult(
                page_id=result.page_id,
                title=result.title,
                source=result.source,
                text=result.text,
                score=alpha * result.semantic_score,
                semantic_score=result.semantic_score,
                keyword_score=0.0,
                position=result.position,
                link=result.link,
                last_updated=result.last_updated,
                chunk_id=result.chunk_id,
                page_hierarchy=result.page_hierarchy,
            )

        # Add / update keyword scores
        for _, chunk_id, kw_score in keyword_results:
            if chunk_id in combined_results:
                r = combined_results[chunk_id]
                r.keyword_score = max(r.keyword_score, kw_score)  # keep the best keyword score
                r.score = alpha * r.semantic_score + (1 - alpha) * r.keyword_score

        # --- 4. Sort + Return Top Results ---
        final_results = sorted(
            combined_results.values(),
            key=lambda x: x.score,
            reverse=True
        )

        return final_results[:final_top_k]

    @staticmethod
    def explain_results(results: List[SearchResult], query: str) -> None:
        """
        Prints a formatted, detailed breakdown of the search results for debugging and analysis.

        Args:
            results (List[SearchResult]): The final list of search results.
            query (str): The original search query.
        """
        print(f"\n{'=' * 60}")
        print(f"SEARCH RESULTS FOR: '{query}'")
        print(f"{'=' * 60}")

        for i, result in enumerate(results, 1):
            # Display all relevant scores and metadata
            print(
                f"\n[{i}] Score: {result.score:.4f} (Semantic: {result.semantic_score:.4f}, Keyword: {result.keyword_score:.4f})")
            print(f"📄 Page: {result.title}")
            print(f"🔗 Link: {result.link}")
            print(f"📍 Hierarchy: {' > '.join(result.page_hierarchy) if result.page_hierarchy else 'N/A'}")
            # Show a truncated snippet of the chunk text
            print(f"📝 Snippet: {result.text[:200]}...")
            print(f"🕒 Last Updated: {result.last_updated}")
            print(f"📊 Chunk: {result.chunk_id} (Position: {result.position})")
