import re
from collections import defaultdict
from typing import List, Optional, Dict

from qdrant_client import QdrantClient
# Import necessary Qdrant models for defining filters
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range
from sentence_transformers import SentenceTransformer

from config.logging_config import logger
# Import application-wide configuration constants
from config.settings import COLLECTION_NAME, RERANK_TOP_K, DEFAULT_TOP_K, HYBRID_ALPHA, SENTENCE_TRANSFORMER
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

    def preprocess_query(self, query: str) -> str:
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

    def semantic_search(self, queries: List[str], top_k: int = DEFAULT_TOP_K,
                        filters: Optional[Dict] = None, final_top_k: int = 3) -> List[SearchResult]:
        """
        Performs enhanced semantic search using one or more queries.

        The results from all queries are aggregated and then re-ranked based on
        score, relevance, and document frequency (hits per page_id).

        Args:
            queries (List[str]): A list of query strings (e.g., original query + refined alternatives).
            top_k (int): The number of initial documents to retrieve per query before aggregation.
            filters (Optional[Dict]): Dictionary of criteria to filter Qdrant points (e.g., page_ids, space_key).
            final_top_k (int): The final number of unique SearchResult objects to return after re-ranking.

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
                    FieldCondition(key="page_id", match=MatchValue(value=filters['page_ids']))
                )
            # Filter by a specific space key
            if 'space_key' in filters:
                conditions.append(
                    FieldCondition(key="space_key", match=MatchValue(value=filters['space_key']))
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
                sr = SearchResult(
                    page_id=r.payload['page_id'],
                    title=r.payload['title'],
                    text=r.payload['text'],
                    tables=r.payload['tables'],
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
                    text=base.text,
                    tables=base.tables,
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

        This is used for context enrichment in RAG systems, providing surrounding text to the initial hit.

        Args:
            result (SearchResult): The core search result object (the 'hit' chunk).
            k (int): The number of chunks to fetch before and after the hit chunk (default is 1).

        Returns:
            List[SearchResult]: A list of adjacent SearchResult objects.
        """
        if k == 0:
            return []
        
        page_id = result.page_id
        
        # Ensure chunk_id is numeric for positional lookups
        try:
            base_chunk_id = int(result.chunk_id)
        except ValueError:
            logger.warning(f"Chunk ID '{result.chunk_id}' is not numeric. Skipping adjacent fetch.")
            return []

        # Generate target chunk IDs for k positions before and k positions after
        target_chunk_ids = [str(base_chunk_id - i) for i in range(1, k + 1)] + \
                           [str(base_chunk_id + i) for i in range(1, k + 1)]

        adjacent_results = []
        for target_id in target_chunk_ids:
            # Query Qdrant for the specific page_id and chunk_id
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
                # Create a SearchResult for the adjacent chunk
                # Assign a slightly lower score to adjacent chunks to differentiate them from the core hit
                adjacent_results.append(SearchResult(
                    page_id=h.payload['page_id'],
                    title=h.payload['title'],
                    text=h.payload['text'],
                    tables=h.payload['tables'],
                    score=result.score * 0.9,  # Apply a minor penalty to the score
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
        """
        Combines semantic (vector) search and keyword (lexical/sparse) search results.

        Scores are blended using a weighting factor ($\alpha$) for final re-ranking (Reciprocal Rank Fusion is an alternative).

        Args:
            query (str): The user's query string.
            top_k (int): The number of final results to return after blending and sorting.
            alpha (float): The weighting factor for blending (0.0=keyword only, 1.0=semantic only).

        Returns:
            List[SearchResult]: A list of SearchResult objects re-ranked by the hybrid score.
        """
        # 1. Get initial semantic search results (using the single query)
        # We use a higher k here (RERANK_TOP_K) to capture a broader initial pool
        semantic_results = self.semantic_search(queries=[query], top_k=RERANK_TOP_K, final_top_k=RERANK_TOP_K)

        # 2. Get keyword (lexical) search scores
        keyword_results = []
        if self.hybrid_search_index.is_fitted:
            # Get keyword search scores (typically using BM25 or similar)
            kw_scores = self.hybrid_search_index.keyword_search(query, top_k=RERANK_TOP_K)

            # Map keyword scores back to the semantic result objects
            for chunk_id, kw_score in kw_scores:
                for sem_result in semantic_results:
                    if sem_result.chunk_id == chunk_id:
                        keyword_results.append((sem_result, kw_score))
                        break

        # 3. Combine and Rerank results
        combined_results = {}

        # Initialize with semantic results (using alpha weight)
        for result in semantic_results:
            combined_results[result.chunk_id] = SearchResult(
                page_id=result.page_id,
                title=result.title,
                text=result.text,
                tables=result.tables,
                score=alpha * result.semantic_score,  # Initial score is only semantic part
                semantic_score=result.semantic_score,
                keyword_score=0.0,
                position=result.position,
                link=result.link,
                last_updated=result.last_updated,
                chunk_id=result.chunk_id,
                page_hierarchy=result.page_hierarchy
            )

        # Add or update with keyword scores
        for result, kw_score in keyword_results:
            if result.chunk_id in combined_results:
                # Store the raw keyword score
                combined_results[result.chunk_id].keyword_score = kw_score
                # Calculate the final hybrid score: score = (alpha * semantic) + ((1 - alpha) * keyword)
                combined_results[result.chunk_id].score = (
                    alpha * result.semantic_score + (1 - alpha) * kw_score
                )

        # Sort the results based on the final hybrid score
        final_results = sorted(combined_results.values(), key=lambda x: x.score, reverse=True)
        return final_results[:top_k]

    def search_by_page_title(self, title_query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Searches for chunks within documents whose titles match a given query string.

        This is a two-step process: 1) find matching page IDs, 2) perform a semantic search constrained to those IDs.

        Args:
            title_query (str): The substring to search for within page titles.
            top_k (int): The number of final chunks to return.

        Returns:
            List[SearchResult]: A list of SearchResult objects from the matching pages.
        """
        # Step 1: Find Page IDs with Matching Titles
        # Scroll through the entire collection (up to 1000 points) just to get titles and page_ids
        # NOTE: A real-world application should use a dedicated index for title search.
        all_results = self.qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=None,
            limit=1000,  # Limits the number of points scanned for efficiency
            with_payload=["title", "page_id"]  # Only retrieve the title and page_id payloads
        )[0]

        matching_page_ids = set()
        title_lower = title_query.lower()
        for point in all_results:
            # Simple substring matching (case-insensitive)
            if title_lower in point.payload['title'].lower():
                matching_page_ids.add(point.payload['page_id'])

        if not matching_page_ids:
            return []

        # Step 2: Perform Semantic Search on the Matching Pages
        # The query is an empty string ("") but the search is limited by the page_ids filter.
        # This effectively returns the top vector points from only the specified pages.
        return self.semantic_search(queries=[""], top_k=top_k, filters={'page_ids': list(matching_page_ids)})

    def explain_results(self, results: List[SearchResult], query: str) -> None:
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
