"""
Module for performing hybrid search (semantic + keyword).
"""
import logging
import re
import numpy as np
import pickle
from typing import List, Optional, Dict, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from search.models import SearchResult
from config import settings

logger = logging.getLogger(__name__)


class HybridSearch:
    """
    Performs hybrid search by combining semantic and keyword-based results.
    """

    def __init__(self, qdrant_client: QdrantClient, embedding_model: SentenceTransformer):
        self.qdrant = qdrant_client
        self.embedding_model = embedding_model
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.document_map: Dict[int, str] = {}
        self._load_tfidf_model()

    def _load_tfidf_model(self) -> None:
        """Loads a pre-trained TF-IDF model and its matrix from disk."""
        try:
            with open('tfidf_model.pkl', 'rb') as f:
                data = pickle.load(f)
                self.tfidf_vectorizer = data['vectorizer']
                self.tfidf_matrix = data['matrix']
                self.document_map = data['document_map']
            logger.info("TF-IDF model loaded from file.")
        except FileNotFoundError:
            logger.warning("TF-IDF model not found. Keyword search will be disabled until it's indexed.")

    def fit_tfidf(self, texts: List[str], chunk_ids: List[str]) -> None:
        """Fits TF-IDF vectorizer and saves the model."""
        if not texts:
            logger.warning("No texts provided to fit TF-IDF model.")
            return

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        logger.info(f"Fitting TF-IDF on {len(texts)} documents...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.document_map = {i: chunk_id for i, chunk_id in enumerate(chunk_ids)}

        with open('tfidf_model.pkl', 'wb') as f:
            pickle.dump({
                'vectorizer': self.tfidf_vectorizer,
                'matrix': self.tfidf_matrix,
                'document_map': self.document_map
            }, f)
        logger.info("TF-IDF model saved successfully.")

    def _embed_text(self, texts: List[str]) -> List[List[float]]:
        """Generates normalized embeddings for text."""
        embeddings = self.embedding_model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return embeddings.tolist()

    def _preprocess_query(self, query: str) -> str:
        """Cleans and optimizes a query string."""
        query = re.sub(r'[^\w\s\-\.]', ' ', query)
        return re.sub(r'\s+', ' ', query.strip())

    def semantic_search(self, query: str, top_k: int = settings.DEFAULT_TOP_K,
                        filters: Optional[Dict] = None) -> List[SearchResult]:
        """Performs a semantic search on the Qdrant collection."""
        query = self._preprocess_query(query)
        query_vector = self._embed_text([query])[0]

        search_filter = None
        if filters:
            conditions = []
            if 'page_ids' in filters:
                conditions.append(FieldCondition(key="page_id", match=MatchValue(value=filters['page_ids'])))
            if 'space_key' in filters:
                conditions.append(FieldCondition(key="space_key", match=MatchValue(value=filters['space_key'])))
            if 'min_text_length' in filters:
                conditions.append(FieldCondition(key="text_length", range=Range(gte=filters['min_text_length'])))
            search_filter = Filter(must=conditions)

        results = self.qdrant.search(
            collection_name=settings.COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=top_k,
            with_payload=True
        )

        return [
            SearchResult(
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
            ) for r in results
        ]

    def keyword_search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Performs a keyword-based search using the TF-IDF index."""
        if not self.tfidf_vectorizer or not self.tfidf_matrix:
            logger.warning("TF-IDF model is not fitted. Cannot perform keyword search.")
            return []

        query_vector = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(self.document_map[idx], similarities[idx])
                for idx in top_indices if similarities[idx] > 0]

    def hybrid_search(self, query: str, top_k: int = settings.DEFAULT_TOP_K,
                      alpha: float = settings.HYBRID_ALPHA) -> List[SearchResult]:
        """Combines semantic and keyword search results with a weighted score."""
        semantic_results = self.semantic_search(query, top_k=settings.RERANK_TOP_K)
        keyword_scores = {
            chunk_id: score for chunk_id, score in self.keyword_search(query, top_k=settings.RERANK_TOP_K)
        }

        combined_results = {}
        for result in semantic_results:
            result.keyword_score = keyword_scores.get(result.chunk_id, 0.0)
            result.score = alpha * result.semantic_score + (1 - alpha) * result.keyword_score
            combined_results[result.chunk_id] = result

        final_results = sorted(combined_results.values(), key=lambda x: x.score, reverse=True)
        return final_results[:top_k]

    def explain_results(self, results: List[SearchResult], query: str) -> None:
        """Prints a user-friendly explanation of the search results."""
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