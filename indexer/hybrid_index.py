import pickle
import os
from typing import List, Tuple

from qdrant_client.local.distances import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from config.logging_config import logger


class HybridSearchIndex:
    """Hybrid search combining semantic and keyword-based search."""

    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.tfidf_matrix = None
        self.document_map = {}  # Maps document index to chunk info
        self.is_fitted = False
        self.save_location = 'data/tfidf_model.pkl'
        os.makedirs(os.path.dirname(self.save_location), exist_ok=True)

    def fit_tfidf(self, texts: List[str], chunk_ids: List[str]):
        """Fit TF-IDF on the document corpus."""
        logger.info(f"Fitting TF-IDF on {len(texts)} documents...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.document_map = {i: chunk_id for i, chunk_id in enumerate(chunk_ids)}
        self.is_fitted = True

        # Save TF-IDF model for persistence
        with open(self.save_location, 'wb') as f:
            pickle.dump({
                'vectorizer': self.tfidf_vectorizer,
                'matrix': self.tfidf_matrix,
                'document_map': self.document_map
            }, f)

    def keyword_search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Perform keyword-based search using TF-IDF."""
        if not self.is_fitted:
            return []

        query_vector = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(self.document_map[idx], similarities[idx])
                   for idx in top_indices if similarities[idx] > 0]

        return results

    def load_tfidf(self):
        if os.path.isfile(self.save_location):
            with open(self.save_location, 'rb') as f:
                data = pickle.load(f)
            if data is not None and 'vectorizer' in data and 'matrix' in data and 'document_map' in data:
                self.tfidf_vectorizer = data['vectorizer']
                self.tfidf_matrix = data['matrix']
                self.document_map = data['document_map']
                self.is_fitted = True
