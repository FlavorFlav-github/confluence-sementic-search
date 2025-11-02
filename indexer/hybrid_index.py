import pickle
import os
from typing import List, Tuple

# Using a simplified cosine similarity function, likely for local development/testing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np

from config.logging_config import logger


class HybridSearchIndex:
    """
    Manages the keyword-based search index component for hybrid search.

    It utilizes the TF-IDF (Term Frequency-Inverse Document Frequency) model
    to represent the document corpus and queries, enabling lexical similarity search.
    The model is fitted and persisted to disk for reusability.
    """

    def __init__(self):
        """
        Initializes the TF-IDF vectorizer and state variables.
        """
        # Configure the TF-IDF vectorizer parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,  # Limits the vocabulary size
            stop_words='english',  # Removes common English stop words
            ngram_range=(1, 2),  # Considers single words (unigrams) and two-word phrases (bigrams)
            min_df=2,  # Ignores terms that appear in fewer than 2 documents (chunks)
            dtype = np.float32
        )
        self.tfidf_matrix = None  # Stores the fitted TF-IDF matrix (document vectors)
        # Maps the internal index of the TF-IDF matrix rows to the actual document chunk ID
        self.document_map = {}
        self.is_fitted = False  # Flag indicating whether the model is trained and ready
        self.save_location = 'data/tfidf_model.pkl'  # File path for model persistence
        # Ensure the directory for saving the model exists
        os.makedirs(os.path.dirname(self.save_location), exist_ok=True)

    def fit_tfidf(self, texts: List[str], chunk_ids: List[str]):
        """
        Fits the TF-IDF vectorizer to the provided document texts and saves the model state.

        Args:
            texts (List[str]): The document corpus (list of chunk texts) to train the model on.
            chunk_ids (List[str]): Corresponding list of unique identifiers for each text chunk.
        """
        logger.info(f"Fitting TF-IDF on {len(texts)} documents...")
        
        # Fit the vectorizer and transform the texts into the TF-IDF sparse matrix
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        # Create the map from matrix index (row number) to the external chunk ID
        self.document_map = {i: chunk_id for i, chunk_id in enumerate(chunk_ids)}
        self.is_fitted = True

        # Save the fitted TF-IDF model and associated data for persistence
        try:
            with open(self.save_location, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.tfidf_vectorizer,
                    'matrix': self.tfidf_matrix,
                    'document_map': self.document_map
                }, f)
            logger.info(f"TF-IDF index successfully saved to {self.save_location}")
        except Exception as e:
            logger.error(f"Failed to save TF-IDF model: {e}")

    def keyword_search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Perform memory-efficient keyword search using sparse dot product.
        """
        if not self.is_fitted:
            logger.warning("TF-IDF model not loaded or fitted.")
            return []

        query_vec = self.tfidf_vectorizer.transform([query])
        query_vec = normalize(query_vec)

        # Sparse dot product for cosine similarity
        similarities = self.tfidf_matrix.dot(query_vec.T).toarray().ravel()

        # Partial top-k selection
        n_docs = similarities.shape[0]
        k = min(top_k, n_docs)
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        results = [(self.document_map[i], float(similarities[i])) for i in top_indices if similarities[i] > 0]
        return results

    def load_tfidf(self):
        """
        Loads the saved TF-IDF vectorizer, matrix, and document map from the persistence file.
        Updates the `is_fitted` flag if successful.
        """
        if os.path.isfile(self.save_location):
            try:
                with open(self.save_location, 'rb') as f:
                    data = pickle.load(f)
                    
                # Robustly check for the presence of all required data
                if data is not None and 'vectorizer' in data and 'matrix' in data and 'document_map' in data:
                    self.tfidf_vectorizer = data['vectorizer']
                    self.tfidf_matrix = data['matrix']
                    self.document_map = data['document_map']
                    self.is_fitted = True
                    logger.info(f"TF-IDF index loaded successfully from {self.save_location}")
                else:
                    logger.warning(f"TF-IDF file at {self.save_location} is incomplete or corrupted.")
            except Exception as e:
                logger.error(f"Failed to load TF-IDF model from {self.save_location}: {e}")
        else:
            logger.info(f"No existing TF-IDF model found at {self.save_location}. Will fit on first index run.")
