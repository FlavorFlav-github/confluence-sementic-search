"""
Module for advanced text processing and keyword extraction.
"""
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from collections import defaultdict
from typing import List

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


class EnhancedTextProcessor:
    """
    Handles text extraction, cleaning, chunking, and keyword extraction.
    """

    def extract_text_from_storage(self, body_storage: dict) -> str:
        """
        Extracts and cleans plain text from Confluence storage format (HTML).

        Args:
            body_storage: A dictionary from the Confluence API response containing HTML.

        Returns:
            The cleaned, plain text string.
        """
        html = body_storage.get("value", "")
        soup = BeautifulSoup(html, "html.parser")

        # Remove irrelevant elements
        for script in soup(["script", "style", "nav", "aside"]):
            script.decompose()

        text = soup.get_text(separator=" ", strip=True)
        # Normalize whitespace
        return re.sub(r'\s+', ' ', text)

    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        Extracts the most frequent non-stopword keywords from a given text.

        Args:
            text: The text string to extract keywords from.
            top_k: The number of keywords to return.

        Returns:
            A list of the top k keywords.
        """
        clean_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        words = [lemmatizer.lemmatize(word) for word in clean_text.split() if word not in stop_words and len(word) > 2]

        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1

        return sorted(word_freq.keys(), key=word_freq.get, reverse=True)[:top_k]

    def smart_chunk_text(self, text: str, max_chars: int, overlap: int, min_chunk: int) -> List[str]:
        """
        Splits text into chunks respecting sentence boundaries for better context.

        Args:
            text: The full text content to chunk.
            max_chars: The maximum character length of a chunk.
            overlap: The character overlap between consecutive chunks.
            min_chunk: The minimum character length for a valid chunk.

        Returns:
            A list of text chunks.
        """
        if len(text) <= max_chars:
            return [text] if len(text) >= min_chunk else []

        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chars and current_chunk:
                if len(current_chunk) >= min_chunk:
                    chunks.append(current_chunk.strip())

                # Create the overlap
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk and len(current_chunk) >= min_chunk:
            chunks.append(current_chunk.strip())

        return chunks