import re
from collections import defaultdict
from typing import List

import nltk
from bs4 import BeautifulSoup
from nltk import WordNetLemmatizer, sent_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

class EnhancedTextProcessor:
    """Advanced text processing with multiple chunking strategies"""

    def __init__(self, chunk_size_limit, min_chunk_size, chunk_size_overlap):
        self.tfidf_vectorizer = None
        self.document_keywords = defaultdict(set)
        self.min_chunk_size = min_chunk_size
        self.chunk_size_limit = chunk_size_limit
        self.chunk_size_overlap = chunk_size_overlap

    def extract_text_from_storage(self, body_storage):
        """Enhanced text extraction with better HTML handling."""
        html = body_storage.get("value", "")
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text with better spacing
        text = soup.get_text(separator=" ", strip=True)

        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text)

        return text

    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract important keywords from text using TF-IDF and NLP."""
        # Clean text
        clean_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        # Tokenize and filter
        words = [word for word in clean_text.split()
                 if word not in stop_words and len(word) > 2]

        # Lemmatize
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

        # Return most frequent unique words
        word_freq = defaultdict(int)
        for word in lemmatized_words:
            word_freq[word] += 1

        return sorted(word_freq.keys(), key=word_freq.get, reverse=True)[:top_k]

    def smart_chunk_text(self, text: str, max_chars: int,
                         min_chunk: int, overlap: int) -> List[str]:
        """Advanced chunking with sentence boundaries and semantic coherence."""
        if len(text) <= max_chars:
            return [text] if len(text) >= min_chunk else []

        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence would exceed limit
            if len(current_chunk) + len(sentence) > max_chars and current_chunk:
                if len(current_chunk) >= min_chunk:
                    chunks.append(current_chunk.strip())

                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > overlap:
                    # Take last part for overlap
                    overlap_text = current_chunk[-overlap:]
                    # Find sentence boundary for cleaner overlap
                    last_period = overlap_text.rfind('.')
                    if last_period > overlap // 2:
                        overlap_text = overlap_text[last_period + 1:]
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # Add final chunk
        if current_chunk and len(current_chunk) >= min_chunk:
            chunks.append(current_chunk.strip())

        return chunks