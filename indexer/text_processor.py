import re
from uuid import uuid4
from collections import defaultdict
from typing import List

import nltk
from bs4 import BeautifulSoup
# Import NLTK components for natural language processing tasks
from nltk import WordNetLemmatizer, sent_tokenize
from nltk.corpus import stopwords, wordnet

# --- NLTK Data Check and Download ---
# This block ensures all necessary NLTK data files (for tokenization, stop words, and lemmatization)
# are present before the class is used.
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("NLTK data not found. Downloading...")
    # Punkt is used for sentence tokenization
    nltk.download('punkt')
    # Stopwords is used for filtering common words
    nltk.download('stopwords')
    # WordNet is used for lemmatization
    nltk.download('wordnet')
_ = stopwords.words('english')
_ = wordnet.synsets('example')

class EnhancedTextProcessor:
    """
    Advanced text processing utility for document ingestion systems (e.g., RAG pipelines).

    It handles HTML extraction, keyword extraction (lexical analysis), and sophisticated text chunking
    to prepare documents for indexing in a vector database.
    """

    def __init__(self, chunk_size_limit: int, min_chunk_size: int, chunk_size_overlap: int):
        """
        Initializes the text processor with configuration parameters for chunking.

        Args:
            chunk_size_limit (int): The maximum number of characters allowed in a single chunk.
            min_chunk_size (int): The minimum number of characters required for a chunk to be kept.
            chunk_size_overlap (int): The number of characters to overlap between consecutive chunks.
        """
        # Placeholder for potential future use (e.g., if TF-IDF model is trained across the corpus)
        self.tfidf_vectorizer = None
        # Stores extracted keywords per document (though currently not used outside this class)
        self.document_keywords = defaultdict(set)
        
        # Chunking configuration
        self.min_chunk_size = min_chunk_size
        self.chunk_size_limit = chunk_size_limit
        self.chunk_size_overlap = chunk_size_overlap

    def extract_text_from_storage(self, body_storage: dict) -> str:
        """
        Extracts clean, readable text from a content storage structure, typically containing HTML.

        Uses BeautifulSoup for robust parsing and cleaning of HTML content.

        Args:
            body_storage (dict): A dictionary containing the content, expected to have a "value" key with HTML string.

        Returns:
            str: The clean, plain text content.
        """
        # Retrieve the raw HTML content
        html = body_storage.get("value", "")
        # Initialize BeautifulSoup parser
        soup = BeautifulSoup(html, "html.parser")

        # Remove elements that do not contain relevant content (e.g., scripts, styles)
        for script in soup(["script", "style"]):
            script.decompose()

        tables_data = {}
        for table in soup.find_all("table"):
            table_id = str(uuid4())
            headers = [th.get_text(strip=True) for th in table.find_all("th")]
            rows = []
            for tr in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all("td")]
                if cells:
                    rows.append(cells)

            # Save structured form
            tables_data[table_id] = {"headers": headers, "rows": rows}

            # Replace table with a placeholder marker
            table.replace_with(f" [[TABLE_ID:{table_id}]] ")

        # Now the text has placeholders
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)

        return text, tables_data

    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        Extracts important keywords from the text using a simple NLP approach:
        stop word removal, lemmatization, and frequency counting.

        Args:
            text (str): The text chunk or full document to analyze.
            top_k (int): The number of top keywords to return.

        Returns:
            List[str]: A list of the most frequent, unique, and meaningful keywords.
        """
        # Clean text: remove non-alphabetic characters and convert to lowercase
        clean_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

        # Initialize NLTK components
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        # Tokenize, filter out stop words and very short words
        words = [word for word in clean_text.split()
                 if word not in stop_words and len(word) > 2]

        # Lemmatize words (reduce them to their base or dictionary form, e.g., 'running' -> 'run')
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

        # Calculate word frequency
        word_freq = defaultdict(int)
        for word in lemmatized_words:
            word_freq[word] += 1

        # Return the top K most frequent unique words
        return sorted(word_freq.keys(), key=word_freq.get, reverse=True)[:top_k]

    def smart_chunk_text(self, text, tables_data, max_chars=500, min_chunk=50, overlap=50):
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > max_chars and current_chunk:
                if len(current_chunk) >= min_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk and len(current_chunk) >= min_chunk:
            chunks.append(current_chunk.strip())

        # Attach tables to chunks
        chunk_objects = []
        for ch in chunks:
            # Find all table IDs in this chunk
            table_ids = re.findall(r"\[\[TABLE_ID:(.*?)\]\]", ch)
            chunk_tables = [tables_data[tid] for tid in table_ids if tid in tables_data]

            # Remove markers from text for embeddings
            clean_text = re.sub(r"\[\[TABLE_ID:.*?\]\]", "[TABLE]", ch)

            chunk_objects.append({
                "text": clean_text,
                "tables": chunk_tables
            })

        return chunk_objects
