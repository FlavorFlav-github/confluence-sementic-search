import re
import unicodedata
from uuid import uuid4
from collections import defaultdict
from typing import List, Tuple, Dict, Any

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

    @staticmethod
    def _json_to_markdown(table_json):
        """
        Converts a JSON table representation to a Markdown table.

        table_json: dict with keys "headers" and "rows"
        """
        headers = table_json.get("headers", [])
        rows = table_json.get("rows", [])

        # Create Markdown header
        md = "| " + " | ".join(headers) + " |\n"
        md += "| " + " | ".join(["-" * len(h) for h in headers]) + " |\n"

        # Add rows
        for row in rows:
            md += "| " + " | ".join(row) + " |\n"

        return md

    @staticmethod
    def _html_to_markdown_table(table):
        """Converts a BeautifulSoup <table> element to Markdown string."""
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if cells:
                rows.append(cells)

        if not headers and rows:
            # Use first row as header if <th> is missing
            headers = rows.pop(0)

        # Build Markdown table
        md = ""
        if headers:
            md += "| " + " | ".join(headers) + " |\n"
            md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in rows:
            md += "| " + " | ".join(row) + " |\n"

        return md.strip(), {"headers": headers, "rows": rows}

    def _clean_text(self, text):
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Normalize unicode characters (accents, etc.)
        text = unicodedata.normalize('NFKC', text)
        # Normalize quotes, apostrophes, dashes
        text = text.replace('‘', "'").replace('’', "'")
        text = text.replace('“', '"').replace('”', '"')
        text = text.replace('–', '-').replace('—', '-')
        # Remove emails
        text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
        # Remove phone numbers (basic)
        text = re.sub(r'\+?\d[\d\s\-]{7,}\d', '[PHONE]', text)
        # Replace unusual symbols with space (except basic punctuation)
        text = re.sub(r'[^\w\s.,!?\'"-]', ' ', text)
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_text_from_storage(self, body_storage: str) -> str:
        """
        Extracts clean, readable text from a content storage structure, typically containing HTML.

        Uses BeautifulSoup for robust parsing and cleaning of HTML content.

        Args:
            body_storage (str): A dictionary containing the content, expected to have a "value" key with HTML string.

        Returns:
            str: The clean, plain text content.
        """
        # Retrieve the raw HTML content
        html = body_storage
        # Initialize BeautifulSoup parser
        soup = BeautifulSoup(html, "html.parser")

        # Remove elements that do not contain relevant content (e.g., scripts, styles)
        for script in soup(["script", "style"]):
            script.decompose()

        for table in soup.find_all("table"):
            md_table, _ = self._html_to_markdown_table(table)

            # Replace the HTML table with Markdown table
            table.replace_with(md_table)

        # Now the text has placeholders
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)

        return text

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

    def _split_markdown_tables(self, text):
        """
        Splits text into segments of normal text and markdown tables.
        Returns a list of dicts: {'type': 'text'|'table', 'content': str}
        """
        segments = []
        pattern = re.compile(r'((?:\|.*\|\n?)+)', re.MULTILINE)
        last_end = 0
        for match in pattern.finditer(text):
            start, end = match.span()
            if start > last_end:
                segments.append({"type": "text", "content": text[last_end:start]})
            segments.append({"type": "table", "content": match.group(1).strip()})
            last_end = end
        if last_end < len(text):
            segments.append({"type": "text", "content": text[last_end:]})
        return segments

    def _chunk_markdown_table(self, table_text, max_chars=500):
        """
        Splits a long markdown table into smaller chunks,
        ensuring headers persist in each sub-chunk.
        """
        lines = [line.strip() for line in table_text.splitlines() if line.strip()]
        if len(lines) <= 2:
            return [table_text]  # small table — no need to chunk

        header = "\n".join(lines[:2])  # header + separator
        data_rows = lines[2:]
        chunks = []
        current_rows = []

        for row in data_rows:
            tentative = header + "\n" + "\n".join(current_rows + [row])
            if len(tentative) > max_chars and current_rows:
                chunks.append(header + "\n" + "\n".join(current_rows))
                current_rows = [row]
            else:
                current_rows.append(row)

        if current_rows:
            chunks.append(header + "\n" + "\n".join(current_rows))

        return chunks

    def smart_chunk_text(self, text, max_chars=500, min_chunk=50, overlap=50):
        """
        Splits text (with markdown tables) into overlapping, clean chunks.
        If a markdown table spans multiple chunks, headers are repeated.
        """
        text = self._clean_text(text)
        segments = self._split_markdown_tables(text)
        chunks = []
        current_chunk = ""

        for seg in segments:
            if seg["type"] == "table":
                table_chunks = self._chunk_markdown_table(seg["content"], max_chars=max_chars)
                for t_chunk in table_chunks:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    chunks.append(t_chunk.strip())
            else:
                sentences = sent_tokenize(seg["content"])
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 > max_chars:
                        if len(current_chunk) >= min_chunk:
                            chunks.append(current_chunk.strip())
                        overlap_text = " ".join(current_chunk.split()[-overlap:]) if overlap > 0 else ""
                        current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk and len(current_chunk) >= min_chunk:
            chunks.append(current_chunk.strip())

        return chunks
