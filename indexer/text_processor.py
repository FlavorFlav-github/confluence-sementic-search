import json
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
    def _html_table_to_markdown(table_html: str) -> str:
        """
        Converts a single HTML <table> element into a well-formatted Markdown table.
        """
        soup = BeautifulSoup(table_html, "html.parser")
        table = soup.find("table")
        if not table:
            return ""

        # Extract headers
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        # Extract rows
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if cells:
                rows.append(cells)

        # Handle case: no <th> — infer header from first row
        if not headers and rows:
            headers = [f"Column {i + 1}" for i in range(len(rows[0]))]

        # Build Markdown
        md_lines = []

        # Header row
        md_lines.append("| " + " | ".join(headers) + " |")
        # Separator
        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        # Data rows
        for r in rows:
            padded = r + [""] * (len(headers) - len(r))  # ensure equal columns
            md_lines.append("| " + " | ".join(padded) + " |")

        return "\n".join(md_lines)

    def _html_table_to_json(self, table):
        """
        Convert an HTML <table> into JSON.
        Handles <th>, missing headers, colspan, and rowspan.
        """
        grid = []  # 2D array of cells
        rowspan_map = {}  # track cells that span multiple rows

        rows = table.find_all("tr")
        for row_idx, tr in enumerate(rows):
            cells = []
            col_idx = 0

            # Fill from previous rowspans first
            while col_idx in rowspan_map and rowspan_map[col_idx]:
                cells.append(rowspan_map[col_idx]["text"])
                rowspan_map[col_idx]["rows_left"] -= 1
                if rowspan_map[col_idx]["rows_left"] <= 0:
                    del rowspan_map[col_idx]
                col_idx += 1

            for cell in tr.find_all(["td", "th"]):
                text = cell.get_text(strip=True)
                colspan = int(cell.get("colspan", 1))
                rowspan = int(cell.get("rowspan", 1))

                for i in range(colspan):
                    cells.append(text)
                    if rowspan > 1:
                        rowspan_map[col_idx + i] = {
                            "text": text,
                            "rows_left": rowspan - 1
                        }
                col_idx += colspan

            grid.append(cells)

        # First row = headers (or fallback)
        headers = grid[0] if grid else []
        if not headers:
            headers = [f"col{i + 1}" for i in range(len(grid[1]))] if len(grid) > 1 else []

        # If any row has more cells than headers, expand headers
        max_cols = max(len(r) for r in grid)
        if len(headers) < max_cols:
            headers += [f"col{i + 1}" for i in range(len(headers), max_cols)]

        # Build JSON rows
        json_rows = []
        for row in grid[1:]:
            row_data = {}
            for i, value in enumerate(row):
                key = headers[i] if i < len(headers) else f"col{i + 1}"
                row_data[key] = value
            json_rows.append(row_data)

        return json.dumps(json_rows, indent=2)

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
            json_table = self._html_table_to_json(table)

            # Replace the HTML table with Markdown table
            table.replace_with(f"\n{json_table}\n")

        # Now the text has placeholders
        text = soup.get_text()
        text = re.sub(r"\n\s*\n+", "\n\n", text)

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

    import re
    import json

    def _split_json_blocks(self, text):
        """
        Splits text into segments of normal text and JSON blocks.
        Returns a list of dicts: {'type': 'text'|'json', 'content': str}
        Only valid JSON blocks are labeled as 'json'.
        """
        segments = []
        # First-pass regex for {...} or [...] blocks
        pattern = re.compile(r'(\{.*?\}|\[.*?\])', re.DOTALL)
        last_end = 0

        for match in pattern.finditer(text):
            start, end = match.span()
            candidate = match.group(1).strip()

            # Add preceding text
            if start > last_end:
                segments.append({"type": "text", "content": text[last_end:start]})

            # Check if candidate is valid JSON
            try:
                json.loads(candidate)
                segments.append({"type": "json", "content": candidate})
            except ValueError:
                # Not valid JSON, treat as text
                segments.append({"type": "text", "content": candidate})

            last_end = end

        # Add remaining text
        if last_end < len(text):
            segments.append({"type": "text", "content": text[last_end:]})

        return segments

    def _chunk_json_block(self, json_text, max_chars=500):
        """
        Splits a large JSON block into smaller chunks while ensuring valid JSON.
        If it's a list of objects, split by array elements.
        """
        try:
            data = json.loads(json_text)
        except Exception:
            # If JSON parsing fails, just return raw text
            return [json_text]

        # If not a list (e.g., dict), keep whole thing
        if not isinstance(data, list):
            return [json_text]

        chunks = []
        current = []
        for item in data:
            tentative = current + [item]
            tentative_str = json.dumps(tentative, indent=2)
            if len(tentative_str) > max_chars and current:
                # Close previous chunk
                chunks.append(json.dumps(current, indent=2))
                current = [item]
            else:
                current.append(item)

        if current:
            chunks.append(json.dumps(current, indent=2))

        return chunks

    def smart_chunk_text(self, text, max_chars=500, min_chunk=50, overlap=50):
        """
        Splits text (with JSON blocks) into overlapping, clean chunks.
        If a JSON block spans multiple chunks, each chunk is valid JSON.
        """
        text = text.strip()
        segments = self._split_json_blocks(text)
        chunks = []
        current_chunk = ""

        for seg in segments:
            if seg["type"] == "json":
                json_chunks = self._chunk_json_block(seg["content"], max_chars=max_chars)
                for j_chunk in json_chunks:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    chunks.append(j_chunk.strip())
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
