import os
import requests
import subprocess
import time
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import uuid
import re
from collections import defaultdict
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue, Range
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# -------------------------
# 1. Enhanced Config
# -------------------------
CONFLUENCE_BASE_URL = "https://confluence.sage.com/rest/api"
CONFLUENCE_API_TOKEN = "CONFLUENCE_API_TOKEN"
CONFLUENCE_ROOT_PAGE_ID = 417798815
SPACE_KEY = "FRCIELESP"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "confluence_pages_sxp_enhanced_v3"

# Enhanced embedding configuration
SENTENCE_TRANSFORMER = 'all-mpnet-base-v2'
EMBEDDING_SIZE = 768

# Advanced text processing configuration
CHUNK_SIZE_LIMIT = 600  # Slightly larger chunks for better context
CHUNK_OVERLAP = 150  # More overlap for better continuity
MIN_CHUNK_SIZE = 50  # Minimum viable chunk size

# Search configuration
DEFAULT_TOP_K = 10
RERANK_TOP_K = 20
HYBRID_ALPHA = 0.7  # Weight for semantic vs keyword search (0.7 = 70% semantic, 30% keyword)


@dataclass
class SearchResult:
    """Enhanced search result with more metadata"""
    page_id: int
    title: str
    text: str
    score: float
    semantic_score: float
    keyword_score: float
    position: int
    link: str
    last_updated: str
    chunk_id: str
    page_hierarchy: List[str] = None


# Initialize models
logger.info("Initializing embedding model and NLP components...")
model_embed = SentenceTransformer(SENTENCE_TRANSFORMER)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# -------------------------
# 2. Enhanced Qdrant Functions
# -------------------------
def start_qdrant():
    """Starts the Qdrant Docker container."""
    logger.info("Attempting to start Qdrant Docker container...")
    try:
        subprocess.run(
            ["docker", "run", "-d", "--name", "qdrant", "-p", "6333:6333", "qdrant/qdrant"],
            check=True
        )
        logger.info("Qdrant container started successfully. Waiting for it to become available...")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Qdrant container: {e}. Is Docker running?")
        raise


def check_and_start_qdrant(timeout=60, retry_delay=5):
    """Enhanced Qdrant connection with better error handling."""
    try:
        qdrant_client = QdrantClient(QDRANT_URL)
        qdrant_client.get_collections()
        logger.info("Qdrant is running and accessible.")
        return qdrant_client
    except Exception:
        logger.info(f"Qdrant is not yet available. Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)

    try:
        start_qdrant()
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                qdrant_client = QdrantClient(QDRANT_URL)
                qdrant_client.get_collections()
                logger.info("Qdrant container started and is now accessible.")
                return qdrant_client
            except Exception:
                time.sleep(retry_delay)
    except Exception as e:
        logger.error(f"Failed to connect to or start Qdrant: {e}")
        exit(1)


logger.info("Checking and starting Qdrant database...")
qdrant = check_and_start_qdrant()


# -------------------------
# 3. Enhanced Text Processing
# -------------------------
class EnhancedTextProcessor:
    """Advanced text processing with multiple chunking strategies"""

    def __init__(self):
        self.tfidf_vectorizer = None
        self.document_keywords = defaultdict(set)

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

    def smart_chunk_text(self, text: str, max_chars: int = CHUNK_SIZE_LIMIT,
                         overlap: int = CHUNK_OVERLAP, min_chunk: int = MIN_CHUNK_SIZE) -> List[str]:
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


# Initialize text processor
text_processor = EnhancedTextProcessor()


# -------------------------
# 4. Enhanced Embedding and Indexing
# -------------------------
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

    def fit_tfidf(self, texts: List[str], chunk_ids: List[str]):
        """Fit TF-IDF on the document corpus."""
        logger.info(f"Fitting TF-IDF on {len(texts)} documents...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.document_map = {i: chunk_id for i, chunk_id in enumerate(chunk_ids)}
        self.is_fitted = True

        # Save TF-IDF model for persistence
        with open('tfidf_model.pkl', 'wb') as f:
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


# Initialize hybrid search
hybrid_index = HybridSearchIndex()


def embed_text(texts: List[str]) -> List[List[float]]:
    """Enhanced embedding with normalization."""
    embeddings = model_embed.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.tolist()


# -------------------------
# 5. Enhanced Confluence API
# -------------------------
def fetch_children(page_id: int, limit: int = 100) -> List[Dict]:
    """Enhanced API fetching with better error handling."""
    url = f"{CONFLUENCE_BASE_URL}/content/{page_id}/child/page"
    params = {
        "limit": limit,
        "expand": "body.storage,version,ancestors,space,metadata.labels"
    }

    try:
        r = requests.get(
            url,
            headers={"Authorization": f"Bearer {CONFLUENCE_API_TOKEN}"},
            params=params,
            timeout=30
        )
        r.raise_for_status()
        return r.json().get("results", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching children for page {page_id}: {e}")
        return []


def build_page_hierarchy(page_data: Dict) -> List[str]:
    """Build hierarchical path for a page."""
    hierarchy = []
    if 'ancestors' in page_data:
        for ancestor in page_data['ancestors']:
            hierarchy.append(ancestor['title'])
    hierarchy.append(page_data['title'])
    return hierarchy


# -------------------------
# 6. Enhanced Indexing Pipeline
# -------------------------
def index_pages(root_page_id: int, reset: bool = False) -> None:
    """Enhanced indexing with hierarchical structure and metadata."""
    if reset:
        logger.info(f"Deleting collection: {COLLECTION_NAME}")
        try:
            qdrant.delete_collection(collection_name=COLLECTION_NAME)
        except Exception:
            pass

    # Create collection with enhanced configuration
    try:
        qdrant.get_collection(COLLECTION_NAME)
        logger.info(f"Collection {COLLECTION_NAME} already exists")
    except Exception:
        logger.info(f"Creating Qdrant collection {COLLECTION_NAME}")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
        )

    queue = [root_page_id]
    visited = set()
    all_texts = []
    all_chunk_ids = []

    while queue:
        current_page_id = queue.pop(0)
        if current_page_id in visited:
            continue
        visited.add(current_page_id)

        logger.info(f"Processing page ID: {current_page_id}")
        children = fetch_children(current_page_id)
        points_to_upsert = []

        for page in children:
            page_id = int(page["id"])
            title = page["title"]

            # Skip root page content but process its children
            if current_page_id != CONFLUENCE_ROOT_PAGE_ID:
                try:
                    body = page["body"]["storage"]
                    last_updated = page["version"]["when"]
                    link = f"https://confluence.sage.com/spaces/{SPACE_KEY}/pages/{page_id}"
                    hierarchy = build_page_hierarchy(page)

                    # Extract and process text
                    text = text_processor.extract_text_from_storage(body)
                    if not text or len(text) < MIN_CHUNK_SIZE:
                        continue

                    # Check if page needs updating
                    try:
                        existing_points = qdrant.scroll(
                            collection_name=COLLECTION_NAME,
                            scroll_filter=Filter(
                                must=[FieldCondition(key="page_id", match=MatchValue(value=page_id))]
                            ),
                            limit=1,
                            with_payload=True
                        )[0]

                        if existing_points and existing_points[0].payload.get("last_updated") == last_updated:
                            logger.info(f"Skipping unchanged page: {title}")
                            queue.append(page_id)
                            continue
                        else:
                            # Delete old chunks for this page
                            qdrant.delete(
                                collection_name=COLLECTION_NAME,
                                points_selector=Filter(
                                    must=[FieldCondition(key="page_id", match=MatchValue(value=page_id))]
                                )
                            )
                    except Exception:
                        pass  # Page not indexed yet

                    # Enhanced chunking
                    text_chunks = text_processor.smart_chunk_text(text)
                    if not text_chunks:
                        continue

                    # Extract keywords for each chunk
                    chunk_embeddings = embed_text(text_chunks)

                    for i, (chunk, embedding) in enumerate(zip(text_chunks, chunk_embeddings)):
                        chunk_id = f"{page_id}_{i}"
                        keywords = text_processor.extract_keywords(chunk)

                        # Store for TF-IDF indexing
                        all_texts.append(chunk)
                        all_chunk_ids.append(chunk_id)

                        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
                        points_to_upsert.append(
                            PointStruct(
                                id=point_id,
                                vector=embedding,
                                payload={
                                    "title": title,
                                    "page_id": page_id,
                                    "chunk_id": chunk_id,
                                    "text": chunk,
                                    "keywords": keywords,
                                    "last_updated": last_updated,
                                    "link": link,
                                    "position": i,
                                    "hierarchy": hierarchy,
                                    "text_length": len(chunk),
                                    "space_key": SPACE_KEY
                                }
                            )
                        )

                    logger.info(f"Prepared {len(text_chunks)} chunks for page: {title}")

                except Exception as e:
                    logger.error(f"Error processing page {title}: {e}")
                    continue
            else:
                logger.info(f"Skipped root page content: {title}")

            queue.append(page_id)

        # Batch upsert
        if points_to_upsert:
            try:
                qdrant.upsert(collection_name=COLLECTION_NAME, points=points_to_upsert)
                logger.info(f"Upserted {len(points_to_upsert)} points to Qdrant")
            except Exception as e:
                logger.error(f"Error upserting points: {e}")

    # Build TF-IDF index for hybrid search
    if all_texts:
        logger.info("Building TF-IDF index for keyword search...")
        hybrid_index.fit_tfidf(all_texts, all_chunk_ids)


# -------------------------
# 7. Advanced Search Functions
# -------------------------
class AdvancedSearch:
    """Advanced search with multiple search modes and ranking."""

    def __init__(self, qdrant_client: QdrantClient):
        self.qdrant = qdrant_client

    def preprocess_query(self, query: str) -> str:
        """Clean and optimize query for better results."""
        # Remove special characters but keep important ones
        query = re.sub(r'[^\w\s\-\.]', ' ', query)
        # Normalize whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        return query

    def semantic_search(self, queries: List[str], top_k: int = DEFAULT_TOP_K,
                        filters: Optional[Dict] = None, final_top_k: int = 3) -> List[SearchResult]:
        """
        Enhanced semantic search:
        - Accepts a list of queries
        - Runs semantic search for each
        - Aggregates results by page_id
        - Reranks based on combined score + frequency
        - Returns top `final_top_k` pages
        """
        aggregated_results = defaultdict(list)

        # Pre-build filters
        search_filter = None
        if filters:
            conditions = []
            if 'page_ids' in filters:
                conditions.append(
                    FieldCondition(key="page_id", match=MatchValue(value=filters['page_ids']))
                )
            if 'space_key' in filters:
                conditions.append(
                    FieldCondition(key="space_key", match=MatchValue(value=filters['space_key']))
                )
            if 'min_text_length' in filters:
                conditions.append(
                    FieldCondition(key="text_length", range=Range(gte=filters['min_text_length']))
                )

            if conditions:
                search_filter = Filter(must=conditions)

        # Process each query
        for query in queries:
            query = self.preprocess_query(query)
            query_vector = embed_text([query])[0]

            results = self.qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=top_k,
                with_payload=True
            )

            for r in results:
                sr = SearchResult(
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
                )
                aggregated_results[sr.page_id].append(sr)

        # Merge + rerank
        merged_results = []
        for page_id, results in aggregated_results.items():
            occurrence = len(results)
            avg_score = sum(r.score for r in results) / occurrence
            max_score = max(r.score for r in results)

            # Weighting: occurrences + score
            combined_score = avg_score + (0.1 * occurrence) + (0.05 * max_score)

            # Take first result as representative (can merge text if needed)
            base = results[0]
            merged_results.append(
                SearchResult(
                    page_id=page_id,
                    title=base.title,
                    text=base.text,
                    score=combined_score,
                    semantic_score=avg_score,
                    keyword_score=0.0,
                    position=base.position,
                    link=base.link,
                    last_updated=base.last_updated,
                    chunk_id=base.chunk_id,
                    page_hierarchy=base.page_hierarchy,
                )
            )

        # Sort by combined_score
        merged_results.sort(key=lambda x: x.score, reverse=True)

        return merged_results[:final_top_k]

    def hybrid_search(self, query: str, top_k: int = DEFAULT_TOP_K,
                      alpha: float = HYBRID_ALPHA) -> List[SearchResult]:
        """Hybrid search combining semantic and keyword approaches."""
        # Get semantic results
        semantic_results = self.semantic_search(query, top_k=RERANK_TOP_K)

        # Get keyword results
        keyword_results = []
        if hybrid_index.is_fitted:
            kw_scores = hybrid_index.keyword_search(query, top_k=RERANK_TOP_K)

            for chunk_id, kw_score in kw_scores:
                # Find corresponding semantic result
                for sem_result in semantic_results:
                    if sem_result.chunk_id == chunk_id:
                        keyword_results.append((sem_result, kw_score))
                        break

        # Combine and rerank results
        combined_results = {}

        # Add semantic results
        for result in semantic_results:
            combined_results[result.chunk_id] = SearchResult(
                page_id=result.page_id,
                title=result.title,
                text=result.text,
                score=alpha * result.semantic_score,
                semantic_score=result.semantic_score,
                keyword_score=0.0,
                position=result.position,
                link=result.link,
                last_updated=result.last_updated,
                chunk_id=result.chunk_id,
                page_hierarchy=result.page_hierarchy
            )

        # Add keyword scores
        for result, kw_score in keyword_results:
            if result.chunk_id in combined_results:
                combined_results[result.chunk_id].keyword_score = kw_score
                combined_results[result.chunk_id].score = (
                    alpha * result.semantic_score + (1 - alpha) * kw_score
                )

        # Sort by combined score and return top results
        final_results = sorted(combined_results.values(), key=lambda x: x.score, reverse=True)
        return final_results[:top_k]

    def search_by_page_title(self, title_query: str, top_k: int = 5) -> List[SearchResult]:
        """Search specifically within pages matching title criteria."""
        # First find pages with matching titles
        all_results = self.qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=None,
            limit=1000,  # Adjust based on your corpus size
            with_payload=["title", "page_id"]
        )[0]

        matching_page_ids = []
        title_lower = title_query.lower()
        for point in all_results:
            if title_lower in point.payload['title'].lower():
                matching_page_ids.append(point.payload['page_id'])

        if not matching_page_ids:
            return []

        # Search within matching pages
        return self.semantic_search("", top_k=top_k, filters={'page_ids': matching_page_ids})

    def explain_results(self, results: List[SearchResult], query: str) -> None:
        """Provide detailed explanation of search results."""
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


class ConfluenceRAG:
    """Complete RAG system: Retrieval + Augmented Generation"""

    def __init__(self, search_system: AdvancedSearch):
        self.search = search_system
        # You can use any LLM API here - OpenAI, Anthropic, local models, etc.
        self.system_prompt = """You are an expert assistant that answers questions based on Confluence documentation.

INSTRUCTIONS:
- Use ONLY the provided context to answer questions
- Be comprehensive but concise
- If information is missing, clearly state what cannot be answered
- Reference specific pages/sections when relevant
- Structure your response clearly with headings if appropriate
- Do not make up information not present in the context"""

    def ask(self, question: str, max_context_chars: int = 4000, top_k: int = 8) -> dict:
        """Ask a question and get a comprehensive answer with sources."""

        # Step 1: Retrieve relevant content
        logger.info(f"🔍 Searching for: '{question}'")
        search_results = self.search.hybrid_search(question, top_k=top_k)

        if not search_results:
            return {
                'question': question,
                'answer': "I couldn't find any relevant information in the Confluence documentation to answer your question.",
                'sources': [],
                'context_used': []
            }

        # Step 2: Prepare context from top results
        context_pieces = []
        used_pages = set()
        total_chars = 0

        for result in search_results:
            # Avoid duplicate content from same page
            page_key = f"{result.page_id}_{result.position}"
            if page_key in used_pages:
                continue

            context_piece = f"[Page: {result.title}]\n{result.text.strip()}\n"

            if total_chars + len(context_piece) <= max_context_chars:
                context_pieces.append({
                    'text': context_piece,
                    'title': result.title,
                    'link': result.link,
                    'score': result.score,
                    'hierarchy': result.page_hierarchy
                })
                used_pages.add(page_key)
                total_chars += len(context_piece)
            else:
                break

        # Step 3: Build the full context
        full_context = "\n".join([piece['text'] for piece in context_pieces])

        # Step 4: Create the prompt
        prompt = f"""{self.system_prompt}

CONTEXT FROM CONFLUENCE DOCUMENTATION:
{full_context}

QUESTION: {question}

Please provide a comprehensive answer based on the context above:"""

        # Step 5: Generate answer (you can replace this with any LLM)
        answer = self._generate_answer_placeholder(question, context_pieces)

        # Step 6: Return structured response
        return {
            'question': question,
            'answer': answer,
            'sources': [
                {
                    'title': piece['title'],
                    'link': piece['link'],
                    'hierarchy': ' > '.join(piece['hierarchy']) if piece['hierarchy'] else '',
                    'relevance_score': piece['score'],
                    'snippet': piece['text'][:200] + "..." if len(piece['text']) > 200 else piece['text']
                }
                for piece in context_pieces
            ],
            'total_results_found': len(search_results),
            'context_pieces_used': len(context_pieces),
            'context_length': total_chars
        }

    def _generate_answer_placeholder(self, question: str, context_pieces: list) -> str:
        """Placeholder for LLM integration. Replace with actual LLM call."""
        # This is where you'd integrate with your preferred LLM:
        # - OpenAI GPT-4: openai.chat.completions.create()
        # - Anthropic Claude: anthropic.messages.create()
        # - Local model: ollama, transformers, etc.
        # - Azure OpenAI: azure-openai client

        # For now, return a structured summary
        pages_found = set(piece['title'] for piece in context_pieces)

        answer = f"""Based on the Confluence documentation, I found relevant information across {len(pages_found)} page(s):

**Pages referenced:** {', '.join(pages_found)}

**Summary of findings:**
"""

        for i, piece in enumerate(context_pieces[:3], 1):  # Top 3 most relevant
            answer += f"\n{i}. **From '{piece['title']}'**: {piece['text'].strip()[:300]}..."

        answer += f"\n\n*Note: This is a placeholder response. Integrate with your preferred LLM (GPT-4, Claude, etc.) for natural language generation.*"

        return answer

    def chat_mode(self):
        """Interactive chat interface for your Confluence knowledge base."""
        print("🤖 Confluence AI Assistant")
        print("Ask me anything about your documentation. Type 'quit' to exit.\n")

        while True:
            question = input("❓ You: ").strip()

            if question.lower() in ['quit', 'exit', 'bye', 'q']:
                print("👋 Goodbye!")
                break

            if not question:
                continue

            print("🤔 Let me search the documentation...")

            try:
                result = self.ask(question)

                print(f"\n🤖 **Answer:**")
                print(result['answer'])

                if result['sources']:
                    print(
                        f"\n📚 **Sources** ({result['context_pieces_used']} of {result['total_results_found']} results used):")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"  {i}. 📄 {source['title']}")
                        if source['hierarchy']:
                            print(f"     📍 {source['hierarchy']}")
                        print(f"     ⭐ Relevance: {source['relevance_score']:.3f}")
                        print(f"     🔗 {source['link']}")
                        print()

                print("─" * 60)

            except Exception as e:
                print(f"❌ Error: {str(e)}")
                logger.error(f"Error in chat mode: {str(e)}")


# -------------------------
# LOCAL LLM OPTIONS & SETUP
# -------------------------

class LocalLLMManager:
    """Manager for different local LLM options"""

    RECOMMENDED_MODELS = {
        "ollama": {
            "phi3.5": {
                "name": "phi3.5:3.8b-mini-instruct-q4_K_M",
                "size": "2.2GB",
                "description": "Microsoft Phi-3.5 - Excellent for RAG, very fast",
                "ram_needed": "4GB",
                "best_for": "General Q&A, great instruction following"
            },
            "llama3.2": {
                "name": "llama3.2:3b-instruct-q4_K_M",
                "size": "1.9GB",
                "description": "Meta Llama 3.2 - Great balance of size/quality",
                "ram_needed": "3GB",
                "best_for": "Conversational, good reasoning"
            },
            "qwen2.5": {
                "name": "qwen2.5:3b-instruct-q4_K_M",
                "size": "1.9GB",
                "description": "Alibaba Qwen2.5 - Excellent for technical content",
                "ram_needed": "3GB",
                "best_for": "Technical docs, coding, structured responses"
            },
            "gemma2": {
                "name": "gemma2:2b-instruct-q4_K_M",
                "size": "1.6GB",
                "description": "Google Gemma2 - Very small but capable",
                "ram_needed": "2GB",
                "best_for": "Resource-constrained environments"
            }
        },
        "transformers": {
            "phi3_mini": {
                "name": "microsoft/Phi-3-mini-4k-instruct",
                "size": "7.6GB",
                "description": "Direct Hugging Face integration",
                "ram_needed": "8GB",
                "best_for": "Full control, no external dependencies"
            }
        }
    }

    @staticmethod
    def print_recommendations():
        """Print model recommendations based on system resources"""
        print("🤖 RECOMMENDED LOCAL LLMs FOR RAG:")
        print("=" * 60)

        print("\n🥇 BEST OVERALL (Recommended):")
        print("   Model: Phi-3.5 Mini (3.8B parameters)")
        print("   Size: 2.2GB | RAM: 4GB | Speed: Very Fast")
        print("   Why: Specifically optimized for instruction following and RAG")

        print("\n🥈 MOST EFFICIENT:")
        print("   Model: Gemma2 2B")
        print("   Size: 1.6GB | RAM: 2GB | Speed: Fastest")
        print("   Why: Smallest but still very capable for Q&A")

        print("\n🥉 BEST FOR TECHNICAL DOCS:")
        print("   Model: Qwen2.5 3B")
        print("   Size: 1.9GB | RAM: 3GB | Speed: Fast")
        print("   Why: Excellent at understanding technical documentation")

        print("\n💻 SYSTEM REQUIREMENTS:")
        print("   Minimum: 4GB RAM, 3GB free disk space")
        print("   Recommended: 8GB RAM, 5GB free disk space")
        print("   CPU: Any modern processor (Apple Silicon/Intel/AMD)")


# -------------------------
# OLLAMA INTEGRATION (RECOMMENDED)
# -------------------------

class OllamaRAG:
    """RAG integration with Ollama local LLMs"""

    def __init__(self, search_system, model_name="phi3.5:3.8b-mini-instruct-q4_K_M",
                 base_url="http://localhost:11434"):
        self.search = search_system
        self.model_name = model_name
        self.base_url = base_url
        self.system_prompt = """You are a helpful AI assistant that answers questions based on provided documentation.

INSTRUCTIONS:
- Answer based ONLY on the provided context
- Be concise but comprehensive
- If information is missing, say so clearly
- Use bullet points or numbered lists when appropriate
- Reference specific sections when relevant"""

    def check_ollama_status(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False

            # Check if our model is installed
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]

            return any(self.model_name in name for name in model_names)
        except:
            return False

    def setup_ollama(self) -> bool:
        """Setup Ollama and install the model"""
        print("🚀 Setting up Ollama...")

        # Check if Ollama is installed
        try:
            subprocess.run(["ollama", "--version"],
                           capture_output=True, check=True)
            print("✅ Ollama is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Ollama not found. Please install from: https://ollama.ai")
            print("   Installation commands:")
            print("   macOS: brew install ollama")
            print("   Linux: curl -fsSL https://ollama.ai/install.sh | sh")
            print("   Windows: Download from https://ollama.ai/download")
            return False

        # Start Ollama server if not running
        if not self.check_ollama_status():
            print("🔄 Starting Ollama server...")
            try:
                # Start server in background
                subprocess.Popen(["ollama", "serve"],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                time.sleep(3)  # Wait for server to start
            except Exception as e:
                print(f"❌ Failed to start Ollama server: {e}")
                return False

        # Install the model if not present
        if not self.check_ollama_status():
            print(f"📥 Installing model: {self.model_name}")
            print("   This may take a few minutes...")
            try:
                result = subprocess.run(
                    ["ollama", "pull", self.model_name],
                    capture_output=True, text=True, timeout=600
                )
                if result.returncode == 0:
                    print("✅ Model installed successfully!")
                else:
                    print(f"❌ Failed to install model: {result.stderr}")
                    return False
            except subprocess.TimeoutExpired:
                print("❌ Model installation timed out")
                return False
            except Exception as e:
                print(f"❌ Error installing model: {e}")
                return False

        return True

    def ask(self, question: str, top_k: int = 3) -> Dict:
        """Ask question with local LLM generation, with query refinement step"""

        # Step 0: Refine query using local LLM
        try:
            refine_prompt = f"""
    You are a search query optimizer for a Confluence knowledge base.

    User question:
    \"\"\"{question}\"\"\"

    Rewrite this into 3 concise, alternative search queries that capture different 
    ways the documentation might describe the answer. 
    Return them as a bullet list, without explanation.
    """
            refined_output = self._generate_with_ollama(refine_prompt)
            # Parse the list of queries (assuming bullet points or line breaks)
            refined_queries = [q.strip("-• ").strip() for q in refined_output.splitlines() if q.strip()]

            # Always include the original user query as a fallback
            if question not in refined_queries:
                refined_queries.insert(0, question)

            logger.info(f"🔍 Refined queries: {refined_queries}")

        except Exception as e:
            logger.error(f"Query refinement failed: {e}")
            refined_queries = [question]

        # Step 1: Search for relevant content using refined queries
        search_results = self.search.semantic_search(refined_queries, top_k=top_k)

        if not search_results:
            return {
                'question': question,
                'answer': "I couldn't find any relevant information to answer your question.",
                'sources': [],
                'model_used': self.model_name
            }

        # Step 2: Prepare context from top results
        context_pieces = []
        for i, result in enumerate(search_results[:top_k], 1):
            context_pieces.append({
                'text': f"[Source {i} - {result.title}]\n{result.text.strip()}",
                'title': result.title,
                'link': result.link,
                'score': result.score
            })

        # Step 3: Build prompt
        context = "\n\n".join([piece['text'] for piece in context_pieces])

        prompt = f"""Based on the following documentation, please answer the user's question:

DOCUMENTATION:
{context}

QUESTION: {question}

Please provide a clear, helpful answer based on the information above:"""

        # Step 4: Generate with local LLM
        try:
            print("🤖 Generating answer with local LLM...")
            answer = self._generate_with_ollama(prompt)
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
            logger.error(f"LLM generation error: {e}")

        return {
            'question': question,
            'answer': answer,
            'sources': [
                {
                    'title': piece['title'],
                    'link': piece['link'],
                    'score': piece['score'],
                    'snippet': piece['text'][:200] + "..."
                }
                for piece in context_pieces
            ],
            'model_used': self.model_name
        }

    def refine_user_query_for_vector_search(self, prompt: str) -> str:

        return None

    def _generate_with_ollama(self, prompt: str) -> str:
        """Generate response using Ollama API"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # Lower for more factual responses
                        "top_p": 0.9,
                        "top_k": 40,
                        "num_predict": 500  # Max tokens
                    }
                },
                timeout=60
            )

            if response.status_code == 200:
                return response.json()["response"].strip()
            else:
                return f"Error: HTTP {response.status_code}"

        except requests.exceptions.Timeout:
            return "Error: Request timed out. Try a smaller model or increase timeout."
        except Exception as e:
            return f"Error communicating with Ollama: {str(e)}"


# -------------------------
# ALTERNATIVE: HUGGING FACE TRANSFORMERS
# -------------------------

class TransformersRAG:
    """RAG with Hugging Face Transformers (no external dependencies)"""

    def __init__(self, search_system, model_name="microsoft/Phi-3-mini-4k-instruct"):
        self.search = search_system
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def setup_model(self):
        """Load model and tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            print(f"📥 Loading {self.model_name}...")
            print("   This may take a few minutes on first run...")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )

            print("✅ Model loaded successfully!")
            return True

        except ImportError:
            print("❌ Please install: pip install transformers torch accelerate")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def ask(self, question: str, top_k: int = 3) -> Dict:
        """Ask question with Transformers model"""
        if self.model is None:
            if not self.setup_model():
                return {'error': 'Model setup failed'}

        # Get search results
        search_results = self.search.semantic_search(question, top_k=top_k)

        # Build context
        context = "\n\n".join([
            f"[{result.title}]\n{result.text}"
            for result in search_results[:top_k]
        ])

        # Generate response
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        answer = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        return {
            'question': question,
            'answer': answer.strip(),
            'sources': [{'title': r.title, 'link': r.link} for r in search_results[:top_k]],
            'model_used': self.model_name
        }


# -------------------------
# SETUP WIZARD
# -------------------------

def setup_wizard():
    """Interactive setup wizard for local LLM"""
    print("🧙‍♂️ LOCAL LLM SETUP WIZARD")
    print("=" * 40)

    LocalLLMManager.print_recommendations()

    print("\n🔧 SETUP OPTIONS:")
    print("1. Ollama (Recommended) - Easy setup, great performance")
    print("2. Hugging Face Transformers - Full control, no external deps")
    print("3. Show system requirements")
    print("4. Skip for now")

    while True:
        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            return setup_ollama_wizard()
        elif choice == "2":
            return setup_transformers_wizard()
        elif choice == "3":
            print_system_requirements()
            continue
        elif choice == "4":
            return None
        else:
            print("Please enter 1, 2, 3, or 4")


def setup_ollama_wizard():
    """Ollama setup wizard"""
    print("\n🦙 OLLAMA SETUP")
    print("Recommended models:")

    models = LocalLLMManager.RECOMMENDED_MODELS["ollama"]
    for i, (key, info) in enumerate(models.items(), 1):
        print(f"{i}. {info['name']}")
        print(f"   Size: {info['size']} | RAM: {info['ram_needed']}")
        print(f"   Best for: {info['best_for']}")
        print()

    while True:
        try:
            choice = int(input(f"Choose model (1-{len(models)}): "))
            if 1 <= choice <= len(models):
                model_key = list(models.keys())[choice - 1]
                model_name = models[model_key]["name"]
                print(f"\n✅ Selected: {model_name}")
                return ("ollama", model_name)
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")


def setup_transformers_wizard():
    """Transformers setup wizard"""
    print("\n🤗 HUGGING FACE TRANSFORMERS SETUP")
    print("This will download and run models directly with Python")
    print("Requires: pip install transformers torch accelerate")
    print("\nRecommended: microsoft/Phi-3-mini-4k-instruct")

    confirm = input("Continue with Transformers? (y/n): ").lower()
    if confirm == 'y':
        return ("transformers", "microsoft/Phi-3-mini-4k-instruct")
    return None


def print_system_requirements():
    """Print detailed system requirements"""
    print("\n💻 SYSTEM REQUIREMENTS:")
    print("-" * 30)
    print("MINIMUM:")
    print("  • 4GB RAM")
    print("  • 3GB free disk space")
    print("  • Any modern CPU")
    print("  • Python 3.8+")
    print()
    print("RECOMMENDED:")
    print("  • 8GB+ RAM")
    print("  • 5GB+ free disk space")
    print("  • GPU (optional, for faster inference)")
    print()
    print("OPERATING SYSTEMS:")
    print("  • macOS (Intel & Apple Silicon)")
    print("  • Linux (Ubuntu, CentOS, etc.)")
    print("  • Windows 10/11")


advanced_search = AdvancedSearch(qdrant)


# -------------------------
# INTEGRATION WITH EXISTING SYSTEM
# -------------------------

def create_local_rag_system(advanced_search):
    """Create RAG system with local LLM"""

    # Run setup wizard
    setup_result = setup_wizard()

    if setup_result is None:
        print("Skipping local LLM setup")
        return None

    llm_type, model_name = setup_result

    if llm_type == "ollama":
        rag_system = OllamaRAG(advanced_search, model_name)

        # Setup Ollama if needed
        if not rag_system.setup_ollama():
            print("❌ Ollama setup failed")
            return None

        print("✅ Ollama RAG system ready!")
        return rag_system

    elif llm_type == "transformers":
        rag_system = TransformersRAG(advanced_search, model_name)
        if not rag_system.setup_model():
            print("❌ Transformers setup failed")
            return None
        print("✅ Transformers RAG system ready!")
        return rag_system

    return None


# -------------------------
# EXAMPLE USAGE
# -------------------------

def demo_local_rag(search_system):
    """Demo the local RAG system"""

    print("🚀 CREATING LOCAL RAG SYSTEM...")
    rag_system = create_local_rag_system(search_system)

    if rag_system is None:
        print("No local RAG system created")
        return

    print("\n🧪 TESTING LOCAL RAG SYSTEM:")
    print("=" * 50)

    # Interactive mode
    print("\n🎯 Ready for interactive Q&A!")
    while True:
        question = input("\n❓ Your question (or 'quit'): ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if question:
            result = rag_system.ask(question)
            print(f"\n🤖 {result['answer']}")

            print(f"\n📚 Sources:")
            for source in result['sources']:
                print(f"  • {source['title']} (Score: {source['score']:.3f})")


if __name__ == "__main__":
    # This would integrate with your existing search system
    demo_local_rag(advanced_search)