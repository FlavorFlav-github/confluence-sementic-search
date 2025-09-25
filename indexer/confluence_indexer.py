import uuid
from typing import List, Dict

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition
from qdrant_client.http.models import MatchValue
from sentence_transformers import SentenceTransformer

from config import settings
from config.logging_config import logger
from indexer import common
from indexer.hybrid_index import HybridSearchIndex
from indexer.text_processor import EnhancedTextProcessor


class ConfluenceIndexer:
    """
    A class to manage the entire Confluence indexing pipeline,
    including API fetching, text embedding, and Qdrant indexing.
    """
    def __init__(self, qdrant_client: QdrantClient, embedding_model: SentenceTransformer,
                 text_processor: EnhancedTextProcessor, hybrid_index: HybridSearchIndex,
                 root_page_id: int, collection_name: str, space_key: str,
                 embedding_size: int, confluence_base_url: str, confluence_api_token: str):

        self.qdrant = qdrant_client
        self.model_embed = embedding_model
        self.text_processor = text_processor
        self.hybrid_index = hybrid_index
        self.ROOT_PAGE_ID = root_page_id
        self.COLLECTION_NAME = collection_name
        self.SPACE_KEY = space_key
        self.EMBEDDING_SIZE = embedding_size
        self.CONFLUENCE_BASE_URL = confluence_base_url
        self.CONFLUENCE_API_TOKEN = confluence_api_token

    def _fetch_children(self, page_id: int, limit: int = 100) -> List[Dict]:
        """Enhanced API fetching with better error handling."""
        url = f"{self.CONFLUENCE_BASE_URL}/content/{page_id}/child/page"
        params = {
            "limit": limit,
            "expand": "body.storage,version,ancestors,space,metadata.labels"
        }

        try:
            r = requests.get(
                url,
                headers={"Authorization": f"Bearer {self.CONFLUENCE_API_TOKEN}"},
                params=params,
                timeout=30
            )
            r.raise_for_status()
            return r.json().get("results", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching children for page {page_id}: {e}")
            return []

    @staticmethod
    def _build_page_hierarchy(page_data: Dict) -> List[str]:
        """Build hierarchical path for a page."""
        hierarchy = []
        if 'ancestors' in page_data:
            for ancestor in page_data['ancestors']:
                hierarchy.append(ancestor['title'])
        hierarchy.append(page_data['title'])
        return hierarchy

    def _initialize_collection(self, reset: bool) -> None:
        """Handles collection creation and optional reset."""
        if reset:
            logger.info(f"Deleting collection: {self.COLLECTION_NAME}")
            try:
                self.qdrant.delete_collection(collection_name=self.COLLECTION_NAME)
            except Exception:
                pass

        try:
            self.qdrant.get_collection(self.COLLECTION_NAME)
            logger.info(f"Collection {self.COLLECTION_NAME} already exists")
        except Exception:
            logger.info(f"Creating Qdrant collection {self.COLLECTION_NAME}")
            self.qdrant.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(size=self.EMBEDDING_SIZE, distance=Distance.COSINE),
            )

    def index_pages(self, reset: bool = False) -> None:
        """Enhanced indexing with hierarchical structure and metadata."""
        self._initialize_collection(reset)

        queue = [self.ROOT_PAGE_ID]
        visited = set()
        all_texts = []
        all_chunk_ids = []
        min_chunk_size = self.text_processor.min_chunk_size
        chunk_size_limit = self.text_processor.chunk_size_limit
        chunk_size_overlap = self.text_processor.chunk_size_overlap

        while queue:
            current_page_id = queue.pop(0)
            if current_page_id in visited:
                continue
            visited.add(current_page_id)

            logger.info(f"Processing page ID: {current_page_id}")
            children = self._fetch_children(current_page_id)
            points_to_upsert = []

            for page in children:
                page_id = int(page["id"])
                title = page["title"]

                # Skip root page content but process its children
                if current_page_id != self.ROOT_PAGE_ID:
                    try:
                        body = page["body"]["storage"]
                        last_updated = page["version"]["when"]
                        link = f"https://confluence.sage.com/spaces/{self.SPACE_KEY}/pages/{page_id}"
                        hierarchy = self._build_page_hierarchy(page)

                        # Extract and process text
                        text = self.text_processor.extract_text_from_storage(body)
                        if not text or len(text) < settings.MIN_CHUNK_SIZE:
                            continue

                        # Check if page needs updating
                        needs_update = self._check_for_update(page_id, last_updated, title)

                        if not needs_update:
                            queue.append(page_id)
                            continue

                        # Delete old chunks if page is being re-indexed
                        if needs_update:
                            self._delete_old_chunks(page_id)

                        # Enhanced chunking
                        text_chunks = self.text_processor.smart_chunk_text(text, chunk_size_limit, min_chunk_size, chunk_size_overlap)
                        if not text_chunks:
                            continue

                        # Embed chunks
                        chunk_embeddings = common.embed_text(self.model_embed, text_chunks)

                        for i, (chunk, embedding) in enumerate(zip(text_chunks, chunk_embeddings)):
                            chunk_id = f"{page_id}_{i}"
                            keywords = self.text_processor.extract_keywords(chunk)

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
                                        "space_key": self.SPACE_KEY
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
            self._batch_upsert(points_to_upsert)

        # Build TF-IDF index for hybrid search
        if all_texts and reset:
            logger.info("Building TF-IDF index for keyword search...")
            self.hybrid_index.fit_tfidf(all_texts, all_chunk_ids)

    def _check_for_update(self, page_id: int, last_updated: str, title: str) -> bool:
        """Checks if a page has been updated since the last index."""
        try:
            existing_points = self.qdrant.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[FieldCondition(key="page_id", match=MatchValue(value=page_id))]
                ),
                limit=1,
                with_payload=True
            )[0]

            if existing_points and existing_points[0].payload.get("last_updated") == last_updated:
                logger.info(f"Skipping unchanged page: {title}")
                return False  # No update needed
            else:
                return True # Update needed (either new or updated)

        except Exception:
            return True # Page not indexed yet

    def _delete_old_chunks(self, page_id: int) -> None:
        """Deletes all chunks associated with a given page ID."""
        try:
            self.qdrant.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=Filter(
                    must=[FieldCondition(key="page_id", match=MatchValue(value=page_id))]
                )
            )
        except Exception as e:
            logger.error(f"Error deleting old chunks for page {page_id}: {e}")


    def _batch_upsert(self, points_to_upsert: List[PointStruct]) -> None:
        """Performs a batch upsert to Qdrant."""
        if points_to_upsert:
            try:
                self.qdrant.upsert(collection_name=self.COLLECTION_NAME, points=points_to_upsert)
                logger.info(f"Upserted {len(points_to_upsert)} points to Qdrant")
            except Exception as e:
                logger.error(f"Error upserting points: {e}")
