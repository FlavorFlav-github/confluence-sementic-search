import logging
from uuid import uuid5, NAMESPACE_URL
from typing import List, Dict, Tuple, Any
import asyncio
import aiohttp
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition
from qdrant_client.http.models import MatchValue
from sentence_transformers import SentenceTransformer

from config.logging_config import logger
from indexer import common
from indexer.data_source.confluence_adapter import ConfluenceAdapter
from indexer.hybrid_index import HybridSearchIndex
from indexer.data_source.notion_adapter import NotionAdapter
from indexer.text_processor import EnhancedTextProcessor


@dataclass
class PageData:
    """Data class for page processing results"""
    points: List[PointStruct]
    texts: List[str]
    chunk_ids: List[str]
    child_id: str

DATA_SOURCE_REF = {
    "confluence": ConfluenceAdapter,
    "notion": NotionAdapter
}

class UniversalIndexer:
    """
    Universal indexer that works with any data source adapter.
    """

    def __init__(self, qdrant_client: QdrantClient, embedding_model: SentenceTransformer,
                 text_processor: EnhancedTextProcessor, hybrid_index: HybridSearchIndex,data_source_name: str,
                 data_source_options: Dict[str, Any], root_page_ids: List[str],
                 collection_name: str, embedding_size: int):
        self.qdrant = qdrant_client
        self.model_embed = embedding_model
        self.text_processor = text_processor
        self.hybrid_index = hybrid_index
        self.data_source_name = data_source_name
        self.data_adapter = DATA_SOURCE_REF.get(data_source_name, None)
        if self.data_adapter is None:
            logging.exception(f"Data adapter {self.data_adapter} not found.")
            raise Exception(f"Data adapter {self.data_adapter} not supported")
        self.data_adapter = self.data_adapter(**data_source_options)
        self.root_page_ids = root_page_ids if isinstance(root_page_ids, list) else [root_page_ids]
        self.COLLECTION_NAME = collection_name
        self.EMBEDDING_SIZE = embedding_size

        self._indexed_pages_cache = {}

    def _build_indexed_pages_cache(self):
        """Build cache of indexed pages with timestamps"""
        if len(self._indexed_pages_cache) > 0:
            return

        logger.info("Building indexed pages cache...")

        try:
            offset = None
            while True:
                records, offset = self.qdrant.scroll(
                    collection_name=self.COLLECTION_NAME,
                    limit=1000,
                    offset=offset,
                    with_payload=["page_id", "last_updated"]
                )

                for record in records:
                    page_id = record.payload.get("page_id")
                    last_updated = record.payload.get("last_updated")
                    if page_id and last_updated:
                        self._indexed_pages_cache[page_id] = last_updated

                if offset is None:
                    break

            logger.info(f"Cached {len(self._indexed_pages_cache)} indexed pages")
        except Exception as e:
            logger.error(f"Error building cache: {e}")
            self._indexed_pages_cache = {}

    def _initialize_collection(self, reset: bool) -> None:
        """Initialize Qdrant collection"""
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

    def _check_for_update_cached(self, page_id: str, last_updated: str) -> bool:
        """Check if page needs update using cache"""
        cached_timestamp = self._indexed_pages_cache.get(page_id)
        return cached_timestamp != last_updated

    async def _process_page_async(self, session: aiohttp.ClientSession, page: Dict,
                                  reset: bool) -> PageData:
        """Process a single page - extract text, generate embeddings, create points"""
        try:
            # Use adapter to get normalized page content
            content = await self.data_adapter.fetch_page_content(session, page)

            if not content or not content.body:
                return PageData([], [], [], content.page_id if content else "")

            # Check if update needed
            needs_update = self._check_for_update_cached(content.page_id, content.last_updated)
            if not needs_update and not reset:
                return PageData([], [], [], content.page_id)

            # Delete old chunks if updating
            if needs_update and not reset:
                self._delete_old_chunks(content.page_id)

            # Extract and chunk text
            text, extracted_table = self.text_processor.extract_text_from_storage(content.body)
            text_chunks = self.text_processor.smart_chunk_text(
                text,
                extracted_table,
                self.text_processor.chunk_size_limit,
                self.text_processor.min_chunk_size,
                self.text_processor.chunk_size_overlap
            )

            if not text_chunks:
                return PageData([], [], [], content.page_id)

            # Batch embed all chunks
            chunk_texts = [x['text'] for x in text_chunks]
            embeddings = common.embed_text(self.model_embed, chunk_texts)

            page_points = []
            tfidf_texts, tfidf_ids = [], []

            for i, (chunk, emb) in enumerate(zip(text_chunks, embeddings)):
                chunk_id = f"{content.page_id}_{i}"
                point_id = str(uuid5(NAMESPACE_URL, chunk_id))
                keywords = self.text_processor.extract_keywords(chunk['text'])

                page_points.append(
                    PointStruct(
                        id=point_id,
                        vector=emb,
                        payload={
                            "title": content.title,
                            "source": self.data_source_name,
                            "page_id": content.page_id,
                            "tables": chunk['tables'],
                            "space_name": content.space_name,
                            "chunk_id": chunk_id,
                            "text": chunk['text'],
                            "keywords": keywords,
                            "last_updated": content.last_updated,
                            "link": content.link,
                            "position": i,
                            "hierarchy": content.hierarchy,
                            "text_length": len(chunk['text']),
                            "space_key": content.space_key
                        }
                    )
                )

                tfidf_texts.append(chunk['text'])
                tfidf_ids.append(chunk_id)

            return PageData(page_points, tfidf_texts, tfidf_ids, content.page_id)

        except Exception as e:
            logger.error(f"Error processing page: {e}")
            return PageData([], [], [], "")

    async def _process_batch_async(self, session: aiohttp.ClientSession, pages: List[Dict],
                                   reset: bool) -> Tuple[List[PointStruct], List[str], List[str], List[str]]:
        """Process batch of pages concurrently"""
        tasks = [self._process_page_async(session, page, reset) for page in pages]
        results = await asyncio.gather(*tasks)

        all_points = []
        all_texts = []
        all_chunk_ids = []
        child_ids = []

        for result in results:
            all_points.extend(result.points)
            all_texts.extend(result.texts)
            all_chunk_ids.extend(result.chunk_ids)
            if result.child_id:
                child_ids.append(result.child_id)

        return all_points, all_texts, all_chunk_ids, child_ids

    async def _index_pages_async(self, reset: bool, batch_size: int = 500, max_concurrent: int = 20):
        """Async indexing implementation"""
        self._initialize_collection(reset)

        if not reset:
            self._build_indexed_pages_cache()

        queue = self.root_page_ids.copy()
        visited = set()
        all_texts, all_chunk_ids = [], []

        connector = aiohttp.TCPConnector(limit=max_concurrent)
        async with aiohttp.ClientSession(connector=connector) as session:
            while queue:
                current_page_id = queue.pop(0)
                if current_page_id in visited:
                    continue
                visited.add(current_page_id)

                logger.info(f"Processing page ID: {current_page_id}")

                # Fetch children using adapter
                children = await self.data_adapter.fetch_children(session, current_page_id)

                if not children:
                    continue

                # Process pages concurrently
                points_to_upsert, texts, chunk_ids, child_ids = await self._process_batch_async(
                    session, children, reset
                )

                # Batch upsert
                for i in range(0, len(points_to_upsert), batch_size):
                    batch = points_to_upsert[i:i + batch_size]
                    self._batch_upsert(batch)

                all_texts.extend(texts)
                all_chunk_ids.extend(chunk_ids)
                queue.extend(child_ids)

        # TF-IDF rebuild
        if all_texts and reset:
            logger.info("Building TF-IDF index...")
            self.hybrid_index.fit_tfidf(all_texts, all_chunk_ids)

    def index_pages(self, reset: bool = False, batch_size: int = 500, max_concurrent: int = 20) -> None:
        """Main entry point for indexing"""
        asyncio.run(self._index_pages_async(reset, batch_size, max_concurrent))

    def _delete_old_chunks(self, page_id: str) -> None:
        """Delete old chunks for a page"""
        try:
            self.qdrant.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=Filter(
                    must=[FieldCondition(key="page_id", match=MatchValue(value=page_id))]
                )
            )
            logger.info(f"Deleted old chunks for page {page_id}")
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")

    def _batch_upsert(self, points_to_upsert: List[PointStruct]) -> None:
        """Batch upsert to Qdrant"""
        if points_to_upsert:
            try:
                self.qdrant.upsert(collection_name=self.COLLECTION_NAME, points=points_to_upsert)
                logger.info(f"Upserted {len(points_to_upsert)} points")
            except Exception as e:
                logger.error(f"Error upserting: {e}")