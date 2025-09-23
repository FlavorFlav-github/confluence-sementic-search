"""
Main indexing pipeline for crawling and processing Confluence pages.
"""
import logging
import uuid
import time
from typing import List
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

from config import settings
from data.text_processor import EnhancedTextProcessor
from data.vector_store import QdrantManager
from api.confluence import ConfluenceClient, build_page_hierarchy
from search.hybrid_search import HybridSearch

logger = logging.getLogger(__name__)


def index_confluence_pages(
    qdrant_manager: QdrantManager,
    confluence_client: ConfluenceClient,
    text_processor: EnhancedTextProcessor,
    embedding_model: SentenceTransformer,
    hybrid_search_index: HybridSearch,
    root_page_id: int,
    reset: bool = False
) -> None:
    """
    Crawls Confluence pages, processes their content, and indexes them in Qdrant.

    Args:
        qdrant_manager: The QdrantManager instance.
        confluence_client: The ConfluenceClient instance.
        text_processor: The EnhancedTextProcessor instance.
        embedding_model: The SentenceTransformer model.
        hybrid_search_index: The HybridSearch instance for TF-IDF indexing.
        root_page_id: The ID of the root page to start crawling from.
        reset: If True, the existing collection will be deleted before re-indexing.
    """
    if reset:
        qdrant_manager.delete_collection(settings.COLLECTION_NAME)

    qdrant_manager.create_collection(settings.COLLECTION_NAME, settings.EMBEDDING_SIZE)

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
        children = confluence_client.fetch_children(current_page_id)
        points_to_upsert = []

        for page in children:
            page_id = int(page["id"])
            title = page["title"]

            # Skip root page content itself but process its children
            if current_page_id == settings.CONFLUENCE_ROOT_PAGE_ID and page_id == settings.CONFLUENCE_ROOT_PAGE_ID:
                continue

            try:
                # Extract metadata
                body = page["body"]["storage"]
                last_updated = page["version"]["when"]
                link = f"https://confluence.sage.com/spaces/{settings.SPACE_KEY}/pages/{page_id}"
                hierarchy = build_page_hierarchy(page)

                # Check if page needs re-indexing
                existing_points = qdrant_manager.get_points_by_page_id(settings.COLLECTION_NAME, page_id)
                if existing_points and existing_points[0].payload.get("last_updated") == last_updated:
                    logger.info(f"Skipping unchanged page: {title}")
                    queue.append(page_id)
                    continue
                else:
                    if existing_points:
                        logger.info(f"Detected updated page: {title}. Re-indexing...")
                        qdrant_manager.delete_points_by_page_id(settings.COLLECTION_NAME, page_id)

                # Process text and create chunks
                text = text_processor.extract_text_from_storage(body)
                if not text or len(text) < settings.MIN_CHUNK_SIZE:
                    continue

                text_chunks = text_processor.smart_chunk_text(
                    text, settings.CHUNK_SIZE_LIMIT, settings.CHUNK_OVERLAP, settings.MIN_CHUNK_SIZE
                )

                if not text_chunks:
                    continue

                chunk_embeddings = [
                    embedding_model.encode(chunk, convert_to_numpy=True, normalize_embeddings=True).tolist()
                    for chunk in text_chunks
                ]

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
                                "space_key": settings.SPACE_KEY
                            }
                        )
                    )

                logger.info(f"Prepared {len(text_chunks)} chunks for page: {title}")

            except Exception as e:
                logger.error(f"Error processing page {title} (ID: {page_id}): {e}")
                continue

            queue.append(page_id)

        qdrant_manager.upsert_points(settings.COLLECTION_NAME, points_to_upsert)

    if all_texts:
        logger.info("Building TF-IDF index for keyword search...")
        hybrid_search_index.fit_tfidf(all_texts, all_chunk_ids)