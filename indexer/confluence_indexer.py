from uuid import uuid5, NAMESPACE_URL
from typing import List, Dict
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from qdrant_client import QdrantClient
# Import Qdrant models for defining collection properties and points
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
    A comprehensive class to manage the entire indexing pipeline for Confluence documentation.

    This class handles fetching content from the Confluence API, processing the text,
    generating vector embeddings and keyword data, and upserting the data into a
    Qdrant vector database. It supports full re-indexing and incremental updates.
    """

    def __init__(self, qdrant_client: QdrantClient, embedding_model: SentenceTransformer,
                 text_processor: EnhancedTextProcessor, hybrid_index: HybridSearchIndex,
                 root_page_id: int, collection_name: str,
                 embedding_size: int, confluence_base_url: str, confluence_api_token: str):
        """
        Initializes the indexer with all required dependencies and configuration.

        Args:
            qdrant_client (QdrantClient): The Qdrant client instance for database operations.
            embedding_model (SentenceTransformer): The model for generating vector embeddings.
            text_processor (EnhancedTextProcessor): The utility for cleaning and chunking text.
            hybrid_index (HybridSearchIndex): The keyword-based index for hybrid search.
            root_page_id (int): The starting page ID for recursive indexing.
            collection_name (str): The name of the Qdrant collection.
            embedding_size (int): The dimension of the vector embeddings.
            confluence_base_url (str): The base URL for the Confluence API.
            confluence_api_token (str): The API token for authentication.
        """
        self.qdrant = qdrant_client
        self.model_embed = embedding_model
        self.text_processor = text_processor
        self.hybrid_index = hybrid_index
        self.saved_spaces_name = {}
        # Store configuration parameters as instance attributes
        self.ROOT_PAGE_ID = root_page_id
        self.COLLECTION_NAME = collection_name
        self.EMBEDDING_SIZE = embedding_size
        self.CONFLUENCE_BASE_URL = confluence_base_url
        self.CONFLUENCE_API_TOKEN = confluence_api_token

    async def _fetch_children_async(self, session: aiohttp.ClientSession, page_id: int, limit: int = 100) -> List[Dict]:
        """
        Fetches the child pages of a given Confluence page from the API.

        Args:
            page_id (int): The ID of the parent page.
            limit (int): The number of child pages to retrieve in one API call.

        Returns:
            List[Dict]: A list of dictionaries, where each dictionary represents a child page.
                        Returns an empty list on failure or no children.
        """
        url = f"{self.CONFLUENCE_BASE_URL}/rest/api/content/{page_id}/child/page"
        params = {
            "limit": limit,
            "expand": "body.storage,version,ancestors,space,metadata.labels"
        }
        headers = {"Authorization": f"Bearer {self.CONFLUENCE_API_TOKEN}"}
        try:
            async with session.get(url, params=params, headers=headers, timeout=30) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("results", [])
        except Exception as e:
            logger.error(f"Error fetching children for page {page_id}: {e}")
            return []

    async def _fetch_all_children(self, page_ids: list, max_concurrent: int = 10):
        """
        Fetch all child pages asynchronously with a concurrency limit.
        """
        connector = aiohttp.TCPConnector(limit=max_concurrent)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self._fetch_children_async(session, pid) for pid in page_ids]
            results = await asyncio.gather(*tasks)
            # Flatten the list of lists
            return [child for page_children in results for child in page_children]
        
    @staticmethod
    def _build_page_hierarchy(page_data: Dict) -> List[str]:
        """
        Constructs a list representing the hierarchical path of a Confluence page.

        Args:
            page_data (Dict): The dictionary containing page information from the API.

        Returns:
            List[str]: A list of page titles from the root down to the current page.
        """
        hierarchy = []
        if 'ancestors' in page_data:
            # Ancestors are returned in reverse order (closest first), so we reverse to get root-first
            for ancestor in page_data['ancestors']:
                hierarchy.append(ancestor['title'])
        hierarchy.append(page_data['title'])
        return hierarchy

    def _initialize_collection(self, reset: bool) -> None:
        """
        Manages the Qdrant collection. Creates it if it doesn't exist, and deletes it first if `reset` is True.

        Args:
            reset (bool): If True, the collection will be deleted and recreated.
        """
        if reset:
            logger.info(f"Deleting collection: {self.COLLECTION_NAME}")
            try:
                self.qdrant.delete_collection(collection_name=self.COLLECTION_NAME)
            except Exception:
                # Ignore errors if the collection does not exist
                pass

        try:
            # Check if the collection already exists
            self.qdrant.get_collection(self.COLLECTION_NAME)
            logger.info(f"Collection {self.COLLECTION_NAME} already exists")
        except Exception:
            # If not, create a new collection with specified vector parameters
            logger.info(f"Creating Qdrant collection {self.COLLECTION_NAME}")
            self.qdrant.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(size=self.EMBEDDING_SIZE, distance=Distance.COSINE),
            )

    def _get_space_name(self, space_id, space_uri):
        space_name = self.saved_spaces_name.get(space_id, None)
        if space_name is None:
            url = f"{self.CONFLUENCE_BASE_URL}{space_uri}"
            headers = {"Authorization": f"Bearer {self.CONFLUENCE_API_TOKEN}"}
            logger.info(f"Running request GET {url}")
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as e:
                logger.error(f"Request GET {url} failed: {e}")
                data = {}
            space_name = data.get("name", None)
            self.saved_spaces_name[space_id] = space_name
        return space_name
        
    def index_pages(self, reset: bool = False, max_workers: int = 8, batch_size: int = 500) -> None:
        """
        The main method to start the indexing process.

        It performs a breadth-first traversal of the Confluence page tree, processes each page,
        and upserts the resulting chunks into Qdrant. It also manages the TF-IDF index.

        Args:
            reset (bool): If True, the entire Qdrant collection will be wiped and re-indexed.
        """
        self._initialize_collection(reset)
        if not isinstance(self.ROOT_PAGE_ID, list):
            queue = [self.ROOT_PAGE_ID]
        else:
            queue = self.ROOT_PAGE_ID
        visited = set()
        all_texts, all_chunk_ids = [], []

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while queue:
            current_page_id = queue.pop(0)
            if current_page_id in visited:
                continue
            visited.add(current_page_id)

            logger.info(f"Fetching children asynchronously for page ID: {current_page_id}")
            children = loop.run_until_complete(self._fetch_all_children([current_page_id]))

            # Threaded processing remains the same
            points_to_upsert = []

            def process_page(page):
                page_id = int(page["id"])
                title = page["title"]
                
                if current_page_id == self.ROOT_PAGE_ID:
                    logger.info(f"Skipped root page content: {title}")
                    return [], []

                try:
                    body = page["body"]["storage"]
                    last_updated = page["version"]["when"]
                    space_uri = page.get("_expandable", {}).get("container", "")
                    space_key = space_uri.split("/")[-1]
                    space_name = self._get_space_name(space_key, space_uri)
                    link = f"{self.CONFLUENCE_BASE_URL}/spaces/{space_key}/pages/{page_id}"
                    hierarchy = self._build_page_hierarchy(page)

                    needs_update = self._check_for_update(page_id, last_updated, title)
                    if not needs_update and not reset:
                        queue.append(page_id)
                        return [], []

                    if needs_update and not reset:
                        self._delete_old_chunks(page_id)

                    text, extracted_table = self.text_processor.extract_text_from_storage(body)
                    text_chunks = self.text_processor.smart_chunk_text(
                        text,
                        extracted_table,
                        self.text_processor.chunk_size_limit,
                        self.text_processor.min_chunk_size,
                        self.text_processor.chunk_size_overlap
                    )

                    if not text_chunks:
                        return [], []

                    embeddings = common.embed_text(self.model_embed, [x['text'] for x in text_chunks])

                    page_points = []
                    tfidf_texts, tfidf_ids = [], []

                    for i, (chunk, emb) in enumerate(zip(text_chunks, embeddings)):
                        chunk_id = f"{page_id}_{i}"
                        point_id = str(uuid5(NAMESPACE_URL, chunk_id))
                        keywords = self.text_processor.extract_keywords(chunk['text'])

                        page_points.append(
                            PointStruct(
                                id=point_id,
                                vector=emb,
                                payload={
                                    "title": title,
                                    "page_id": page_id,
                                    "tables": chunk['tables'],
                                    "space_name": space_name,
                                    "chunk_id": chunk_id,
                                    "text": chunk['text'],
                                    "keywords": keywords,
                                    "last_updated": last_updated,
                                    "link": link,
                                    "position": i,
                                    "hierarchy": hierarchy,
                                    "text_length": len(chunk['text']),
                                    "space_key": space_key
                                }
                            )
                        )

                        tfidf_texts.append(chunk['text'])
                        tfidf_ids.append(chunk_id)

                    return page_points, (tfidf_texts, tfidf_ids)

                except Exception as e:
                    logger.error(f"Error processing page {title}: {e}")
                    return [], []

            # ThreadPoolExecutor for page processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_page, page) for page in children]
                for future in as_completed(futures):
                    page_points, tfidf_data = future.result()
                    if page_points:
                        points_to_upsert.extend(page_points)
                    if tfidf_data:
                        texts, ids = tfidf_data
                        all_texts.extend(texts)
                        all_chunk_ids.extend(ids)

            # Batch upsert
            for i in range(0, len(points_to_upsert), batch_size):
                batch = points_to_upsert[i:i + batch_size]
                self._batch_upsert(batch)

            # Add child pages to queue
            for page in children:
                queue.append(int(page["id"]))

        # TF-IDF rebuild
        if all_texts and reset:
            logger.info("Building TF-IDF index for keyword search...")
            self.hybrid_index.fit_tfidf(all_texts, all_chunk_ids)
        elif not reset:
            logger.info("No TF-IDF rebuild requested. Using existing index.")

    def _check_for_update(self, page_id: int, last_updated: str, title: str) -> bool:
        """
        Checks if a page has been updated by comparing its `last_updated` timestamp in Qdrant.

        Args:
            page_id (int): The page ID to check.
            last_updated (str): The new `last_updated` timestamp from the API.
            title (str): The page title for logging.

        Returns:
            bool: True if an update is needed (page is new or timestamp differs), False otherwise.
        """
        try:
            # Query Qdrant for any point with the given page_id
            existing_points = self.qdrant.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[FieldCondition(key="page_id", match=MatchValue(value=page_id))]
                ),
                limit=1,
                with_payload=True  # Retrieve the payload to check the timestamp
            )[0]

            if existing_points and existing_points[0].payload.get("last_updated") == last_updated:
                logger.info(f"Skipping unchanged page: {title}")
                return False  # Page is already up-to-date
            else:
                return True # Page is new or has been updated

        except Exception:
            # If the scroll operation fails (e.g., page not found), assume it's a new page that needs indexing
            return True

    def _delete_old_chunks(self, page_id: int) -> None:
        """
        Deletes all chunks associated with a specific page ID from Qdrant.
        This is a necessary step for updating a page's content.

        Args:
            page_id (int): The ID of the page whose chunks should be deleted.
        """
        try:
            self.qdrant.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=Filter(
                    must=[FieldCondition(key="page_id", match=MatchValue(value=page_id))]
                )
            )
            logger.info(f"Deleted old chunks for page {page_id}")
        except Exception as e:
            logger.error(f"Error deleting old chunks for page {page_id}: {e}")

    def _batch_upsert(self, points_to_upsert: List[PointStruct]) -> None:
        """
        Inserts or updates a list of points (chunks) into Qdrant in a single batch operation.

        Args:
            points_to_upsert (List[PointStruct]): A list of `PointStruct` objects to be upserted.
        """
        if points_to_upsert:
            try:
                self.qdrant.upsert(collection_name=self.COLLECTION_NAME, points=points_to_upsert)
                logger.info(f"Upserted {len(points_to_upsert)} points to Qdrant")
            except Exception as e:
                logger.error(f"Error upserting points: {e}")
