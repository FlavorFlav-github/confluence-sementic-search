"""
Module for managing the Qdrant vector database.
Handles connection, collection management, and data operations.
"""
import logging
import subprocess
import time
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from config import settings
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class QdrantManager:
    """
    Manages the Qdrant connection and operations.
    """

    def __init__(self, url: str):
        self.url = url
        self.client = self._check_and_connect()

    def _start_qdrant_container(self) -> None:
        """Starts the Qdrant Docker container if not running."""
        logger.info("Attempting to start Qdrant Docker container...")
        try:
            subprocess.run(
                ["docker", "run", "-d", "--name", "qdrant_search", "-p", "6333:6333", "qdrant/qdrant"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            logger.info("Qdrant container started successfully. Waiting for it to become available...")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Qdrant container: {e}. Is Docker running?")
            raise

    def _check_and_connect(self, timeout: int = 60, retry_delay: int = 5) -> QdrantClient:
        """
        Checks for and connects to Qdrant, starting the container if necessary.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                client = QdrantClient(self.url)
                client.get_collections()
                logger.info("Qdrant is running and accessible.")
                return client
            except Exception:
                logger.warning(f"Qdrant not available. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                try:
                    self._start_qdrant_container()
                except Exception:
                    pass  # Silently fail if it can't start, the loop will retry
        logger.error("Timed out waiting for Qdrant to start.")
        raise ConnectionError("Failed to connect to Qdrant after multiple retries.")

    def get_client(self) -> QdrantClient:
        """Returns the Qdrant client instance."""
        return self.client

    def create_collection(self, collection_name: str, size: int) -> None:
        """Creates a new Qdrant collection."""
        try:
            self.client.get_collection(collection_name)
            logger.info(f"Collection {collection_name} already exists.")
        except Exception:
            logger.info(f"Creating collection: {collection_name}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE),
            )

    def delete_collection(self, collection_name: str) -> None:
        """Deletes a Qdrant collection."""
        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Collection {collection_name} deleted.")
        except Exception as e:
            logger.warning(f"Could not delete collection {collection_name}: {e}")

    def upsert_points(self, collection_name: str, points: List[PointStruct]) -> None:
        """Upserts a list of points into a collection."""
        if not points:
            return
        try:
            self.client.upsert(collection_name=collection_name, points=points)
            logger.info(f"Upserted {len(points)} points to {collection_name}.")
        except Exception as e:
            logger.error(f"Error upserting points to Qdrant: {e}")

    def get_points_by_page_id(self, collection_name: str, page_id: int) -> List[Any]:
        """Retrieves points by page ID to check for updates."""
        try:
            points, _ = self.client.scroll(
                collection_name=collection_name,
                scroll_filter={"must": [{"key": "page_id", "match": {"value": page_id}}]},
                limit=1,
                with_payload=True
            )
            return points
        except Exception as e:
            logger.error(f"Error fetching points for page {page_id}: {e}")
            return []

    def delete_points_by_page_id(self, collection_name: str, page_id: int) -> None:
        """Deletes all points associated with a given page ID."""
        self.client.delete(
            collection_name=collection_name,
            points_selector={"filter": {"must": [{"key": "page_id", "match": {"value": page_id}}]}}
        )