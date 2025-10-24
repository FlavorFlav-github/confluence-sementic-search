import time
from typing import Dict, Any, List, Union

from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.http.models import FieldCondition, MatchValue, Filter
from qdrant_client import QdrantClient

from config.logging_config import logger
from collections import defaultdict


def get_qdrant_client(url: str, max_wait_seconds: int = 30) -> QdrantClient:
    """
    Create Qdrant client with retry logic.

    Args:
        url: Qdrant connection URL
        max_wait_seconds: Maximum time to wait for connection (default: 30)

    Returns:
        QdrantClient instance

    Raises:
        Exception: If connection fails after max_wait_seconds
    """
    start_time = time.time()
    attempt = 0

    while True:
        elapsed = time.time() - start_time

        if elapsed >= max_wait_seconds:
            raise Exception(
                f"Failed to connect to Qdrant at {url} after {max_wait_seconds} seconds"
            )

        attempt += 1
        try:
            print(f"Attempting to connect to Qdrant at {url} (attempt {attempt})...")
            client = QdrantClient(url=url, timeout=5)

            # Test the connection
            client.get_collections()

            print(f"✓ Successfully connected to Qdrant after {elapsed:.2f} seconds")
            return client

        except (ResponseHandlingException, UnexpectedResponse, Exception) as e:
            remaining = max_wait_seconds - elapsed

            if remaining <= 0:
                raise Exception(
                    f"Failed to connect to Qdrant: {str(e)}"
                ) from e

            # Exponential backoff with cap at 5 seconds
            wait_time = min(2 ** (attempt - 1), 5)
            wait_time = min(wait_time, remaining)

            print(f"⚠ Connection failed: {str(e)}")
            print(f"  Retrying in {wait_time:.1f}s... ({remaining:.1f}s remaining)")

            time.sleep(wait_time)

def get_documents_by_metadata(
    client,
    collection_name: str,
    filters: Dict[str, Any] = None,
    limit: int = 100,
    with_vectors: bool = False,
    with_payload: bool = True,
    return_count_only: bool = False,
    page_id_field: str = "page_id",
    space_field: str = "space_name"
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Retrieve documents from a Qdrant collection based on metadata filters.
    Can also return counts of documents and unique page IDs.
    In count mode, the function aggregates counts by space_name and ignores 'limit'.

    Args:
        client: QdrantClient instance.
        collection_name (str): Name of the collection to query.
        filters (Dict[str, Any], optional): Metadata filters. Defaults to None.
        limit (int): Max number of documents to fetch (ignored in count mode). Defaults to 100.
        with_vectors (bool): Include vector embeddings. Defaults to False.
        with_payload (bool): Include payload in results. Defaults to True.
        return_count_only (bool): If True, return counts instead of full documents. Defaults to False.
        page_id_field (str): Field in payload representing unique page ID. Defaults to "page_id".
        space_field (str): Field in payload representing the space name. Defaults to "space_name".

    Returns:
        Union[List[Dict[str, Any]], Dict[str, Any]]:
            - If return_count_only=False: list of documents with id, payload, vector
            - If return_count_only=True: dict with total and per-space aggregates
    """
    # Build Qdrant filter structure if filters are provided
    query_filter = None
    if filters:
        conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
        query_filter = Filter(must=conditions)

    try:
        if return_count_only:
            # Scroll through all points for accurate counts
            offset = None
            total_documents = 0
            unique_pages_global = set()
            space_counts = defaultdict(lambda: {"total_documents": 0, "unique_pages": set()})

            while True:
                points, next_offset = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=query_filter,
                    limit=1000,  # chunk size per request
                    offset=offset,
                    with_vectors=False,
                    with_payload=True,  # needed for page_id and space_name
                )

                for point in points:
                    total_documents += 1
                    if point.payload:
                        # Count unique page
                        if page_id_field in point.payload:
                            unique_pages_global.add(point.payload[page_id_field])

                        # Aggregate per space
                        space_name = point.payload.get(space_field, "unknown")
                        space_counts[space_name]["total_documents"] += 1
                        if page_id_field in point.payload:
                            space_counts[space_name]["unique_pages"].add(point.payload[page_id_field])

                if not next_offset:
                    break
                offset = next_offset

            # Convert sets to counts
            for space in space_counts:
                space_counts[space]["unique_pages"] = len(space_counts[space]["unique_pages"])

            return {
                "total_documents": total_documents,
                "unique_pages": len(unique_pages_global),
                "per_space": dict(space_counts)
            }

        # Normal mode: return documents with optional limit, vectors, and payload
        result = client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_vectors=with_vectors,
            with_payload=with_payload,
        )
        points, _ = result

        documents = []
        for point in points:
            documents.append({
                "id": point.id,
                "payload": point.payload if with_payload else None,
                "vector": point.vector if with_vectors else None,
            })
        return documents

    except Exception as e:
        logger.error(f"Failed to retrieve documents from Qdrant: {e}")
        if return_count_only:
            return {"total_documents": 0, "unique_pages": 0, "detail": {}}
        return []


from qdrant_client import QdrantClient
from typing import List, Dict, Any


def get_qdrant_stats(client) -> List[Dict[str, Any]]:
    """
    Get comprehensive information about all collections in Qdrant.

    Args:
        client: Qdratnt client instance.

    Returns:
        List of dictionaries containing detailed information about each collection
    """
    # Get all collections
    collections = client.get_collections()

    collections_data = []

    for collection in collections.collections:
        collection_name = collection.name

        # Get detailed collection information
        collection_info = client.get_collection(collection_name=collection_name)

        # Extract and organize the information
        collection_dict = {
            "name": collection_name,
            "status": collection_info.status,
            "optimizer_status": collection_info.optimizer_status,
            "points_count": collection_info.points_count,
            "vectors_count": collection_info.vectors_count,
            "segments_count": collection_info.segments_count,
            "indexed_vectors_count": collection_info.indexed_vectors_count,
            "config": {
                "params": {
                    "vectors": collection_info.config.params.vectors,
                    "shard_number": collection_info.config.params.shard_number,
                    "replication_factor": collection_info.config.params.replication_factor,
                    "write_consistency_factor": collection_info.config.params.write_consistency_factor,
                },
                "hnsw_config": collection_info.config.hnsw_config,
                "optimizer_config": collection_info.config.optimizer_config,
                "wal_config": collection_info.config.wal_config,
            },
            "payload_schema": collection_info.payload_schema if hasattr(collection_info, 'payload_schema') else {}
        }

        collections_data.append(collection_dict)

    return collections_data
