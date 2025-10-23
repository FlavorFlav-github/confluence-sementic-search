import subprocess
import time
from typing import Dict, Any, List, Union

from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, MatchValue, Filter

from config import settings
from config.logging_config import logger
from config.settings import COLLECTION_NAME
from collections import defaultdict


def start_qdrant():
    """
    Starts the Qdrant vector database as a Docker container.

    This function executes the necessary `docker run` command to pull (if needed)
    and start the container in detached mode (-d), name it 'qdrant', and map
    the default port 6333.

    Raises:
        subprocess.CalledProcessError: If the Docker command fails (e.g., Docker service is not running).
    """
    logger.info("Attempting to start Qdrant Docker container...")
    
    try:
        # Check if container already exists
        result = subprocess.run(
            ["docker", "ps", "-a", "-q", "-f", "name=qdrant"],
            capture_output=True, text=True
        )
        container_id = result.stdout.strip()
        
        if container_id:
            print(f"Found existing Qdrant container: {container_id}")
            # If exists but stopped, start it
            subprocess.run(["docker", "start", container_id], check=True)
            print(f"Started existing Qdrant container: {container_id}")
            return
    except subprocess.CalledProcessError:    
        try:
            # Command to run Qdrant container
            subprocess.run(
                ["docker", "run", "-d", "--name", "qdrant", "-p", "6333:6333", "qdrant/qdrant"],
                check=True  # Raise an error if the command fails
            )
            logger.info("Qdrant container started successfully. Waiting for it to become available...")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Qdrant container: {e}. Is Docker running?")
            raise


def check_and_start_qdrant(timeout: int = 60, retry_delay: int = 5) -> QdrantClient:
    """
    Connects to Qdrant, attempting to start the Docker container if the connection fails.

    This function first attempts to connect directly. If that fails, it assumes the service
    is down, attempts to start it via Docker, and then enters a retry loop to wait for
    the service to become healthy.

    Args:
        timeout (int): The total time (in seconds) to wait for Qdrant to start and become accessible.
        retry_delay (int): The time (in seconds) to wait between connection attempts.

    Returns:
        QdrantClient: An initialized and successfully connected Qdrant client object.

    Raises:
        SystemExit: If starting or connecting to Qdrant fails after all retries.
    """
    logger.info("Checking and starting Qdrant database...")

    # --- Initial Connection Check ---
    try:
        # Attempt to create a client and call a simple method (get_collections)
        # to verify connectivity and readiness.
        qdrant_client = QdrantClient(settings.QDRANT_URL)
        qdrant_client.get_collections()
        logger.info("Qdrant is running and accessible.")
        return qdrant_client
    except Exception:
        # If connection fails, log and pause before attempting to start the service.
        logger.info(f"Qdrant is not yet available. Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)

    # --- Attempt to Start and Retry Connection ---
    try:
        # Try to start the Qdrant Docker container
        start_qdrant()
        start_time = time.time()
        
        # Enter a loop to poll the Qdrant service until timeout
        while time.time() - start_time < timeout:
            try:
                # Re-attempt connection
                qdrant_client = QdrantClient(settings.QDRANT_URL)
                qdrant_client.get_collections()
                logger.info("Qdrant container started and is now accessible.")
                return qdrant_client
            except Exception:
                # Connection failed, wait and retry
                time.sleep(retry_delay)
        
        # If the loop finishes without success
        raise TimeoutError(f"Qdrant did not become accessible within the {timeout} second timeout.")
        
    except Exception as e:
        # Catch any errors during the startup attempt or the connection loop
        logger.error(f"Failed to connect to or start Qdrant: {e}")
        # Exit the application as the required service is unavailable
        exit(1)

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


# TODO : Remove those for production
def main():
    check_and_start_qdrant()
    print(get_documents_by_metadata(
        QdrantClient(settings.QDRANT_URL),COLLECTION_NAME, {'title': '4.3.6.9 - Create and update policies, permissions and product roles'}, with_vectors=False, with_payload=True, return_count_only=False))

if __name__ == "__main__":
    main()