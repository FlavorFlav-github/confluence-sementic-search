import subprocess
import time

from qdrant_client import QdrantClient

from config import settings
from config.logging_config import logger

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
    logger.info("Checking and starting Qdrant database...")
    try:
        qdrant_client = QdrantClient(settings.QDRANT_URL)
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
                qdrant_client = QdrantClient(settings.QDRANT_URL)
                qdrant_client.get_collections()
                logger.info("Qdrant container started and is now accessible.")
                return qdrant_client
            except Exception:
                time.sleep(retry_delay)
    except Exception as e:
        logger.error(f"Failed to connect to or start Qdrant: {e}")
        exit(1)