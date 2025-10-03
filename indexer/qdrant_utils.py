import subprocess
import time

from qdrant_client import QdrantClient

from config import settings
from config.logging_config import logger


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
