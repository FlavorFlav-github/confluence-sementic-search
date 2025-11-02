import re
import subprocess
import time
import unicodedata

import redis
import json
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime
from config.logging_config import logger
from config.settings import REDIS_HOST, REDIS_PORT


class RAGCacheHelper:
    """
    Redis-based caching system for RAG queries.
    Caches answers and invalidates them based on collection updates.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0,
                 ttl_days: int = 30, password: Optional[str] = None):
        """
        Initialize Redis cache helper.

        Args:
            host: Redis host address
            port: Redis port
            db: Redis database number
            ttl_days: Time-to-live for cache entries in days
            password: Redis password (if required)
        """
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True
        )
        self.ttl_seconds = ttl_days * 24 * 3600

        # Key prefixes for organization
        self.CACHE_PREFIX = "rag:cache:"
        self.COLLECTION_UPDATE_PREFIX = "rag:collection:update:"

        logger.info(f"✅ Redis cache initialized at {host}:{port}")

    def _generate_cache_key(self, question: str, collection_name: str,
                            top_k: int = 10, final_top_k: int = 3,
                            score_threshold: float = 0) -> str:
        """
        Generate a unique cache key based on query parameters.

        Args:
            question: The user's question
            collection_name: Name of the Qdrant collection
            top_k: Number of initial results
            final_top_k: Number of final results
            score_threshold: Minimum score threshold

        Returns:
            SHA256 hash as cache key
        """
        # Normalize the question text
        normalized_question = unicodedata.normalize('NFC', question)
        normalized_question = normalized_question.lower().strip()
        normalized_question = re.sub(r'\s+', '', normalized_question)

        # Normalize collection name similarly
        normalized_collection = unicodedata.normalize('NFC', collection_name)
        normalized_collection = normalized_collection.lower().strip()

        # Format numeric parameters consistently
        # Use fixed decimal places for float to avoid floating point issues
        score_str = f"{score_threshold:.6f}"

        # Create cache input with all parameters
        cache_input = f"{normalized_collection}:{normalized_question}:{top_k}:{final_top_k}:{score_str}"

        # Generate hash
        cache_hash = hashlib.sha256(cache_input.encode('utf-8')).hexdigest()

        return f"{self.CACHE_PREFIX}{normalized_collection}:{cache_hash}"

    def _get_collection_update_key(self, collection_name: str) -> str:
        """Get the Redis key for storing collection last update timestamp."""
        return f"{self.COLLECTION_UPDATE_PREFIX}{collection_name}"

    def start_redis(self):
        """
        Starts a Redis Docker container if it doesn't exist or is stopped.

        - Checks for an existing container named 'redis'
        - Starts it if stopped
        - Builds/runs a new one if missing

        Raises:
            subprocess.CalledProcessError: If the Docker commands fail.
        """
        logger.info("Attempting to start Redis Docker container...")

        try:
            # Check if container already exists
            result = subprocess.run(
                ["docker", "ps", "-a", "-q", "-f", "name=redis"],
                capture_output=True, text=True, check=True
            )
            container_id = result.stdout.strip()

            if container_id:
                logger.info(f"Found existing Redis container: {container_id}")

                # Check if it's running
                running = subprocess.run(
                    ["docker", "ps", "-q", "-f", "name=redis"],
                    capture_output=True, text=True, check=True
                ).stdout.strip()

                if running:
                    logger.info("Redis container is already running.")
                    return

                # If exists but stopped, start it
                subprocess.run(["docker", "start", "redis"], check=True)
                logger.info("Started existing Redis container.")
                return

            # If no existing container, run a new one
            subprocess.run(
                [
                    "docker", "run", "-d",
                    "--name", "redis",
                    "-p", "6379:6379",
                    "redis:latest"
                ],
                check=True
            )
            logger.info("Redis container started successfully.")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Redis container: {e}. Is Docker running?")
            raise

    def check_and_start_redis(self, timeout: int = 60, retry_delay: int = 5):
        """
        Ensures Redis is running locally (Docker), starting it if needed.

        Attempts to connect to Redis, starts the container if unreachable,
        and retries until timeout.

        Args:
            timeout (int): Seconds to wait for Redis to become ready.
            retry_delay (int): Seconds to wait between connection retries.

        Returns:
            redis.Redis: A connected Redis client instance.

        Raises:
            SystemExit: If unable to start or connect to Redis.
        """
        logger.info("Checking and starting Redis service...")

        redis_host = REDIS_HOST
        redis_port = REDIS_PORT

        # --- Try initial connection ---
        try:
            client = redis.Redis(host=redis_host, port=redis_port)
            client.ping()
            logger.info("Redis is running and accessible.")
            return client
        except ConnectionError:
            logger.info("Redis not available. Attempting to start container...")
            time.sleep(retry_delay)

        # --- Start Redis container and retry connection ---
        try:
            self.start_redis()
            start_time = time.time()

            while time.time() - start_time < timeout:
                try:
                    client = redis.Redis(host=redis_host, port=redis_port)
                    client.ping()
                    logger.info("Redis container started and accessible.")
                    return client
                except ConnectionError:
                    time.sleep(retry_delay)

            raise TimeoutError(f"Redis did not become accessible within {timeout} seconds.")

        except Exception as e:
            logger.error(f"Failed to connect to or start Redis: {e}")
            exit(1)

    def set_collection_update_time(self, collection_name: str,
                                   update_timestamp: Optional[float] = None) -> None:
        """
        Store the last update timestamp for a collection.

        Args:
            collection_name: Name of the collection
            update_timestamp: Unix timestamp (if None, uses current time)
        """
        if update_timestamp is None:
            update_timestamp = datetime.now().timestamp()

        key = self._get_collection_update_key(collection_name)
        self.redis_client.set(key, str(update_timestamp))
        logger.info(f"📝 Updated timestamp for collection '{collection_name}': {update_timestamp}")

    def get_collection_update_time(self, collection_name: str) -> Optional[float]:
        """
        Get the last update timestamp for a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Unix timestamp or None if not found
        """
        key = self._get_collection_update_key(collection_name)
        timestamp = self.redis_client.get(key)
        return float(timestamp) if timestamp else None

    def get_cached_answer(self, question: str, collection_name: str,
                          collection_last_update: Optional[float] = None,
                          top_k: int = 10, final_top_k: int = 3,
                          score_threshold: float = 0) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached answer if valid.

        Args:
            question: The user's question
            collection_name: Name of the Qdrant collection
            collection_last_update: Unix timestamp of collection's last update
            top_k: Number of initial results
            final_top_k: Number of final results
            score_threshold: Minimum score threshold

        Returns:
            Cached answer dict or None if not found/invalid
        """
        cache_key = self._generate_cache_key(question, collection_name, top_k,
                                             final_top_k, score_threshold)

        # Try to get cached data
        cached_data = self.redis_client.get(cache_key)

        if not cached_data:
            logger.info(f"🔍 Cache MISS for question: '{question[:50]}...'")
            return None

        # Parse cached data
        try:
            cached_entry = json.loads(cached_data)
        except json.JSONDecodeError:
            logger.warning(f"⚠️ Invalid cache data for key: {cache_key}")
            self.redis_client.delete(cache_key)
            return None

        # Check if cache is still valid based on collection update time
        if collection_last_update:
            cached_time = cached_entry.get('cached_at', 0)
            if collection_last_update > cached_time:
                logger.info(f"🔄 Cache INVALID (collection updated): '{question[:50]}...'")
                self.redis_client.delete(cache_key)
                return None

        logger.info(f"✅ Cache HIT for question: '{question[:50]}...'")
        return cached_entry.get('answer')

    def cache_answer(self, question: str, answer: Dict[str, Any],
                     collection_name: str, top_k: int = 10,
                     final_top_k: int = 3, score_threshold: float = 0) -> None:
        """
        Cache an answer in Redis.

        Args:
            question: The user's question
            answer: The answer dictionary to cache
            collection_name: Name of the Qdrant collection
            top_k: Number of initial results
            final_top_k: Number of final results
            score_threshold: Minimum score threshold
        """
        cache_key = self._generate_cache_key(question, collection_name, top_k,
                                             final_top_k, score_threshold)

        # Create cache entry with metadata
        cache_entry = {
            'answer': answer,
            'cached_at': datetime.now().timestamp(),
            'collection': collection_name
        }

        # Store in Redis with TTL
        self.redis_client.setex(
            cache_key,
            int(self.ttl_seconds),
            json.dumps(cache_entry)
        )

        logger.info(f"💾 Cached answer for: '{question[:50]}...'")

    def invalidate_collection_cache(self, collection_name: str) -> int:
        """
        Invalidate all cached entries for a specific collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Number of keys deleted
        """
        pattern = f"{self.CACHE_PREFIX}{collection_name}:*"
        deleted_count = 0

        # Scan for all matching keys and delete them
        for key in self.redis_client.scan_iter(match=pattern, count=100):
            self.redis_client.delete(key)
            deleted_count += 1

        logger.info(f"🗑️ Invalidated {deleted_count} cache entries for collection '{collection_name}'")
        return deleted_count

    def get_cache_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get cache statistics.

        Args:
            collection_name: Optional collection name to filter stats

        Returns:
            Dictionary with cache statistics
        """
        if collection_name:
            pattern = f"{self.CACHE_PREFIX}{collection_name}:*"
        else:
            pattern = f"{self.CACHE_PREFIX}*"

        keys = list(self.redis_client.scan_iter(match=pattern, count=100))

        stats = {
            'total_cached_queries': len(keys),
            'collection': collection_name or 'all',
            'redis_info': {
                'used_memory_human': self.redis_client.info('memory')['used_memory_human'],
                'connected_clients': self.redis_client.info('clients')['connected_clients']
            }
        }

        return stats

    def clear_all_cache(self) -> int:
        """
        Clear all RAG cache entries.

        Returns:
            Number of keys deleted
        """
        pattern = f"{self.CACHE_PREFIX}*"
        deleted_count = 0

        for key in self.redis_client.scan_iter(match=pattern, count=100):
            self.redis_client.delete(key)
            deleted_count += 1

        logger.info(f"🗑️ Cleared all cache: {deleted_count} entries deleted")
        return deleted_count

    def health_check(self) -> bool:
        """
        Check if Redis connection is healthy.

        Returns:
            True if connection is working
        """
        try:
            self.redis_client.ping()
            return True
        except redis.ConnectionError:
            logger.error("❌ Redis connection failed")
            return False