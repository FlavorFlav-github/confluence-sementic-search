import time
from functools import wraps
from typing import Optional, Dict
from config.logging_config import logger

class Timer:
    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed: Optional[float] = None
        self._start: Optional[float] = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self  # allows `as t_outer`

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.perf_counter()
        self.elapsed = end - self._start
        logger.info(f"[TIME] {self.label}: {self.elapsed:.6f} seconds")

def timed(label: str = None, store: Optional[Dict[str, float]] = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            elapsed = end - start
            key = label or func.__name__
            if store is not None:
                store[key] = elapsed
            logger.info(f"[TIME] {key}: {elapsed:.6f} seconds")
            return result
        return wrapper
    return decorator

def async_timed(label: str = None, store: Optional[Dict[str, float]] = None):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            end = time.perf_counter()
            elapsed = end - start
            key = label or func.__name__
            if store is not None:
                store[key] = elapsed
            logger.info(f"[TIME] {key}: {elapsed:.6f} seconds")
            return result
        return wrapper
    return decorator