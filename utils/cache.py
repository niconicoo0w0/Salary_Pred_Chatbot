# cache.py — Cache management with TTL (time-to-live)
import time
import json
import hashlib
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta
from .config import config
from .logger import logger


class TTLCache:
    """Cache container with expiration time (TTL)."""

    def __init__(self, ttl_hours: int = 24):
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.ttl = timedelta(hours=ttl_hours)

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached value if not expired; otherwise return None."""

        if key not in self.cache:
            return None

        value, timestamp = self.cache[key]

        # Legacy support: convert float timestamp to datetime
        if isinstance(timestamp, float):
            timestamp = datetime.fromtimestamp(timestamp)

        # Check expiration (remove if expired)
        if datetime.now() - timestamp > self.ttl:
            del self.cache[key]
            logger.debug(f"Cache expired for key: {key}")
            return None

        logger.debug(f"Cache hit for key: {key}")
        return value

    def set(self, key: str, value: Any) -> None:
        """Store a value with current timestamp."""
        self.cache[key] = (value, datetime.now())
        logger.debug(f"Cache set for key: {key}")

    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        logger.info("Cache cleared")

    def cleanup_expired(self) -> int:
        """Remove expired entries; return number removed."""
        now = datetime.now()
        expired_keys = []

        for key, (_, timestamp) in self.cache.items():
            if isinstance(timestamp, float):
                timestamp = datetime.fromtimestamp(timestamp)

            if now - timestamp > self.ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

        return len(expired_keys)

    def size(self) -> int:
        """Return number of active (non-expired) cached items."""
        return len(self.cache)


# Global cache instance created only when caching is enabled
cache = TTLCache(ttl_hours=config.CACHE_TTL_HOURS) if config.ENABLE_CACHE else None


def get_cache_key(*args, **kwargs) -> str:
    """
    Generate a unique hash key for function call arguments.
    Ensures consistent ordering of keyword arguments for deterministic hashing.
    """
    key_data = {
        "args": args,
        "kwargs": sorted(kwargs.items())
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)

    # MD5 hashing for compact cache keys
    return hashlib.md5(key_str.encode()).hexdigest()


def cached(ttl_hours: Optional[int] = None):
    """
    Decorator to transparently cache function results.
    - Uses global `cache` instance
    - Skips caching if disabled via config

    Example:
        @cached(ttl_hours=12)
        def expensive_call(...):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not config.ENABLE_CACHE or cache is None:
                return func(*args, **kwargs)

            key = get_cache_key(func.__name__, *args, **kwargs)

            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result

            result = func(*args, **kwargs)
            cache.set(key, result)
            return result

        return wrapper
    return decorator
