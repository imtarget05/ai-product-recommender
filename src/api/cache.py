"""Caching layer supporting Redis with automatic In-Memory TTL fallback."""
import time
import json
from typing import Optional, Any
from src.config import settings

class CacheManager:
    """Provides high performance caching for Top-K recommendation results."""

    def __init__(self):
        self.memory_cache = {}
        self.memory_expiry = {}
        self.redis_client = None
        self._init_redis()

    def _init_redis(self):
        try:
            import redis
            if settings.REDIS_URL:
                client = redis.from_url(
                    settings.REDIS_URL,
                    socket_timeout=1.5,
                    decode_responses=True
                )
            else:
                client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    password=settings.REDIS_PASSWORD,
                    socket_timeout=1.5,
                    decode_responses=True
                )
            client.ping()
            self.redis_client = client
            print("🚀 Connected to Redis Cache server.")
        except Exception:
            self.redis_client = None
            # Fallback to local memory cache silently

    def get(self, key: str) -> Optional[Any]:
        """Retrieve key from cache."""
        # Try Redis first if available
        if self.redis_client:
            try:
                val = self.redis_client.get(key)
                if val:
                    return json.loads(val)
            except Exception:
                pass

        # Fallback to Memory Cache
        now = time.time()
        if key in self.memory_cache:
            if now < self.memory_expiry.get(key, 0):
                return self.memory_cache[key]
            else:
                del self.memory_cache[key]
                if key in self.memory_expiry:
                    del self.memory_expiry[key]
        return None

    def set(self, key: str, value: Any, ttl_seconds: int = 300):
        """Store key-value pair with TTL."""
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl_seconds, json.dumps(value))
                return
            except Exception:
                pass

        # Fallback to Memory Cache
        self.memory_cache[key] = value
        self.memory_expiry[key] = time.time() + ttl_seconds

    def invalidate_user(self, user_id: int):
        """Invalidate all cached recommendations for a specific user."""
        prefix = f"rec:user:{user_id}:"
        # Redis delete by pattern
        if self.redis_client:
            try:
                keys = self.redis_client.keys(f"{prefix}*")
                if keys:
                    self.redis_client.delete(*keys)
            except Exception:
                pass

        # In-memory delete
        keys_to_del = [k for k in self.memory_cache.keys() if k.startswith(prefix)]
        for k in keys_to_del:
            self.memory_cache.pop(k, None)
            self.memory_expiry.pop(k, None)

cache_manager = CacheManager()
