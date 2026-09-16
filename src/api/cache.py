"""Caching layer supporting Redis with automatic In-Memory TTL fallback.
Enhanced with O(1) User Cache Versioning (eliminating blocking Redis KEYS *)
and thread-safe LRU memory cache.
"""
import time
import json
import threading
from typing import Optional, Any, Dict
from src.config import settings

class CacheManager:
    """Provides high-performance, non-blocking caching for Top-K recommendation results."""

    def __init__(self, max_memory_entries: int = 5000):
        self.max_memory_entries = max_memory_entries
        self.memory_cache: Dict[str, Any] = {}
        self.memory_expiry: Dict[str, float] = {}
        self.user_versions: Dict[int, int] = {}
        self._lock = threading.Lock()
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

    def get_user_version(self, user_id: int) -> int:
        """Get the current cache version number for a user (O(1))."""
        if self.redis_client:
            try:
                v = self.redis_client.get(f"user_ver:{user_id}")
                return int(v) if v is not None else 1
            except Exception:
                pass

        with self._lock:
            return self.user_versions.get(user_id, 1)

    def get_recommendation_cache(self, user_id: int, strategy: str, top_k: int) -> Optional[Any]:
        """Convenience getter using user cache versioning."""
        version = self.get_user_version(user_id)
        key = f"rec:u:{user_id}:v:{version}:s:{strategy}:k:{top_k}"
        return self.get(key)

    def set_recommendation_cache(self, user_id: int, strategy: str, top_k: int, value: Any, ttl_seconds: int = 300):
        """Convenience setter using user cache versioning."""
        version = self.get_user_version(user_id)
        key = f"rec:u:{user_id}:v:{version}:s:{strategy}:k:{top_k}"
        self.set(key, value, ttl_seconds=ttl_seconds)

    def get(self, key: str) -> Optional[Any]:
        """Retrieve key from cache."""
        if self.redis_client:
            try:
                val = self.redis_client.get(key)
                if val:
                    return json.loads(val)
            except Exception:
                pass

        now = time.time()
        with self._lock:
            if key in self.memory_cache:
                if now < self.memory_expiry.get(key, 0):
                    return self.memory_cache[key]
                else:
                    self.memory_cache.pop(key, None)
                    self.memory_expiry.pop(key, None)
        return None

    def set(self, key: str, value: Any, ttl_seconds: int = 300):
        """Store key-value pair with TTL and capacity protection."""
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl_seconds, json.dumps(value))
                return
            except Exception:
                pass

        now = time.time()
        with self._lock:
            # Enforce max memory capacity (LRU-like purge of expired entries)
            if len(self.memory_cache) >= self.max_memory_entries:
                expired = [k for k, exp in self.memory_expiry.items() if now >= exp]
                for k in expired[:1000]:
                    self.memory_cache.pop(k, None)
                    self.memory_expiry.pop(k, None)
                # If still at capacity, evict oldest items
                if len(self.memory_cache) >= self.max_memory_entries:
                    excess = len(self.memory_cache) - self.max_memory_entries + 100
                    for k in list(self.memory_cache.keys())[:excess]:
                        self.memory_cache.pop(k, None)
                        self.memory_expiry.pop(k, None)

            self.memory_cache[key] = value
            self.memory_expiry[key] = now + ttl_seconds

    def invalidate_user(self, user_id: int):
        """Invalidate user recommendations in O(1) via atomic version increment."""
        if self.redis_client:
            try:
                self.redis_client.incr(f"user_ver:{user_id}")
                # Set TTL on version key so it naturally expires after 7 days
                self.redis_client.expire(f"user_ver:{user_id}", 86400 * 7)
            except Exception:
                pass

        with self._lock:
            self.user_versions[user_id] = self.user_versions.get(user_id, 1) + 1

cache_manager = CacheManager()

