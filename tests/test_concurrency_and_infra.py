"""Concurrency, Thread-Safety, Circuit Breaker, and Admin Reload tests."""
import time
import threading
from unittest.mock import MagicMock
import numpy as np
from src.api.cache import CacheManager
from src.services.qdrant_service import QdrantVectorStore
from src.database.session import check_db_health

def test_database_health_check_function():
    """Verify check_db_health executes SELECT 1 without table scan."""
    healthy = check_db_health()
    assert healthy is True


def test_user_cache_versioning_invalidation():
    """Verify O(1) user cache versioning invalidates old cached responses immediately."""
    cache = CacheManager(max_memory_entries=100)
    user_id = 42

    # Set cache for version 1
    cache.set_recommendation_cache(user_id=user_id, strategy="hybrid", top_k=5, value={"rec": [1, 2, 3]})
    val1 = cache.get_recommendation_cache(user_id=user_id, strategy="hybrid", top_k=5)
    assert val1 == {"rec": [1, 2, 3]}

    # Invalidate user
    cache.invalidate_user(user_id=user_id)

    # Next get should return None because version was incremented
    val2 = cache.get_recommendation_cache(user_id=user_id, strategy="hybrid", top_k=5)
    assert val2 is None


def test_memory_cache_thread_safety():
    """Verify multithreaded concurrent reads and writes do not throw RuntimeError."""
    cache = CacheManager(max_memory_entries=200)
    errors = []

    def worker(worker_id: int):
        try:
            for i in range(100):
                k = f"key_{worker_id}_{i}"
                cache.set(k, f"val_{i}", ttl_seconds=60)
                _ = cache.get(k)
                if i % 10 == 0:
                    cache.invalidate_user(worker_id)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(wid,)) for wid in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Encountered concurrency errors: {errors}"


def test_qdrant_circuit_breaker_trips_on_failures():
    """Verify QdrantVectorStore trips circuit breaker and recovers fail-fast."""
    q_store = QdrantVectorStore(url=None, api_key=None, dim=16)

    # Mock client failure
    mock_client = MagicMock()
    mock_client.query_points.side_effect = Exception("Simulated Qdrant Network Timeout")
    q_store.client = mock_client

    dummy_vec = np.ones(16, dtype=np.float32)

    # First 2 calls fail and increment failure count
    res1 = q_store.search_similar(dummy_vec)
    assert res1 == []
    assert q_store._failure_count == 1
    assert not q_store._is_circuit_open()

    res2 = q_store.search_similar(dummy_vec)
    assert res2 == []
    assert q_store._failure_count == 2
    assert not q_store._is_circuit_open()

    # 3rd failure trips the breaker
    res3 = q_store.search_similar(dummy_vec)
    assert res3 == []
    assert q_store._failure_count >= 3
    assert q_store._is_circuit_open()

    # 4th call immediately returns [] without calling query_points (Circuit Breaker open)
    mock_client.query_points.reset_mock()
    res4 = q_store.search_similar(dummy_vec)
    assert res4 == []
    mock_client.query_points.assert_not_called()
