"""Real-container integration tests for Qdrant + Redis (beyond mocks).

Uses testcontainers-python (QdrantContainer + RedisContainer) against
ephemeral Docker containers. No MagicMock here — real ANN search and
real Redis round-trips.

Graceful skip contract:
- If ``testcontainers`` is not installed -> skip.
- If Docker daemon is unreachable -> skip (CI without Docker stays green).
- If a container image cannot be pulled/started -> skip that test.

Existing ``MagicMock`` unit tests (tests/test_regressions.py,
tests/test_qdrant.py) are untouched and remain the fast default suite.
"""

import shutil
import uuid

import numpy as np
import pytest

try:
    from testcontainers.qdrant import QdrantContainer
    from testcontainers.redis import RedisContainer

    _TESTCONTAINERS_IMPORTED = True
except Exception:  # pragma: no cover - import guard for envs without testcontainers
    QdrantContainer = None  # type: ignore
    RedisContainer = None  # type: ignore
    _TESTCONTAINERS_IMPORTED = False


def _docker_available() -> bool:
    """Best-effort Docker daemon check (binary + daemon ping)."""
    if shutil.which("docker") is None:
        return False
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


_DOCKER_AVAILABLE = _docker_available()
_CONTAINER_TESTS_READY = _TESTCONTAINERS_IMPORTED and _DOCKER_AVAILABLE

pytestmark = pytest.mark.skipif(
    not _CONTAINER_TESTS_READY,
    reason="Docker unavailable or testcontainers not installed; skipping real-container tests.",
)


def _require_containers():
    if not _CONTAINER_TESTS_READY:
        pytest.skip("Docker unavailable or testcontainers not installed.")


def _qdrant_url(container) -> str:
    """Resolve Qdrant HTTP URL across testcontainers versions."""
    for attr in ("get_api_url", "get_url", "get_connection_url"):
        getter = getattr(container, attr, None)
        if callable(getter):
            try:
                return str(getter())
            except Exception:
                continue
    host = container.get_container_host_ip()
    port = container.get_exposed_port(6333)
    return f"http://{host}:{port}"


def _redis_client(container):
    """Build a redis client for a started RedisContainer."""
    import redis

    host = container.get_container_host_ip()
    port = int(container.get_exposed_port(6379))
    # Newer testcontainers expose get_connection_url()/get_client(); prefer explicit host/port.
    try:
        get_client = getattr(container, "get_client", None)
        if callable(get_client):
            return get_client()
    except Exception:
        pass
    return redis.Redis(host=host, port=port, decode_responses=True)


@pytest.fixture()
def qdrant_url():
    _require_containers()
    try:
        with QdrantContainer() as container:
            url = _qdrant_url(container)
            yield url
    except Exception as exc:
        pytest.skip(f"QdrantContainer could not start: {exc}")


@pytest.fixture()
def redis_client():
    _require_containers()
    try:
        with RedisContainer() as container:
            client = _redis_client(container)
            client.ping()
            yield client
            try:
                client.close()
            except Exception:
                pass
    except Exception as exc:
        # pytest.skip raises Skipped; don't swallow it as a startup failure.
        if isinstance(exc, pytest.skip.Exception):
            raise
        pytest.skip(f"RedisContainer could not start: {exc}")


def test_qdrant_container_upsert_and_search(qdrant_url):
    """Real Qdrant: upsert 3 vectors, ANN search returns the nearest neighbour."""
    from src.services.qdrant_service import QdrantVectorStore

    collection = f"test_products_{uuid.uuid4().hex[:8]}"
    store = QdrantVectorStore(url=qdrant_url, api_key=None, collection_name=collection, dim=4)
    assert store.is_available()

    product_ids = [101, 102, 103]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],  # very similar to 101
        ]
    )
    metas = [
        {"title": "Smartphone A", "category": "Tech"},
        {"title": "Jacket B", "category": "Fashion"},
        {"title": "Smartphone C", "category": "Tech"},
    ]
    assert store.upsert_products(product_ids, embeddings, metas) is True

    hits = store.search_similar(np.array([1.0, 0.0, 0.0, 0.0]), top_k=2, exclude_ids=[101])
    assert len(hits) > 0
    top_pid, top_score, payload = hits[0]
    assert top_pid == 103
    assert top_score > 0.8
    assert payload["category"] == "Tech"


def test_qdrant_container_category_filter(qdrant_url):
    """Real Qdrant: payload filter restricts ANN results to a category."""
    from src.services.qdrant_service import QdrantVectorStore

    collection = f"test_filter_{uuid.uuid4().hex[:8]}"
    store = QdrantVectorStore(url=qdrant_url, api_key=None, collection_name=collection, dim=4)
    assert store.is_available()

    product_ids = [201, 202]
    embeddings = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    metas = [
        {"title": "Phone", "category": "Tech"},
        {"title": "Coat", "category": "Fashion"},
    ]
    assert store.upsert_products(product_ids, embeddings, metas) is True

    hits = store.search_similar(
        np.array([1.0, 0.0, 0.0, 0.0]), top_k=5, filter_category="Fashion"
    )
    assert [pid for pid, _, _ in hits] == [202]


def test_redis_container_recommendation_cache_roundtrip(redis_client, monkeypatch):
    """Real Redis: CacheManager versioned write -> read -> invalidate."""
    from src.api.cache import CacheManager

    monkeypatch.setattr(CacheManager, "_init_redis", lambda self: None)
    cache = CacheManager(max_memory_entries=10)
    cache.redis_client = redis_client

    cache.set_recommendation_cache(42, "hybrid", 5, {"items": [1, 2, 3]})
    assert cache.get_recommendation_cache(42, "hybrid", 5) == {"items": [1, 2, 3]}
    cache.invalidate_user(42)
    assert cache.get_recommendation_cache(42, "hybrid", 5) is None
