"""Deterministic regressions: no live Redis, Qdrant or LLM required."""
from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.api.cache import CacheManager
from src.database.models import Interaction, Product
from src.models.collaborative import CollaborativeRecommender
from src.models.content_based import ContentBasedRecommender
from src.services.qdrant_service import QdrantVectorStore


@pytest.fixture
def cache(monkeypatch):
    monkeypatch.setattr(CacheManager, "_init_redis", lambda self: None)
    return CacheManager(max_memory_entries=10)


def test_memory_cache_owns_nested_values(cache):
    value = {"recommendations": [{"product_id": 1}], "cached": False}
    cache.set("key", value)
    value["recommendations"][0]["product_id"] = 99
    result = cache.get("key")
    assert result["recommendations"][0]["product_id"] == 1
    result["recommendations"].clear()
    result["cached"] = True
    assert cache.get("key") == {"recommendations": [{"product_id": 1}], "cached": False}


def test_first_redis_invalidation_changes_version(cache):
    values = {}
    redis = MagicMock()
    redis.get.side_effect = values.get
    redis.setex.side_effect = lambda key, ttl, value: values.__setitem__(key, value)

    def incr(key):
        values[key] = int(values.get(key, 0)) + 1
        return values[key]

    redis.incr.side_effect = incr
    cache.redis_client = redis
    cache.set_recommendation_cache(42, "hybrid", 5, {"items": [1]})
    assert cache.get_recommendation_cache(42, "hybrid", 5) == {"items": [1]}
    cache.invalidate_user(42)
    assert cache.get_recommendation_cache(42, "hybrid", 5) is None


@pytest.mark.parametrize("mode", ["none", "unavailable", "error", "empty", "success"])
def test_content_based_qdrant_fallback(mode):
    products = [
        Product(id=1, title="gaming keyboard blue", category="Tech", price=100),
        Product(id=2, title="gaming keyboard red", category="Tech", price=110),
        Product(id=3, title="summer dress", category="Fashion", price=40),
    ]
    event = Interaction(user_id=1, product_id=1, event_type="click", weight=2,
                        timestamp=datetime.utcnow())
    model = ContentBasedRecommender()
    model.fit(products, [event])
    expected = model.recommend(1, top_k=2)
    assert expected
    store = MagicMock(spec=QdrantVectorStore)
    store.is_available.return_value = mode != "unavailable"
    store.search_similar.return_value = [(2, 0.9, {})] if mode == "success" else []
    if mode == "error":
        store.search_similar.side_effect = RuntimeError("offline")
    model.qdrant_store = None if mode == "none" else store
    result = model.recommend(1, top_k=2)
    assert all(r["product_id"] != 1 for r in result)
    if mode == "success":
        assert result[0]["product_id"] == 2
        assert "Qdrant" in result[0]["model"]
    else:
        assert result == expected
    if mode in ("none", "unavailable"):
        store.search_similar.assert_not_called()


@pytest.mark.parametrize("n_users,n_items", [(1, 3), (3, 1), (2, 3)])
def test_collaborative_rectangular_fallback(n_users, n_items, monkeypatch):
    # Force the numerical fallback as well as naturally degenerate shapes.
    monkeypatch.setattr("src.models.collaborative.TruncatedSVD.fit_transform",
                        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("SVD failure")))
    products = [Product(id=i + 1, title=f"Product {i}", category="Tech", price=10)
                for i in range(n_items)]
    events = [Interaction(user_id=u + 1, product_id=i + 1, event_type="click", weight=2)
              for u in range(n_users) for i in range(n_items)]
    model = CollaborativeRecommender()
    model.fit(products, events)
    assert model.user_factors.shape[1] == model.item_factors.shape[1]
    recs = model.recommend(1, top_k=n_items, exclude_interacted=False)
    assert len(recs) == n_items
    assert all(np.isfinite(r["score"]) for r in recs)


def test_collaborative_empty_refit_clears_old_state():
    model = CollaborativeRecommender()
    model.fit([Product(id=1, title="A", category="T", price=1)],
              [Interaction(user_id=1, product_id=1, event_type="click", weight=2)])
    model.fit([], [])
    assert model.recommend(1, exclude_interacted=False) == []
    assert model.similar_items(1) == []
