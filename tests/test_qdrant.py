"""Unit tests for Qdrant Vector Store Service."""
import pytest
import numpy as np
from src.services.qdrant_service import QdrantVectorStore

def test_qdrant_in_memory_lifecycle():
    # Use in-memory Qdrant instance
    store = QdrantVectorStore(url=None, api_key=None, collection_name="test_products", dim=4)
    assert store.client is not None

    product_ids = [101, 102, 103]
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0]  # Very similar to 101
    ])
    metas = [
        {"title": "Smartphone A", "category": "Tech"},
        {"title": "Jacket B", "category": "Fashion"},
        {"title": "Smartphone C", "category": "Tech"}
    ]

    success = store.upsert_products(product_ids, embeddings, metas)
    assert success is True

    # Search similar to item 101
    query = np.array([1.0, 0.0, 0.0, 0.0])
    hits = store.search_similar(query, top_k=2, exclude_ids=[101])
    assert len(hits) > 0
    # Item 103 should be top match
    top_pid, top_score, payload = hits[0]
    assert top_pid == 103
    assert top_score > 0.8
    assert payload["category"] == "Tech"
