"""Unit tests for Ranking & Diversity Reranker."""
import pytest
from datetime import datetime, timedelta
from src.database.models import Product
from src.ranking.reranker import ProductReranker

def test_reranker_sorting_and_suppression():
    product_dict = {
        1: Product(id=1, title="Macbook M3", category="Laptop", rating_avg=4.9, created_at=datetime.utcnow() - timedelta(days=2)),
        2: Product(id=2, title="Bàn phím cơ", category="Phụ kiện", rating_avg=4.5, created_at=datetime.utcnow() - timedelta(days=10)),
        3: Product(id=3, title="Chuột Gaming", category="Phụ kiện", rating_avg=4.0, created_at=datetime.utcnow() - timedelta(days=30)),
    }

    candidates = [
        {"product_id": 1, "score": 0.9, "model": "CF", "reason": "Reason 1"},
        {"product_id": 2, "score": 0.8, "model": "CB", "reason": "Reason 2"},
        {"product_id": 3, "score": 0.7, "model": "CF", "reason": "Reason 3"},
    ]

    reranker = ProductReranker()

    # Suppose user already bought product 1 recently
    results = reranker.rerank(
        candidates=candidates,
        product_dict=product_dict,
        recently_purchased_ids=[1],
        top_k=2
    )

    # Product 1 should be suppressed
    pids = [r["product_id"] for r in results]
    assert 1 not in pids
    assert 2 in pids
    assert 3 in pids
    assert results[0]["final_score"] >= results[1]["final_score"]
