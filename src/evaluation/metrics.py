"""Recommendation System Evaluation Metrics (Phase 6).
Computes standard offline ranking & retrieval metrics:
Precision@K, Recall@K, NDCG@K, MAP@K, HitRate@K, and Catalog Coverage.
"""
import math
from typing import List, Set, Dict, Any
import numpy as np

def precision_at_k(actual: Set[int], predicted: List[int], k: int = 10) -> float:
    """Precision@K = (# of recommended items in top-K that are relevant) / K."""
    if not predicted or k <= 0:
        return 0.0
    pred_k = predicted[:k]
    hits = len(set(pred_k) & actual)
    return hits / float(k)


def recall_at_k(actual: Set[int], predicted: List[int], k: int = 10) -> float:
    """Recall@K = (# of recommended items in top-K that are relevant) / (# of relevant items)."""
    if not actual or not predicted or k <= 0:
        return 0.0
    pred_k = predicted[:k]
    hits = len(set(pred_k) & actual)
    return hits / float(len(actual))


def hit_rate_at_k(actual: Set[int], predicted: List[int], k: int = 10) -> float:
    """HitRate@K = 1.0 if at least one relevant item is in top-K, else 0.0."""
    if not actual or not predicted:
        return 0.0
    pred_k = predicted[:k]
    return 1.0 if len(set(pred_k) & actual) > 0 else 0.0


def average_precision_at_k(actual: Set[int], predicted: List[int], k: int = 10) -> float:
    """Average Precision @ K."""
    if not actual or not predicted or k <= 0:
        return 0.0

    pred_k = predicted[:k]
    score = 0.0
    num_hits = 0

    for i, item in enumerate(pred_k):
        if item in actual:
            num_hits += 1
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def ndcg_at_k(actual: Set[int], predicted: List[int], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance."""
    if not actual or not predicted or k <= 0:
        return 0.0

    pred_k = predicted[:k]
    dcg = 0.0
    for i, item in enumerate(pred_k):
        if item in actual:
            dcg += 1.0 / math.log2(i + 2) # i + 2 since rank is 1-indexed

    # Ideal DCG: all actual items at top ranks
    idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(actual), k)))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def graded_ndcg_at_k(actual_weights: Dict[int, float], predicted: List[int], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain with graded relevance weights.
    Graded relevance: e.g. purchase=5.0, add_to_cart=3.5, click=2.0, view=1.0.
    DCG = \sum_{i=1}^k (2^{rel_i} - 1) / log2(i + 1)
    """
    if not actual_weights or not predicted or k <= 0:
        return 0.0

    pred_k = predicted[:k]
    dcg = 0.0
    for i, item in enumerate(pred_k):
        rel = float(actual_weights.get(item, 0.0))
        if rel > 0.0:
            dcg += (math.pow(2.0, rel) - 1.0) / math.log2(i + 2)

    # Ideal ranking: sort actual weights descending
    sorted_rels = sorted(actual_weights.values(), reverse=True)[:k]
    idcg = sum((math.pow(2.0, r) - 1.0) / math.log2(idx + 2) for idx, r in enumerate(sorted_rels))
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def evaluate_recommender(
    recommender,
    test_user_items: Dict[int, Set[int]],
    k: int = 10,
    total_catalog_size: int = 0,
    reranker = None,
    product_dict = None
) -> Dict[str, float]:
    """Benchmark a recommendation model across multiple evaluation users.
    Optionally evaluates full ranking pipeline when reranker is supplied.
    """
    precisions = []
    recalls = []
    hit_rates = []
    ndcgs = []
    maps = []
    all_recommended_items = set()

    for uid, actual_items in test_user_items.items():
        if not actual_items:
            continue

        raw_recs = recommender.recommend(uid, top_k=k * 2 if reranker else k, exclude_interacted=True)
        if reranker and product_dict:
            final_recs = reranker.rerank(raw_recs, product_dict=product_dict, top_k=k)
            pred_ids = [r["product_id"] for r in final_recs]
        else:
            pred_ids = [r["product_id"] for r in raw_recs[:k]]

        all_recommended_items.update(pred_ids)

        precisions.append(precision_at_k(actual_items, pred_ids, k))
        recalls.append(recall_at_k(actual_items, pred_ids, k))
        hit_rates.append(hit_rate_at_k(actual_items, pred_ids, k))
        ndcgs.append(ndcg_at_k(actual_items, pred_ids, k))
        maps.append(average_precision_at_k(actual_items, pred_ids, k))

    coverage = (len(all_recommended_items) / total_catalog_size) if total_catalog_size > 0 else 0.0

    return {
        f"Precision@{k}": round(float(np.mean(precisions)) if precisions else 0.0, 4),
        f"Recall@{k}": round(float(np.mean(recalls)) if recalls else 0.0, 4),
        f"HitRate@{k}": round(float(np.mean(hit_rates)) if hit_rates else 0.0, 4),
        f"NDCG@{k}": round(float(np.mean(ndcgs)) if ndcgs else 0.0, 4),
        f"MAP@{k}": round(float(np.mean(maps)) if maps else 0.0, 4),
        "CatalogCoverage": round(coverage, 4),
        "EvaluatedUsers": len(precisions)
    }

