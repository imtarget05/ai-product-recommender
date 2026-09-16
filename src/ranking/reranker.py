"""Ranking Layer (Giai đoạn 4 - Candidate Reranking).
Fine-ranks candidate items using multi-objective features:
relevance score, product quality/rating, freshness, category diversity, and purchase suppression.
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict
from src.database.models import Product

class ProductReranker:
    """Reranks candidate generation output to balance relevance, quality, and diversity."""

    def __init__(
        self,
        weight_model: float = 0.70,
        weight_rating: float = 0.20,
        weight_freshness: float = 0.10,
        max_per_category: int = 3
    ):
        self.weight_model = weight_model
        self.weight_rating = weight_rating
        self.weight_freshness = weight_freshness
        self.max_per_category = max_per_category

    def rerank(
        self,
        candidates: List[Dict[str, Any]],
        product_dict: Dict[int, Product],
        recently_purchased_ids: Optional[List[int]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Rerank candidates based on multi-feature scoring and diversity constraints."""
        if not candidates:
            return []

        purchased_set = set(recently_purchased_ids or [])
        scored_candidates = []
        now = datetime.utcnow()

        for cand in candidates:
            pid = cand["product_id"]

            # Filter out recently purchased items (anti-fatigue)
            if pid in purchased_set:
                continue

            product = product_dict.get(pid)
            if not product:
                continue

            # Feature 1: Model candidate score (normalized)
            model_score = min(max(cand.get("score", 0.5), 0.0), 1.0)

            # Feature 2: Rating feature (0.0 to 1.0)
            rating_score = (product.rating_avg or 4.0) / 5.0

            # Feature 3: Freshness feature (newer items receive slight boost)
            freshness_score = 0.5
            if product.created_at:
                days_old = (now - product.created_at).days
                freshness_score = max(0.0, 1.0 - (days_old / 180.0))

            # Composite final ranking score
            final_score = (
                self.weight_model * model_score +
                self.weight_rating * rating_score +
                self.weight_freshness * freshness_score
            )

            scored_candidates.append({
                "product_id": pid,
                "title": product.title,
                "category": product.category,
                "price": product.price,
                "rating_avg": product.rating_avg,
                "image_url": product.image_url,
                "base_score": round(model_score, 4),
                "final_score": round(final_score, 4),
                "model": cand.get("model", "Ranking"),
                "reason": cand.get("reason", "Phù hợp với hồ sơ mua sắm của bạn")
            })

        # Sort descending by final score
        scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)

        # Apply category diversity constraint (avoid flooding with 1 category)
        diverse_results = []
        category_counts = defaultdict(int)
        overflow_pool = []

        for item in scored_candidates:
            cat = item["category"]
            if category_counts[cat] < self.max_per_category:
                diverse_results.append(item)
                category_counts[cat] += 1
            else:
                overflow_pool.append(item)

            if len(diverse_results) >= top_k:
                break

        # If diversity constraint left us with fewer than top_k items, fill from overflow
        if len(diverse_results) < top_k:
            for item in overflow_pool:
                diverse_results.append(item)
                if len(diverse_results) >= top_k:
                    break

        return diverse_results[:top_k]
