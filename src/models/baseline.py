"""Popularity-Based Baseline Recommender.
Serves as the benchmark model and primary cold-start fallback strategy.
Enhanced with Bayesian average rating smoothing and real-time interaction suppression.
"""
from collections import defaultdict
from typing import List, Dict, Any, Optional
from src.database.models import Product, Interaction
from src.models.base import BaseRecommender
from src.config import settings

class PopularityRecommender(BaseRecommender):
    """Recommends products based on interaction frequency and overall sales popularity."""

    def __init__(self):
        super().__init__(name="PopularityBaseline")
        self.product_scores: Dict[int, float] = {}
        self.ranked_products: List[int] = []
        self.products_by_category: Dict[str, List[int]] = defaultdict(list)
        self.product_dict: Dict[int, Product] = {}
        self.user_interacted_items: Dict[int, set] = defaultdict(set)

    def fit(self, products: List[Product], interactions: List[Interaction]) -> None:
        """Calculate weighted popularity score for each item using Bayesian rating smoothing."""
        self.product_dict = {p.id: p for p in products}

        # Bayesian rating smoothing prior
        prior_weight = 5.0
        prior_mean = 4.0

        scores = defaultdict(float)
        for p in products:
            v = float(p.rating_count or 0)
            r = float(p.rating_avg or prior_mean)
            # Weighted Bayesian Rating
            bayesian_rating = (v * r + prior_weight * prior_mean) / (v + prior_weight)
            scores[p.id] = bayesian_rating * 0.2

        # Index user interactions and accumulate real event weights
        self.user_interacted_items.clear()
        for inter in interactions:
            weight = inter.weight or settings.EVENT_WEIGHTS.get(inter.event_type, 1.0)
            scores[inter.product_id] += float(weight)
            self.user_interacted_items[inter.user_id].add(inter.product_id)

        # Normalize scores to [0.0, 1.0] safely
        max_score = max(scores.values()) if scores and max(scores.values()) > 0 else 1.0
        self.product_scores = {pid: round(s / max_score, 4) for pid, s in scores.items()}

        # Sort ranked product IDs descending
        self.ranked_products = sorted(
            self.product_scores.keys(),
            key=lambda pid: self.product_scores[pid],
            reverse=True
        )

        # Group by category
        self.products_by_category = defaultdict(list)
        for pid in self.ranked_products:
            p = self.product_dict.get(pid)
            if p:
                self.products_by_category[p.category].append(pid)

        self.is_fitted = True

    def add_interaction(self, user_id: int, product_id: int, weight: float = 1.0) -> None:
        """Register interaction in real time and slightly increment product popularity."""
        self.user_interacted_items[user_id].add(product_id)
        if product_id in self.product_scores:
            # Gentle increment to preserve score scale
            self.product_scores[product_id] = min(1.0, self.product_scores[product_id] + 0.001 * weight)

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_interacted: bool = True
    ) -> List[Dict[str, Any]]:
        """Return top-K most popular products, prioritizing un-interacted items with graceful backfill."""
        if not self.is_fitted:
            return []

        interacted = self.user_interacted_items.get(user_id, set()) if exclude_interacted else set()
        results = []

        # First pass: items user hasn't interacted with
        for pid in self.ranked_products:
            if pid not in interacted:
                results.append({
                    "product_id": pid,
                    "score": self.product_scores.get(pid, 0.5),
                    "model": self.name,
                    "reason": "Sản phẩm xu hướng bán chạy nhất thị trường"
                })
                if len(results) >= top_k:
                    return results

        # Second pass (fallback for small catalogs): fill remaining slots from popular items
        if len(results) < top_k and exclude_interacted:
            for pid in self.ranked_products:
                if pid in interacted:
                    results.append({
                        "product_id": pid,
                        "score": self.product_scores.get(pid, 0.5),
                        "model": self.name,
                        "reason": "Sản phẩm xu hướng bán chạy nhất thị trường"
                    })
                    if len(results) >= top_k:
                        break

        return results

    def recommend_by_category(self, category: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top popular products in a specific category."""
        pids = self.products_by_category.get(category, [])[:top_k]
        return [
            {
                "product_id": pid,
                "score": self.product_scores.get(pid, 0.5),
                "model": self.name,
                "reason": f"Sản phẩm nổi bật trong danh mục {category}"
            }
            for pid in pids
        ]

