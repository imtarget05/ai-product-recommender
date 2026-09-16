"""Popularity-Based Baseline Recommender.
Serves as the benchmark model and primary cold-start fallback strategy.
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

    def fit(self, products: List[Product], interactions: List[Interaction]) -> None:
        """Calculate weighted popularity score for each item."""
        self.product_dict = {p.id: p for p in products}

        # Initialize scores with base product rating
        scores = defaultdict(float)
        for p in products:
            # Baseline score from average rating and rating count
            scores[p.id] = (p.rating_avg or 3.5) * 0.1 + min((p.rating_count or 0) * 0.01, 1.0)

        # Accumulate scores from real interaction events
        for inter in interactions:
            weight = inter.weight or settings.EVENT_WEIGHTS.get(inter.event_type, 1.0)
            scores[inter.product_id] += float(weight)

        # Normalize scores to [0.0, 1.0]
        max_score = max(scores.values()) if scores else 1.0
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

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_interacted: bool = True
    ) -> List[Dict[str, Any]]:
        """Return top-K most popular products."""
        if not self.is_fitted:
            return []

        results = []
        for pid in self.ranked_products[:top_k]:
            results.append({
                "product_id": pid,
                "score": self.product_scores.get(pid, 0.5),
                "model": self.name,
                "reason": "Sản phẩm xu hướng bán chạy nhất thị trường"
            })
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
