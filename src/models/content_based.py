"""Content-Based Filtering Recommender.
Recommends products similar to what the user has previously viewed, purchased, or liked.
"""
from collections import defaultdict
from typing import List, Dict, Any, Optional
from src.database.models import Product, Interaction
from src.models.base import BaseRecommender
from src.features.text_embedder import ItemEmbedder
from src.features.user_profiler import UserProfiler

class ContentBasedRecommender(BaseRecommender):
    """Recommends products matching user textual and metadata profile."""

    def __init__(self, embedder: Optional[ItemEmbedder] = None):
        super().__init__(name="ContentBased")
        self.embedder = embedder or ItemEmbedder(embedding_dim=64)
        self.profiler = UserProfiler(self.embedder)
        self.user_interactions: Dict[int, List[Interaction]] = defaultdict(list)
        self.product_dict: Dict[int, Product] = {}

    def fit(self, products: List[Product], interactions: List[Interaction]) -> None:
        """Fit item embedder on catalog and index user interactions."""
        self.product_dict = {p.id: p for p in products}

        # Embed all products
        self.embedder.fit_transform(products)

        # Index interactions by user
        self.user_interactions.clear()
        for inter in interactions:
            self.user_interactions[inter.user_id].append(inter)

        self.is_fitted = True

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_interacted: bool = True
    ) -> List[Dict[str, Any]]:
        """Recommend products matching the user's implicit profile."""
        if not self.is_fitted:
            return []

        interactions = self.user_interactions.get(user_id, [])
        if not interactions:
            return [] # Cold-start fallback handled at Hybrid layer

        user_vec = self.profiler.build_user_vector(interactions)
        if user_vec is None:
            return []

        exclude_ids = [inter.product_id for inter in interactions] if exclude_interacted else []
        matched = self.profiler.match_user_to_items(user_vec, exclude_item_ids=exclude_ids, top_k=top_k)

        # Find the user's most interacted product/category for a nice explanation
        last_interacted_pid = interactions[-1].product_id
        ref_prod = self.product_dict.get(last_interacted_pid)
        reason_ref = ref_prod.category if ref_prod else "sở thích gần đây"

        results = []
        for pid, score in matched:
            results.append({
                "product_id": pid,
                "score": round(score, 4),
                "model": self.name,
                "reason": f"Phù hợp với sở thích về '{reason_ref}' bạn từng xem"
            })
        return results

    def similar_items(
        self,
        product_id: int,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Return similar products given an item id."""
        if not self.is_fitted:
            return []

        sim_items = self.embedder.compute_similarity(product_id, top_k=top_k)
        target_prod = self.product_dict.get(product_id)
        target_name = target_prod.title[:30] if target_prod else "sản phẩm này"

        results = []
        for pid, score in sim_items:
            results.append({
                "product_id": pid,
                "score": round(score, 4),
                "model": self.name,
                "reason": f"Sản phẩm có tính năng & phân khúc tương đồng với '{target_name}'"
            })
        return results
