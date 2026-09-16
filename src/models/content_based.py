"""Content-Based Filtering Recommender.
Recommends products similar to what the user has previously viewed, purchased, or liked.
Supports Qdrant Vector DB for cloud-native ANN search with seamless local fallback.
"""
from collections import defaultdict
from typing import List, Dict, Any, Optional
import numpy as np
from src.database.models import Product, Interaction
from src.models.base import BaseRecommender
from src.features.text_embedder import ItemEmbedder
from src.features.user_profiler import UserProfiler
from src.services.qdrant_service import QdrantVectorStore

class ContentBasedRecommender(BaseRecommender):
    """Recommends products matching user textual and metadata profile."""

    def __init__(
        self,
        embedder: Optional[ItemEmbedder] = None,
        qdrant_store: Optional[QdrantVectorStore] = None
    ):
        super().__init__(name="ContentBased")
        self.embedder = embedder or ItemEmbedder(embedding_dim=64)
        self.profiler = UserProfiler(self.embedder)
        self.qdrant_store = qdrant_store
        self.user_interactions: Dict[int, List[Interaction]] = defaultdict(list)
        self.product_dict: Dict[int, Product] = {}

    def fit(self, products: List[Product], interactions: List[Interaction]) -> None:
        """Fit item embedder on catalog and index user interactions."""
        self.product_dict = {p.id: p for p in products}

        # Embed all products
        embeddings = self.embedder.fit_transform(products)

        # Index vectors into Qdrant if vector store is available
        if self.qdrant_store:
            try:
                pids = [p.id for p in products]
                metas = [
                    {
                        "title": p.title,
                        "category": p.category,
                        "price": p.price,
                        "rating_avg": p.rating_avg
                    }
                    for p in products
                ]
                self.qdrant_store.upsert_products(pids, embeddings, metas)
            except Exception as e:
                print(f"⚠️ Notice: Qdrant indexing skipped: {e}")

        # Index interactions by user
        self.user_interactions.clear()
        for inter in interactions:
            self.user_interactions[inter.user_id].append(inter)

        self.is_fitted = True

    def add_interaction(self, interaction: Interaction) -> None:
        """Register a new user interaction in memory immediately for real-time recommendation."""
        self.user_interactions[interaction.user_id].append(interaction)

    def index_new_product(self, product: Product) -> None:
        """Dynamically embed and index a new product in real time (Item Cold-Start support)."""
        self.product_dict[product.id] = product
        if hasattr(self.embedder, "transform_single"):
            vec = self.embedder.transform_single(product)
            if self.qdrant_store and self.qdrant_store.client:
                try:
                    meta = {
                        "title": product.title,
                        "category": product.category,
                        "price": product.price,
                        "rating_avg": product.rating_avg
                    }
                    self.qdrant_store.upsert_products([product.id], np.array([vec]), [meta])
                except Exception:
                    pass


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

        # Find the user's most interacted product/category for an intuitive explanation
        last_interacted_pid = interactions[-1].product_id
        ref_prod = self.product_dict.get(last_interacted_pid)
        reason_ref = ref_prod.category if ref_prod else "sở thích gần đây"

        # Try Qdrant ANN search first if available
        if self.qdrant_store and self.qdrant_store.client:
            try:
                qdrant_hits = self.qdrant_store.search_similar(
                    query_vector=user_vec,
                    top_k=top_k,
                    exclude_ids=exclude_ids
                )
                if qdrant_hits:
                    return [
                        {
                            "product_id": pid,
                            "score": round(score, 4),
                            "model": f"{self.name} (Qdrant ANN)",
                            "reason": f"Phù hợp với sở thích về '{reason_ref}' bạn từng xem"
                        }
                        for pid, score, _ in qdrant_hits
                    ]
            except Exception:
                pass # Fallback to local cosine matching

        # Local NumPy/Cosine matching fallback
        matched = self.profiler.match_user_to_items(user_vec, exclude_item_ids=exclude_ids, top_k=top_k)
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

        target_prod = self.product_dict.get(product_id)
        target_name = target_prod.title[:30] if target_prod else "sản phẩm này"

        # Try Qdrant search
        target_vec = self.embedder.get_embedding(product_id)
        if self.qdrant_store and target_vec is not None:
            try:
                hits = self.qdrant_store.search_similar(
                    query_vector=target_vec,
                    top_k=top_k,
                    exclude_ids=[product_id]
                )
                if hits:
                    return [
                        {
                            "product_id": pid,
                            "score": round(score, 4),
                            "model": f"{self.name} (Qdrant)",
                            "reason": f"Sản phẩm có tính năng & phân khúc tương đồng với '{target_name}'"
                        }
                        for pid, score, _ in hits
                    ]
            except Exception:
                pass

        # Local similarity fallback
        sim_items = self.embedder.compute_similarity(product_id, top_k=top_k)
        results = []
        for pid, score in sim_items:
            results.append({
                "product_id": pid,
                "score": round(score, 4),
                "model": self.name,
                "reason": f"Sản phẩm có tính năng & phân khúc tương đồng với '{target_name}'"
            })
        return results
