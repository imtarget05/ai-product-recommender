"""Hybrid Recommendation System.
Blends Collaborative Filtering, Content-Based Filtering, and Popularity Baseline
with intelligent Cold-Start handling and score normalization.
"""
from typing import List, Dict, Any, Optional
from collections import defaultdict
from src.database.models import Product, Interaction
from src.models.base import BaseRecommender
from src.models.baseline import PopularityRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender
from src.config import settings

class HybridRecommender(BaseRecommender):
    """Ensemble Hybrid Recommender combining CF, CB, and Popularity."""

    def __init__(
        self,
        cf_weight: float = 0.6,
        cb_weight: float = 0.4,
        cold_start_threshold: int = 3
    ):
        super().__init__(name="HybridRecommender")
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.cold_start_threshold = cold_start_threshold

        self.popularity_model = PopularityRecommender()
        self.content_model = ContentBasedRecommender()
        self.cf_model = CollaborativeRecommender(n_factors=settings.CF_LATENT_FACTORS)

        self.user_interaction_counts: Dict[int, int] = defaultdict(int)
        self.product_dict: Dict[int, Product] = {}

    def fit(self, products: List[Product], interactions: List[Interaction]) -> None:
        """Fit all component models simultaneously."""
        self.product_dict = {p.id: p for p in products}

        # Count interactions per user to detect cold-start users
        self.user_interaction_counts.clear()
        for inter in interactions:
            self.user_interaction_counts[inter.user_id] += 1

        print("Fitting Popularity Baseline...")
        self.popularity_model.fit(products, interactions)

        print("Fitting Content-Based Model...")
        self.content_model.fit(products, interactions)

        print("Fitting Collaborative Filtering Model...")
        self.cf_model.fit(products, interactions)

        self.is_fitted = True
        print("✅ Hybrid Recommender successfully trained all sub-models.")

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_interacted: bool = True
    ) -> List[Dict[str, Any]]:
        """Generate hybrid recommendations with automatic cold-start routing."""
        if not self.is_fitted:
            return []

        inter_count = self.user_interaction_counts.get(user_id, 0)

        # -------------------------------------------------------------
        # Scenario 1: Brand New User (Zero interaction cold-start)
        # -------------------------------------------------------------
        if inter_count == 0:
            results = self.popularity_model.recommend(user_id, top_k=top_k)
            for item in results:
                item["model"] = "Popularity (Cold-start Fallback)"
                item["reason"] = "Sản phẩm xu hướng được cộng đồng mua sắm nhiều nhất hôm nay"
            return results

        # -------------------------------------------------------------
        # Scenario 2: Light user (1-2 interactions: early warm cold-start)
        # CF factors are not yet stable; prioritize Content-Based
        # -------------------------------------------------------------
        if inter_count < self.cold_start_threshold:
            cb_recs = self.content_model.recommend(user_id, top_k=top_k, exclude_interacted=exclude_interacted)
            if len(cb_recs) < top_k:
                # Backfill with popularity
                pop_recs = self.popularity_model.recommend(user_id, top_k=top_k)
                existing_pids = set(r["product_id"] for r in cb_recs)
                for pop in pop_recs:
                    if pop["product_id"] not in existing_pids:
                        pop["model"] = "Popularity Fallback"
                        cb_recs.append(pop)
                        if len(cb_recs) >= top_k:
                            break
            return cb_recs[:top_k]

        # -------------------------------------------------------------
        # Scenario 3: Established User (Full Hybrid Ensemble)
        # -------------------------------------------------------------
        # Retrieve candidate pools (retrieve 2x top_k for high-quality ranking)
        pool_size = max(top_k * 2, 20)
        cf_recs = self.cf_model.recommend(user_id, top_k=pool_size, exclude_interacted=exclude_interacted)
        cb_recs = self.content_model.recommend(user_id, top_k=pool_size, exclude_interacted=exclude_interacted)

        merged_scores = defaultdict(float)
        reasons = {}
        model_contributions = defaultdict(list)

        # Ingest CF candidates
        for item in cf_recs:
            pid = item["product_id"]
            merged_scores[pid] += self.cf_weight * item["score"]
            model_contributions[pid].append("CF")
            reasons[pid] = item["reason"]

        # Ingest Content-Based candidates
        for item in cb_recs:
            pid = item["product_id"]
            merged_scores[pid] += self.cb_weight * item["score"]
            model_contributions[pid].append("Content-Based")
            # If item was also recommended by CF, provide a combined rationale!
            if pid in reasons:
                reasons[pid] = "Được nhiều người cùng sở thích lựa chọn & trùng khớp đặc tính bạn quan tâm"
            else:
                reasons[pid] = item["reason"]

        # Sort merged candidate pool descending
        sorted_pids = sorted(merged_scores.keys(), key=lambda x: merged_scores[x], reverse=True)

        results = []
        for pid in sorted_pids[:top_k]:
            models_used = " + ".join(model_contributions[pid])
            results.append({
                "product_id": pid,
                "score": round(float(merged_scores[pid]), 4),
                "model": f"Hybrid ({models_used})",
                "reason": reasons.get(pid, "Gợi ý cá nhân hóa dựa trên tổ hợp hành vi của bạn")
            })

        # Backfill if not enough candidates
        if len(results) < top_k:
            pop_recs = self.popularity_model.recommend(user_id, top_k=top_k)
            existing_pids = set(r["product_id"] for r in results)
            for pop in pop_recs:
                if pop["product_id"] not in existing_pids:
                    pop["model"] = "Popularity Backfill"
                    results.append(pop)
                    if len(results) >= top_k:
                        break

        return results[:top_k]

    def similar_items(self, product_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return similar items combining content similarity and collaborative co-views."""
        cb_sim = self.content_model.similar_items(product_id, top_k=top_k)
        if cb_sim:
            return cb_sim
        return self.cf_model.similar_items(product_id, top_k=top_k)
