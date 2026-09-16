"""Hybrid Recommendation System.
Blends Collaborative Filtering, Content-Based Filtering, and Popularity Baseline
with candidate-level score normalization, intelligent Cold-Start handling,
and real-time closed-loop updates.
"""
from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np
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
        cold_start_threshold: int = 3,
        qdrant_store: Optional[Any] = None
    ):
        super().__init__(name="HybridRecommender")
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.cold_start_threshold = cold_start_threshold
        self.qdrant_store = qdrant_store

        self.popularity_model = PopularityRecommender()
        self.content_model = ContentBasedRecommender(qdrant_store=qdrant_store)
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

    def on_new_interaction(self, interaction: Interaction) -> None:
        """Propagate real-time interaction to all sub-models in memory."""
        self.user_interaction_counts[interaction.user_id] += 1
        self.content_model.add_interaction(interaction)
        self.cf_model.add_interaction(interaction.user_id, interaction.product_id)
        weight = float(interaction.weight or 1.0)
        self.popularity_model.add_interaction(interaction.user_id, interaction.product_id, weight=weight)

    def index_new_product(self, product: Product) -> None:
        """Index a newly published product across all sub-models (Item Cold-Start)."""
        self.product_dict[product.id] = product
        self.content_model.index_new_product(product)
        if product.id not in self.popularity_model.product_scores:
            self.popularity_model.product_scores[product.id] = 0.2
            self.popularity_model.ranked_products.append(product.id)

    @staticmethod
    def _normalize_candidate_scores(candidates: List[Dict[str, Any]]) -> Dict[int, float]:
        """Normalize candidate scores to [0.0, 1.0] using Min-Max scaling."""
        if not candidates:
            return {}
        scores = [float(c["score"]) for c in candidates]
        min_s, max_s = min(scores), max(scores)
        if max_s > min_s:
            return {c["product_id"]: (float(c["score"]) - min_s) / (max_s - min_s) for c in candidates}
        return {c["product_id"]: 1.0 for c in candidates}

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
            results = self.popularity_model.recommend(user_id, top_k=top_k, exclude_interacted=exclude_interacted)
            for item in results:
                item["model"] = "Popularity (Cold-start Fallback)"
                item["reason"] = "Sản phẩm xu hướng được cộng đồng mua sắm nhiều nhất hôm nay"
            return results

        # -------------------------------------------------------------
        # Scenario 2: Light user (1-2 interactions: early warm cold-start)
        # Prioritize Content-Based to immediately reflect recent clicks
        # -------------------------------------------------------------
        if inter_count < self.cold_start_threshold:
            cb_recs = self.content_model.recommend(user_id, top_k=top_k, exclude_interacted=exclude_interacted)
            if len(cb_recs) < top_k:
                # Backfill with popularity
                pop_recs = self.popularity_model.recommend(user_id, top_k=top_k, exclude_interacted=exclude_interacted)
                existing_pids = set(r["product_id"] for r in cb_recs)
                for pop in pop_recs:
                    if pop["product_id"] not in existing_pids:
                        pop["model"] = "Popularity Fallback"
                        cb_recs.append(pop)
                        if len(cb_recs) >= top_k:
                            break
            return cb_recs[:top_k]

        # -------------------------------------------------------------
        # Scenario 3: Established User (Full Hybrid Ensemble with Normalization)
        # -------------------------------------------------------------
        pool_size = max(top_k * 2, 20)
        cf_recs = self.cf_model.recommend(user_id, top_k=pool_size, exclude_interacted=exclude_interacted)
        cb_recs = self.content_model.recommend(user_id, top_k=pool_size, exclude_interacted=exclude_interacted)

        # Apply candidate-level Min-Max normalization to prevent scale dominance
        cf_scaled = self._normalize_candidate_scores(cf_recs)
        cb_scaled = self._normalize_candidate_scores(cb_recs)

        merged_scores = defaultdict(float)
        reasons = {}
        model_contributions = defaultdict(list)

        # Ingest CF candidates
        for item in cf_recs:
            pid = item["product_id"]
            scaled_s = cf_scaled.get(pid, 0.5)
            merged_scores[pid] += self.cf_weight * scaled_s
            model_contributions[pid].append("CF")
            reasons[pid] = item["reason"]

        # Ingest Content-Based candidates
        for item in cb_recs:
            pid = item["product_id"]
            scaled_s = cb_scaled.get(pid, 0.5)
            merged_scores[pid] += self.cb_weight * scaled_s
            model_contributions[pid].append("Content-Based")
            if pid in reasons:
                reasons[pid] = "Được nhiều người cùng sở thích lựa chọn & trùng khớp đặc tính bạn quan tâm"
            else:
                reasons[pid] = item["reason"]

        # Sort merged candidate pool descending
        sorted_pids = sorted(merged_scores.keys(), key=lambda x: merged_scores[x], reverse=True)

        results = []
        for pid in sorted_pids[:top_k]:
            models_used = " + ".join(model_contributions[pid])
            raw_composite_score = float(merged_scores[pid])
            results.append({
                "product_id": pid,
                "score": round(min(raw_composite_score, 1.0), 4),
                "model": f"Hybrid ({models_used})",
                "reason": reasons.get(pid, "Gợi ý cá nhân hóa dựa trên tổ hợp hành vi của bạn")
            })

        # Backfill if not enough candidates
        if len(results) < top_k:
            pop_recs = self.popularity_model.recommend(user_id, top_k=top_k, exclude_interacted=exclude_interacted)
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

