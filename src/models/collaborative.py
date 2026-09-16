"""Collaborative Filtering Recommender using Matrix Factorization & Latent Factors.
Decomposes user-item interaction matrix into latent representation spaces.
Hardened against small matrices, NaN anomalies, and negative cosine scores.
"""
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from src.database.models import Product, Interaction
from src.models.base import BaseRecommender
from src.config import settings

class CollaborativeRecommender(BaseRecommender):
    """Latent Factor Matrix Factorization Collaborative Filtering."""

    def __init__(self, n_factors: int = 16):
        super().__init__(name="CollaborativeFiltering")
        self.n_factors = n_factors
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.item_to_idx: Dict[int, int] = {}
        self.idx_to_item: Dict[int, int] = {}
        self.user_factors: Optional[np.ndarray] = None # (U, K)
        self.item_factors: Optional[np.ndarray] = None # (I, K)
        self.user_interacted_items: Dict[int, set] = defaultdict(set)
        self.svd_model: Optional[TruncatedSVD] = None

    def fit(self, products: List[Product], interactions: List[Interaction]) -> None:
        """Construct sparse interaction matrix and decompose into user/item latent factors."""
        # A refit replaces the previous snapshot, including empty datasets.
        self.is_fitted = False
        self.user_to_idx.clear()
        self.idx_to_user.clear()
        self.item_to_idx.clear()
        self.idx_to_item.clear()
        self.user_interacted_items.clear()
        self.user_factors = None
        self.item_factors = None
        self.svd_model = None
        if not products or not interactions:
            return

        # Unique user and item index mappings
        unique_users = sorted(list(set(inter.user_id for inter in interactions)))
        unique_items = sorted(list(set(p.id for p in products)))

        self.user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        self.idx_to_user = {i: uid for i, uid in enumerate(unique_users)}
        self.item_to_idx = {pid: i for i, pid in enumerate(unique_items)}
        self.idx_to_item = {i: pid for i, pid in enumerate(unique_items)}

        # Build sparse user-item interaction matrix
        rows, cols, data = [], [], []
        inter_weights = defaultdict(float)

        self.user_interacted_items.clear()
        for inter in interactions:
            if inter.user_id in self.user_to_idx and inter.product_id in self.item_to_idx:
                u_idx = self.user_to_idx[inter.user_id]
                i_idx = self.item_to_idx[inter.product_id]
                weight = float(inter.weight or settings.EVENT_WEIGHTS.get(inter.event_type, 1.0))
                inter_weights[(u_idx, i_idx)] += weight
                self.user_interacted_items[inter.user_id].add(inter.product_id)

        for (u_idx, i_idx), w in inter_weights.items():
            rows.append(u_idx)
            cols.append(i_idx)
            # Log transform implicit feedback to stabilize variance
            data.append(float(np.log1p(w)))

        if not rows:
            return

        n_users = len(unique_users)
        n_items = len(unique_items)
        sparse_mat = csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)

        # Latent factor decomposition via Truncated SVD
        # SVD requires n_components < min(n_users, n_items)
        max_possible_factors = min(n_users, n_items) - 1
        # Shared degenerate fallback: represent the interaction matrix exactly
        # in one-hot basis so dot(user_factors[u], item_factors[i]) reproduces
        # the raw affinity. This keeps factor widths consistent for both the
        # degenerate-shape branch and the SVD failure fallback below.
        def _exact_factorization(matrix: np.ndarray) -> None:
            n_users, n_items = matrix.shape
            if n_users <= n_items:
                # Each user is a basis vector; item vectors carry per-user weights.
                self.user_factors = np.eye(n_users, dtype=np.float32)
                self.item_factors = matrix.T.astype(np.float32)
            else:
                # Item vectors are the basis; user vector carries per-item weights.
                self.user_factors = matrix.astype(np.float32)
                self.item_factors = np.eye(n_items, dtype=np.float32)

        if max_possible_factors < 1:
            # Degenerate matrix (e.g. 1 user or 1 item)
            _exact_factorization(sparse_mat.toarray())
        else:
            actual_factors = min(self.n_factors, max_possible_factors)
            try:
                self.svd_model = TruncatedSVD(n_components=actual_factors, random_state=42)
                self.user_factors = self.svd_model.fit_transform(sparse_mat)
                self.item_factors = self.svd_model.components_.T # (I, K)
            except Exception:
                self.svd_model = None
                _exact_factorization(sparse_mat.toarray())

        # Sanitize NaNs
        self.user_factors = np.nan_to_num(self.user_factors, nan=0.0)
        self.item_factors = np.nan_to_num(self.item_factors, nan=0.0)

        # L2 Normalize factors safely
        u_norms = np.linalg.norm(self.user_factors, axis=1, keepdims=True)
        u_norms[u_norms == 0.0] = 1.0
        self.user_factors = self.user_factors / u_norms

        i_norms = np.linalg.norm(self.item_factors, axis=1, keepdims=True)
        i_norms[i_norms == 0.0] = 1.0
        self.item_factors = self.item_factors / i_norms

        self.is_fitted = True

    def add_interaction(self, user_id: int, product_id: int) -> None:
        """Register real-time interaction for suppression & warm tracking."""
        self.user_interacted_items[user_id].add(product_id)

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_interacted: bool = True
    ) -> List[Dict[str, Any]]:
        """Predict user-item affinity scores via dot product in latent space."""
        if not self.is_fitted or user_id not in self.user_to_idx:
            return [] # Cold-start user

        u_idx = self.user_to_idx[user_id]
        u_vector = self.user_factors[u_idx] # (K,)

        # Dot product with all items
        scores = np.dot(self.item_factors, u_vector) # (I,)
        scores = np.nan_to_num(scores, nan=-1.0)
        scores = scores.copy()

        # Exclude interacted items
        if exclude_interacted:
            interacted = self.user_interacted_items.get(user_id, set())
            for pid in interacted:
                if pid in self.item_to_idx:
                    scores[self.item_to_idx[pid]] = -1.0

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            raw_score = float(scores[idx])
            if raw_score > 0.0:
                # Map score to [0, 1] range
                norm_score = max(0.0, min(raw_score, 1.0))
                results.append({
                    "product_id": self.idx_to_item[idx],
                    "score": round(norm_score, 4),
                    "model": self.name,
                    "reason": "Người dùng có sở thích tương tự bạn cũng yêu thích sản phẩm này"
                })
        return results

    def similar_items(
        self,
        product_id: int,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Find similar items using item latent embeddings."""
        if not self.is_fitted or product_id not in self.item_to_idx:
            return []

        i_idx = self.item_to_idx[product_id]
        item_vector = self.item_factors[i_idx]

        sim_scores = np.dot(self.item_factors, item_vector)
        sim_scores = np.nan_to_num(sim_scores, nan=-1.0)
        sim_scores = sim_scores.copy()
        sim_scores[i_idx] = -1.0

        top_indices = np.argsort(sim_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            raw_score = float(sim_scores[idx])
            if raw_score > 0.0:
                norm_score = max(0.0, min(raw_score, 1.0))
                results.append({
                    "product_id": self.idx_to_item[idx],
                    "score": round(norm_score, 4),
                    "model": self.name,
                    "reason": "Thường được mua hoặc quan tâm cùng với sản phẩm này"
                })
        return results

