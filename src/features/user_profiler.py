"""User Profiling: Builds user preference embeddings from interaction history."""
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.preprocessing import normalize
from src.features.text_embedder import ItemEmbedder
from src.database.models import Interaction

class UserProfiler:
    """Aggregates user interaction history into a single user interest vector."""

    def __init__(self, item_embedder: ItemEmbedder):
        self.item_embedder = item_embedder

    def build_user_vector(
        self,
        interactions: List[Interaction],
        time_decay_factor: float = 0.98
    ) -> Optional[np.ndarray]:
        """Compute weighted preference vector from user interactions.

        Uses interaction weights (view=1.0, click=2.0, cart=3.5, purchase=5.0)
        and optional recency decay.
        """
        if not interactions or self.item_embedder.embeddings is None:
            return None

        # Sort by timestamp ascending
        sorted_interactions = sorted(
            interactions,
            key=lambda x: x.timestamp if x.timestamp else 0
        )

        n = len(sorted_interactions)
        weighted_vectors = []
        total_weight = 0.0

        for i, inter in enumerate(sorted_interactions):
            vec = self.item_embedder.get_embedding(inter.product_id)
            if vec is None:
                continue

            # Base event weight
            weight = float(inter.weight if inter.weight else 1.0)

            # Time decay: more recent events have higher weight
            decay = time_decay_factor ** (n - 1 - i)
            effective_weight = weight * decay

            weighted_vectors.append(vec * effective_weight)
            total_weight += effective_weight

        if not weighted_vectors or total_weight == 0.0:
            return None

        user_vec = np.sum(weighted_vectors, axis=0) / total_weight
        return normalize(user_vec.reshape(1, -1), norm="l2")[0]

    def match_user_to_items(
        self,
        user_vec: np.ndarray,
        exclude_item_ids: Optional[List[int]] = None,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Score all catalog items against user preference vector."""
        if self.item_embedder.embeddings is None:
            return []

        scores = np.dot(self.item_embedder.embeddings, user_vec) # Cosine similarity
        scores = scores.copy()

        # Mask excluded items (e.g., already purchased)
        if exclude_item_ids:
            for pid in exclude_item_ids:
                if pid in self.item_embedder.id_to_idx:
                    idx = self.item_embedder.id_to_idx[pid]
                    scores[idx] = -1.0

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.0:
                results.append((self.item_embedder.idx_to_id[idx], score))
        return results
