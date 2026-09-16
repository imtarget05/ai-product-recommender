"""User Profiling: Builds user preference embeddings from interaction history.
Includes true timestamp-based exponential time decay, recency weighting, and numerical safety.
"""
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from sklearn.preprocessing import normalize
from src.features.text_embedder import ItemEmbedder
from src.database.models import Interaction

class UserProfiler:
    """Aggregates user interaction history into a single user interest vector."""

    def __init__(self, item_embedder: ItemEmbedder, half_life_days: float = 7.0):
        self.item_embedder = item_embedder
        self.half_life_days = half_life_days
        # Lambda for half-life decay: exp(-lambda * delta_days) = 0.5 => lambda = ln(2) / half_life
        self.decay_lambda = np.log(2.0) / max(self.half_life_days, 0.1)

    def _parse_timestamp_seconds(self, ts: Union[datetime, int, float, None]) -> float:
        """Parse timestamp into epoch seconds."""
        if ts is None:
            return 0.0
        if isinstance(ts, datetime):
            return ts.timestamp()
        if isinstance(ts, (int, float)):
            # Handle millisecond timestamps from RetailRocket/Kaggle
            if ts > 1e11:
                return float(ts) / 1000.0
            return float(ts)
        return 0.0

    def build_user_vector(
        self,
        interactions: List[Interaction],
        reference_time: Optional[datetime] = None
    ) -> Optional[np.ndarray]:
        """Compute weighted preference vector from user interactions.

        Uses interaction event weights (view=1.0, click=2.0, cart=3.5, purchase=5.0)
        and real-time exponential decay based on time elapsed since interaction.
        """
        if not interactions or self.item_embedder.embeddings is None:
            return None

        # Sort by timestamp ascending
        sorted_interactions = sorted(
            interactions,
            key=lambda x: self._parse_timestamp_seconds(x.timestamp)
        )

        ref_sec = reference_time.timestamp() if reference_time else None
        if ref_sec is None:
            # Anchor to the latest event timestamp if reference_time is not provided
            max_ts = max(self._parse_timestamp_seconds(x.timestamp) for x in sorted_interactions)
            ref_sec = max_ts if max_ts > 0 else datetime.utcnow().timestamp()

        n = len(sorted_interactions)
        weighted_vectors = []
        total_weight = 0.0

        for i, inter in enumerate(sorted_interactions):
            vec = self.item_embedder.get_embedding(inter.product_id)
            if vec is None:
                continue

            # Base event weight
            weight = float(inter.weight if inter.weight else 1.0)

            # Exponential time decay based on delta days
            event_sec = self._parse_timestamp_seconds(inter.timestamp)
            if event_sec > 0 and ref_sec >= event_sec:
                delta_days = (ref_sec - event_sec) / 86400.0
                decay = float(np.exp(-self.decay_lambda * delta_days))
            else:
                # Fallback to rank decay if timestamps are missing
                decay = float(0.98 ** (n - 1 - i))

            effective_weight = max(weight * decay, 0.001)

            weighted_vectors.append(vec * effective_weight)
            total_weight += effective_weight

        if not weighted_vectors or total_weight <= 0.0:
            return None

        user_vec = np.sum(weighted_vectors, axis=0) / total_weight
        user_vec = np.nan_to_num(user_vec, nan=0.0)

        norm = np.linalg.norm(user_vec)
        if norm == 0.0:
            return None

        return (user_vec / norm).astype(np.float32)

    def match_user_to_items(
        self,
        user_vec: np.ndarray,
        exclude_item_ids: Optional[List[int]] = None,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Score all catalog items against user preference vector."""
        if self.item_embedder.embeddings is None or len(self.item_embedder.embeddings) == 0:
            return []

        scores = np.dot(self.item_embedder.embeddings, user_vec) # Cosine similarity
        scores = np.nan_to_num(scores, nan=-1.0)
        scores = scores.copy()

        # Mask excluded items (e.g., already purchased or interacted)
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

