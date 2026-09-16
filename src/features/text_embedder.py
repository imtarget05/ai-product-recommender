"""Feature Engineering: Item Metadata Embedding Generator.
Converts product title, category, description, and tags into dense vector embeddings.
Uses TF-IDF + TruncatedSVD for instantaneous, dependency-light embeddings,
with seamless extensibility for Sentence-Transformers.
"""
import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from src.config import settings
from src.database.models import Product

class ItemEmbedder:
    """Computes and stores dense vector representations for product catalog items."""

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None
        self.product_ids: List[int] = []
        self.embeddings: Optional[np.ndarray] = None # Shape: (N, embedding_dim)
        self.id_to_idx: Dict[int, int] = {}
        self.idx_to_id: Dict[int, int] = {}

    def _prepare_text(self, product: Product) -> str:
        """Combine product metadata fields into a rich textual document."""
        parts = [
            product.title or "",
            product.category or "",
            product.tags or "",
            product.description or "",
        ]
        return " ".join(parts).lower()

    def fit_transform(self, products: List[Product]) -> np.ndarray:
        """Fit feature extractor on product list and compute embeddings."""
        if not products:
            raise ValueError("No products provided to embedder.")

        texts = [self._prepare_text(p) for p in products]
        self.product_ids = [p.id for p in products]
        self.id_to_idx = {pid: i for i, pid in enumerate(self.product_ids)}
        self.idx_to_id = {i: pid for i, pid in enumerate(self.product_ids)}

        # TF-IDF on unigrams and bigrams
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )
        tfidf_matrix = self.vectorizer.fit_transform(texts)

        # Truncated SVD for dense semantic embedding
        n_components = min(self.embedding_dim, tfidf_matrix.shape[1] - 1, len(products) - 1)
        if n_components < 2:
            n_components = min(2, tfidf_matrix.shape[1])

        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        dense_vecs = self.svd.fit_transform(tfidf_matrix)

        # Ensure exact embedding dimension by zero-padding if catalog is small
        if dense_vecs.shape[1] < self.embedding_dim:
            pad_width = self.embedding_dim - dense_vecs.shape[1]
            dense_vecs = np.pad(dense_vecs, ((0, 0), (0, pad_width)), mode="constant")

        # L2 Normalize so dot product equals cosine similarity
        self.embeddings = normalize(dense_vecs, norm="l2", axis=1)
        return self.embeddings

    def save(self, output_dir: Optional[Path] = None):
        """Save fitted embedder and embedding matrix to disk."""
        target_dir = output_dir or settings.EMBEDDINGS_DIR
        os.makedirs(target_dir, exist_ok=True)

        matrix_path = target_dir / "item_embeddings.npy"
        np.save(matrix_path, self.embeddings)

        meta_path = target_dir / "item_embedder_meta.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump({
                "product_ids": self.product_ids,
                "id_to_idx": self.id_to_idx,
                "idx_to_id": self.idx_to_id,
                "vectorizer": self.vectorizer,
                "svd": self.svd,
                "embedding_dim": self.embedding_dim
            }, f)

    def load(self, output_dir: Optional[Path] = None) -> bool:
        """Load precomputed embeddings from disk if available."""
        target_dir = output_dir or settings.EMBEDDINGS_DIR
        matrix_path = target_dir / "item_embeddings.npy"
        meta_path = target_dir / "item_embedder_meta.pkl"

        if not matrix_path.exists() or not meta_path.exists():
            return False

        self.embeddings = np.load(matrix_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
            self.product_ids = meta["product_ids"]
            self.id_to_idx = meta["id_to_idx"]
            self.idx_to_id = meta["idx_to_id"]
            self.vectorizer = meta["vectorizer"]
            self.svd = meta["svd"]
            self.embedding_dim = meta["embedding_dim"]
        return True

    def get_embedding(self, product_id: int) -> Optional[np.ndarray]:
        """Return embedding vector for a given product ID."""
        if self.embeddings is None or product_id not in self.id_to_idx:
            return None
        idx = self.id_to_idx[product_id]
        return self.embeddings[idx]

    def compute_similarity(self, product_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """Find most similar items by cosine similarity."""
        if self.embeddings is None or product_id not in self.id_to_idx:
            return []

        target_idx = self.id_to_idx[product_id]
        target_vec = self.embeddings[target_idx] # (D,)

        # Dot product with all normalized embeddings -> Cosine similarity
        scores = np.dot(self.embeddings, target_vec) # (N,)

        # Exclude self
        scores[target_idx] = -1.0

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.idx_to_id[idx], float(scores[idx])) for idx in top_indices if scores[idx] > 0]
