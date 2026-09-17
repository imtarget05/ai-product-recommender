"""Feature Engineering: Item Metadata Embedding Generator.
Converts product title, category, description, and tags into dense vector embeddings.
Hardened against small catalogs (N <= 1), empty texts, stopwords, and NaN/Inf anomalies.
"""
import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from src.config import settings
from src.database.models import Product

class EmbeddingProvider(ABC):
    @abstractmethod
    def fit_transform(self, products: List[Product]) -> np.ndarray:
        pass

    @abstractmethod
    def transform_single(self, product: Product) -> np.ndarray:
        pass

class TfIdfSvdEmbedder(EmbeddingProvider):
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None
    
    def _prepare_text(self, product: Product) -> str:
        parts = [
            str(product.title or ""),
            str(product.category or ""),
            str(product.tags or ""),
            str(product.description or ""),
        ]
        clean_text = " ".join(filter(None, parts)).lower().strip()
        return clean_text if clean_text else "san pham mac dinh"
        
    def fit_transform(self, products: List[Product]) -> np.ndarray:
        if not products:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        texts = [self._prepare_text(p) for p in products]
        
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b"
        )
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        n_samples, n_features = tfidf_matrix.shape

        if n_samples >= 2 and n_features >= 2:
            n_components = min(self.embedding_dim, n_features - 1, n_samples - 1)
            n_components = max(1, n_components)
            if n_components >= 2 and n_features >= 2:
                try:
                    self.svd = TruncatedSVD(n_components=n_components, random_state=42)
                    dense_vecs = self.svd.fit_transform(tfidf_matrix)
                except Exception:
                    self.svd = None
                    dense_vecs = tfidf_matrix.toarray()
            else:
                self.svd = None
                dense_vecs = tfidf_matrix.toarray()
        else:
            self.svd = None
            dense_vecs = tfidf_matrix.toarray()

        dense_vecs = np.nan_to_num(dense_vecs, nan=0.0, posinf=0.0, neginf=0.0)

        if dense_vecs.shape[1] < self.embedding_dim:
            pad_width = self.embedding_dim - dense_vecs.shape[1]
            dense_vecs = np.pad(dense_vecs, ((0, 0), (0, pad_width)), mode="constant")
        elif dense_vecs.shape[1] > self.embedding_dim:
            dense_vecs = dense_vecs[:, :self.embedding_dim]

        norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return dense_vecs / norms

    def transform_single(self, product: Product) -> np.ndarray:
        if self.vectorizer is None:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        text = self._prepare_text(product)
        tfidf_vec = self.vectorizer.transform([text])

        if self.svd is not None:
            dense_vec = self.svd.transform(tfidf_vec)
        else:
            dense_vec = tfidf_vec.toarray()

        dense_vec = np.nan_to_num(dense_vec, nan=0.0, posinf=0.0, neginf=0.0)
        if dense_vec.shape[1] < self.embedding_dim:
            pad = self.embedding_dim - dense_vec.shape[1]
            dense_vec = np.pad(dense_vec, ((0, 0), (0, pad)), mode="constant")
        elif dense_vec.shape[1] > self.embedding_dim:
            dense_vec = dense_vec[:, :self.embedding_dim]

        norm = np.linalg.norm(dense_vec)
        if norm > 0.0:
            dense_vec = dense_vec / norm
        return dense_vec[0]

class ItemEmbedder:
    """Computes and stores dense vector representations for product catalog items."""

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.product_ids: List[int] = []
        self.embeddings: Optional[np.ndarray] = None # Shape: (N, embedding_dim)
        self.id_to_idx: Dict[int, int] = {}
        self.idx_to_id: Dict[int, int] = {}
        
        # Abstraction layer support
        self.provider = TfIdfSvdEmbedder(embedding_dim=embedding_dim)

    def fit_transform(self, products: List[Product]) -> np.ndarray:
        self.product_ids = [int(p.id) for p in products]
        self.id_to_idx = {pid: i for i, pid in enumerate(self.product_ids)}
        self.idx_to_id = {i: pid for i, pid in enumerate(self.product_ids)}
        
        self.embeddings = self.provider.fit_transform(products)
        return self.embeddings

    def transform_single(self, product: Product) -> np.ndarray:
        return self.provider.transform_single(product)

    def save(self, output_dir: Optional[Path] = None):
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
                "vectorizer": self.provider.vectorizer,
                "svd": self.provider.svd,
                "embedding_dim": self.embedding_dim
            }, f)

    def load(self, output_dir: Optional[Path] = None) -> bool:
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
            self.provider.vectorizer = meta["vectorizer"]
            self.provider.svd = meta["svd"]
            self.embedding_dim = meta["embedding_dim"]
        return True

    def get_embedding(self, product_id: int) -> Optional[np.ndarray]:
        if self.embeddings is None or product_id not in self.id_to_idx:
            return None
        idx = self.id_to_idx[product_id]
        return self.embeddings[idx]

    def compute_similarity(self, product_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.embeddings is None or product_id not in self.id_to_idx:
            return []
        target_idx = self.id_to_idx[product_id]
        target_vec = self.embeddings[target_idx]
        scores = np.dot(self.embeddings, target_vec)
        scores = np.nan_to_num(scores, nan=-1.0)
        scores[target_idx] = -1.0
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.idx_to_id[idx], float(scores[idx])) for idx in top_indices if scores[idx] > 0]
