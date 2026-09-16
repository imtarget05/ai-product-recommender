"""Qdrant Cloud & Local Vector Database Service.
Handles ANN Vector Search, collection lifecycle, and product embedding indexing.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from src.config import settings

class QdrantVectorStore:
    """Manages vector collection in Qdrant Cloud / Local."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        dim: int = 64
    ):
        self.url = url or settings.QDRANT_URL
        self.api_key = api_key or settings.QDRANT_API_KEY
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.dim = dim
        self.client: Optional[QdrantClient] = None
        self._init_client()

    def _init_client(self):
        """Initialize connection to Qdrant Cloud or in-memory fallback."""
        try:
            if self.url and self.api_key:
                # Qdrant Cloud Cluster
                self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=5.0)
                print(f"🌐 Connected to Qdrant Cloud: {self.url}")
            elif self.url:
                # Local or self-hosted Qdrant URL
                self.client = QdrantClient(url=self.url, timeout=5.0)
                print(f"📦 Connected to Qdrant: {self.url}")
            else:
                # Local In-Memory Qdrant for development/testing
                self.client = QdrantClient(":memory:")
                print("🧠 Initialized In-Memory Qdrant Vector DB.")

            self._ensure_collection()
        except Exception as e:
            print(f"⚠️ Warning initializing Qdrant: {e}. Falling back to :memory: mode.")
            self.client = QdrantClient(":memory:")
            self._ensure_collection()

    def _ensure_collection(self):
        """Ensure vector collection exists with cosine distance."""
        if not self.client:
            return

        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.dim,
                    distance=qmodels.Distance.COSINE
                )
            )
            print(f"✅ Created Qdrant collection '{self.collection_name}' (dim={self.dim}, metric=COSINE).")

    def upsert_products(
        self,
        product_ids: List[int],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Upsert product vectors and payload metadata into Qdrant."""
        if not self.client or len(product_ids) == 0:
            return False

        points = []
        for i, pid in enumerate(product_ids):
            native_pid = int(pid)
            vec = [float(x) for x in embeddings[i]]
            payload = metadatas[i].copy() if metadatas and i < len(metadatas) else {}
            payload["product_id"] = native_pid

            points.append(
                qmodels.PointStruct(
                    id=native_pid,
                    vector=vec,
                    payload=payload
                )
            )

        # Batch upsert (chunks of 100)
        batch_size = 100
        for i in range(0, len(points), batch_size):
            chunk = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=chunk
            )
        return True

    def search_similar(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        exclude_ids: Optional[List[int]] = None,
        filter_category: Optional[str] = None
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Perform Approximate Nearest Neighbors (ANN) vector search."""
        if not self.client:
            return []

        # Build search filters
        must_conditions = []
        if filter_category:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="category",
                    match=qmodels.MatchValue(value=filter_category)
                )
            )

        q_filter = qmodels.Filter(must=must_conditions) if must_conditions else None

        # ANN Search
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            query_filter=q_filter,
            limit=top_k + (len(exclude_ids) if exclude_ids else 0),
            with_payload=True
        ).points

        exclude_set = set(exclude_ids or [])
        results = []
        for hit in hits:
            pid = hit.payload.get("product_id", hit.id)
            if pid in exclude_set:
                continue
            results.append((int(pid), float(hit.score), hit.payload or {}))
            if len(results) >= top_k:
                break

        return results

    def get_vector(self, product_id: int) -> Optional[np.ndarray]:
        """Retrieve stored embedding vector for a product."""
        if not self.client:
            return None
        records = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[product_id],
            with_vectors=True
        )
        if records and records[0].vector:
            return np.array(records[0].vector)
        return None
