"""Edge case, numerical anomaly, and cold-start boundary tests."""
import pytest
import numpy as np
from datetime import datetime
from src.database.models import Product, Interaction
from src.features.text_embedder import ItemEmbedder
from src.features.user_profiler import UserProfiler
from src.models.collaborative import CollaborativeRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender
from src.api.schemas import InteractionCreate

def test_empty_catalog_handling():
    """Verify models do not crash when initialized on empty catalogs."""
    embedder = ItemEmbedder(embedding_dim=32)
    embeddings = embedder.fit_transform([])
    assert embeddings.shape == (0, 32)

    cf = CollaborativeRecommender(n_factors=8)
    cf.fit([], [])
    assert cf.recommend(user_id=1, top_k=5) == []

    cb = ContentBasedRecommender(embedder=embedder)
    cb.fit([], [])
    assert cb.recommend(user_id=1, top_k=5) == []

    hybrid = HybridRecommender()
    hybrid.fit([], [])
    assert hybrid.recommend(user_id=1, top_k=5) == []


def test_degenerate_svd_single_user_single_item():
    """Verify SVD handles 1 user and 1 item without crashing."""
    p = Product(id=101, title="Solo Item", category="Test", price=100.0)
    inter = Interaction(id=1, user_id=99, product_id=101, event_type="view", weight=1.0)

    cf = CollaborativeRecommender(n_factors=16)
    cf.fit([p], [inter])
    assert cf.is_fitted
    # No NaN in factors
    assert not np.isnan(cf.user_factors).any()
    assert not np.isnan(cf.item_factors).any()


def test_empty_metadata_text_embedder():
    """Verify embedder handles products with None / empty string metadata."""
    p1 = Product(id=1, title="", category=None, description="", tags=None)
    p2 = Product(id=2, title=None, category="", description=None, tags="")

    embedder = ItemEmbedder(embedding_dim=32)
    vecs = embedder.fit_transform([p1, p2])
    assert vecs.shape == (2, 32)
    assert not np.isnan(vecs).any()
    assert not np.isinf(vecs).any()

    # Test single real-time transformation
    p3 = Product(id=3, title=None, category=None)
    vec3 = embedder.transform_single(p3)
    assert vec3.shape == (32,)
    assert not np.isnan(vec3).any()


def test_user_profiler_time_decay_and_timestamp_types():
    """Verify user profiler handles int timestamps (epoch ms) and missing timestamps."""
    embedder = ItemEmbedder(embedding_dim=16)
    p1 = Product(id=1, title="P1", category="C1")
    p2 = Product(id=2, title="P2", category="C2")
    embedder.fit_transform([p1, p2])

    profiler = UserProfiler(embedder, half_life_days=7.0)

    # Mixed timestamps: datetime, epoch int ms, None
    inters = [
        Interaction(id=1, user_id=1, product_id=1, event_type="view", weight=1.0, timestamp=datetime.utcnow()),
        Interaction(id=2, user_id=1, product_id=2, event_type="cart", weight=3.5, timestamp=1600000000000),
        Interaction(id=3, user_id=1, product_id=1, event_type="purchase", weight=5.0, timestamp=None)
    ]
    u_vec = profiler.build_user_vector(inters)
    assert u_vec is not None
    assert u_vec.shape == (16,)
    assert not np.isnan(u_vec).any()
    # Norm is 1.0 (normalized)
    assert np.isclose(np.linalg.norm(u_vec), 1.0, atol=1e-4)


def test_interaction_create_rating_validation():
    """Verify Pydantic rejects rating event when rating_value is missing."""
    with pytest.raises(ValueError):
        InteractionCreate(
            user_id=1,
            product_id=2,
            event_type="rating",
            rating_value=None # Must fail
        )

    # Valid rating event
    valid = InteractionCreate(
        user_id=1,
        product_id=2,
        event_type="rating",
        rating_value=4.5
    )
    assert valid.rating_value == 4.5


def test_real_time_interaction_updates_warm_user():
    """Verify calling on_new_interaction immediately activates Content-Based for a new user."""
    p1 = Product(id=1, title="Mechanical Keyboard Blue Switch", category="Gear", price=100.0)
    p2 = Product(id=2, title="Mechanical Keyboard Red Switch", category="Gear", price=110.0)
    p3 = Product(id=3, title="Floral Summer Dress", category="Fashion", price=40.0)

    hybrid = HybridRecommender(cold_start_threshold=3)
    hybrid.fit([p1, p2, p3], [])

    # User 888 is brand new -> Popularity fallback
    recs0 = hybrid.recommend(user_id=888, top_k=2)
    assert "Popularity" in recs0[0]["model"]

    # User 888 clicks product 1 (Keyboard)
    click_event = Interaction(id=10, user_id=888, product_id=1, event_type="click", weight=2.0, timestamp=datetime.utcnow())
    hybrid.on_new_interaction(click_event)

    # Immediately recommend again for User 888 -> Must now use Content-Based matching the keyboard!
    recs1 = hybrid.recommend(user_id=888, top_k=2, exclude_interacted=True)
    assert len(recs1) > 0
    # Keyboard 2 (product 2) should be recommended due to content similarity with Keyboard 1
    rec_pids = [r["product_id"] for r in recs1]
    assert 2 in rec_pids
    assert 1 not in rec_pids # Excluded because user already clicked it!
