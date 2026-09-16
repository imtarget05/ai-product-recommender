"""Unit tests for Recommendation Models."""
import pytest
from datetime import datetime
from src.database.models import Product, Interaction
from src.models.baseline import PopularityRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender
from src.models.hybrid import HybridRecommender

@pytest.fixture
def mock_data():
    products = [
        Product(id=1, title="Laptop Gaming RTX", category="Laptop", price=25000000.0, tags="laptop, gaming, rtx", description="High performance", rating_avg=4.8, rating_count=100),
        Product(id=2, title="Chuột Gaming không dây", category="Phụ kiện", price=1200000.0, tags="chuot, gaming, wireless", description="DPI cao", rating_avg=4.6, rating_count=80),
        Product(id=3, title="Bàn phím cơ cơ khí", category="Phụ kiện", price=1800000.0, tags="ban phim, gaming, led", description="Switch blue", rating_avg=4.7, rating_count=90),
        Product(id=4, title="Váy đầm dạ hội nữ", category="Thời trang", price=500000.0, tags="vay, thoi trang, nu", description="Vải lụa cao cấp", rating_avg=4.5, rating_count=40),
        Product(id=5, title="Áo thun polo nam", category="Thời trang", price=250000.0, tags="ao, polo, nam", description="Cotton co giãn", rating_avg=4.3, rating_count=60),
    ]

    interactions = [
        # User 1: Tech/Gaming fan
        Interaction(id=1, user_id=1, product_id=1, event_type="purchase", weight=5.0, timestamp=datetime.utcnow()),
        Interaction(id=2, user_id=1, product_id=2, event_type="click", weight=2.0, timestamp=datetime.utcnow()),
        Interaction(id=3, user_id=1, product_id=3, event_type="add_to_cart", weight=3.5, timestamp=datetime.utcnow()),
        # User 2: Tech/Gaming fan
        Interaction(id=4, user_id=2, product_id=1, event_type="view", weight=1.0, timestamp=datetime.utcnow()),
        Interaction(id=5, user_id=2, product_id=2, event_type="purchase", weight=5.0, timestamp=datetime.utcnow()),
        # User 3: Fashion fan
        Interaction(id=6, user_id=3, product_id=4, event_type="purchase", weight=5.0, timestamp=datetime.utcnow()),
        Interaction(id=7, user_id=3, product_id=5, event_type="click", weight=2.0, timestamp=datetime.utcnow()),
    ]
    return products, interactions

def test_popularity_recommender(mock_data):
    products, interactions = mock_data
    model = PopularityRecommender()
    model.fit(products, interactions)

    recs = model.recommend(user_id=1, top_k=3)
    assert len(recs) == 3
    assert all("score" in r and "product_id" in r for r in recs)

def test_content_based_recommender(mock_data):
    products, interactions = mock_data
    model = ContentBasedRecommender()
    model.fit(products, interactions)

    # User 1 interacted with items 1, 2, 3 -> Should recommend from remaining catalog (items 4, 5)
    recs = model.recommend(user_id=1, top_k=2, exclude_interacted=True)
    assert len(recs) > 0
    recommended_ids = [r["product_id"] for r in recs]
    # Interacted items (1, 2, 3) must NOT be in recommendations
    assert 1 not in recommended_ids
    assert 2 not in recommended_ids
    assert 3 not in recommended_ids

    # Similar items for product 1 (Laptop)
    sims = model.similar_items(product_id=1, top_k=2)
    assert len(sims) > 0

def test_collaborative_recommender(mock_data):
    products, interactions = mock_data
    model = CollaborativeRecommender(n_factors=2)
    model.fit(products, interactions)

    recs = model.recommend(user_id=2, top_k=2, exclude_interacted=True)
    assert len(recs) > 0

def test_hybrid_recommender_established_and_cold_start(mock_data):
    products, interactions = mock_data
    hybrid = HybridRecommender(cold_start_threshold=2)
    hybrid.fit(products, interactions)

    # Established user (User 1 has 3 interactions)
    recs_user1 = hybrid.recommend(user_id=1, top_k=3)
    assert len(recs_user1) > 0

    # Cold start user (User 999 has 0 interactions)
    recs_cold = hybrid.recommend(user_id=999, top_k=3)
    assert len(recs_cold) == 3
    assert "Cold-start" in recs_cold[0]["model"]
