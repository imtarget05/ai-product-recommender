"""Unit tests for Database and Models."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.models import Base, Product, User, Interaction, SearchLog

@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_create_product_and_user(in_memory_db):
    prod = Product(
        title="Bàn phím cơ Bluetooth",
        category="Thiết bị máy tính",
        price=1500000.0,
        description="Bàn phím cơ gõ êm",
        rating_avg=4.8,
        rating_count=50
    )
    in_memory_db.add(prod)

    user = User(
        username="test_user",
        email="test@recsys.local",
        segment="tech_geek"
    )
    in_memory_db.add(user)
    in_memory_db.commit()

    assert prod.id is not None
    assert user.id is not None
    assert prod.to_dict()["title"] == "Bàn phím cơ Bluetooth"
    assert user.to_dict()["segment"] == "tech_geek"

def test_interaction_recording(in_memory_db):
    prod = Product(title="Tai nghe ANC", category="Phụ kiện", price=1000000.0)
    user = User(username="user_inter", email="u@test.local")
    in_memory_db.add_all([prod, user])
    in_memory_db.commit()

    inter = Interaction(
        user_id=user.id,
        product_id=prod.id,
        event_type="purchase",
        weight=5.0
    )
    in_memory_db.add(inter)
    in_memory_db.commit()

    assert inter.id is not None
    assert inter.event_type == "purchase"
    assert inter.weight == 5.0
