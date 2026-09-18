"""Integration and Unit tests for Cognitive Agent, RAG Search & Shopping Assistant."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from src.api.main import app
from src.database.session import SessionLocal
from src.database.models import Product, Interaction
from src.agent.groq_client import GroqClientManager, groq_client
from src.agent.rag_search import RAGSearchEngine
from src.agent.explainer import DynamicExplainer
from src.agent.shopping_agent import ConversationalShoppingAssistant
from src.commerce.cart_gateway import DisabledCartGateway, InMemoryCartGateway


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def db_session():
    db = SessionLocal()
    yield db
    db.close()


def test_groq_client_circuit_breaker():
    """Verify circuit breaker trips on consecutive failures and recovers."""
    manager = GroqClientManager()
    manager._failure_count = 0
    manager._circuit_open_until = 0.0

    assert not manager._is_circuit_open()

    # Simulate 3 failures
    manager._record_failure()
    assert not manager._is_circuit_open()
    manager._record_failure()
    assert not manager._is_circuit_open()
    manager._record_failure()
    assert manager._is_circuit_open()

    # Reset upon success
    manager._record_success()
    assert not manager._is_circuit_open()
    assert manager._failure_count == 0


def test_rag_search_heuristic_intent_parsing():
    """Verify regex heuristic fallback parses prices and categories accurately."""
    engine = RAGSearchEngine()
    categories = ["Laptop", "Điện thoại", "Thời trang", "Âm thanh", "Đồng hồ"]

    # 1. Under max price
    intent1 = engine._heuristic_parse("tìm laptop dưới 20 triệu", categories)
    assert intent1["category"] == "Laptop"
    assert intent1["max_price"] == 20000000.0

    # 2. Price range
    intent2 = engine._heuristic_parse("tai nghe âm thanh từ 500k đến 2 triệu", categories)
    assert intent2["category"] == "Âm thanh"
    assert intent2["min_price"] == 500000.0
    assert intent2["max_price"] == 2000000.0

    # 3. Short price notation
    intent3 = engine._heuristic_parse("đồng hồ nam dưới 1.5tr", categories)
    assert intent3["category"] == "Đồng hồ"
    assert intent3["max_price"] == 1500000.0


def test_rag_search_execution(db_session: Session):
    """Verify RAG hybrid search pipeline retrieves filtered products and synthesizes reply."""
    engine = RAGSearchEngine()
    result = engine.search("tìm sản phẩm dưới 15 triệu", db=db_session, top_k=3)

    assert "query" in result
    assert "intent" in result
    assert "reply" in result
    assert len(result["reply"]) > 0
    assert "products" in result
    assert len(result["products"]) <= 3
    assert result["latency_ms"] >= 0

    for prod in result["products"]:
        assert prod["price"] <= 15000000.0 or result["intent"].get("max_price") == 15000000.0


def test_dynamic_explainer(db_session: Session):
    """Verify dynamic rationale generator creates compelling explanations."""
    explainer = DynamicExplainer()
    product = db_session.query(Product).first()
    assert product is not None

    interactions = db_session.query(Interaction).limit(3).all()
    product_dict = {product.id: product}

    explanation = explainer.explain(
        product=product,
        recent_interactions=interactions,
        product_dict=product_dict
    )
    assert isinstance(explanation, str)
    assert len(explanation.strip()) > 5


def test_shopping_assistant_unconfigured_cart_action_is_a_proposal(db_session: Session):
    """A missing commerce adapter must not be presented as a completed cart update."""
    assistant = ConversationalShoppingAssistant(cart_gateway=DisabledCartGateway())
    sample_product = db_session.query(Product).first()
    assert sample_product is not None
    user_id = 999

    messages = [
        {"role": "user", "content": f"Tôi muốn thêm sản phẩm #{sample_product.id} vào giỏ hàng"}
    ]

    res = assistant.process_chat(messages, user_id=user_id, db=db_session)
    assert "reply" in res
    assert res.get("action") is not None
    assert res["action"]["type"] == "add_to_cart"
    assert res["action"]["product_id"] == sample_product.id
    assert res["action"]["status"] == "PROPOSED"
    assert "thành công" not in res["reply"].lower()

    # A recommendation event is not evidence that a cart was mutated.
    recorded = (
        db_session.query(Interaction)
        .filter(Interaction.user_id == user_id, Interaction.product_id == sample_product.id, Interaction.event_type == "add_to_cart")
        .first()
    )
    assert recorded is None


def test_shopping_assistant_records_acknowledged_cart_action_once_per_idempotency_key(db_session: Session):
    """An acknowledged gateway completion is logged once even when the request is retried."""
    gateway = InMemoryCartGateway()
    assistant = ConversationalShoppingAssistant(cart_gateway=gateway)
    sample_product = db_session.query(Product).first()
    assert sample_product is not None
    user_id = 998
    messages = [{"role": "user", "content": f"Tôi muốn thêm sản phẩm #{sample_product.id} vào giỏ hàng"}]

    first = assistant.process_chat(messages, user_id=user_id, db=db_session, idempotency_key="cart-request-998")
    second = assistant.process_chat(messages, user_id=user_id, db=db_session, idempotency_key="cart-request-998")

    assert first["action"]["status"] == "COMPLETED"
    assert second["action"]["status"] == "COMPLETED"
    assert first["action"]["action_id"] == second["action"]["action_id"]
    assert len(gateway.actions) == 1
    assert (
        db_session.query(Interaction)
        .filter(Interaction.user_id == user_id, Interaction.product_id == sample_product.id, Interaction.event_type == "add_to_cart")
        .count()
        == 1
    )
    db_session.query(Interaction).filter(
        Interaction.user_id == user_id,
        Interaction.product_id == sample_product.id,
        Interaction.event_type == "add_to_cart",
    ).delete()
    db_session.commit()


def test_shopping_assistant_requires_idempotency_key_for_configured_cart_gateway(db_session: Session):
    """Configured cart writes without a client key are rejected rather than guessed."""
    assistant = ConversationalShoppingAssistant(cart_gateway=InMemoryCartGateway())
    sample_product = db_session.query(Product).first()
    assert sample_product is not None
    result = assistant.process_chat(
        [{"role": "user", "content": f"Thêm sản phẩm #{sample_product.id} vào giỏ hàng"}],
        user_id=997,
        db=db_session,
    )
    assert result["action"]["status"] == "FAILED"
    assert "idempotency" in result["action"]["detail"].lower()


def test_agent_api_status(client):
    """Verify GET /api/v1/agent/status returns health and model config."""
    response = client.get("/api/v1/agent/status")
    assert response.status_code == 200
    data = response.json()
    assert "groq_available" in data
    assert "chat_model" in data
    assert "fast_model" in data


def test_agent_api_search(client):
    """Verify POST /api/v1/agent/search returns valid RAG schema."""
    payload = {
        "query": "laptop mỏng nhẹ dưới 30 triệu",
        "user_id": 1,
        "top_k": 4
    }
    response = client.post("/api/v1/agent/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["query"]
    assert "intent" in data
    assert "reply" in data
    assert "products" in data
    assert isinstance(data["products"], list)
    assert len(data["products"]) <= 4
    if data["products"]:
        p = data["products"][0]
        assert "product_id" in p
        assert "title" in p
        assert "price" in p


def test_agent_api_chat(client):
    """Verify POST /api/v1/agent/chat handles multi-turn conversation."""
    payload = {
        "messages": [
            {"role": "user", "content": "Xin chào, tôi cần tìm một đôi giày thể thao thoải mái"}
        ],
        "user_id": 5
    }
    response = client.post("/api/v1/agent/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert len(data["reply"]) > 0
    assert "suggested_products" in data
    assert "latency_ms" in data


def test_agent_api_explain(client):
    """Verify POST /api/v1/agent/explain returns personalized rationale."""
    payload = {
        "product_id": 1,
        "user_id": 10,
        "strategy": "hybrid"
    }
    response = client.post("/api/v1/agent/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["product_id"] == 1
    assert "explanation" in data
    assert len(data["explanation"]) > 0

    # Non-existent product 404 test
    err_res = client.post("/api/v1/agent/explain", json={"product_id": 9999999})
    assert err_res.status_code == 404


def test_offline_fallback_resilience(db_session: Session):
    """Verify agent operates reliably even when Groq is completely mocked out or offline."""
    with patch.object(groq_client, "is_available", return_value=False):
        engine = RAGSearchEngine()
        res = engine.search("tìm phụ kiện điện thoại dưới 500k", db=db_session, top_k=2)
        assert res is not None
        assert "reply" in res
        assert len(res["products"]) <= 2

        explainer = DynamicExplainer()
        prod = db_session.query(Product).first()
        exp = explainer.explain(prod)
        assert len(exp) > 0
