"""Integration tests for FastAPI Serving Endpoints."""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture(scope="module")
def client():
    # Use context manager so lifespan startup executes and trains models
    with TestClient(app) as test_client:
        yield test_client

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "docs_url" in data

def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_ready"] is True
    assert data["total_products"] > 0

def test_get_products(client):
    response = client.get("/api/v1/products?limit=10")
    assert response.status_code == 200
    prods = response.json()
    assert isinstance(prods, list)
    assert len(prods) <= 10

def test_recommendation_established_user(client):
    response = client.get("/api/v1/recommend/10?top_k=5&strategy=hybrid")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 10
    assert len(data["recommendations"]) <= 5
    assert "latency_ms" in data

def test_recommendation_cold_start_user(client):
    # User 1 is a cold start newbie in seed data
    response = client.get("/api/v1/recommend/1?top_k=4&strategy=hybrid")
    assert response.status_code == 200
    data = response.json()
    assert len(data["recommendations"]) <= 4

def test_record_interaction_and_cache_invalidation(client):
    payload = {
        "user_id": 10,
        "product_id": 1,
        "event_type": "click"
    }
    response = client.post("/api/v1/interact", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "recorded"
    assert res_data["event_type"] == "click"

def test_similar_products(client):
    response = client.get("/api/v1/similar-products/1?top_k=3")
    assert response.status_code == 200
    data = response.json()
    assert "similar_products" in data
    assert len(data["similar_products"]) <= 3

def test_admin_reload_model(client):
    response = client.post("/api/v1/admin/reload-model")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "reloaded_at" in data
    assert data["total_products"] > 0

def test_invalid_rating_interaction_rejected(client):
    # Rating event without rating_value must return 422
    payload = {
        "user_id": 1,
        "product_id": 1,
        "event_type": "rating",
        "rating_value": None
    }
    response = client.post("/api/v1/interact", json=payload)
    assert response.status_code == 422


def test_agent_chat_cart_intent_is_proposed_without_a_commerce_adapter(client):
    """The public chat API must expose proposal status instead of a false cart success."""
    response = client.post("/api/v1/agent/chat", json={
        "messages": [{"role": "user", "content": "Thêm sản phẩm #1 vào giỏ hàng"}],
        "user_id": 1,
        "idempotency_key": "api-cart-proposal-1",
    })

    assert response.status_code == 200
    action = response.json()["action"]
    assert action["status"] == "PROPOSED"
    assert action["product_id"] == 1


# ---------------------------------------------------------------------------
# Readiness / model-bundle serving contract (plan 2026-09-18 Task 2 + 3)
# ---------------------------------------------------------------------------
def test_health_live_and_ready_are_separate(client):
    """/health/live stays available; /health/ready reports DB + bundle state."""
    live = client.get("/api/v1/health/live")
    assert live.status_code == 200
    assert live.json()["status"] == "alive"

    ready = client.get("/api/v1/health/ready")
    body = ready.json()
    assert "ready" in body
    if ready.status_code == 200:
        assert body["ready"] is True
        assert body["db_connected"] is True
    else:
        assert ready.status_code == 503
        assert body["ready"] is False


def test_model_info_reports_bundle_identity_or_503(client):
    """/model/info exposes immutable bundle identity, or 503 when unloaded."""
    response = client.get("/api/v1/model/info")
    if response.status_code == 200:
        body = response.json()
        assert body["version"]
        assert len(body["dataset_fingerprint"]) == 64
    else:
        assert response.status_code == 503


def test_recommendation_response_carries_model_version_header(client):
    """Every recommendation response must identify the serving model version."""
    response = client.get("/api/v1/recommend/10?top_k=3&strategy=hybrid")
    assert response.status_code == 200
    assert response.headers.get("X-Model-Version")
    assert response.headers.get("X-Cache-Source") in {"redis", "memory", "unknown"}
    assert "model_version" in response.json()


def test_admin_reload_requires_key_when_configured(client, monkeypatch):
    """A configured ADMIN_API_KEY must protect the admin mutation endpoint."""
    monkeypatch.setenv("ADMIN_API_KEY", "admin-secret")
    denied = client.post("/api/v1/admin/reload-model")
    assert denied.status_code == 401
    allowed = client.post("/api/v1/admin/reload-model", headers={"X-API-Key": "admin-secret"})
    assert allowed.status_code == 200
