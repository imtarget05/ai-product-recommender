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

