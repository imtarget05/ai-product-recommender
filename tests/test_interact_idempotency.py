"""Idempotent interaction ingest: the same X-Idempotency-Key returns the
original stored response without inserting a duplicate Interaction."""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _payload():
    return {"user_id": 10, "product_id": 1, "event_type": "view"}


def test_replay_returns_original_without_duplicate(client):
    import uuid

    headers = {"X-Idempotency-Key": f"rec-{uuid.uuid4().hex[:8]}"}
    first = client.post("/api/v1/interact", json=_payload(), headers=headers)
    assert first.status_code == 200
    second = client.post("/api/v1/interact", json=_payload(), headers=headers)
    assert second.status_code == 200
    assert second.json()["interaction_id"] == first.json()["interaction_id"]


def test_no_key_keeps_current_behavior(client):
    first = client.post("/api/v1/interact", json=_payload())
    second = client.post("/api/v1/interact", json=_payload())
    assert first.status_code == 200 == second.status_code
    assert second.json()["interaction_id"] != first.json()["interaction_id"]
