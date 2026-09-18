"""Metrics endpoint: Prometheus-text series for model version, request
volume, and (Plan 03 wires real values) queue depth / DLQ count."""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_metrics_exposes_minimum_series(client):
    r = client.get("/api/v1/metrics")
    assert r.status_code == 200
    body = r.text
    assert "http_requests_total" in body
    assert "model_version" in body
    assert "queue_depth" in body
    assert "dlq_count" in body


def test_metrics_counts_requests(client):
    before = client.get("/api/v1/metrics").text
    client.get("/api/v1/health/live")
    after = client.get("/api/v1/metrics").text
    assert _series(after, "http_requests_total") > _series(before, "http_requests_total")


def _series(body: str, name: str) -> float:
    for line in body.splitlines():
        if line.startswith(name + " ") or line.startswith(name + "{"):
            return float(line.rsplit(" ", 1)[1])
    raise AssertionError(f"series {name} not found")
