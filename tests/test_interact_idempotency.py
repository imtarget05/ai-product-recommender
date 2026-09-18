"""Idempotent interaction ingest: the same X-Idempotency-Key returns the
original stored response without inserting a duplicate Interaction."""
import os
import subprocess
import sys

# Isolate from the dev database and ensure the 002 durable-workflow tables
# (idempotency_keys, ...) exist: init_db() only creates base metadata tables.
# Must be set before importing the app (engine binds at import time).
DB = "/tmp/recsys_idem_test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{DB}"

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    # The app engine binds DATABASE_URL at first import, which may happen
    # in another test module before this one. Migrate + seed BOTH the
    # isolated DB and the default dev DB so the table exists regardless
    # of which engine the app bound.
    from src.config import PROJECT_ROOT

    default_db = PROJECT_ROOT / "data" / "recsys.db"
    for db_path in (DB, str(default_db)):
        if db_path == DB and os.path.exists(DB):
            os.remove(DB)
        env = dict(os.environ)
        env["DATABASE_URL"] = f"sqlite:///{db_path}"
        proc = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
    # Seed the rows the endpoint requires (fresh DBs have none).
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session

    for db_path in (DB, str(default_db)):
        eng = create_engine(f"sqlite:///{db_path}")
        with Session(eng) as s:
            s.execute(
                text("INSERT OR IGNORE INTO users (id, username) VALUES (10, 'idem_test_user')")
            )
            s.execute(
                text(
                    "INSERT OR IGNORE INTO products (id, title, category, price) "
                    "VALUES (1, 'Idem Test Product', 'test', 9.99)"
                )
            )
            s.commit()
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
