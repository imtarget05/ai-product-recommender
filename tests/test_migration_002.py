"""Migration 002 creates the shared durable-workflow tables and reverses cleanly.

Runs alembic in a subprocess with DATABASE_URL pointed at a scratch SQLite
file: alembic env.py always resolves the URL from settings at import time,
so in-process overrides cannot isolate the test database.
"""
import os
import subprocess

from sqlalchemy import create_engine, inspect

DB = "/tmp/recsys_mig002_test.db"
TABLES = (
    "idempotency_keys",
    "jobs",
    "outbox_events",
    "dead_letters",
    "audit_events",
    "model_registry",
)


def _env(fresh=False):
    if fresh and os.path.exists(DB):
        os.remove(DB)
    env = dict(os.environ)
    env["DATABASE_URL"] = f"sqlite:///{DB}"
    return env


def _run(*args, fresh=False):
    return subprocess.run(
        [".venv/bin/python", "-m", "alembic", *args],
        capture_output=True,
        text=True,
        env=_env(fresh),
        check=False,
    )


def _names():
    return set(inspect(create_engine(f"sqlite:///{DB}")).get_table_names())


def test_upgrade_creates_tables():
    proc = _run("upgrade", "head", fresh=True)
    assert proc.returncode == 0, proc.stderr
    names = _names()
    for expected in TABLES:
        assert expected in names


def test_downgrade_removes_only_002_tables():
    assert _run("upgrade", "head", fresh=True).returncode == 0
    proc = _run("downgrade", "-1")
    assert proc.returncode == 0, proc.stderr
    names = _names()
    for expected in TABLES:
        assert expected not in names
    assert "products" in names
