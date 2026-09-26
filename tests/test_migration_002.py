"""Migration 002 creates the shared durable-workflow tables and reverses cleanly.

Runs alembic in a subprocess with DATABASE_URL pointed at a scratch SQLite
file: alembic env.py always resolves the URL from settings at import time,
so in-process overrides cannot isolate the test database.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from sqlalchemy import create_engine, inspect

# tempfile.gettempdir() rather than a hardcoded /tmp: on Windows "/tmp" is not
# guaranteed to exist and does not match the directory the OS actually uses.
DB = str(Path(tempfile.gettempdir()) / "recsys_mig002_test.db")
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
        [sys.executable, "-m", "alembic", *args],
        capture_output=True,
        text=True,
        env=_env(fresh),
        check=False,
    )


def _names():
    # The engine must be disposed: an undisposed engine keeps the SQLite file
    # handle open, and the next test's os.remove(DB) then fails with
    # PermissionError on Windows (POSIX allows unlinking an open file).
    engine = create_engine(f"sqlite:///{DB}")
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


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
