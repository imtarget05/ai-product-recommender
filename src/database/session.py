"""Database connection and session management.
Production-hardened with connection pooling, pool_pre_ping, and transaction safety.
"""
import os
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from src.config import settings
from src.database.models import Base

db_url = settings.get_database_url
connect_args = {}
engine_kwargs = {
    "echo": False,
    "future": True
}

if db_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
else:
    # Production PostgreSQL / Railway settings
    engine_kwargs.update({
        "pool_pre_ping": True,
        "pool_size": 20,
        "max_overflow": 30,
        "pool_recycle": 1800,
        "pool_timeout": 30
    })

engine = create_engine(
    db_url,
    connect_args=connect_args,
    **engine_kwargs
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all database tables."""
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    Base.metadata.create_all(bind=engine)


def check_db_health() -> bool:
    """Fast liveness check using SELECT 1 without counting tables."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def get_db():
    """FastAPI dependency for DB session with explicit rollback on error."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_context():
    """Context manager for standalone scripts/background tasks."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

