"""Centralized Configuration for RecSys-AI."""
import os
from typing import Optional
from pathlib import Path
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # App
    APP_NAME: str = "RecSys-AI"
    DEBUG: bool = True
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Paths
    BASE_DIR: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
    EMBEDDINGS_DIR: Path = PROJECT_ROOT / "data" / "embeddings"
    DB_PATH: Path = PROJECT_ROOT / "data" / "recsys.db"

    # Database (Railway Postgres or SQLite)
    DATABASE_URL: str = f"sqlite:///{PROJECT_ROOT}/data/recsys.db"

    @property
    def get_database_url(self) -> str:
        """Fix Railway Postgres URL schema from postgres:// to postgresql:// if needed."""
        url = self.DATABASE_URL
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return url

    # Redis Cache (Railway Redis or localhost)
    REDIS_URL: Optional[str] = None # e.g. redis://default:password@host:port
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    CACHE_TTL_SECONDS: int = 300

    # Vector DB (Qdrant Cloud Free Tier / Local Qdrant)
    QDRANT_URL: Optional[str] = None # e.g. https://xxxx-xxxx.eu-central.aws.cloud.qdrant.io:6333
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "product_embeddings"
    VECTOR_DIMENSION: int = 64

    # LLM & Cognitive Agent (Groq LPU)
    GROQ_API_KEY: Optional[str] = None
    GROQ_CHAT_MODEL: str = "qwen/qwen3.8-27b"
    GROQ_FAST_MODEL: str = "groq/compound-mini"
    GROQ_TIMEOUT_SECONDS: float = 4.0


    # Recommendation Hyperparameters

    TOP_K_DEFAULT: int = 10
    CF_LATENT_FACTORS: int = 32
    CF_REGULARIZATION: float = 0.05
    CF_ITERATIONS: int = 15
    HYBRID_CF_WEIGHT: float = 0.6
    HYBRID_CB_WEIGHT: float = 0.4
    COLD_START_THRESHOLD: int = 3

    # Implicit Feedback Event Weights
    # Matching the infographic: view, click, add_to_cart, purchase, rating
    EVENT_WEIGHTS: dict = {
        "view": 1.0,
        "click": 2.0,
        "add_to_cart": 3.5,
        "purchase": 5.0,
        "rating": 4.0
    }

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }

settings = Settings()
