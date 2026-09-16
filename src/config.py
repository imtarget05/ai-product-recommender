"""Centralized Configuration for RecSys-AI."""
import os
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

    # Database
    DATABASE_URL: str = f"sqlite:///{PROJECT_ROOT}/data/recsys.db"

    # Redis Cache
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    CACHE_TTL_SECONDS: int = 300

    # Recommendation Hyperparameters
    TOP_K_DEFAULT: int = 10
    CF_LATENT_FACTORS: int = 32
    CF_REGULARIZATION: float = 0.05
    CF_ITERATIONS: int = 15
    HYBRID_CF_WEIGHT: float = 0.6
    HYBRID_CB_WEIGHT: float = 0.4
    COLD_START_THRESHOLD: int = 3

    # Implicit Feedback Event Weights
    # As shown in the infographic: view, click, add_to_cart, purchase, rating
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
