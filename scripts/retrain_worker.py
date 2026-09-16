"""Batch Retraining Worker for RecSys-AI.
Periodically trains recommendation models on accumulated database interactions,
persists model artifacts, and triggers zero-downtime hot-reload on serving API.
"""
import os
import sys
import time
import requests
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.session import get_db_context
from src.database.models import Product, Interaction
from src.models.hybrid import HybridRecommender
from src.config import settings

def run_retrain(api_base_url: str = "http://127.0.0.1:8000"):
    """Execute model training and notify serving API."""
    print("=" * 65)
    print(f"🔄 RECSYS-AI BATCH RETRAIN WORKER [{datetime.utcnow().isoformat()}]")
    print("=" * 65)

    start_t = time.time()
    with get_db_context() as db:
        products = db.query(Product).all()
        interactions = db.query(Interaction).all()

        if not products:
            print("❌ No products found in database to retrain.")
            return False

        print(f"📦 Loaded {len(products)} products, {len(interactions)} interactions.")

        # Train new hybrid model instance
        hybrid = HybridRecommender(
            cf_weight=settings.HYBRID_CF_WEIGHT,
            cb_weight=settings.HYBRID_CB_WEIGHT,
            cold_start_threshold=settings.COLD_START_THRESHOLD
        )
        print("🧠 Fitting sub-models (Popularity, Content-Based, Latent CF)...")
        hybrid.fit(products, interactions)

        # Persist embedder
        print(f"💾 Saving embeddings to {settings.EMBEDDINGS_DIR}...")
        hybrid.content_model.embedder.save(settings.EMBEDDINGS_DIR)

    elapsed = round(time.time() - start_t, 2)
    print(f"✅ Retraining complete in {elapsed}s.")

    # Hot-reload serving API
    reload_url = f"{api_base_url.rstrip('/')}/api/v1/admin/reload-model"
    print(f"📡 Triggering zero-downtime hot reload via {reload_url}...")
    try:
        resp = requests.post(reload_url, timeout=10.0)
        if resp.status_code == 200:
            print(f"🎉 Serving API successfully reloaded: {resp.json().get('message')}")
        else:
            print(f"⚠️ Serving API reload responded with HTTP {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"ℹ️ Could not notify API ({e}). If API is not running, models will load on next startup.")

    return True

if __name__ == "__main__":
    api_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    run_retrain(api_base_url=api_url)
