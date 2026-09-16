"""ETL Pipeline for RetailRocket E-Commerce Dataset (Kaggle).
Processes:
1. events.csv (view, addtocart, transaction)
2. item_properties.csv (metadata)
3. Computes dense embeddings and syncs with PostgreSQL & Qdrant Cloud!
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from src.config import settings
from src.database.session import init_db, get_db_context
from src.database.models import Product, User, Interaction
from src.features.text_embedder import ItemEmbedder
from src.services.qdrant_service import QdrantVectorStore

RETAILROCKET_DIR = settings.RAW_DATA_DIR / "retailrocket"

def generate_mock_retailrocket_if_missing(max_events: int = 5000):
    """If Kaggle dataset is not yet downloaded, create sample RetailRocket formatted files."""
    RETAILROCKET_DIR.mkdir(parents=True, exist_ok=True)
    events_file = RETAILROCKET_DIR / "events.csv"
    items_file = RETAILROCKET_DIR / "item_properties.csv"

    if not events_file.exists():
        print("ℹ️ RetailRocket files not found in data/raw/retailrocket/.")
        print("💡 Generating a compliant RetailRocket sample format for testing...")
        import random
        now_ms = int(datetime.utcnow().timestamp() * 1000)

        # 50 unique items, 100 unique visitors
        events_data = []
        event_types = ["view", "addtocart", "transaction"]
        weights = [0.7, 0.2, 0.1]

        for _ in range(max_events):
            t = now_ms - random.randint(0, 30 * 86400 * 1000)
            vid = random.randint(1000, 1100)
            iid = random.randint(5001, 5050)
            ev = random.choices(event_types, weights=weights, k=1)[0]
            txid = random.randint(10000, 99999) if ev == "transaction" else None
            events_data.append({"timestamp": t, "visitorid": vid, "event": ev, "itemid": iid, "transactionid": txid})

        pd.DataFrame(events_data).to_csv(events_file, index=False)

        # Item properties
        categories = ["Electronics", "Fashion Apparel", "Home Living", "Sporting Goods", "Books"]
        props_data = []
        for iid in range(5001, 5051):
            cat = random.choice(categories)
            props_data.append({"timestamp": now_ms, "itemid": iid, "property": "categoryid", "value": cat})
            props_data.append({"timestamp": now_ms, "itemid": iid, "property": "title", "value": f"{cat} Item #{iid}"})

        pd.DataFrame(props_data).to_csv(items_file, index=False)
        print(f"✅ Generated sample RetailRocket dataset at {RETAILROCKET_DIR}")


def ingest_retailrocket(limit_interactions: int = 5000):
    """Main ETL ingestion for RetailRocket."""
    print("=" * 65)
    print("🚀 INGESTING RETAILROCKET DATASET -> POSTGRES & QDRANT CLOUD")
    print("=" * 65)

    generate_mock_retailrocket_if_missing()

    events_path = RETAILROCKET_DIR / "events.csv"
    items_path = RETAILROCKET_DIR / "item_properties.csv"

    print("📖 Loading events and item properties...")
    events_df = pd.read_csv(events_path)
    print(f"Total raw events: {len(events_df)}")

    # Load item categories if available
    item_category_map = {}
    item_title_map = {}
    if items_path.exists():
        props_df = pd.read_csv(items_path)
        cat_rows = props_df[props_df["property"] == "categoryid"]
        for _, row in cat_rows.iterrows():
            item_category_map[int(row["itemid"])] = str(row["value"])

        title_rows = props_df[props_df["property"] == "title"]
        for _, row in title_rows.iterrows():
            item_title_map[int(row["itemid"])] = str(row["value"])

    # Clean and filter events
    events_df = events_df.dropna(subset=["visitorid", "itemid", "event"]).head(limit_interactions)
    events_df["itemid"] = events_df["itemid"].astype(int)
    events_df["visitorid"] = events_df["visitorid"].astype(int)

    unique_item_ids = sorted(events_df["itemid"].unique())
    unique_user_ids = sorted(events_df["visitorid"].unique())

    print(f"Found {len(unique_item_ids)} unique items and {len(unique_user_ids)} unique users.")

    init_db()
    with get_db_context() as db:
        # Ingest Products
        print("📦 Upserting Products into Database...")
        products = []
        for iid in unique_item_ids:
            iid_int = int(iid)
            cat = item_category_map.get(iid_int, "Retail Catalog")
            title = item_title_map.get(iid_int, f"Retail Product #{iid_int}")
            price = float((abs(hash(str(iid_int))) % 900 + 100) * 1000)

            prod = db.query(Product).filter(Product.id == iid_int).first()
            if not prod:
                prod = Product(
                    id=iid_int,
                    title=title,
                    category=cat,
                    price=price,
                    description=f"RetailRocket catalog product {title}. Category: {cat}.",
                    tags=f"{cat.lower()}, retail, ecommerce",
                    image_url="https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=500",
                    rating_avg=4.5,
                    rating_count=20
                )
                db.add(prod)
            products.append(prod)
        db.commit()

        # Ingest Users
        print("👥 Upserting Users...")
        for uid in unique_user_ids:
            uid_int = int(uid)
            user = db.query(User).filter(User.id == uid_int).first()
            if not user:
                user = User(
                    id=uid_int,
                    username=f"retail_visitor_{uid_int}",
                    email=f"visitor{uid_int}@retailrocket.net",
                    segment="ecommerce_shopper"
                )
                db.add(user)
        db.commit()

        # Ingest Interactions
        print("⚡ Ingesting Events (view, addtocart, transaction)...")
        event_weight_map = {
            "view": 1.0,
            "addtocart": 3.5,
            "transaction": 5.0
        }

        # Clear old for this batch
        db.query(Interaction).filter(Interaction.user_id.in_(unique_user_ids)).delete(synchronize_session=False)

        for _, row in events_df.iterrows():
            ev = str(row["event"]).lower()
            ev_type = "add_to_cart" if ev == "addtocart" else ("purchase" if ev == "transaction" else "view")
            weight = event_weight_map.get(ev, 1.0)
            t_stamp = datetime.fromtimestamp(row["timestamp"] / 1000.0) if row["timestamp"] > 1e11 else datetime.utcnow()

            inter = Interaction(
                user_id=int(row["visitorid"]),
                product_id=int(row["itemid"]),
                event_type=ev_type,
                weight=weight,
                timestamp=t_stamp
            )
            db.add(inter)
        db.commit()

        # Compute Embeddings and Upsert to Qdrant inside active session
        print("🧠 Generating dense embeddings for RetailRocket catalog...")
        embedder = ItemEmbedder(embedding_dim=settings.VECTOR_DIMENSION)
        embeddings = embedder.fit_transform(products)

        print("🌐 Syncing vectors with Qdrant Vector DB...")
        qdrant = QdrantVectorStore(dim=settings.VECTOR_DIMENSION)
        metas = [
            {"title": p.title, "category": p.category, "price": p.price}
            for p in products
        ]
        qdrant.upsert_products(unique_item_ids, embeddings, metas)

    print("=" * 65)
    print("🎉 RETAILROCKET INGESTION & QDRANT VECTOR SYNC COMPLETE!")
    print(f" - Products Indexed: {len(products)}")
    print(f" - Users Ingested: {len(unique_user_ids)}")
    print(f" - Events Recorded: {len(events_df)}")
    print(f" - Qdrant Collection: '{settings.QDRANT_COLLECTION_NAME}'")
    print("=" * 65)


if __name__ == "__main__":
    ingest_retailrocket()
