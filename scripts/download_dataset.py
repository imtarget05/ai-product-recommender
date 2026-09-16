"""Dataset Downloader and Importer for RecSys-AI.
Supports:
1. MovieLens-100k / MovieLens-Latest-Small (Free direct zip download, zero credentials needed)
2. RetailRocket Kaggle Dataset (events.csv, item_properties.csv)
"""
import os
import sys
import zipfile
import urllib.request
from pathlib import Path
import pandas as pd
from src.config import settings
from src.database.session import init_db, get_db_context
from src.database.models import Product, User, Interaction

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def download_and_ingest_movielens():
    """Download MovieLens-Latest-Small and ingest into RecSys-AI schema."""
    raw_dir = settings.RAW_DATA_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ml-latest-small.zip"
    extract_dir = raw_dir / "ml-latest-small"

    if not (extract_dir / "ratings.csv").exists():
        print(f"📥 Downloading MovieLens dataset from {MOVIELENS_URL}...")
        urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
        print("📦 Extracting zip file...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
        print("✅ Download & extraction complete!")

    print("📊 Loading CSVs into pandas...")
    movies_df = pd.read_csv(extract_dir / "movies.csv")
    ratings_df = pd.read_csv(extract_dir / "ratings.csv")

    init_db()
    with get_db_context() as db:
        print(f"Ingesting {len(movies_df)} movies into products table...")
        # Clean current
        db.query(Interaction).delete()
        db.query(Product).delete()
        db.query(User).delete()
        db.commit()

        # Ingest top 500 movies to keep fast
        subset_movies = movies_df.head(500)
        valid_movie_ids = set(subset_movies["movieId"])

        for _, row in subset_movies.iterrows():
            genres = str(row["genres"]).replace("|", ", ")
            p = Product(
                id=int(row["movieId"]),
                title=str(row["title"]),
                category=genres.split(",")[0] if genres else "General",
                price=round(float(abs(hash(str(row["title"]))) % 800000 + 100000), -3),
                tags=genres,
                description=f"Movie: {row['title']} (Genres: {genres})",
                image_url="https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500"
            )
            db.add(p)
        db.commit()

        # Ingest users and ratings
        subset_ratings = ratings_df[ratings_df["movieId"].isin(valid_movie_ids)].head(5000)
        user_ids = sorted(list(set(subset_ratings["userId"])))

        print(f"Ingesting {len(user_ids)} users...")
        for uid in user_ids:
            u = User(id=int(uid), username=f"movielens_user_{uid}", segment="cinephile")
            db.add(u)
        db.commit()

        print(f"Ingesting {len(subset_ratings)} interactions...")
        for _, row in subset_ratings.iterrows():
            rating = float(row["rating"])
            inter = Interaction(
                user_id=int(row["userId"]),
                product_id=int(row["movieId"]),
                event_type="rating",
                rating_value=rating,
                weight=rating,
                timestamp=pd.to_datetime(row["timestamp"], unit="s")
            )
            db.add(inter)
        db.commit()

    print("🎉 MovieLens dataset successfully ingested!")

if __name__ == "__main__":
    download_and_ingest_movielens()
