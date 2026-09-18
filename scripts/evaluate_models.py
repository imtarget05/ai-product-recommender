"""Benchmarking & Offline Evaluation Script.
Evaluates Popularity, Content-Based, Collaborative, Hybrid, and Full Reranked Pipeline
using strict Global Cutoff Timestamp (T_cutoff) to prevent future data leakage.
"""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import os
import sys
from collections import defaultdict

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.session import get_db_context
from src.database.models import Product, Interaction
from src.models.baseline import PopularityRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender
from src.models.hybrid import HybridRecommender
from src.ranking.reranker import ProductReranker
from src.evaluation.metrics import evaluate_recommender

def run_evaluation(test_ratio: float = 0.2, top_k: int = 5, output=None,
                   dataset_label: str = "local snapshot; provenance unverified"):
    if not 0 < test_ratio < 1 or top_k < 1:
        raise ValueError("Require 0 < test_ratio < 1 and top_k >= 1")
    print("=" * 70)
    print("🎯 RECSYS-AI MODEL BENCHMARK & OFFLINE EVALUATION (STRICT TEMPORAL)")
    print("=" * 70)

    with get_db_context() as db:
        products = db.query(Product).order_by(Product.id).all()
        interactions = (
            db.query(Interaction)
            .order_by(Interaction.timestamp.asc(), Interaction.id.asc())
            .all()
        )

        if not products or not interactions:
            raise ValueError("No data available for evaluation; seed an isolated database first")

        total_catalog_size = len(products)
        product_dict = {p.id: p for p in products}
        print(f"Loaded {total_catalog_size} catalog products and {len(interactions)} interactions.")

        # Strict Global Cutoff Timestamp Split (prevents future leakage across users)
        split_idx = int(len(interactions) * (1.0 - test_ratio))
        cutoff_timestamp = interactions[split_idx].timestamp
        print(f"Global Cutoff Timestamp (T_cutoff): {cutoff_timestamp}")

        train_interactions = [i for i in interactions if i.timestamp <= cutoff_timestamp]
        test_events = [i for i in interactions if i.timestamp > cutoff_timestamp]

        test_user_items = defaultdict(set)
        for inter in test_events:
            # Positive signals in test phase
            if inter.event_type in ["click", "add_to_cart", "purchase"] or (inter.rating_value and inter.rating_value >= 4.0):
                test_user_items[inter.user_id].add(inter.product_id)

        if not test_user_items:
            raise ValueError("Temporal holdout contains no positive evaluation users")

        print(f"Split completed: {len(train_interactions)} train interactions, {len(test_user_items)} active test users.")
        print("-" * 70)

        # Initialize models
        models = [
            PopularityRecommender(),
            ContentBasedRecommender(),
            CollaborativeRecommender(n_factors=16),
            HybridRecommender(cf_weight=0.6, cb_weight=0.4)
        ]

        results = {}
        for model in models:
            print(f"⚡ Training & evaluating: {model.name}...")
            model.fit(products, train_interactions)
            metrics = evaluate_recommender(
                recommender=model,
                test_user_items=test_user_items,
                k=top_k,
                total_catalog_size=total_catalog_size
            )
            results[model.name] = metrics

        # Evaluate Full Hybrid + Reranker Pipeline
        print(f"⚡ Evaluating Full Pipeline: Hybrid + Multi-Objective Reranker...")
        reranker = ProductReranker()
        pipeline_metrics = evaluate_recommender(
            recommender=models[-1], # Hybrid
            test_user_items=test_user_items,
            k=top_k,
            total_catalog_size=total_catalog_size,
            reranker=reranker,
            product_dict=product_dict
        )
        results["Pipeline (Hybrid + Reranker)"] = pipeline_metrics

        print("\n" + "=" * 70)
        print(f"{'Model / Pipeline':<30} | {'P@5':<8} | {'R@5':<8} | {'NDCG@5':<8} | {'Coverage':<8}")
        print("-" * 70)
        for name, m in results.items():
            print(f"{name:<30} | {m[f'Precision@{top_k}']:<8.4f} | {m[f'Recall@{top_k}']:<8.4f} | {m[f'NDCG@{top_k}']:<8.4f} | {m['CatalogCoverage']:<8.4f}")
        print("=" * 70)
        print("💡 Production Insights:")
        print(" - Events use a temporal split; catalog metadata is not historical.")
        print(" - Full Pipeline balances relevance, Bayesian quality ratings, and category diversity.")
        print("=" * 70)

        # Hash evaluated inputs without publishing user-level records or DB credentials.
        snapshot = {
            "products": [[getattr(p, col.name) for col in Product.__table__.columns]
                         for p in products],
            "interactions": [[getattr(i, col.name) for col in Interaction.__table__.columns]
                             for i in interactions],
        }
        report = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "label": dataset_label,
                "sha256": hashlib.sha256(json.dumps(snapshot, default=str, sort_keys=True)
                                         .encode()).hexdigest(),
                "products": len(products), "interactions": len(interactions),
                "train_events": len(train_interactions), "test_events": len(test_events),
                "evaluated_users": len(test_user_items),
            },
            "protocol": {
                "top_k": top_k, "requested_test_ratio": test_ratio,
                "cutoff_timestamp": cutoff_timestamp.isoformat(),
                "split": "train <= cutoff; test > cutoff; timestamp ties stay in train",
                "positives": "click, add_to_cart, purchase, or rating >= 4",
                "exclude_interacted": True,
                "content_backend": "local embeddings; no Qdrant store supplied",
                "cf_factors": 16, "hybrid_cf_factors": models[-1].cf_model.n_factors,
                "hybrid_weights": {"cf": 0.6, "cb": 0.4},
            },
            "environment": {
                "python": platform.python_version(), "platform": platform.platform(),
                "packages": {name: version(name) for name in
                             ("numpy", "scipy", "scikit-learn", "sqlalchemy")},
            },
            "limitations": [
                "Local snapshot provenance is not independently verified; not a RetailRocket benchmark.",
                "Catalog metadata/ratings are current, not cutoff snapshots; future leakage remains possible.",
                "Reranker freshness uses wall-clock time, not the historical cutoff.",
                "Repeated train items remain in test relevance although recommendations exclude them.",
                "Pipeline does not pass purchase history to reranker; not identical to API serving.",
                "Binary metrics only; graded-NDCG helper is not wired into this evaluation.",
                "Standalone CF uses 16 factors; Hybrid uses configured factors; not a controlled ablation.",
            ],
            "results": results,
        }
        if output is not None:
            destination = Path(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                              allow_nan=False) + "\n", encoding="utf-8")
            print(f"Saved evaluation report: {destination.resolve()}")

        # Immutable serving bundle: artifacts + manifest written atomically so
        # replicas load the same version instead of training process-local models.
        bundle_dir = Path(os.getenv("MODEL_BUNDLE_PATH", "data/model_bundle"))
        try:
            from src.serving.model_bundle import write_model_bundle

            bundle_version = os.getenv("MODEL_VERSION", "recsys-hybrid-v001")
            artifacts = {
                "evaluation_report.json": json.dumps(
                    report, ensure_ascii=False, allow_nan=False, indent=2
                ).encode("utf-8"),
            }
            schema = {
                "recommendation_fields": ["product_id", "score", "model", "reason"],
                "strategies": ["hybrid", "collaborative", "content_based", "popularity"],
                "hybrid_weights": {"cf": 0.6, "cb": 0.4},
            }
            write_model_bundle(
                dest=bundle_dir,
                version=bundle_version,
                dataset_fingerprint=report["dataset"]["sha256"],
                schema=schema,
                artifacts=artifacts,
                payload={"hybrid_cf_factors": models[-1].cf_model.n_factors,
                         "cf_factors": 16},
            )
            print(f"Wrote immutable model bundle: {bundle_dir}")
        except Exception as exc:  # bundle export must not break evaluation
            print(f"Bundle export skipped: {exc}")
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset-label", default="local snapshot; provenance unverified")
    args = parser.parse_args()
    run_evaluation(args.test_ratio, args.top_k, args.output, args.dataset_label)

