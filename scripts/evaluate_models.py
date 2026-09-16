"""Benchmarking & Offline Evaluation Script.
Evaluates Popularity, Content-Based, Collaborative, Hybrid, and Full Reranked Pipeline
using strict Global Cutoff Timestamp (T_cutoff) to prevent future data leakage.
"""
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

def run_evaluation(test_ratio: float = 0.2, top_k: int = 5):
    print("=" * 70)
    print("🎯 RECSYS-AI MODEL BENCHMARK & OFFLINE EVALUATION (STRICT TEMPORAL)")
    print("=" * 70)

    with get_db_context() as db:
        products = db.query(Product).all()
        interactions = (
            db.query(Interaction)
            .order_by(Interaction.timestamp.asc())
            .all()
        )

        if not products or not interactions:
            print("❌ No data available for evaluation. Run seed data script first.")
            return

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
        print(" - Strict temporal split ensures zero future data leakage.")
        print(" - Full Pipeline balances relevance, Bayesian quality ratings, and category diversity.")
        print("=" * 70)

if __name__ == "__main__":
    run_evaluation()

