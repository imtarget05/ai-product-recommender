"""Benchmarking & Offline Evaluation Script.
Evaluates Popularity, Content-Based, Collaborative, and Hybrid recommenders
using train/test interaction split.
"""
from collections import defaultdict
from src.database.session import get_db_context
from src.database.models import Product, Interaction
from src.models.baseline import PopularityRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender
from src.models.hybrid import HybridRecommender
from src.evaluation.metrics import evaluate_recommender

def run_evaluation(test_ratio: float = 0.2, top_k: int = 5):
    print("=" * 65)
    print("🎯 RECSYS-AI MODEL BENCHMARK & OFFLINE EVALUATION")
    print("=" * 65)

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
        print(f"Loaded {total_catalog_size} catalog products and {len(interactions)} interactions.")

        # Temporal split by user: last 20% interactions per user placed in test set
        user_inter_map = defaultdict(list)
        for inter in interactions:
            user_inter_map[inter.user_id].append(inter)

        train_interactions = []
        test_user_items = defaultdict(set)

        for uid, user_events in user_inter_map.items():
            if len(user_events) < 5:
                # Keep in train to allow model to learn anything
                train_interactions.extend(user_events)
                continue

            split_idx = int(len(user_events) * (1.0 - test_ratio))
            train_interactions.extend(user_events[:split_idx])

            # Test set contains items user interacted with in test phase
            for inter in user_events[split_idx:]:
                # Only evaluate on positive signals (click, cart, purchase, high rating)
                if inter.event_type in ["click", "add_to_cart", "purchase"] or (inter.rating_value and inter.rating_value >= 4.0):
                    test_user_items[uid].add(inter.product_id)

        print(f"Split completed: {len(train_interactions)} train interactions, {len(test_user_items)} test users.")
        print("-" * 65)

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

        print("\n" + "=" * 65)
        print(f"{'Model':<25} | {'P@5':<8} | {'R@5':<8} | {'NDCG@5':<8} | {'Coverage':<8}")
        print("-" * 65)
        for name, m in results.items():
            print(f"{name:<25} | {m[f'Precision@{top_k}']:<8.4f} | {m[f'Recall@{top_k}']:<8.4f} | {m[f'NDCG@{top_k}']:<8.4f} | {m['CatalogCoverage']:<8.4f}")
        print("=" * 65)
        print("💡 Insights:")
        print(" - Hybrid combines high precision from Content-Based with diversity from Collaborative Filtering.")
        print(" - Popularity offers a strong baseline but suffers from lower catalog coverage.")
        print("=" * 65)

if __name__ == "__main__":
    run_evaluation()
