"""FastAPI Route Handlers for Recommendation System Serving."""
import time
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.database.session import get_db
from src.database.models import Product, User, Interaction
from src.api.schemas import (
    ProductItem,
    RecommendedProduct,
    RecommendationResponse,
    InteractionCreate,
    InteractionResponse,
    UserItem,
    HealthResponse
)
from src.api.cache import cache_manager
from src.ranking.reranker import ProductReranker
from src.config import settings

router = APIRouter(prefix="/api/v1")

# Global singleton state loaded during startup
app_state = {
    "hybrid_model": None,
    "product_dict": {},
    "reranker": ProductReranker(),
    "ready": False
}


@router.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    """Check health and system readiness."""
    total_prods = db.query(Product).count()
    total_users = db.query(User).count()
    total_inters = db.query(Interaction).count()

    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version="0.1.0",
        models_ready=app_state["ready"],
        total_products=total_prods,
        total_users=total_users,
        total_interactions=total_inters
    )


@router.get("/recommend/{user_id}", response_model=RecommendationResponse)
def get_recommendations(
    user_id: int,
    top_k: int = Query(default=10, ge=1, le=50),
    strategy: str = Query(default="hybrid", pattern="^(hybrid|collaborative|content_based|popularity)$"),
    bypass_cache: bool = Query(default=False),
    db: Session = Depends(get_db)
):
    """Retrieve Top-K recommended products for a user.

    Strategies supported:
    - `hybrid`: Collaborative Filtering + Content-Based + Cold-start fallback
    - `collaborative`: Latent factor Matrix Factorization
    - `content_based`: Metadata & Text Embeddings
    - `popularity`: Highest overall popularity & sales
    """
    start_time = time.time()

    if not app_state["ready"] or not app_state["hybrid_model"]:
        raise HTTPException(status_code=503, detail="Models are still initializing. Please retry in a few seconds.")

    # Check cache first
    cache_key = f"rec:user:{user_id}:strat:{strategy}:k:{top_k}"
    if not bypass_cache:
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            cached_result["cached"] = True
            cached_result["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            return cached_result

    # Select model strategy
    hybrid = app_state["hybrid_model"]
    if strategy == "hybrid":
        candidates = hybrid.recommend(user_id, top_k=top_k * 2)
    elif strategy == "collaborative":
        candidates = hybrid.cf_model.recommend(user_id, top_k=top_k * 2)
    elif strategy == "content_based":
        candidates = hybrid.content_model.recommend(user_id, top_k=top_k * 2)
    elif strategy == "popularity":
        candidates = hybrid.popularity_model.recommend(user_id, top_k=top_k * 2)
    else:
        candidates = hybrid.recommend(user_id, top_k=top_k * 2)

    # If pure model returned empty (e.g., pure CF on a cold-start user), fallback gracefully
    if not candidates:
        candidates = hybrid.popularity_model.recommend(user_id, top_k=top_k * 2)
        for c in candidates:
            c["reason"] = "Sản phẩm xu hướng nổi bật (Cold-start fallback)"

    # Get recent purchases to avoid recommending already bought items
    recent_purchases = (
        db.query(Interaction.product_id)
        .filter(Interaction.user_id == user_id, Interaction.event_type == "purchase")
        .all()
    )
    purchased_ids = [p[0] for p in recent_purchases]

    # Candidate Reranking (Phase 4: Multi-objective scoring & diversity)
    reranked = app_state["reranker"].rerank(
        candidates=candidates,
        product_dict=app_state["product_dict"],
        recently_purchased_ids=purchased_ids,
        top_k=top_k
    )

    recommended_items = [
        RecommendedProduct(
            product_id=item["product_id"],
            title=item["title"],
            category=item["category"],
            price=item["price"],
            rating_avg=item["rating_avg"],
            image_url=item["image_url"],
            score=item["final_score"],
            model=item["model"],
            reason=item["reason"]
        )
        for item in reranked
    ]

    latency = round((time.time() - start_time) * 1000, 2)
    response_data = {
        "user_id": user_id,
        "count": len(recommended_items),
        "strategy": strategy,
        "cached": False,
        "latency_ms": latency,
        "recommendations": [item.model_dump() for item in recommended_items]
    }

    # Store in cache
    cache_manager.set(cache_key, response_data, ttl_seconds=settings.CACHE_TTL_SECONDS)
    return response_data


@router.get("/similar-products/{product_id}")
def get_similar_products(product_id: int, top_k: int = Query(default=6, ge=1, le=20)):
    """Get similar products based on content embeddings and co-interactions."""
    if not app_state["ready"] or not app_state["hybrid_model"]:
        raise HTTPException(status_code=503, detail="Models are initializing.")

    hybrid = app_state["hybrid_model"]
    sim_items = hybrid.similar_items(product_id, top_k=top_k)

    results = []
    for item in sim_items:
        prod = app_state["product_dict"].get(item["product_id"])
        if prod:
            results.append({
                "product_id": prod.id,
                "title": prod.title,
                "category": prod.category,
                "price": prod.price,
                "rating_avg": prod.rating_avg,
                "image_url": prod.image_url,
                "similarity_score": item["score"],
                "reason": item["reason"]
            })
    return {"product_id": product_id, "similar_products": results}


@router.post("/interact", response_model=InteractionResponse)
def record_interaction(interaction: InteractionCreate, db: Session = Depends(get_db)):
    """Record real-time user behavior (view, click, add_to_cart, purchase, rating).
    Automatically invalidates cache for closed-loop real-time updates.
    """
    valid_events = ["view", "click", "add_to_cart", "purchase", "rating"]
    if interaction.event_type not in valid_events:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid event_type. Must be one of {valid_events}"
        )

    # Check existence
    user = db.query(User).filter(User.id == interaction.user_id).first()
    product = db.query(Product).filter(Product.id == interaction.product_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    weight = settings.EVENT_WEIGHTS.get(interaction.event_type, 1.0)
    if interaction.rating_value is not None and interaction.event_type == "rating":
        weight = interaction.rating_value

    new_inter = Interaction(
        user_id=interaction.user_id,
        product_id=interaction.product_id,
        event_type=interaction.event_type,
        rating_value=interaction.rating_value,
        weight=weight,
        timestamp=datetime.utcnow()
    )
    db.add(new_inter)
    db.commit()
    db.refresh(new_inter)

    # Invalidate cache for this user immediately
    cache_manager.invalidate_user(interaction.user_id)

    # If hybrid model is in memory, update interaction count
    if app_state["hybrid_model"]:
        app_state["hybrid_model"].user_interaction_counts[interaction.user_id] += 1

    return InteractionResponse(
        status="recorded",
        interaction_id=new_inter.id,
        user_id=new_inter.user_id,
        product_id=new_inter.product_id,
        event_type=new_inter.event_type,
        weight=new_inter.weight,
        timestamp=new_inter.timestamp.isoformat()
    )


@router.get("/products", response_model=List[ProductItem])
def list_products(
    category: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Retrieve catalog products with optional category and search filters."""
    query = db.query(Product)
    if category:
        query = query.filter(Product.category == category)
    if search:
        query = query.filter(Product.title.ilike(f"%{search}%"))
    return query.limit(limit).all()


@router.get("/users", response_model=List[UserItem])
def list_users(limit: int = 50, db: Session = Depends(get_db)):
    """Retrieve users with their total interaction counts."""
    users = db.query(User).limit(limit).all()
    user_ids = [u.id for u in users]

    counts = dict(
        db.query(Interaction.user_id, func.count(Interaction.id))
        .filter(Interaction.user_id.in_(user_ids))
        .group_by(Interaction.user_id)
        .all()
    )

    return [
        UserItem(
            id=u.id,
            username=u.username,
            email=u.email,
            segment=u.segment or "general",
            interaction_count=counts.get(u.id, 0)
        )
        for u in users
    ]
