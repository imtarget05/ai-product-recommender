"""FastAPI Application Main Entry Point."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import settings
from src.database.session import init_db, SessionLocal
from src.database.models import Product, Interaction
from src.models.hybrid import HybridRecommender
from src.api.routes import router as api_router, app_state
from src.api.agent_routes import router as agent_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler: Initializes DB and pretrains/loads RecSys models."""
    print("🚀 Initializing RecSys-AI Database & Models...")
    init_db()

    # Load data and fit recommendation models
    db = SessionLocal()
    try:
        products = db.query(Product).all()
        interactions = db.query(Interaction).all()

        print(f"📊 Loaded {len(products)} products and {len(interactions)} interactions from database.")

        if products and interactions:
            from src.services.qdrant_service import QdrantVectorStore
            qdrant_store = QdrantVectorStore()

            hybrid = HybridRecommender(
                cf_weight=settings.HYBRID_CF_WEIGHT,
                cb_weight=settings.HYBRID_CB_WEIGHT,
                cold_start_threshold=settings.COLD_START_THRESHOLD,
                qdrant_store=qdrant_store
            )
            hybrid.fit(products, interactions)

            app_state["hybrid_model"] = hybrid
            app_state["product_dict"] = {p.id: p for p in products}
            app_state["qdrant_store"] = qdrant_store
            app_state["ready"] = True
            print("🌟 RecSys-AI Serving Engine is READY!")
        else:
            print("⚠️ No data found in database. Run 'python -m scripts.generate_seed_data' first.")
    finally:
        db.close()

    yield

    print("🛑 Shutting down RecSys-AI Serving Engine.")


def create_app() -> FastAPI:
    """Create and configure FastAPI instance."""
    app = FastAPI(
        title="RecSys-AI Recommendation Engine",
        description="Autonomous Industrial Vision & Recommendation System API (Collaborative, Content-Based, Hybrid)",
        version="0.1.0",
        lifespan=lifespan
    )

    # Enable CORS for frontend and Streamlit
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routers
    app.include_router(api_router)
    app.include_router(agent_router)

    # Request counter for /metrics (excludes the metrics scrape itself).
    from starlette.middleware.base import BaseHTTPMiddleware
    from src.api.routes import bump_request_count

    class _MetricsMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            response = await call_next(request)
            if request.url.path != "/api/v1/metrics":
                bump_request_count()
            return response

    app.add_middleware(_MetricsMiddleware)

    @app.get("/")
    def root():
        return {
            "name": settings.APP_NAME,
            "status": "online",
            "docs_url": "/docs",
            "recommend_endpoint": "/api/v1/recommend/{user_id}",
            "agent_search_endpoint": "/api/v1/agent/search",
            "agent_chat_endpoint": "/api/v1/agent/chat",
            "architecture": "Collaborative Filtering + Content-Based + Hybrid + Reranker + RAG LLM Agent"
        }

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=settings.DEBUG)
