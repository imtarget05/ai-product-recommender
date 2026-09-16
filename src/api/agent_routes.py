"""FastAPI Route Handlers for Cognitive Agent & RAG Assistant."""
import time
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.database.session import get_db
from src.database.models import Product, Interaction
from src.api.schemas import (
    AgentSearchRequest,
    AgentSearchResponse,
    AgentChatRequest,
    AgentChatResponse,
    AgentAction,
    AgentExplainRequest,
    AgentExplainResponse,
    RecommendedProduct
)
from src.agent.groq_client import groq_client
from src.agent.rag_search import RAGSearchEngine
from src.agent.shopping_agent import ConversationalShoppingAssistant
from src.agent.explainer import explainer
from src.api.routes import app_state
from src.config import settings

router = APIRouter(prefix="/api/v1/agent", tags=["Cognitive Agent & RAG Assistant"])

# Lazy singletons for agent engines
_rag_engine: Optional[RAGSearchEngine] = None
_shopping_assistant: Optional[ConversationalShoppingAssistant] = None


def get_rag_engine() -> RAGSearchEngine:
    global _rag_engine
    if _rag_engine is None:
        qdrant_store = app_state.get("qdrant_store")
        _rag_engine = RAGSearchEngine(qdrant_store=qdrant_store)
    return _rag_engine


def get_shopping_assistant() -> ConversationalShoppingAssistant:
    global _shopping_assistant
    if _shopping_assistant is None:
        _shopping_assistant = ConversationalShoppingAssistant(rag_engine=get_rag_engine())
    return _shopping_assistant


@router.get("/status")
def get_agent_status():
    """Check availability of Groq LPU and Agent services."""
    return {
        "groq_available": groq_client.is_available(),
        "chat_model": settings.GROQ_CHAT_MODEL,
        "fast_model": settings.GROQ_FAST_MODEL,
        "timeout_seconds": settings.GROQ_TIMEOUT_SECONDS,
        "circuit_breaker_active": groq_client._is_circuit_open()
    }


@router.post("/search", response_model=AgentSearchResponse)
def agent_semantic_search(
    request: AgentSearchRequest,
    db: Session = Depends(get_db)
):
    """Execute RAG Semantic Search with natural language query parsing and vector retrieval."""
    engine = get_rag_engine()
    res = engine.search(
        query=request.query,
        db=db,
        top_k=request.top_k,
        user_id=request.user_id
    )

    products = [
        RecommendedProduct(
            product_id=p["product_id"],
            title=p["title"],
            category=p["category"],
            price=p["price"],
            rating_avg=p["rating_avg"],
            image_url=p.get("image_url"),
            score=p["score"],
            model=p["model"],
            reason=p["reason"]
        )
        for p in res.get("products", [])
    ]

    return AgentSearchResponse(
        query=res["query"],
        intent=res["intent"],
        reply=res["reply"],
        products=products,
        latency_ms=res["latency_ms"]
    )


@router.post("/chat", response_model=AgentChatResponse)
def agent_conversational_chat(
    request: AgentChatRequest,
    db: Session = Depends(get_db)
):
    """Multi-turn Conversational Shopping Assistant with intent detection and tool-calling."""
    assistant = get_shopping_assistant()
    raw_messages = [m.model_dump() for m in request.messages]

    result = assistant.process_chat(
        messages=raw_messages,
        user_id=request.user_id,
        db=db
    )

    suggested = [
        RecommendedProduct(
            product_id=p["product_id"],
            title=p["title"],
            category=p["category"],
            price=p["price"],
            rating_avg=p["rating_avg"],
            image_url=p.get("image_url"),
            score=p["score"],
            model=p["model"],
            reason=p["reason"]
        )
        for p in result.get("suggested_products", [])
    ]

    action_obj = None
    if result.get("action"):
        action_obj = AgentAction(**result["action"])

    return AgentChatResponse(
        reply=result["reply"],
        suggested_products=suggested,
        action=action_obj,
        latency_ms=result["latency_ms"]
    )


@router.post("/explain", response_model=AgentExplainResponse)
def agent_explain_recommendation(
    request: AgentExplainRequest,
    db: Session = Depends(get_db)
):
    """Generate dynamic personalized explanation for why a product was recommended."""
    product = db.query(Product).filter(Product.id == request.product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail=f"Product {request.product_id} not found")

    recent_interactions = None
    if request.user_id:
        recent_interactions = (
            db.query(Interaction)
            .filter(Interaction.user_id == request.user_id)
            .order_by(Interaction.timestamp.desc())
            .limit(5)
            .all()
        )

    product_dict = app_state.get("product_dict") or {
        p.id: p for p in db.query(Product).all()
    }

    explanation_text = explainer.explain(
        product=product,
        recent_interactions=recent_interactions,
        product_dict=product_dict,
        strategy_name=request.strategy
    )

    return AgentExplainResponse(
        product_id=request.product_id,
        user_id=request.user_id,
        explanation=explanation_text
    )
