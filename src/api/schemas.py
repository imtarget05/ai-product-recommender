"""Pydantic Request & Response Schemas for FastAPI Endpoints."""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator

class ProductItem(BaseModel):
    id: int
    title: str
    category: str
    price: float
    description: Optional[str] = None
    image_url: Optional[str] = None
    tags: Optional[str] = None
    rating_avg: float = 0.0
    rating_count: int = 0

    model_config = {"from_attributes": True}

class RecommendedProduct(BaseModel):
    product_id: int
    title: str
    category: str
    price: float
    rating_avg: float
    image_url: Optional[str] = None
    score: float
    model: str
    reason: str

class RecommendationResponse(BaseModel):
    user_id: int
    count: int
    strategy: str
    cached: bool = False
    latency_ms: float
    recommendations: List[RecommendedProduct]

class InteractionCreate(BaseModel):
    user_id: int = Field(..., ge=1, description="ID của người dùng (>= 1)")
    product_id: int = Field(..., ge=1, description="ID của sản phẩm tương tác (>= 1)")
    event_type: str = Field(..., description="view, click, add_to_cart, purchase, rating")
    rating_value: Optional[float] = Field(None, ge=1.0, le=5.0, description="Giá trị đánh giá 1-5 sao")

    @model_validator(mode="after")
    def validate_rating_event(self):
        if self.event_type == "rating" and self.rating_value is None:
            raise ValueError("rating_value is required when event_type is 'rating'")
        return self

class InteractionResponse(BaseModel):
    status: str
    interaction_id: int
    user_id: int
    product_id: int
    event_type: str
    weight: float
    timestamp: str

class UserItem(BaseModel):
    id: int
    username: str
    email: Optional[str] = None
    segment: str
    interaction_count: int = 0

class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    models_ready: bool
    db_connected: bool = True
    total_products: Optional[int] = None
    total_users: Optional[int] = None
    total_interactions: Optional[int] = None

class AdminReloadResponse(BaseModel):
    status: str
    message: str
    total_products: int
    total_interactions: int
    reloaded_at: str


# ==========================================
# Cognitive Agent & RAG Assistant Schemas
# ==========================================

class AgentSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Truy vấn tìm kiếm bằng ngôn ngữ tự nhiên")
    user_id: Optional[int] = Field(None, ge=1, description="ID người dùng để cá nhân hóa (tùy chọn)")
    top_k: int = Field(default=5, ge=1, le=20, description="Số lượng sản phẩm tối đa")


class AgentSearchResponse(BaseModel):
    query: str
    intent: dict
    reply: str
    products: List[RecommendedProduct]
    latency_ms: float


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class AgentAction(BaseModel):
    type: str = Field(..., description="Loại hành động: add_to_cart, search, compare...")
    product_id: Optional[int] = None
    product_title: Optional[str] = None
    detail: Optional[str] = None


class AgentChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1, description="Lịch sử các lượt chat")
    user_id: int = Field(default=1, ge=1, description="ID người dùng đang tương tác")


class AgentChatResponse(BaseModel):
    reply: str
    suggested_products: List[RecommendedProduct] = []
    action: Optional[AgentAction] = None
    latency_ms: float


class AgentExplainRequest(BaseModel):
    product_id: int = Field(..., ge=1, description="ID sản phẩm cần giải thích lý do gợi ý")
    user_id: Optional[int] = Field(None, ge=1, description="ID người dùng nhận gợi ý")
    strategy: str = Field(default="hybrid", description="Chiến lược gợi ý (hybrid, collaborative, content_based...)")


class AgentExplainResponse(BaseModel):
    product_id: int
    user_id: Optional[int] = None
    explanation: str


