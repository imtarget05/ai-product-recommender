# Technical Design: RAG Semantic Search & Conversational Shopping Assistant

- **Status**: Approved for Planning
- **Author**: Engineering Team
- **Date**: 2026-09-16
- **Target Repository**: `/Users/mainguyenbinhtan/Downloads/RecSys-AI`

---

## 1. Executive Summary & Business Objectives

Following modern e-commerce paradigms established by Amazon Rufus and Sephora Virtual Artist, this system introduces an intelligent cognitive layer to RecSys-AI. It bridges natural language user intentions with vector database retrieval (Qdrant Cloud) and recommendation model ranking.

Key Capabilities:
1. **RAG Natural Language Search**: Allows buyers to express unstructured needs ("tìm laptop gaming mỏng nhẹ dưới 25 triệu cấu hình mạnh"), parses structured purchase constraints, and executes hybrid retrieval (Vector ANN + SQL filters).
2. **Conversational Shopping Assistant**: Multi-turn dialogue capable of asking clarifying questions, comparing specifications of alternative products, and executing cart actions.
3. **Dynamic Recommendation Explanations**: Uses LPU-accelerated LLMs (Groq Llama-3) to produce personalized explanations based on past interaction history and product metadata.
4. **Resilient Architecture**: Zero-downtime offline fallback mechanism ensuring the system continues operating seamlessly even during upstream network disruptions.

---

## 2. System Architecture & Components

```text
                               ┌────────────────────────┐
                               │  User Query / Message  │
                               └───────────┬────────────┘
                                           │
                                           ▼
                            ┌───────────────────────────────┐
                            │      LLM Router & Client      │
                            │   (Groq LPU Llama-3.3-70B)    │
                            └───────┬───────────────┬───────┘
                                    │               │
            [RAG Semantic Search]   │               │   [Shopping Consultation]
                                    ▼               ▼
     ┌────────────────────────────────┐   ┌────────────────────────────────┐
     │      RAG Search Engine         │   │   Conversational Assistant     │
     │  1. LLM Parser -> JSON Intent │   │  • Intent & Memory Tracking    │
     │  2. Qdrant ANN Cosine Search   │   │  • Tool: compare_products      │
     │  3. Hard Filters (Price/Cat)   │   │  • Tool: add_to_cart           │
     └──────────────┬─────────────────┘   └───────────────┬────────────────┘
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                       ┌────────────────────────────────┐
                       │  Dynamic Explanation Generator │
                       │    (Llama-3.1-8b Sub-50ms)     │
                       │  User History + Product Spec   │
                       └───────────────┬────────────────┘
                                       ▼
                       ┌────────────────────────────────┐
                       │   Response Payload: Message,   │
                       │  Product Cards & Cart Action   │
                       └────────────────────────────────┘
```

### Module Responsibilities:

1. **`src/agent/groq_client.py`**:
   - Manages connection to Groq Cloud API using `groq` official SDK.
   - Configurable models:
     - `llama-3.3-70b-versatile` for deep intent reasoning and multi-turn chat.
     - `llama-3.1-8b-instant` for ultra-low latency intent parsing and explanation generation.
   - Enforces timeout (3.5s) and automatic Circuit Breaker fallback.

2. **`src/agent/rag_search.py`**:
   - Parses natural query into structured filters: categories, maximum/minimum price bounds, semantic keywords.
   - Encodes semantic query into dense vectors and queries Qdrant Cloud collection (`product_embeddings`).
   - Applies SQL catalog bounds and generates contextual synthesized summary.

3. **`src/agent/shopping_agent.py`**:
   - Stateful agent loop maintaining conversation context.
   - Internal tool invocation:
     - `tool_search(query, category, max_price)`
     - `tool_compare(product_ids)`
     - `tool_add_to_cart(user_id, product_id)`

4. **`src/agent/explainer.py`**:
   - Synthesizes personalized rationale for Top-K candidate items using user interaction history and product metadata.

---

## 3. Data Models & API Specifications

### New Schemas (`src/api/schemas.py` additions):

```python
class AgentSearchRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Natural language search query")
    user_id: Optional[int] = Field(default=None, description="Optional user ID for personalized re-ranking")
    top_k: int = Field(default=5, ge=1, le=20)

class AgentSearchResponse(BaseModel):
    query: str
    intent_summary: str
    reply: str
    products: List[RecommendedProduct]
    latency_ms: float

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str

class AgentChatRequest(BaseModel):
    user_id: int = Field(..., ge=1)
    messages: List[ChatMessage]

class AgentAction(BaseModel):
    action_type: str # "add_to_cart" | "view_product" | "compare"
    product_id: Optional[int] = None
    product_ids: Optional[List[int]] = None

class AgentChatResponse(BaseModel):
    reply: str
    suggested_products: List[RecommendedProduct] = []
    action: Optional[AgentAction] = None
    latency_ms: float
```

### Endpoints (`src/api/agent_routes.py`):
- `POST /api/v1/agent/search`
- `POST /api/v1/agent/chat`
- `POST /api/v1/agent/explain`

---

## 4. Fault Tolerance & Fallback Engine

When the Groq API key is missing or upstream network is unavailable:
1. **Search Fallback**:
   - Heuristic regex extracts budget ("dưới X triệu" -> `max_price = X * 1,000,000`).
   - Query embedding matches against Qdrant Cloud or local Cosine Similarity.
   - Generates deterministic natural summary.
2. **Chat Fallback**:
   - Rule-based keyword responder matching "mua", "giá", "tìm", "so sánh".
   - Executes product lookup directly from the database and returns structured cards.
3. **Explanation Fallback**:
   - Category and tag matching heuristics generating formatted rationales.

---

## 5. UI Integration (`src/ui/app.py`)

New tab added: **"🤖 Trợ lý AI Mua sắm (AI Shopping Assistant)"**:
- Streamlit chat interface (`st.chat_message`, `st.chat_input`).
- Live rendering of product cards directly beneath assistant responses.
- Interactive action buttons: "👁️ Xem chi tiết", "🛒 Thêm vào giỏ", "💳 Mua ngay" triggering real-time feedback loop.

---

## 6. Verification Plan

1. **Unit Tests (`tests/test_agent.py`)**:
   - Intent parsing accuracy on price and category constraints.
   - RAG retrieval returning relevant products.
   - Multi-turn conversation handling.
   - Offline fallback behavior when API key is unset or mocked to error.
2. **Integration Tests**:
   - `POST /api/v1/agent/search` endpoint status 200 with non-empty products.
   - `POST /api/v1/agent/chat` handling tool invocation and returning suggested products.
3. **Live UI Verification**:
   - Chatting with shopping assistant via Streamlit.
   - Executing search query "tìm laptop gaming dưới 30 triệu".
