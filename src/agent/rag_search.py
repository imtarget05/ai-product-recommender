"""RAG Semantic Search Engine.
Parses natural language intent, retrieves vector candidates from Qdrant Cloud,
applies catalog attribute filters, and synthesizes answers via Groq LPU.
"""
import re
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session

from src.database.models import Product
from src.features.text_embedder import ItemEmbedder
from src.services.qdrant_service import QdrantVectorStore
from src.agent.groq_client import groq_client
from src.config import settings

class RAGSearchEngine:
    """Semantic Natural Language Search combining Vector Retrieval and LLM Intent Parsing."""

    def __init__(
        self,
        embedder: Optional[ItemEmbedder] = None,
        qdrant_store: Optional[QdrantVectorStore] = None
    ):
        self.embedder = embedder or ItemEmbedder(embedding_dim=settings.VECTOR_DIMENSION)
        self.qdrant_store = qdrant_store or QdrantVectorStore()

    def parse_intent(self, query: str, known_categories: List[str]) -> Dict[str, Any]:
        """Parse natural language query into structured criteria using Groq LLM with regex fallback."""
        # 1. Heuristic Regex parsing as baseline
        fallback_intent = self._heuristic_parse(query, known_categories)

        if not groq_client.is_available():
            return fallback_intent

        system_prompt = (
            "Bạn là trợ lý trích xuất ý định tìm kiếm sản phẩm e-commerce. "
            "Hãy phân tích câu hỏi của người dùng và trả về DUY NHẤT một JSON hợp lệ (không markdown block) "
            "với định dạng: "
            '{"category": string hoặc null, "min_price": number hoặc null, "max_price": number hoặc null, "semantic_query": string}'
        )
        user_prompt = f"Danh mục có sẵn: {', '.join(known_categories)}\nCâu tìm kiếm: '{query}'"

        try:
            raw_res = groq_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=settings.GROQ_FAST_MODEL,
                temperature=0.1,
                max_tokens=200
            )
            if raw_res:
                # Clean possible markdown wrapping
                clean_json = re.sub(r"^```json\s*|\s*```$", "", raw_res.strip(), flags=re.MULTILINE)
                parsed = json.loads(clean_json)
                return {
                    "category": parsed.get("category") or fallback_intent.get("category"),
                    "min_price": float(parsed["min_price"]) if parsed.get("min_price") else fallback_intent.get("min_price"),
                    "max_price": float(parsed["max_price"]) if parsed.get("max_price") else fallback_intent.get("max_price"),
                    "semantic_query": parsed.get("semantic_query") or query
                }
        except Exception:
            pass

        return fallback_intent

    def _heuristic_parse(self, query: str, known_categories: List[str]) -> Dict[str, Any]:
        """Robust regex-based intent parser for offline execution."""
        q_lower = query.lower()

        # Category match
        matched_cat = None
        for cat in known_categories:
            if cat.lower() in q_lower:
                matched_cat = cat
                break

        # Price matching: range ("từ X đến Y"), max ("dưới X"), min ("trên X")
        max_price = None
        min_price = None

        m_range = re.search(
            r"(?:từ\s*)?(\d+(?:\.\d+)?)\s*(triệu|tr|trieu|k|nghìn|ngàn|đ|vnd)?\s*(?:đến|tới|-)\s*(\d+(?:\.\d+)?)\s*(triệu|tr|trieu|k|nghìn|ngàn|đ|vnd)?",
            q_lower
        )
        if m_range:
            def _calc_val(v_str, u_str, default_u=""):
                v = float(v_str)
                u = u_str or default_u
                if u in ["triệu", "tr", "trieu"]:
                    return v * 1_000_000
                elif u in ["k", "nghìn", "ngàn"]:
                    return v * 1_000
                elif v < 100:
                    return v * 1_000_000
                return v

            u2 = m_range.group(4) or ""
            u1 = m_range.group(2) or u2
            min_price = _calc_val(m_range.group(1), u1, default_u=u2)
            max_price = _calc_val(m_range.group(3), u2, default_u=u1)
        else:
            m_under = re.search(r"(?:dưới|tầm|<|<=)\s*(\d+(?:\.\d+)?)\s*(triệu|tr|trieu|k|nghìn|ngàn|đ|vnd)?", q_lower)
            if m_under:
                val = float(m_under.group(1))
                unit = m_under.group(2) or ""
                if unit in ["triệu", "tr", "trieu"]:
                    max_price = val * 1_000_000
                elif unit in ["k", "nghìn", "ngàn"]:
                    max_price = val * 1_000
                elif val < 100:
                    max_price = val * 1_000_000
                else:
                    max_price = val

            m_above = re.search(r"(?:trên|>|>=)\s*(\d+(?:\.\d+)?)\s*(triệu|tr|trieu|k|nghìn|ngàn|đ|vnd)?", q_lower)
            if m_above:
                val = float(m_above.group(1))
                unit = m_above.group(2) or ""
                if unit in ["triệu", "tr", "trieu"]:
                    min_price = val * 1_000_000
                elif unit in ["k", "nghìn", "ngàn"]:
                    min_price = val * 1_000
                elif val < 100:
                    min_price = val * 1_000_000
                else:
                    min_price = val

        return {
            "category": matched_cat,
            "min_price": min_price,
            "max_price": max_price,
            "semantic_query": query
        }


    def search(
        self,
        query: str,
        db: Session,
        top_k: int = 5,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute complete RAG search pipeline."""
        start_t = time.time()

        # Fetch known categories
        all_products = db.query(Product).all()
        prod_map = {p.id: p for p in all_products}
        categories = sorted(list(set(p.category for p in all_products if p.category)))

        # 1. Parse intent
        intent = self.parse_intent(query, categories)

        # 2. Vector retrieval via Qdrant or Embedder
        target_prod = Product(
            id=0,
            title=intent.get("semantic_query") or query,
            category=intent.get("category") or "",
            tags=query,
            description=query
        )
        query_vec = self.embedder.transform_single(target_prod)

        candidate_hits = []
        if self.qdrant_store and self.qdrant_store.is_available() if hasattr(self.qdrant_store, "is_available") else True:
            try:
                candidate_hits = self.qdrant_store.search_similar(
                    query_vector=query_vec,
                    top_k=top_k * 3,
                    filter_category=intent.get("category")
                )
            except Exception:
                candidate_hits = []

        # 3. Filter by price bounds & category in SQL
        matched_products = []
        hit_ids = set(h[0] for h in candidate_hits)

        # First add vector hits that satisfy price bounds
        for pid, score, _ in candidate_hits:
            p = prod_map.get(pid)
            if not p:
                continue
            if intent.get("max_price") and p.price > intent["max_price"]:
                continue
            if intent.get("min_price") and p.price < intent["min_price"]:
                continue
            matched_products.append((p, score))
            if len(matched_products) >= top_k:
                break

        # If not enough candidates, backfill via keyword search
        if len(matched_products) < top_k:
            kw = intent.get("semantic_query") or query
            kw_query = db.query(Product)
            if intent.get("category"):
                kw_query = kw_query.filter(Product.category == intent["category"])
            if intent.get("max_price"):
                kw_query = kw_query.filter(Product.price <= intent["max_price"])
            if intent.get("min_price"):
                kw_query = kw_query.filter(Product.price >= intent["min_price"])

            fill_prods = kw_query.limit(top_k * 2).all()
            for p in fill_prods:
                if p.id not in [m[0].id for m in matched_products]:
                    matched_products.append((p, 0.75))
                    if len(matched_products) >= top_k:
                        break

        # 4. Generate LLM synthesis response
        synthesis = self._synthesize_response(query, intent, [p for p, _ in matched_products])

        latency = round((time.time() - start_t) * 1000, 2)
        return {
            "query": query,
            "intent": intent,
            "reply": synthesis,
            "products": [
                {
                    "product_id": p.id,
                    "title": p.title,
                    "category": p.category,
                    "price": p.price,
                    "rating_avg": p.rating_avg,
                    "image_url": p.image_url,
                    "score": round(score, 4),
                    "model": "RAG Semantic Search",
                    "reason": f"Phù hợp với tiêu chí tìm kiếm '{query}'"
                }
                for p, score in matched_products
            ],
            "latency_ms": latency
        }

    def _synthesize_response(self, query: str, intent: Dict[str, Any], products: List[Product]) -> str:
        """Synthesize natural explanation of retrieved products."""
        if not products:
            return f"Rất tiếc, hiện tại hệ thống chưa tìm thấy sản phẩm nào khớp hoàn toàn với yêu cầu '{query}'. Bạn có thể nới lỏng mức giá hoặc tìm kiếm theo danh mục khác nhé!"

        prod_titles = ", ".join([f"'{p.title}' ({p.price:,.0f} đ)" for p in products[:3]])

        if groq_client.is_available():
            prompt = (
                f"Người dùng vừa tìm kiếm: '{query}'. "
                f"Hệ thống đã tìm thấy {len(products)} sản phẩm nổi bật: {prod_titles}. "
                "Hãy viết 1-2 câu tư vấn mua hàng ngắn gọn, tự nhiên, nêu rõ lý do các sản phẩm này phù hợp với mong muốn của khách."
            )
            try:
                res = groq_client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model=settings.GROQ_FAST_MODEL,
                    max_tokens=150,
                    temperature=0.3
                )
                if res and res.strip():
                    return res.strip()
            except Exception:
                pass

        # Heuristic fallback response
        return f"Dưới đây là {len(products)} sản phẩm phù hợp nhất với yêu cầu '{query}', được tuyển chọn kỹ lưỡng về mức giá và đánh giá cao từ cộng đồng:"
