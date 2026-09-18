"""Conversational Shopping Assistant Agent.
Handles multi-turn dialogue, product comparison, contextual recommendation,
and executes cart actions using Groq LPU.
"""
import json
import re
import time
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from src.database.models import Product, Interaction
from src.agent.groq_client import groq_client
from src.agent.ollama_client import ollama_client
from src.agent.rag_search import RAGSearchEngine
from src.commerce.cart_gateway import (
    CartActionResult,
    CartActionStatus,
    CartGateway,
    DisabledCartGateway,
)
from src.config import settings

class ConversationalShoppingAssistant:
    """Intelligent shopping agent capable of consultation, comparison, and cart action execution."""

    def __init__(
        self,
        rag_engine: Optional[RAGSearchEngine] = None,
        cart_gateway: Optional[CartGateway] = None,
    ):
        self.rag_engine = rag_engine or RAGSearchEngine()
        self.cart_gateway = cart_gateway or DisabledCartGateway()
        self._logged_cart_action_ids: set[str] = set()

    def process_chat(
        self,
        messages: List[Dict[str, str]],
        user_id: int,
        db: Session,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process multi-turn conversation and generate assistant reply with action triggers."""
        start_t = time.time()

        all_products = db.query(Product).all()
        prod_map = {p.id: p for p in all_products}

        # Latest user message
        last_user_msg = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        q_lower = last_user_msg.lower()

        # Check for cart intent ("thêm vào giỏ", "mua sản phẩm #12", "cho vào giỏ hàng")
        cart_action = self._detect_cart_intent(last_user_msg, prod_map)
        suggested_prods = []

        # Recommendation telemetry is separate from a durable cart acknowledgement.
        if cart_action and cart_action.get("product_id"):
            pid = cart_action["product_id"]
            p = prod_map.get(pid)
            title = p.title if p else f"#{pid}"
            action_result = self._submit_cart_action(user_id, pid, idempotency_key)
            if action_result.status == CartActionStatus.COMPLETED:
                self._record_completed_cart_interaction(action_result, user_id, db)
                reply = f"🛒 Đã thêm sản phẩm '{title}' vào giỏ hàng của bạn thành công!"
            elif action_result.status == CartActionStatus.PROPOSED:
                reply = f"🛒 Tôi đã chuẩn bị thêm sản phẩm '{title}' vào giỏ hàng. Vui lòng xác nhận trong hệ thống thương mại để hoàn tất."
            else:
                reply = f"Tôi chưa thể thêm sản phẩm '{title}' vào giỏ hàng. {action_result.detail}"
            latency = round((time.time() - start_t) * 1000, 2)
            return {
                "reply": reply,
                "suggested_products": [self._format_product(p)] if p else [],
                "action": {**cart_action, **self._format_cart_action_result(action_result)},
                "latency_ms": latency
            }

        # Otherwise perform RAG search to supply context to the LLM
        rag_res = self.rag_engine.search(last_user_msg, db=db, top_k=4, user_id=user_id)
        suggested_raw = rag_res.get("products", [])
        suggested_prods = suggested_raw

        context_items = [f"ID {p['product_id']}: '{p['title']}' (Giá: {p['price']:,.0f} đ, Đánh giá: {p['rating_avg']} sao, Danh mục: {p['category']})" for p in suggested_raw]
        context_str = "\n".join(context_items)

        system_instruction = (
            "Bạn là Chuyên gia Tư vấn Mua sắm AI thông minh, thân thiện của sàn thương mại điện tử RecSys-AI. "
            "Nhiệm vụ của bạn là giải đáp thắc mắc của khách hàng, phân tích và so sánh tính năng/giá thành các sản phẩm được cung cấp, "
            "và hướng dẫn khách hàng lựa chọn sản phẩm phù hợp nhất với nhu cầu của họ. "
            "Trả lời ngắn gọn, lịch sự, chuyên nghiệp bằng tiếng Việt, tránh dài dòng."
        )

        llm_messages = [{"role": "system", "content": system_instruction}]
        # Append last few turns of chat history for context continuity
        for m in messages[-4:]:
            llm_messages.append({"role": m["role"], "content": m["content"]})

        if context_str:
            llm_messages.append({
                "role": "system",
                "content": f"Dữ liệu sản phẩm thực tế phù hợp trong kho:\n{context_str}\nHãy tham khảo dữ liệu trên để tư vấn trực tiếp cho khách."
            })

        reply_content = None
        # M1 local plan: Ollama (qwen2.5:3b) first when OLLAMA_URL is set,
        # then Groq, then heuristic fallback.
        if ollama_client.is_available():
            reply_content = ollama_client.chat_completion(
                messages=llm_messages,
                model=settings.OLLAMA_CHAT_MODEL,
                temperature=0.3,
                max_tokens=350
            )
        if not reply_content and groq_client.is_available():
            reply_content = groq_client.chat_completion(
                messages=llm_messages,
                model=settings.GROQ_CHAT_MODEL,
                temperature=0.3,
                max_tokens=350
            )

        if not reply_content:
            # Heuristic fallback reply
            reply_content = rag_res.get("reply") or f"Tôi đã tìm thấy một số sản phẩm phù hợp với yêu cầu '{last_user_msg}' của bạn. Bạn hãy xem qua các gợi ý bên dưới nhé!"

        latency = round((time.time() - start_t) * 1000, 2)
        return {
            "reply": reply_content,
            "suggested_products": suggested_prods,
            "action": None,
            "latency_ms": latency
        }

    def _detect_cart_intent(self, text: str, prod_map: Dict[int, Product]) -> Optional[Dict[str, Any]]:
        """Detect if user asks to add an item to cart and extract product ID."""
        t_lower = text.lower()
        is_cart_kw = (
            bool(re.search(r"(?:thêm|bỏ|cho)\b.*?\b(?:vào giỏ|vào cart|giỏ hàng)", t_lower))
            or any(kw in t_lower for kw in ["thêm vào giỏ", "cho vào giỏ", "bỏ vào giỏ", "mua sản phẩm", "mua món", "chốt đơn", "thêm vào giỏ hàng"])
        )
        if is_cart_kw:
            # Look for product ID like #12 or id 12 or sản phẩm #12
            m = re.search(r"(?:#|id\s*|sản phẩm\s*#?|món\s*#?)(\d+)", t_lower)
            if m:
                pid = int(m.group(1))
                if pid in prod_map:
                    p = prod_map[pid]
                    return {
                        "type": "add_to_cart",
                        "product_id": pid,
                        "product_title": p.title if p else f"#{pid}",
                        "detail": f"Thêm sản phẩm #{pid} vào giỏ hàng"
                    }

            # Alternatively, check matching by product name in text
            for pid, p in prod_map.items():
                if p.title and len(p.title) > 3 and p.title.lower() in t_lower:
                    return {
                        "type": "add_to_cart",
                        "product_id": pid,
                        "product_title": p.title,
                        "detail": f"Thêm sản phẩm '{p.title}' vào giỏ hàng"
                    }

        return None

    def _submit_cart_action(
        self,
        user_id: int,
        product_id: int,
        idempotency_key: Optional[str],
    ) -> CartActionResult:
        if getattr(self.cart_gateway, "requires_idempotency_key", True) and not idempotency_key:
            return CartActionResult(
                status=CartActionStatus.FAILED,
                action_id=None,
                product_id=product_id,
                detail="An idempotency key is required for configured cart actions.",
            )
        return self.cart_gateway.add_item(user_id, product_id, idempotency_key or "proposal")

    def _record_completed_cart_interaction(
        self,
        action_result: CartActionResult,
        user_id: int,
        db: Session,
    ) -> None:
        if not action_result.action_id or action_result.action_id in self._logged_cart_action_ids:
            return
        db.add(Interaction(
            user_id=user_id,
            product_id=action_result.product_id,
            event_type="add_to_cart",
            weight=3.5,
        ))
        db.commit()
        self._logged_cart_action_ids.add(action_result.action_id)

    @staticmethod
    def _format_cart_action_result(result: CartActionResult) -> Dict[str, Any]:
        return {
            "status": result.status.value,
            "action_id": result.action_id,
            "product_id": result.product_id,
            "detail": result.detail,
        }

    def _format_product(self, p: Product) -> Dict[str, Any]:
        """Convert Product DB instance to standard recommendation dict."""
        return {
            "product_id": p.id,
            "title": p.title,
            "category": p.category,
            "price": p.price,
            "rating_avg": p.rating_avg,
            "image_url": p.image_url,
            "score": 1.0,
            "model": "Shopping Assistant",
            "reason": "Sản phẩm được chọn vào giỏ hàng"
        }

shopping_assistant = ConversationalShoppingAssistant()
