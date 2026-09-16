"""Dynamic Explanation Generator.
Produces personalized natural language rationales for recommended products
based on user behavior and item metadata using Groq LPU.
"""
from typing import List, Optional, Dict
from src.database.models import Product, Interaction
from src.agent.groq_client import groq_client
from src.config import settings

class DynamicExplainer:
    """Generates human-like, persuasive recommendation rationales."""

    def explain(
        self,
        product: Product,
        recent_interactions: Optional[List[Interaction]] = None,
        product_dict: Optional[Dict[int, Product]] = None,
        strategy_name: str = "Hybrid"
    ) -> str:
        """Generate dynamic rationale with LLM and fallback to contextual templates."""
        ref_titles = []
        if recent_interactions and product_dict:
            for inter in recent_interactions[-3:]:
                ref_prod = product_dict.get(inter.product_id)
                if ref_prod and ref_prod.title not in ref_titles:
                    ref_titles.append(ref_prod.title)

        # 1. Try Groq LPU Fast Model
        if groq_client.is_available():
            history_str = f"Lịch sử vừa xem gần đây: {', '.join(ref_titles)}" if ref_titles else "Người dùng đang tìm kiếm sản phẩm chất lượng cao."
            prompt = (
                f"Sản phẩm đề xuất: '{product.title}' (Danh mục: {product.category}, Giá: {product.price:,.0f} đ, Đánh giá: {product.rating_avg} sao).\n"
                f"{history_str}\n"
                "Hãy viết 1 câu giải thích ngắn gọn (dưới 25 từ) bằng tiếng Việt thật tự nhiên lý do tại sao người này nên mua hoặc xem sản phẩm này."
            )
            try:
                res = groq_client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model=settings.GROQ_FAST_MODEL,
                    max_tokens=60,
                    temperature=0.4
                )
                if res and res.strip():
                    return res.strip().strip('"')
            except Exception:
                pass

        # 2. Heuristic Contextual Template Fallback
        if ref_titles:
            return f"Phù hợp với sở thích của bạn sau khi xem '{ref_titles[-1][:30]}', sở hữu đánh giá cao ⭐ {product.rating_avg}."
        if product.rating_avg and product.rating_avg >= 4.5:
            return f"Sản phẩm bán chạy hàng đầu danh mục {product.category} với điểm đánh giá xuất sắc {product.rating_avg} ⭐."
        return f"Gợi ý phù hợp dựa trên phân tích đặc tính sản phẩm và xu hướng người dùng cùng sở thích."

explainer = DynamicExplainer()
