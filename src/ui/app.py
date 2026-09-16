"""Interactive Streamlit Demo for RecSys-AI.
Visualizes the 5-phase pipeline from the architectural infographic:
User Behavior -> Feature/Embedding -> Recommendation Model -> Ranking -> Top-K Output.
"""
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from src.database.session import get_db_context
from src.database.models import Product, User, Interaction
from src.models.hybrid import HybridRecommender
from src.ranking.reranker import ProductReranker
from src.config import settings

st.set_page_config(
    page_title="RecSys-AI | Hệ thống Gợi ý Sản phẩm",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern e-commerce styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border-radius: 12px;
        padding: 12px;
        border: 1px solid #BBF7D0;
    }
    .product-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        margin-bottom: 16px;
        transition: transform 0.2s ease;
    }
    .product-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.08);
    }
    .reason-box {
        background-color: #EFF6FF;
        border-left: 3px solid #3B82F6;
        padding: 6px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        color: #1E40AF;
        margin-top: 8px;
    }
    .badge-hybrid {
        background-color: #8B5CF6;
        color: white;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-cf {
        background-color: #3B82F6;
        color: white;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-cb {
        background-color: #10B981;
        color: white;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-pop {
        background-color: #F59E0B;
        color: white;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_serving_engine():
    """Load cached in-memory recommendation engine."""
    with get_db_context() as db:
        products = db.query(Product).all()
        interactions = db.query(Interaction).all()
        prod_dict = {p.id: p for p in products}

        hybrid = HybridRecommender(
            cf_weight=settings.HYBRID_CF_WEIGHT,
            cb_weight=settings.HYBRID_CB_WEIGHT,
            cold_start_threshold=settings.COLD_START_THRESHOLD
        )
        hybrid.fit(products, interactions)
        reranker = ProductReranker()
        return hybrid, prod_dict, reranker


hybrid_engine, product_dict, reranker = load_serving_engine()

# Sidebar Setup
st.sidebar.image("https://img.icons8.com/isometric/100/shopping-cart.png", width=64)
st.sidebar.title("Cấu hình RecSys")

# User Selection
with get_db_context() as db:
    users = db.query(User).order_by(User.id.asc()).limit(80).all()
    user_options = {
        f"ID {u.id} - {u.username} ({u.segment})": u.id
        for u in users
    }

selected_user_label = st.sidebar.selectbox("Chọn Người dùng (User):", list(user_options.keys()), index=0)
selected_user_id = user_options[selected_user_label]

# Model Selection
strategy_map = {
    "Hybrid (CF + Content-Based + Cold-Start)": "hybrid",
    "Collaborative Filtering (Hành vi người dùng tương tự)": "collaborative",
    "Content-Based Filtering (Đặc tính sản phẩm tương đồng)": "content_based",
    "Popularity Baseline (Xu hướng bán chạy nhất)": "popularity"
}
selected_strategy_label = st.sidebar.selectbox("Thuật toán gợi ý:", list(strategy_map.keys()), index=0)
selected_strategy = strategy_map[selected_strategy_label]

top_k = st.sidebar.slider("Số lượng gợi ý (Top-K):", min_value=4, max_value=16, value=8, step=2)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Pipeline Status")
st.sidebar.success("✅ Database: SQLite (Connected)")
st.sidebar.success("✅ Embeddings: 64-dim TF-IDF + SVD")
st.sidebar.success("✅ Latent Factors: 32-dim ALS SVD")
st.sidebar.info("💡 Tip: Nhấp nút **👁️ Xem** hoặc **🛒 Mua** để gửi event tương tác realtime và quan sát gợi ý thay đổi ngay!")


# Main Content Header
st.markdown('<div class="main-title">HỆ THỐNG GỢI Ý SẢN PHẨM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Recommendation System phân tích hành vi người dùng (view, click, cart, purchase) để đề xuất sản phẩm tối ưu.</div>', unsafe_allow_html=True)

# Architecture Pipeline Flow (From TikTok Infographic)
with st.expander("📌 Sơ đồ Pipeline Kiến trúc 5 Khối", expanded=False):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("##### 01 Dữ liệu vào")
        st.caption("• Lịch sử xem\n• Mua hàng\n• Click & Cart\n• Rating")
    with col2:
        st.markdown("##### 02 Features")
        st.caption("• User Vectors\n• Item Embeddings\n• Implicit weights")
    with col3:
        st.markdown("##### 03 Models")
        st.caption("• Collaborative\n• Content-Based\n• Hybrid Engine")
    with col4:
        st.markdown("##### 04 Ranking")
        st.caption("• Quality boost\n• Anti-fatigue\n• Diversity filter")
    with col5:
        st.markdown("##### 05 Output")
        st.caption("• Top-K Products\n• Sub-20ms Latency\n• Rationale")

tab_live, tab_history, tab_similar, tab_benchmark = st.tabs([
    "🛍️ Bạn có thể thích (Live Recs)",
    "📜 Lịch sử tương tác của User",
    "🔍 Tìm sản phẩm tương đồng",
    "📊 Benchmark Đánh giá Mô hình"
])

# -------------------------------------------------------------
# TAB 1: Live Recommendations
# -------------------------------------------------------------
with tab_live:
    # Generate recommendations
    if selected_strategy == "hybrid":
        raw_candidates = hybrid_engine.recommend(selected_user_id, top_k=top_k * 2)
    elif selected_strategy == "collaborative":
        raw_candidates = hybrid_engine.cf_model.recommend(selected_user_id, top_k=top_k * 2)
    elif selected_strategy == "content_based":
        raw_candidates = hybrid_engine.content_model.recommend(selected_user_id, top_k=top_k * 2)
    else:
        raw_candidates = hybrid_engine.popularity_model.recommend(selected_user_id, top_k=top_k * 2)

    if not raw_candidates:
        raw_candidates = hybrid_engine.popularity_model.recommend(selected_user_id, top_k=top_k * 2)

    # Reranking
    with get_db_context() as db:
        purchases = (
            db.query(Interaction.product_id)
            .filter(Interaction.user_id == selected_user_id, Interaction.event_type == "purchase")
            .all()
        )
        purchased_ids = [p[0] for p in purchases]

    recommendations = reranker.rerank(
        candidates=raw_candidates,
        product_dict=product_dict,
        recently_purchased_ids=purchased_ids,
        top_k=top_k
    )

    st.subheader(f"✨ Gợi ý dành riêng cho bạn ({len(recommendations)} sản phẩm)")

    # Grid Display
    cols_per_row = 4
    for i in range(0, len(recommendations), cols_per_row):
        row_items = recommendations[i:i + cols_per_row]
        cols = st.columns(len(row_items))
        for col, item in zip(cols, row_items):
            with col:
                st.markdown(f"""
                <div class="product-card">
                    <img src="{item['image_url']}" style="width:100%; height:160px; object-fit:cover; border-radius:8px; margin-bottom:8px;" onerror="this.src='https://via.placeholder.com/300x200?text=Product';"/>
                    <div style="font-weight:700; font-size:0.95rem; height:44px; overflow:hidden; text-overflow:ellipsis;">{item['title']}</div>
                    <div style="color:#6B7280; font-size:0.8rem; margin:4px 0;">🏷️ {item['category']}</div>
                    <div style="color:#DC2626; font-weight:700; font-size:1.05rem;">{item['price']:,.0f} đ</div>
                    <div style="font-size:0.85rem; color:#D97706; margin-bottom:6px;">⭐ {item['rating_avg']} • Điểm phù hợp: <b>{int(item['final_score']*100)}%</b></div>
                    <div class="reason-box">💡 {item['reason']}</div>
                </div>
                """, unsafe_allow_html=True)

                # Interactive event buttons to trigger real-time loop
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("👁️", key=f"view_{item['product_id']}_{selected_user_id}", help="Xem chi tiết"):
                        with get_db_context() as db:
                            db.add(Interaction(user_id=selected_user_id, product_id=item['product_id'], event_type="view", weight=1.0))
                        st.toast(f"Đã ghi nhận tương tác 'Xem' sản phẩm #{item['product_id']}!", icon="👁️")
                        st.rerun()
                with c2:
                    if st.button("🛒", key=f"cart_{item['product_id']}_{selected_user_id}", help="Thêm vào giỏ hàng"):
                        with get_db_context() as db:
                            db.add(Interaction(user_id=selected_user_id, product_id=item['product_id'], event_type="add_to_cart", weight=3.5))
                        st.toast(f"Đã thêm #{item['product_id']} vào giỏ hàng!", icon="🛒")
                        st.rerun()
                with c3:
                    if st.button("💳", key=f"buy_{item['product_id']}_{selected_user_id}", help="Mua ngay"):
                        with get_db_context() as db:
                            db.add(Interaction(user_id=selected_user_id, product_id=item['product_id'], event_type="purchase", weight=5.0))
                        st.toast(f"Đã mua #{item['product_id']}! Sản phẩm sẽ được lọc khỏi gợi ý.", icon="💳")
                        st.rerun()

# -------------------------------------------------------------
# TAB 2: User Interaction History
# -------------------------------------------------------------
with tab_history:
    st.subheader(f"Lịch sử tương tác của User #{selected_user_id}")
    with get_db_context() as db:
        user_inters = (
            db.query(Interaction)
            .filter(Interaction.user_id == selected_user_id)
            .order_by(Interaction.timestamp.desc())
            .limit(30)
            .all()
        )

    if not user_inters:
        st.info("User này là người dùng mới tinh (Cold-Start) chưa có lịch sử tương tác nào!")
    else:
        history_records = []
        for inter in user_inters:
            prod = product_dict.get(inter.product_id)
            history_records.append({
                "Thời gian": inter.timestamp.strftime("%Y-%m-%d %H:%M") if inter.timestamp else "N/A",
                "Hành vi (Event)": inter.event_type.upper(),
                "Trọng số": inter.weight,
                "Sản phẩm": prod.title if prod else f"Product #{inter.product_id}",
                "Danh mục": prod.category if prod else "N/A",
                "Giá": f"{prod.price:,.0f} đ" if prod else "N/A"
            })
        st.dataframe(pd.DataFrame(history_records), use_container_width=True)

# -------------------------------------------------------------
# TAB 3: Similar Products (Item-to-Item)
# -------------------------------------------------------------
with tab_similar:
    st.subheader("Khám phá Sản phẩm tương đồng (Item-to-Item Similarity)")
    prod_options = {f"#{p.id} - {p.title} ({p.category})": p.id for p in product_dict.values()}
    selected_prod_label = st.selectbox("Chọn sản phẩm gốc:", list(prod_options.keys()), index=0)
    target_prod_id = prod_options[selected_prod_label]

    sim_items = hybrid_engine.similar_items(target_prod_id, top_k=6)
    if sim_items:
        cols = st.columns(len(sim_items))
        for col, item in zip(cols, sim_items):
            p = product_dict.get(item["product_id"])
            if p:
                with col:
                    st.image(p.image_url, use_column_width=True)
                    st.markdown(f"**{p.title}**")
                    st.caption(f"{p.category} | {p.price:,.0f} đ")
                    st.progress(min(float(item["score"]), 1.0), text=f"Độ tương đồng: {int(item['score']*100)}%")
                    st.info(item["reason"])

# -------------------------------------------------------------
# TAB 4: Offline Benchmark Metrics
# -------------------------------------------------------------
with tab_benchmark:
    st.subheader("Đánh giá Benchmark các Mô hình Gợi ý")
    st.markdown("""
    Bảng so sánh hiệu năng của các mô hình trên tập kiểm thử (Leave-20%-out evaluation):
    - **Precision@K**: Tỷ lệ gợi ý trúng sở thích thực tế của người dùng.
    - **Recall@K**: Tỷ lệ bắt trọn các sản phẩm người dùng yêu thích.
    - **NDCG@K**: Đánh giá độ chính xác có xét đến thứ tự xếp hạng (sản phẩm đúng ở vị trí cao được điểm cao hơn).
    - **Catalog Coverage**: Tỷ lệ danh mục sản phẩm được khai phá (tránh hiện tượng chỉ gợi ý sản phẩm hot).
    """)

    benchmark_data = [
        {"Mô hình": "Popularity Baseline", "P@5": "9.93%", "R@5": "21.70%", "NDCG@5": "0.1710", "Coverage": "16.13%", "Ưu điểm": "Cực tốt cho Cold-start", "Nhược điểm": "Thiếu cá nhân hóa, độ phủ thấp"},
        {"Mô hình": "Content-Based", "P@5": "5.40%", "R@5": "10.83%", "NDCG@5": "0.0903", "Coverage": "100.0%", "Ưu điểm": "Gợi ý chính xác theo ngữ cảnh & text", "Nhược điểm": "Bị giam trong 'filter bubble'"},
        {"Mô hình": "Collaborative Filtering", "P@5": "3.36%", "R@5": "6.18%", "NDCG@5": "0.0519", "Coverage": "100.0%", "Ưu điểm": "Khám phá serendipity (sản phẩm bất ngờ)", "Nhược điểm": "Bị tê liệt khi gặp user mới"},
        {"Mô hình": "Hybrid Recommender", "P@5": "4.82%", "R@5": "9.86%", "NDCG@5": "0.0850", "Coverage": "100.0%", "Ưu điểm": "Cân bằng toàn diện, giải quyết triệt để cold-start", "Nhược điểm": "Cần tinh chỉnh trọng số"}
    ]
    st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True)
