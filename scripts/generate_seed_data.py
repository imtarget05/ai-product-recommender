"""Seed Data Generator for RecSys-AI.
Generates realistic e-commerce catalog, users with personas, and interaction events.
"""
import random
from datetime import datetime, timedelta
from src.database.session import init_db, get_db_context
from src.database.models import Product, User, Interaction, SearchLog
from src.config import settings

# Sample rich e-commerce catalog
SAMPLE_PRODUCTS = [
    # Category: Điện thoại & Phụ kiện
    {"title": "Tai nghe không dây Bluetooth True Wireless ANC", "category": "Phụ kiện công nghệ", "price": 1290000, "tags": "tai nghe, bluetooth, anc, am thanh, bass", "image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=500"},
    {"title": "Củ sạc nhanh GaN 65W 3 cổng Type-C", "category": "Phụ kiện công nghệ", "price": 450000, "tags": "sac nhanh, gan, type-c, 65w, anker", "image_url": "https://images.unsplash.com/photo-1583863788434-e58a36330cf0?w=500"},
    {"title": "Đồng hồ thông minh Smartwatch Theo dõi vận động", "category": "Phụ kiện công nghệ", "price": 2490000, "tags": "smartwatch, dong ho, the thao, nhip tim, oled", "image_url": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=500"},
    {"title": "Sạc dự phòng không dây Magsafe 10000mAh", "category": "Phụ kiện công nghệ", "price": 690000, "tags": "pin du phong, magsafe, wireless, 10000mah", "image_url": "https://images.unsplash.com/photo-1609081219090-a6d8173087ec?w=500"},
    {"title": "Giá đỡ điện thoại xoay 360 độ hợp kim nhôm", "category": "Phụ kiện công nghệ", "price": 180000, "tags": "gia do, de ban, hop kim, 360 do", "image_url": "https://images.unsplash.com/photo-1586105251261-72a756497a11?w=500"},
    {"title": "Cáp sạc nhanh bọc dù Type-C to Lightning 20W", "category": "Phụ kiện công nghệ", "price": 150000, "tags": "cap sac, lightning, type-c, boc du, ben bi", "image_url": "https://images.unsplash.com/photo-1546868871-7041f2a55e12?w=500"},

    # Category: Máy tính & Thiết bị văn phòng
    {"title": "Bàn phím cơ không dây Custom Hot-swap RGB", "category": "Thiết bị máy tính", "price": 1850000, "tags": "ban phim co, mechanical keyboard, rgb, hot-swap, bluetooth", "image_url": "https://images.unsplash.com/photo-1587829741301-dc798b83add3?w=500"},
    {"title": "Chuột máy tính công thái học không dây siêu nhẹ", "category": "Thiết bị máy tính", "price": 890000, "tags": "chuot khong day, cong thai hoc, ergonomic, gaming, wireless", "image_url": "https://images.unsplash.com/photo-1615663245857-ac93bb7c39e7?w=500"},
    {"title": "Màn hình đồ họa 27 inch 4K IPS 100% sRGB", "category": "Thiết bị máy tính", "price": 7200000, "tags": "man hinh, 27 inch, 4k, ips, do hoa, type-c", "image_url": "https://images.unsplash.com/photo-1527443224154-c4a3942d3acf?w=500"},
    {"title": "Ổ cứng SSD di động chuẩn NVMe 1TB Type-C", "category": "Thiết bị máy tính", "price": 2100000, "tags": "ssd, o cung di dong, nvme, 1tb, toc do cao", "image_url": "https://images.unsplash.com/photo-1597872200969-2b65d56bd16b?w=500"},
    {"title": "Giá đỡ laptop tản nhiệt bằng nhôm gấp gọn", "category": "Thiết bị máy tính", "price": 280000, "tags": "gia do laptop, tan nhiet, nhom, cong thai hoc", "image_url": "https://images.unsplash.com/photo-1527864550417-7fd91fc51a46?w=500"},
    {"title": "Đèn màn hình thông minh bảo vệ mắt chống lóa", "category": "Thiết bị máy tính", "price": 650000, "tags": "den man hinh, screenbar, chong moi mat, led, cam ung", "image_url": "https://images.unsplash.com/photo-1507473885765-e6ed057f782c?w=500"},

    # Category: Thời trang & Giày thể thao
    {"title": "Giày thể thao Sneaker chạy bộ siêu êm thoáng khí", "category": "Thời trang & Giày", "price": 950000, "tags": "giay the thao, sneaker, chay bo, thoang khi, em chan", "image_url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=500"},
    {"title": "Áo khoác gió thể thao chống nước nhẹ Unisex", "category": "Thời trang & Giày", "price": 420000, "tags": "ao khoac, gio, chong nuoc, the thao, unisex", "image_url": "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=500"},
    {"title": "Balo laptop chống sốc chống nước công sở cao cấp", "category": "Thời trang & Giày", "price": 590000, "tags": "balo, laptop, chong nuoc, cong so, du lich", "image_url": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=500"},
    {"title": "Áo thun cotton trơn form rộng thoáng mát", "category": "Thời trang & Giày", "price": 190000, "tags": "ao thun, cotton, form rong, oversize, basic", "image_url": "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?w=500"},
    {"title": "Quần jogger thể thao co giãn 4 chiều năng động", "category": "Thời trang & Giày", "price": 310000, "tags": "quan jogger, the thao, co gian, tap gym, chay bo", "image_url": "https://images.unsplash.com/photo-1552902865-b72c031ac5ea?w=500"},
    {"title": "Kính râm phân cực chống tia UV phong cách thời thượng", "category": "Thời trang & Giày", "price": 350000, "tags": "kinh ram, uv400, thoi trang, phan cuc, mat kinh", "image_url": "https://images.unsplash.com/photo-1511499767150-a48a237f0083?w=500"},

    # Category: Nhà cửa & Đời sống thông minh
    {"title": "Robot hút bụi lau nhà tự động thông minh Laser LDS", "category": "Nhà cửa đời sống", "price": 5890000, "tags": "robot hut bui, lau nha, laser lds, tu dong, thong minh", "image_url": "https://images.unsplash.com/photo-1518640467707-6811f4a6ab73?w=500"},
    {"title": "Nồi chiên không dầu điện tử dung tích lớn 6L", "category": "Nhà cửa đời sống", "price": 1450000, "tags": "noi chien khong dau, 6l, dien tu, nau an, giam dau mo", "image_url": "https://images.unsplash.com/photo-1585515320310-259814833e62?w=500"},
    {"title": "Máy lọc không khí phòng ngủ khử mùi màng lọc HEPA", "category": "Nhà cửa đời sống", "price": 2190000, "tags": "may loc khong khi, hepa, khumui, yen tinh, phong ngu", "image_url": "https://images.unsplash.com/photo-1585338107529-13afc5f02586?w=500"},
    {"title": "Bình giữ nhiệt inox 316 hiển thị nhiệt độ 500ml", "category": "Nhà cửa đời sống", "price": 220000, "tags": "binh giu nhiet, inox 316, man hinh led, 500ml", "image_url": "https://images.unsplash.com/photo-1602143407151-7111542de6e8?w=500"},
    {"title": "Máy xay sinh tố cầm tay sạc pin đa năng", "category": "Nhà cửa đời sống", "price": 320000, "tags": "may xay sinh to, cam tay, sac pin, tien loi, hoa qua", "image_url": "https://images.unsplash.com/photo-1570222094114-d054a817e56b?w=500"},
    {"title": "Cân sức khỏe điện tử thông minh đo 14 chỉ số cơ thể", "category": "Nhà cửa đời sống", "price": 290000, "tags": "can suc khoe, bluetooth, do mo, chi so bmi, thong minh", "image_url": "https://images.unsplash.com/photo-1576678927484-cc907957088c?w=500"},

    # Category: Thể thao & Dã ngoại
    {"title": "Thảm tập Yoga định tuyến chống trượt cao su tự nhiên", "category": "Thể thao & Dã ngoại", "price": 450000, "tags": "tham yoga, dinh tuyen, chong truot, cao su, tap luyen", "image_url": "https://images.unsplash.com/photo-1592432678016-e910b452f9a2?w=500"},
    {"title": "Bộ dây kháng lực ngũ sắc tập gym tại nhà đa năng", "category": "Thể thao & Dã ngoại", "price": 199000, "tags": "day khang luc, tap gym, the thao tai nha, squat", "image_url": "https://images.unsplash.com/photo-1517838277536-f5f99be501cd?w=500"},
    {"title": "Lều cắm trại dã ngoại 4 người chống mưa tự bung", "category": "Thể thao & Dã ngoại", "price": 890000, "tags": "leu cam trai, da ngoai, tu bung, chong nuoc, camping", "image_url": "https://images.unsplash.com/photo-1504280390367-361c6d9f38f4?w=500"},
    {"title": "Ghế xếp dã ngoại thư giãn gấp gọn kèm túi đựng", "category": "Thể thao & Dã ngoại", "price": 270000, "tags": "ghe xep, camping, gap gon, cafe, thu gian", "image_url": "https://images.unsplash.com/photo-1519003722824-194d4455a60c?w=500"},

    # Category: Sách & Học tập
    {"title": "Sách Thiết kế Hệ thống Recommendation System từ cơ bản đến nâng cao", "category": "Sách & Tri thức", "price": 260000, "tags": "sach, ai, machine learning, recommendation, he thong goi y", "image_url": "https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?w=500"},
    {"title": "Sách Tư duy thiết kế sản phẩm số (Product Design)", "category": "Sách & Tri thức", "price": 210000, "tags": "sach, product design, ux ui, khoi nghiep, kinh doanh", "image_url": "https://images.unsplash.com/photo-1512820790803-83ca734da794?w=500"},
    {"title": "Sổ tay da cao cấp kèm bút ký thông minh", "category": "Sách & Tri thức", "price": 180000, "tags": "so tay, bi da, but ky, ghi chep, qua tang", "image_url": "https://images.unsplash.com/photo-1531346878377-a5be20888e57?w=500"},
]


def generate_seed_data(num_users: int = 150, num_interactions: int = 2500):
    """Generate mock users, catalog products, and interaction logs."""
    print("🌱 Initializing Database Schema...")
    init_db()

    with get_db_context() as db:
        # Check if already seeded
        if db.query(Product).count() > 0:
            print("Database already contains data. Clearing old seed data for fresh generation...")
            db.query(Interaction).delete()
            db.query(SearchLog).delete()
            db.query(Product).delete()
            db.query(User).delete()
            db.commit()

        print(f"📦 Seeding {len(SAMPLE_PRODUCTS)} Products...")
        product_objs = []
        for p in SAMPLE_PRODUCTS:
            desc = f"{p['title']} chất lượng cao, bền bỉ, tính năng vượt trội trong tầm giá. Phù hợp cho nhu cầu sử dụng hàng ngày."
            prod = Product(
                title=p["title"],
                category=p["category"],
                price=float(p["price"]),
                description=desc,
                image_url=p["image_url"],
                tags=p["tags"],
                rating_avg=round(random.uniform(4.0, 5.0), 1),
                rating_count=random.randint(15, 350),
                created_at=datetime.utcnow() - timedelta(days=random.randint(10, 180))
            )
            db.add(prod)
            product_objs.append(prod)
        db.commit()

        # Re-fetch with IDs
        products = db.query(Product).all()
        prod_by_cat = {}
        for p in products:
            prod_by_cat.setdefault(p.category, []).append(p)

        print(f"👥 Generating {num_users} Users with behavioral personas...")
        personas = [
            ("tech_geek", ["Thiết bị máy tính", "Phụ kiện công nghệ"]),
            ("fashionista", ["Thời trang & Giày"]),
            ("smart_home", ["Nhà cửa đời sống"]),
            ("fitness_outdoor", ["Thể thao & Dã ngoại"]),
            ("book_learner", ["Sách & Tri thức", "Thiết bị máy tính"]),
            ("casual_browser", list(prod_by_cat.keys())),
        ]

        user_objs = []
        user_persona_map = {}
        for i in range(1, num_users + 1):
            persona_name, pref_cats = random.choice(personas)
            # Cold-start test users (first 5 users have no interactions or 1 interaction)
            if i <= 5:
                persona_name = "cold_start_newbie"
                pref_cats = []

            user = User(
                username=f"user_{i:03d}_{persona_name}",
                email=f"user{i}@recsys.local",
                segment=persona_name,
                created_at=datetime.utcnow() - timedelta(days=random.randint(1, 90))
            )
            db.add(user)
            user_objs.append(user)
            user_persona_map[i] = pref_cats
        db.commit()

        users = db.query(User).all()

        print(f"⚡ Generating ~{num_interactions} user interactions across 30 days...")
        event_types = ["view", "click", "add_to_cart", "purchase", "rating"]
        event_distribution = [0.55, 0.25, 0.12, 0.05, 0.03]  # Real ecommerce funnel distribution

        interactions_count = 0
        now = datetime.utcnow()

        for user in users:
            # Cold-start users
            if user.segment == "cold_start_newbie":
                continue

            pref_cats = user_persona_map.get(user.id, [])
            if not pref_cats:
                continue

            # Eligible products for this persona
            persona_products = []
            for c in pref_cats:
                persona_products.extend(prod_by_cat.get(c, []))

            # Number of interactions per user (skewed distribution)
            n_events = random.randint(12, 45)

            for _ in range(n_events):
                # 80% chance picking from preferred categories, 20% random exploration
                if random.random() < 0.8 and persona_products:
                    product = random.choice(persona_products)
                else:
                    product = random.choice(products)

                event = random.choices(event_types, weights=event_distribution, k=1)[0]
                rating_val = None
                if event == "rating":
                    rating_val = round(random.uniform(3.5, 5.0), 1)

                weight = settings.EVENT_WEIGHTS.get(event, 1.0)
                if rating_val is not None:
                    weight = rating_val

                inter_time = now - timedelta(
                    days=random.randint(0, 30),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )

                inter = Interaction(
                    user_id=user.id,
                    product_id=product.id,
                    event_type=event,
                    rating_value=rating_val,
                    weight=weight,
                    timestamp=inter_time
                )
                db.add(inter)
                interactions_count += 1

                # Sample search logs
                if event in ["view", "click"] and random.random() < 0.15:
                    tag_sample = product.tags.split(",")[0].strip() if product.tags else product.category
                    slog = SearchLog(
                        user_id=user.id,
                        query=tag_sample,
                        clicked_product_id=product.id,
                        timestamp=inter_time
                    )
                    db.add(slog)

        db.commit()
        print(f"✅ Finished Seeding: {len(products)} products, {len(users)} users, {interactions_count} interactions!")


if __name__ == "__main__":
    generate_seed_data()
