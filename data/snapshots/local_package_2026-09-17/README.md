# Snapshot: local_package_2026-09-17

Nguồn: thư mục rời `recsys_local_package_full 2/` (xuất từ Colab T4, 2026-09-17 18:59),
đã được hợp nhất vào RecSys-AI ngày 2026-09-18.

## Nội dung
- `train_recsys_T4.ipynb` (bản 11 cells, đầy đủ) → đã chép đè lên
  `RecSys-AI/colab/train_recsys_T4.ipynb` (bản cũ 10 cells).
- `recsys.db` + `item_embeddings.npy` + `item_embedder_meta.pkl` → **giữ tại đây,
  KHÔNG đè lên data live**. Lý do:
  - Snapshot db: 31 products / 150 users / 4310 interactions (không có bảng alembic_version).
  - Data live (`RecSys-AI/data/`): 81 products / 251 users / 29069 interactions,
    db mới hơn (2026-09-18 01:55), embeddings lớn hơn (41600 vs 16000 bytes).
  - Kết luận: data live đầy đủ và mới hơn → snapshot này chỉ để tham khảo/khôi phục.
- Thư mục `reports/` trong package rỗng → không mang theo.

## Khôi phục (nếu cần)
```bash
cp data/snapshots/local_package_2026-09-17/recsys.db data/recsys.db
cp data/snapshots/local_package_2026-09-17/item_* data/embeddings/
```
Lưu ý: các file `*.db`, `data/embeddings/*` đã bị `.gitignore` loại khỏi git,
snapshot cũng không được commit — chỉ tồn tại local.
