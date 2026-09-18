# STATELESS FREE-INFRA — Hướng dẫn cấu hình free-tier ($0)

RecSys-AI chạy stateless trên free-tier: artifact qua storage interface,
DATABASE_URL trỏ Neon/Supabase free, không path cứng local.

## 1. Storage backends (`STORAGE_BACKEND`)

| Giá trị   | Dùng khi                          | Cấu hình |
|-----------|-----------------------------------|----------|
| `local`   | dev / CI / single-instance        | `OUT_DIR` = thư mục gốc artifact (default `./data/embeddings`) |
| `s3mock`  | test integration, S3-compat mock  | `S3_ENDPOINT` = URL mock server |
| `r2`      | Cloudflare R2 (S3-compat, free)   | `R2_ENDPOINT` = `https://<account>.r2.cloudflarestorage.com/<bucket>` |

Backend nằm ở `src/storage/` — stdlib-only (urllib), không SDK ngoài:

- `base.py` — ABC: `save(key, bytes)`, `load(key)`, `exists(key)`, `delete(key)`
- `local_disk.py` — LocalDisk, atomic write (tmp + rename)
- `s3mock.py` — S3Mock/R2-compat (PUT/GET/HEAD/DELETE)
- `factory.py` — `get_storage()` đọc `STORAGE_BACKEND`

## 2. ItemEmbedder qua storage

`ItemEmbedder.save(storage=...)` / `.load(storage=...)` persist artifact
(`item_embeddings.npy`, `item_embedder_meta.pkl`) qua storage interface.
Giữ backcompat: `save(output_dir=...)` / `load(output_dir=...)` như cũ.

Routes đọc artifact qua `src/api/routes.py::_storage_load_artifact(key)` —
không đọc path cứng local. Qdrant wiring giữ nguyên contract.

## 3. DATABASE_URL (Neon/Supabase free)

- Default: `sqlite:///./data/recsys.db`
- Free-tier postgres: `DATABASE_URL=postgresql://user:pass@host/db`

## 4. Chạy test

```bash
pytest tests/test_storage_stateless.py -v
```

## 5. Deploy free-tier gợi ý

- API: Render/Railway free — stateless, scale-to-zero.
- DB: Neon/Supabase free qua `DATABASE_URL`.
- Artifact/vector: R2 + Qdrant cloud free tier.
