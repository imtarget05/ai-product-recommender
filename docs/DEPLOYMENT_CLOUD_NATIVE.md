# DEPLOYMENT CHỐT: Cloud-Native Traditional ML (Non-LLM core)

> Single deployment target. Core recommend không phụ thuộc LLM.

## 1. Two-stage pipeline (critical path, 100% Non-LLM)

```
Postgres (events/catalog) ─┬─▶ Retrieve: SVD CF + Qdrant ANN (TF-IDF 64-dim) + Popularity
                           │   Hybrid ensemble α·CF + (1−α)·CB, top 2K candidates
                           ▼
                    Rank: multi-objective reranker
                    (rating boost · freshness · category diversity · purchase suppression)
                           ▼
                    FastAPI /api/v1/recommend → Redis cache → client
```

- Retrieve và Rank đo latency **riêng biệt** mỗi request
  (`retrieve_latency` / `rank_latency`, log `[Metrics] ML Pipeline`,
  `src/api/routes.py`). Tuning/scale từng stage độc lập.
- Cold-start: user < `COLD_START_THRESHOLD` interactions → Popularity fallback.

## 2. Scale & SLA

| Mảnh | Scale | Ghi chú |
|---|---|---|
| FastAPI serving | Horizontal replicas (stateless, Railway/autoscale) | In-memory hybrid model + `POST /admin/reload-model` hot-reload |
| Redis | Shared cache, per-user versioning O(1) invalidate | TTL `CACHE_TTL_SECONDS=300` |
| Postgres | Managed (Railway), Alembic migrate on deploy | Catalog + interactions + logs |
| Qdrant | Cloud Free Tier, ANN Cosine, 64-dim | Re-index khi đổi embedding |

**SLA: p95 end-to-end `/recommend` < 50ms** (cache hit ≪ 50ms; cache miss =
Retrieve + Rank + 1 DB query purchases). Retrieve vs Rank breakdown trong log
cho phép phát hiện stage nào vượt ngân sách.

## 3. Training (Colab) vs Inference (Cloud) — tách bạch

| | Training (offline, Colab) | Inference (online, Cloud) |
|---|---|---|
| Nơi chạy | `colab/train_recsys_T4.ipynb` (GPU T4) | FastAPI replicas + Redis + Qdrant/Postgres |
| Việc | Classical fitting: SVD latent factors, TF-IDF 64-dim, export artifacts | Retrieve SVD/Vector → Rank reranker → cache → serve |
| Input | RetailRocket `events.csv` / `item_properties.csv` | Live Postgres + Qdrant + in-memory model |
| LLM? | Không | Không (core). LLM chỉ ở `/agent/*` optional |
| Cadence | Theo batch / khi data drift | Realtime per-request, hot-reload qua `/admin/reload-model` |

## 4. LLM optional (ngoài critical path)

- `POST /api/v1/agent/search|chat|explain` — Ollama local (`qwen2.5:3b`)
  → Groq → heuristic fallback. Circuit-breaker 3 fails / 30s.
- Ollama/Groq down → shopping agent trả heuristic, **`/recommend` không ảnh hưởng**.
- Cấu hình: `OLLAMA_URL` unset = Groq-only; cả hai unset = heuristic thuần.
