# REPAIR PLAN — ai-product-recommender

Audit: 2026-09-25. The code was **cloned, installed and executed** on Windows. Every "VERIFIED" below means it was run.

---

## Current State

FastAPI hybrid recommender. Two-stage classical-ML critical path (TruncatedSVD collaborative filtering + TF-IDF/SVD content-based, fused as `α·CF + (1−α)·CB`), a reranker, Redis caching with an in-memory TTL fallback, optional Qdrant ANN, Alembic migrations, a shopping-assistant agent on a separate router, and a Streamlit UI. 72 test functions.

**Execution note that matters:** on a bare `git clone` the suite reports **18 failed, 63 passed**. Seventeen of those failures are not bugs — CI runs `python -m scripts.generate_seed_data` before `pytest` (`.github/workflows/ci.yml:37-40`) and the README does not tell you so. After seeding, the suite reports **1 failed, 80 passed**, and that single failure is a genuine Windows-only defect.

---

## Verified Working Features

Verified after repair: **81 passed, 0 failed.**

- **Two-stage pure-ML critical path with no LLM in `/recommend`.** `src/api/routes.py:192-232` calls only `hybrid.recommend` / `cf_model` / `content_model` / `popularity_model` and the reranker. The LLM lives in a separate router (`src/api/agent_routes.py`, mounted at `main.py:98`). This claim is **true and verifiable by reading two files.**
- **Ensemble maths.** `src/models/hybrid.py:135-156` — per-source Min-Max normalisation then `α·CF + (1−α)·CB`, with weights from `settings.HYBRID_CF_WEIGHT` / `HYBRID_CB_WEIGHT` (`main.py:52-53`).
- **Cold-start routing by interaction count** — `src/models/hybrid.py:100-125`, covered by `tests/test_api.py:42`.
- **Content embedder with a hard dimension contract** — `src/features/text_embedder.py:26,47-83`, with padding/truncation to 64 dims at `:75-79`.
- **Reranker with three real signals** — Bayesian rating (`:58-61`), freshness (`:64-70`), category cap (`:18`) in `src/ranking/reranker.py`.
- **Per-request latency instrumentation** — `src/api/routes.py:232` emits exactly `[Metrics] ML Pipeline - Strategy: …, Retrieve: Xms, Rank: Yms`, as claimed.
- **Fail-closed model-bundle verification** — `src/serving/model_bundle.py:33-64` validates manifest fields and per-artifact SHA-256, with negative tests at `tests/test_models.py:87-140` (checksum mismatch, missing field).
- **Real Alembic migrations with an upgrade/downgrade test** — `alembic/versions/001_initial_schema.py`, `002_durable_workflow.py` (idempotency keys + outbox); `Procfile:1` and `docker/Dockerfile:51` both run `alembic upgrade head` before uvicorn.
- **Redis with graceful fallback and O(1) invalidation** — `src/api/cache.py:24-46,60-70` using a user-version counter.
- **Cart-action boundary** — `src/commerce/cart_gateway.py` + `src/agent/shopping_agent.py`; with no adapter the assistant emits `PROPOSED` (`tests/test_api.py:88`); with an adapter it **requires** an idempotency key and records exactly once (`tests/test_agent.py:109,137,167`).
- **Genuinely honest evaluation reporting** — `data/evaluation/seed_results.json:5` literally states `"Synthetic seed snapshot; not RetailRocket"` and lists 7 stated limitations (`:38-46`) including leakage caveats. This is the most honest eval documentation in the portfolio and should be highlighted in an interview.
- Real relational schema with FKs and composite indexes — `src/database/models.py:90-93`.

---

## Broken Features

### B1 — The production serving path cannot serve · **P0**

`src/api/main.py:36-39` returns from the lifespan **immediately after loading the bundle**, so `app_state["hybrid_model"]` is never populated in production. Nothing anywhere in the codebase reconstructs a model from the bundle — `data/model_bundle/serving_payload.json` contains only two hyperparameters (`{"hybrid_cf_factors":32,"cf_factors":16}`), no factors, no vectors, no catalogue.

Therefore every `GET /api/v1/recommend/{user_id}` reaches `src/api/routes.py:172-173` and returns **503**, while `/health/ready` reports ready.

This is the headline architectural problem: the "immutable bundle" and "train/inference separation" story is **integrity-checked but functionally unused**.

### B2 — The Docker image deletes the bundle it requires

`docker/Dockerfile:33` runs `rm -rf data && mkdir -p data`. The model bundle lives in `data/model_bundle/`. So an `APP_ENV=production` deploy crashes earlier than B1, at `main.py:28-30`, with `MODEL_BUNDLE_INVALID`.

### B3 — SQLite-only SQL on the documented Postgres target

`src/api/routes.py:372` uses `INSERT OR REPLACE`. That syntax does not exist in PostgreSQL. The documented target is Railway / Neon / Render Postgres (`railway.json`, `render.yaml`, README). The idempotency feature will 500 the moment `X-Idempotency-Key` is used in production. Only exercised against SQLite in tests, so CI cannot catch it.

### B4 — Readiness does not check the model · **P0**

`src/api/routes.py:87` computes `ready = db_ok and bundle is not None and cache_ok`. It **never checks `app_state["hybrid_model"]`**. A pod that answers every `/recommend` with 503 reports `ready: true`. Compounded by `render.yaml` health-checking `/health/live` anyway, so even a correct `/health/ready` would not gate the deploy.

### B5 — A test that validates a file nothing ships

`tests/recsys/test_dockerfile.py:8-18` asserts that `src/recsys/Dockerfile` has ≥2 `FROM` stages. That Dockerfile is **not referenced by CI, by `docker-compose.yml`, by `render.yaml`, or by `railway.json`.** Worse, it is internally broken: its runtime stage runs `python -m pytest tests/` **without ever copying `tests/`** into the stage. So the only "Docker test" in the suite validates a dead file that could not work if used.

---

## Half-implemented Features

### H1 — Qdrant is an adapter with no evidence of a live cluster

`src/services/qdrant_service.py` and `tests/integration/test_qdrant_container.py:121-174` (real Qdrant upsert/search + Redis roundtrip) are real, but the container tests are **opt-in and not in CI**. `CLOUD_VERIFIED: NO`.

### H2 — Ollama/Groq are deliberately off the critical path

`src/agent/groq_client.py` has a circuit breaker (`tests/test_agent.py:30`) and heuristic fallback (`tests/test_agent.py:52,250`). This is correct design and honestly documented. `DEV_ONLY`, keep as is.

### H3 — The Colab notebook is not the serving path

`colab/train_recsys_T4.ipynb` exists, but serving does **not** consume its exported artifacts — see B1. The "training happens in Colab, serving loads artifacts" story is `NOT VERIFIED`.

### H4 — `/metrics` reports hardcoded zeros

`src/api/routes.py:476-491` hardcodes `queue_depth 0` and `dlq_count 0`, with a comment admitting the durable queue is unwired. That is a metric stub being served as a metric.

---

## Documentation Claims Not Verified

| Claim | Location | Status |
|---|---|---|
| "**SLA p95 < 50ms**" | `README:11` | **FALSE / UNSUPPORTED.** There is no p95 measurement anywhere in the repository. The only latency instrumentation is a per-request **mean** log line (`routes.py:232`) and a self-reported `latency_ms` in the response body (`:249-255`). `README:29` already retracts a sub-20 ms claim, yet the 50 ms line is left in the header block. Delete it. |
| "Baseline 48 passed, 2 warnings" | `README:29` | **STALE.** 72 test functions exist today. The README hedges ("xem CI để biết số hiện tại") which is honest but unhelpful. |
| "Immutable model bundle … fail-closed" | `README` | **PARTIALLY VERIFIED.** Verification is real and strict; **use of the bundle is missing** (B1, B2). |
| "`/health/ready` (503 khi thiếu DB/bundle/cache)" | `README:8` | **PARTIALLY VERIFIED.** The endpoints are separated and the 503 body is correct, but readiness is computed wrongly (B4). |
| "GitHub Actions: pytest offline → Docker build" | `README` | **VERIFIED.** `.github/workflows/ci.yml:17-54`, with Redis/Qdrant/Groq explicitly blanked at `:21-27` to force the offline paths, and the image build re-running the whole suite as a build layer. |
| "Alpinebic auto migrate on deploy" | `README:15` | **VERIFIED** — `Procfile:1`, `docker/Dockerfile:51`. |
| "PostgreSQL on Railway, Redis, Qdrant Cloud" | `README` | **CONFIGURED.** All `sync: false`. Nothing `CLOUD_VERIFIED`. |
| RetailRocket evaluation | `README:12` | **DOCUMENTED ONLY, and honestly labelled.** `data/raw/` is empty (`.gitkeep`); both eval JSONs state the data is synthetic. |

---

## Security Problems

### S1 · P0 — No authentication anywhere

`/recommend/{user_id}`, `/users`, `/products` and `/interact` are fully anonymous. `user_id` is a **trusted path parameter** (`routes.py:156`). Anyone can read the user list — **including email addresses** (`routes.py:437-462`) — and write interactions for any user. There is no rate limiting.

### S2 · P0 — Fail-open admin endpoint

`src/api/routes.py:387-389`: `if not expected: return True`. With `ADMIN_API_KEY` unset, `POST /admin/reload-model` is unauthenticated — and it triggers a **full in-process model refit** (`routes.py:399-416`). `render.yaml` never sets that key, so a stock Render deployment is open. The comparison at `:390` is a plain `!=`, not constant-time.

### S3 · Verifiable positives — keep these

- **CORS is correct.** Explicit `CORS_ORIGINS`, `allow_credentials=False` (`src/api/main.py:86-94`). This is better than most repos in the portfolio, including CreditFlow and ApexInspect.
- **No SQL injection.** Parameterised ORM throughout. The product search explicitly escapes LIKE wildcards (`routes.py:432-433`) — a nice touch.
- **Solid input validation.** Pydantic v2 schemas (`src/api/schemas.py`), `ge`/`le` bounds on `user_id`/`top_k` (`routes.py:156-157`), an event allowlist (`:301-306`), rating validation (`tests/test_api.py:76`).
- **No committed secrets.** `.env.example` only; `.gitignore` covers `.env`; `render.yaml:30-45` all `sync: false`.
- **No sensitive logging.** The `[Metrics]` line logs only ids, strategy and timings (`routes.py:232`).

---

## Testing Gaps

### T1 — The Windows-only test failure · **REPAIRED**

`tests/test_migration_002.py:43` built a SQLAlchemy engine and **never disposed it**, so the SQLite file handle stayed open in the pytest process. The next test's `os.remove(DB)` at line 26 then raised `PermissionError: [WinError 32]`. POSIX allows unlinking an open file; Windows does not. Secondary: line 13 hardcoded `/tmp/recsys_mig002_test.db`, a path that does not exist on Windows.

**Fix applied:** `_names()` now wraps the engine in `try/finally: engine.dispose()`, and `DB` uses `tempfile.gettempdir()`. **Verified: 81 passed, 0 failed.**

### T2 — The seed step is undocumented

17 of the original 18 failures were simply "no seed data". Anyone cloning the repo and running `pytest` sees 18 red tests and concludes the project is broken. Add the seed command to the README's Local Development section, or add a `conftest.py` hook that seeds automatically.

### T3 — No E2E and no load tests

No test drives HTTP → DB → model → response and asserts on ranking output beyond "is a list". No load test, despite a p95 claim in the README.

### T4 — Container integration tests are not in CI

`tests/integration/test_qdrant_container.py` is real and valuable but opt-in, so the Redis and Qdrant paths are never exercised by the pipeline.

### T5 — About 12 of 72 tests are file/config shape checks

Including the Dockerfile test theatre at B5. Do not quote 72 as behavioural coverage.

---

## Deployment Gaps

- **CD: NOT IMPLEMENTED.** There is no deploy workflow. `railway.json` and `render.yaml` are `autoDeploy` **configs**, not pipelines, and no pipeline step verifies a deploy.
- **Cloud: CONFIGURED_ONLY across the board** — Railway, Render, Neon Postgres, Redis, R2, Qdrant Cloud. `CLOUD_VERIFIED: NO`.
- **Two compose files that disagree** — `docker-compose.yml` and `docker/docker-compose.yml` are near-duplicates (pgvector/pg16 + redis:7 + backend) with a hardcoded `postgrespassword` at `:15`/`:51` and drift risk between them.
- **One dead Dockerfile** (`src/recsys/Dockerfile`) that could not build even if used.
- **Three overlapping deploy targets** for one service (`railway.json` + `render.yaml` + `Procfile`) — config sprawl.

---

## Recruiter-facing Problems

1. **Production cannot serve recommendations.** A reviewer who sets `APP_ENV=production` and calls `/recommend` gets a 503 and a health check that says everything is fine. This is the worst possible first impression for a project whose entire pitch is the serving path.
2. **SQLite syntax in a Postgres deployment.** A three-line bug that breaks the idempotency feature in production, invisible to CI because CI only runs SQLite.
3. **An open admin endpoint that triggers a full refit.** Fail-open auth is a pattern interviewers probe for specifically.
4. **"SLA p95 < 50ms" in the header** with no measurement behind it. Deleting one line removes the largest credibility risk in the README.
5. **A test that validates an unused Dockerfile.** If a reviewer spots that the "Docker test" tests a file nothing ships, they will start looking for other tests that do not test anything.
6. **17 of 18 initial test failures on a fresh clone** is a five-minute impression problem that a single README line fixes.

---

## Repair Tasks

### P0 — blocking

- [x] **P0-1 Fix the Windows-only test failure in `tests/test_migration_002.py`.** *(DONE — dispose the engine, use `tempfile.gettempdir()`. Verified 81 passed.)*
- [ ] **P0-2 Make the production path actually serve.** Either (a) define a real serialisation format and have `main.py:36-39` reconstruct the hybrid model from the bundle, or (b) drop the bundle-from-Notebook story and always fit at startup from the DB. **Option (b) is smaller and more honest.** Whichever is chosen, add a test that boots with `APP_ENV=production` and asserts `/api/v1/recommend/1` returns 200 with a non-empty ranking.
- [ ] **P0-3 Stop `docker/Dockerfile` from deleting the bundle.** Remove `rm -rf data` at `:33`, or move the bundle outside `data/`. Test: build the image, run it, assert `/health/ready` is true.
- [ ] **P0-4 Replace `INSERT OR REPLACE` at `src/api/routes.py:372`** with a dialect-portable upsert (`sqlalchemy.dialects.postgresql.insert(...).on_conflict_do_update(...)` with a SQLite fallback). Add a test that runs the interact-idempotency path against a **real Postgres service container**, not only SQLite.
- [ ] **P0-5 Make `/health/ready` check the model.** Add `app_state["hybrid_model"] is not None` at `routes.py:87`. Test: clear the model, assert ready is false.
- [ ] **P0-6 Close the fail-open admin check at `routes.py:387-389`.** Fail closed when `ADMIN_API_KEY` is unset in production, and use constant-time comparison. Test: unset key + production → 503/401.
- [ ] **P0-7 Add authentication to `/interact`, `/users` and `/recommend`.** At minimum, require an API key on all write paths and stop returning user emails from `/users`.

### P1 — important

- [ ] **P1-1 Delete the `SLA p95 < 50ms` line** from `README:11`, or add a real latency benchmark and commit its output. Do not leave an unmeasured number in the header.
- [ ] **P1-2 Document the seed step** in the README's Local Development section, and add it to a `Makefile` target so `make test` does the right thing.
- [ ] **P1-3 Delete `src/recsys/Dockerfile` and `tests/recsys/test_dockerfile.py`.** Then delete one of the two duplicate compose files. A test that asserts properties of an unused file is worse than no test.
- [ ] **P1-4 Reconcile the "48 passed" badge** with the real count.
- [ ] **P1-5 Add a real CD workflow** with a post-deploy health assertion that **fails** the job on a bad response.
- [ ] **P1-6 Replace the hardcoded `postgrespassword`** in both compose files with an env var that has no default.
- [ ] **P1-7 Gitignore `data/snapshots/**/*.npy`** — a committed model artefact.
- [ ] **P1-8 Implement `/metrics` honestly** or remove `queue_depth` and `dlq_count` from it.

### P2 — nice-to-have

- [ ] **P2-1** Add a load test so a latency number, if ever quoted, is measured.
- [ ] **P2-2** Add the Qdrant/Redis container tests to CI as a second job.
- [ ] **P2-3** Add a real E2E test: HTTP → DB write → model update → changed recommendation.
- [ ] **P2-4** Consolidate to a single deploy target and delete the others.
- [ ] **P2-5** Write `docs/RECRUITER-EVIDENCE.md` mapping each CV bullet → implementation file → test → runtime observation.
