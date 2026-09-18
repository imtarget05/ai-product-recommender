# Limitations and evidence

## Evaluation

`data/evaluation/seed_results.json` is a measured **synthetic local seed snapshot**,
not RetailRocket and not production evidence. It includes four models plus an
optional reranker, dataset fingerprint, counts, cutoff, parameters, package
versions and limitations. The snapshot was evaluated after API tests, which add
interactions. Raw database rows are not published. A fingerprint identifies the
snapshot but does not make it reconstructible.

The seed generator uses unseeded randomness and wall-clock timestamps; fresh
runs will differ. It clears existing data: only run it against a disposable DB.
To generate a new independent report from the repository root:

```sh
export DATABASE_URL="sqlite:////tmp/recsys-disposable-evaluation.db"
export QDRANT_URL='' QDRANT_API_KEY='' GROQ_API_KEY=''
python -m scripts.generate_seed_data
python -m scripts.evaluate_models --output /tmp/recsys-evaluation.json --dataset-label 'synthetic seed'
```

This preserves the checked-in report. Never point the seed command at production.
For real-data evaluation, select a prepared database via DATABASE_URL and omit
seeding. RetailRocket raw files are ignored by Git; obtain the dataset separately
and inspect the ingestion script. Clean checkout ETL is not yet reproducible.

Event splitting is temporal, but catalog metadata/ratings are current snapshots;
freshness uses current time. Thus **zero future leakage is not established**.
Repeated items remain in holdout relevance although recommendations exclude
train-seen items. Graded-NDCG exists as a helper but is not used in the aggregate
report. Standalone CF and Hybrid use different factor counts. Evaluation reranking
does not pass API purchase history. Do not infer Hybrid superiority from this run.

## Serving and deployment

- Sub-20ms claims were removed: no production latency benchmark exists. A future
  benchmark must separate cache hits, misses, concurrency, warmup, payload size,
  network and cloud-service time; TestClient timing is not network latency.
- Docker build was attempted locally but blocked by an unavailable daemon.
  GitHub Actions execution, non-root container startup and live PostgreSQL,
  Redis, Qdrant and Groq deployments still require validation.
- Requirements have broad version ranges, not a dependency lockfile.
- Serving now loads an immutable, checksum-verified bundle (`MODEL_BUNDLE_PATH`)
  in production; `RECSYS_TRAIN_ON_START` is a development-only escape hatch.
  The bundle pins the evaluation report fingerprint and hybrid weights, so two
  replicas serve the same version — but the bundle does not yet contain the
  fitted CF/content matrices, so a production replica still refits from the
  database at startup unless a future bundle adds serialized payloads.
- Admin reload and interaction endpoints accept deployment-level authentication
  via `ADMIN_API_KEY`; when the variable is empty (local development) they are
  unauthenticated, so it MUST be set before public exposure.
- Liveness (`/health/live`) and readiness (`/health/ready`) are separate.
  Readiness returns HTTP 503 and requires DB + bundle, plus Redis when
  `APP_ENV=production` and `CACHE_MODE=redis`. Load balancers must route only
  ready instances. `/health` remains a legacy informational probe that can
  return HTTP 200 with `status=degraded`.
- CORS no longer uses a wildcard with credentials; origins come from
  `CORS_ORIGINS` (default `http://localhost:3000`).
- Console logging is not a complete monitoring/alerting system.

Secrets belong only in local environment or deployment secret storage. Never
publish .env files; rotate credentials if they were exposed in logs or tools.
