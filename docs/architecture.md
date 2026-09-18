# Architecture and operational decisions

## Request flow

Streamlit → FastAPI `/api/v1/recommend/{user_id}` → Redis cache or process-local
memory fallback → candidate generation → reranker → response validation.

Candidate generators: weighted popularity, SVD collaborative filtering, metadata
content embeddings, and Hybrid with normalized candidate scores (default CF 0.6,
CB 0.4). Cold-start users use popularity. Content search uses Qdrant when supplied
and available, otherwise local similarity. API startup loads SQL catalog/events
and trains models; it does not load a versioned production model artifact.

The reranker combines relevance, Bayesian ratings, freshness, category limits,
and purchase suppression. API interactions persist to SQL, invalidate the user's
cache version, and update the in-process model. Alembic manages SQL schema;
SQLite is the development default, PostgreSQL is the deployment option.

## Decisions and trade-offs

- Redis is a cache, **not a message broker**. Memory fallback keeps a single
  process available but cannot coordinate replicas or survive restarts.
- Optional Qdrant/Groq integrations have fallback paths. Offline tests prove
  those paths, not cloud deployment correctness.
- CI runs on pushes and pull requests to main: Python 3.11, dependency install,
  synthetic SQLite seed, pytest, then Docker build with a second pytest gate.
  Repository branch protection must separately require the checks; a workflow
  alone does not prevent direct pushes.
- Docker uses an unprivileged runtime user. Test-generated data is removed before
  runtime so Alembic can initialize a fresh database. Populate data explicitly
  before expecting recommendation readiness. Railway currently uses Nixpacks,
  so Docker hardening does not automatically change Railway deployment.

## Observability today

Health JSON exposes database connectivity and model readiness; HTTP 200 alone
is not a readiness guarantee. Docker's curl probe checks HTTP availability only.
Startup logs, Uvicorn logs, circuit-breaker messages, and per-response latency
exist. No centralized metrics, tracing, alerts, or latency SLO has been verified.
Production monitoring should separately track ready state, request p50/p95/p99,
errors, cache hits, fallback usage, model version and evaluation quality.
