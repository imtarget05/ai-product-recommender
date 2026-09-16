# System verification — 2026-09-16

## Verified in this audit

- Full existing suite plus regressions: **48 passed**, 2 dependency deprecation warnings, 15.72 s.
- Focused regressions: **11 passed**, 5.34 s.
- Python compilation of source, scripts and tests: passed.
- `git diff --check`: passed.
- Full suite used a temporary SQLite backup of the existing database. Qdrant/Groq credentials were overridden with empty environment variables; Redis pointed to a closed local port. This verifies local fallback behavior, not cloud connectivity.
- No independent subagents were available. Work was organized by subsystem and independent checks were batched; these must not be represented as separate agents.

## Confirmed defects and changes

### Cache

Memory cache values were shared with callers, allowing nested mutations to corrupt subsequent responses. Deep copies now isolate both writes and reads. A regression checks mutation of nested recommendation objects.

The first Redis invalidation previously reused the initial recommendation version: missing version defaulted to 1, while Redis INCR on a missing key also produced 1. Missing Redis versions now default to 0. A mocked Redis regression covers the first invalidation. Distributed outage/recovery and in-flight stale-writer races are not covered by this fix.

### Content-based / Qdrant

The availability method is now public. Content-based recommendations use local cosine fallback for missing or unavailable stores, exceptions and empty results. All five cases, including successful ANN retrieval, pass deterministic tests. An intermediate indentation error was observed; the latest source restored the availability guard and passes compilation. The availability method checks client/circuit state only; it does not perform a network health probe.

### Collaborative filtering

Regression tests reproduced two defects even after the original tests passed:

1. The SVD exception fallback used an interaction matrix and its transpose as factors, giving incompatible latent widths on rectangular data.
2. Empty refits retained previously fitted state and returned stale recommendations.

Refits now reset mappings, factors, suppression state and fitted status. Both degenerate matrices and SVD errors use consistent factors in a shared basis, choosing the smaller basis dimension. Both paths pass through normalization. This is a correctness fallback, not a scalable replacement for sparse factorization: dense conversion can still be expensive for large catalogs.

### Other working-tree changes reviewed

The tree also contains pagination/search bounds, literal LIKE wildcard escaping, event-type schema validation, zero-signal ANN query suppression and changed Groq model defaults. The full suite passes with these changes. Live Groq model availability was not validated. Do not infer that every behavior of these changes has dedicated regression coverage.

## Remaining priorities before production

1. **Test isolation by default:** existing integration fixtures use configured services/database. This run explicitly isolated them; plain pytest with real credentials may still access services or write interactions. Add a central isolated test configuration before CI automation.
2. **Startup readiness:** startup currently requires both products and interactions; a new catalog with no events cannot serve popularity cold-start recommendations. Global readiness/model state also needs lifecycle reset tests.
3. **Authentication and authorization:** review admin reload and user-scoped actions before exposing the service publicly. User IDs supplied by clients are not a substitute for authenticated ownership.
4. **Search relevance:** the fallback labeled keyword search does not actually apply the query text; it filters attributes and takes catalog rows. Returning arbitrary items with a fixed relevance score is not evidence of semantic relevance. Add meaningful query matching and no-match tests.
5. **Retrieval correctness:** enforce category/price bounds on vector hits, validate LLM-produced intent, and verify embedder/index version consistency across retrains.
6. **Evaluation validity:** temporal interaction splitting alone does not prove zero leakage. Product metadata/ratings may contain future information, and targets previously seen in training conflict with exclude-interacted serving. Define repeat-purchase vs novel-item evaluation and report them separately.
7. **Concurrency:** validate reload/interaction races, stale cache writers and Redis outage/recovery. Passing a thread-safety smoke test is not a distributed-consistency guarantee.
8. **Deployment verification:** PostgreSQL migrations, live Redis/Qdrant/Groq, Docker startup, browser UI and load/latency budgets still require dedicated runs.
9. **Dependencies:** two warnings remain for Starlette/httpx TestClient integration and the deprecated AnyIO BlockingPortal alias. Resolve with a compatible dependency update, not untested package additions.

## Baseline measurement, not a business claim

An earlier local offline evaluation reported pipeline Precision@5 = 0.0590, Recall@5 = 0.1053 and NDCG@5 = 0.0973. These are historical baseline observations on the local dataset, not a verified post-fix improvement or evidence of conversion/revenue uplift. Real value requires trustworthy evaluation data and an online experiment with guardrails.
