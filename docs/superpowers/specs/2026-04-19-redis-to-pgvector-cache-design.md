# Design: Replace Redis Semantic Cache with pgvector

**Date:** 2026-04-19
**Status:** Approved

---

## Problem

The current semantic cache (`app/utils/semantic_cache.py`) uses Redis as a backing store but provides none of Redis's actual value. On every lookup it:

1. Issues a full `SCAN` over all `cache:incident:*` keys
2. Fetches each key individually
3. Computes cosine similarity in Python against every stored embedding

This is O(n) per lookup and grows linearly with the number of cached incidents. It also requires a second infrastructure dependency (Redis) alongside the Postgres/Supabase instance that already exists and already stores resolved incidents with embeddings in the `resolved_incidents` table, backed by an ivfflat ANN index (`vector_cosine_ops`).

---

## Solution

Delete the Redis cache entirely. Add a single `cache_lookup` function to `app/utils/incident_store.py` that queries `resolved_incidents` using the existing pgvector ivfflat index. The lookup becomes O(log n) and requires no new infrastructure.

---

## What Is Deleted

| Artifact | Reason |
|---|---|
| `app/utils/semantic_cache.py` | Entire module replaced |
| `tests/test_semantic_cache.py` | Tests for deleted module |
| `config.py: redis_url` | No Redis dependency |
| `config.py: semantic_cache_enabled` | Artifact of early testing; cache always runs when `postgres_dsn` is set |
| `config.py: cache_ttl_seconds` | Meaningless for Postgres rows; incidents persist until deleted |
| `requirements.txt: redis>=7.4.0` | No Redis dependency |

---

## What Is Added

### `incident_store.cache_lookup`

```python
async def cache_lookup(
    dsn: str,
    error_summary: str,
    threshold: float = 0.95,
) -> dict | None:
```

**Behaviour:**
- Returns `None` immediately if `dsn` or `error_summary` is empty.
- Embeds `error_summary` using `text-embedding-3-small` (reuses `_embed_text`).
- Runs against `resolved_incidents`:

```sql
SELECT recommended_action, error_summary,
       1 - (embedding <=> $1::vector) AS similarity
FROM resolved_incidents
WHERE embedding IS NOT NULL
ORDER BY embedding <=> $1::vector
LIMIT 1;
```

- Returns the row dict if `similarity >= threshold`, else `None`.
- Swallows all errors (logs warning, returns `None`) — cache misses are non-fatal.

**Why cosine only (no hybrid FTS):**
At a 0.95 threshold, semantic similarity alone is already a high-precision gate. Hybrid search is appropriate for broad recall (SOP lookup); it is not appropriate here. If testing reveals false positives or misses, a hybrid approach can be layered on top without changing the interface.

---

## What Is Changed

### `app/nodes/router.py`

- `_check_cache` calls `incident_store.cache_lookup(dsn=settings.postgres_dsn, error_summary=..., threshold=settings.cache_similarity_threshold)`.
- Remove import of `app.utils.semantic_cache`.
- Remove `semantic_cache_enabled` guard — cache lookup runs whenever `postgres_dsn` is non-empty.

### `app/nodes/verification.py`

- `_persist_resolution` drops the entire Redis `cache_store` block.
- `save_resolved_incident` already writes `embedding` to `resolved_incidents` — this is the store path, no new code needed.
- Remove import of `app.utils.semantic_cache`.

### `app/config.py`

- Remove: `redis_url`, `semantic_cache_enabled`, `cache_ttl_seconds`.
- Keep: `cache_similarity_threshold` (consumed by `router.py`).

---

## Data Flow After Migration

```
Alert received
    │
    ▼
router_node
    │ embed(error_summary)
    │ SELECT ... ORDER BY embedding <=> $1 LIMIT 1
    │ similarity >= 0.95?
    ├── YES → return cached recommended_action → human_gate (short-circuit)
    └── NO  → processor → researcher → action → verification_node
                                                        │
                                                        ▼
                                              save_resolved_incident
                                              (writes embedding to resolved_incidents)
```

---

## Tests

Delete `tests/test_semantic_cache.py`.

Add to `tests/test_incident_store.py`:

| Test | Scenario |
|---|---|
| `test_cache_lookup_hit` | Similarity above threshold → returns dict with `recommended_action` |
| `test_cache_lookup_miss_below_threshold` | Nearest row exists but similarity < 0.95 → returns None |
| `test_cache_lookup_empty_dsn` | `dsn=""` → returns None immediately, no DB call |
| `test_cache_lookup_embedding_failure` | OpenAI call raises → returns None, logs warning |
| `test_cache_lookup_db_error` | asyncpg raises → returns None, logs warning |

All DB interactions mocked via `unittest.mock.AsyncMock`.

---

## Infrastructure

No migrations required. `resolved_incidents` table and its ivfflat index already exist (`migrations/003_resolved_incidents.sql`, `bootstrap.sql`).
