# Redis → pgvector Semantic Cache Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the O(n) Redis-backed semantic cache with a threshold-filtered pgvector ANN lookup on the existing `resolved_incidents` table, eliminating Redis as a dependency entirely.

**Architecture:** Add `cache_lookup` to `app/utils/incident_store.py` — it embeds the query and runs a single `ORDER BY embedding <=> $1::vector LIMIT 1` query against the existing ivfflat index, returning the row only if `1 - distance >= threshold`. Update `router.py` to call it. Simplify `verification.py` to drop the now-redundant Redis store block (`save_resolved_incident` already writes embeddings). Delete `semantic_cache.py`, its tests, and all Redis config/deps.

**Tech Stack:** asyncpg, pgvector (`<=>` cosine distance operator), OpenAI text-embedding-3-small, pytest-asyncio, unittest.mock

---

## File Map

| File | Action | Change |
|---|---|---|
| `tests/test_incident_store.py` | Modify | Add 5 `cache_lookup` tests to existing file |
| `app/utils/incident_store.py` | Modify | Add `cache_lookup` function |
| `app/nodes/router.py` | Modify | `_check_cache` calls `incident_store.cache_lookup`; drop `semantic_cache_enabled` guard |
| `app/nodes/verification.py` | Modify | Simplify `_persist_resolution` — remove Redis block |
| `app/config.py` | Modify | Remove `redis_url`, `semantic_cache_enabled`, `cache_ttl_seconds` |
| `app/utils/semantic_cache.py` | Delete | Entire module gone |
| `tests/test_semantic_cache.py` | Delete | Tests for deleted module |
| `requirements.txt` | Modify | Remove `redis>=7.4.0` |

---

### Task 1: Write failing tests for `cache_lookup`

**Files:**
- Modify: `tests/test_incident_store.py` (append a new `TestCacheLookup` class)

- [ ] **Step 1: Append `TestCacheLookup` to `tests/test_incident_store.py`**

Add this class at the end of the file:

```python
class TestCacheLookup:
    @pytest.mark.asyncio
    async def test_cache_lookup_hit(self):
        """Returns dict when nearest row similarity >= threshold."""
        fake_row = {
            "recommended_action": "restart the pod",
            "error_summary": "OOMKilled in namespace default",
            "similarity": 0.97,
        }
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=fake_row)
        mock_conn.close = AsyncMock()

        with patch("app.utils.incident_store._embed_text", return_value=[0.1] * 1536), \
             patch("asyncpg.connect", return_value=mock_conn):
            result = await cache_lookup("postgresql://test", "OOMKilled pod crashing")

        assert result is not None
        assert result["recommended_action"] == "restart the pod"

    @pytest.mark.asyncio
    async def test_cache_lookup_miss_below_threshold(self):
        """Returns None when nearest row similarity < threshold."""
        fake_row = {
            "recommended_action": "restart the pod",
            "error_summary": "OOMKilled in namespace default",
            "similarity": 0.80,
        }
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=fake_row)
        mock_conn.close = AsyncMock()

        with patch("app.utils.incident_store._embed_text", return_value=[0.1] * 1536), \
             patch("asyncpg.connect", return_value=mock_conn):
            result = await cache_lookup("postgresql://test", "high CPU on worker node", threshold=0.95)

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_lookup_empty_dsn(self):
        """Returns None immediately when dsn is empty — no DB call made."""
        with patch("asyncpg.connect") as mock_connect:
            result = await cache_lookup("", "OOMKilled pod crashing")

        assert result is None
        mock_connect.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_lookup_embedding_failure(self):
        """Returns None when _embed_text returns empty list."""
        with patch("app.utils.incident_store._embed_text", return_value=[]), \
             patch("asyncpg.connect") as mock_connect:
            result = await cache_lookup("postgresql://test", "OOMKilled pod crashing")

        assert result is None
        mock_connect.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_lookup_db_error(self):
        """Returns None and logs warning when asyncpg raises."""
        with patch("app.utils.incident_store._embed_text", return_value=[0.1] * 1536), \
             patch("asyncpg.connect", side_effect=Exception("connection refused")):
            result = await cache_lookup("postgresql://test", "OOMKilled pod crashing")

        assert result is None
```

Also update the import line at the top of the file from:

```python
from app.utils.incident_store import fetch_similar_incidents, save_resolved_incident
```

to:

```python
from app.utils.incident_store import cache_lookup, fetch_similar_incidents, save_resolved_incident
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/Joseph/llmops/autonomous-sre && pytest tests/test_incident_store.py::TestCacheLookup -v
```

Expected: 5 failures — `ImportError: cannot import name 'cache_lookup' from 'app.utils.incident_store'`

---

### Task 2: Implement `cache_lookup` in `incident_store.py`

**Files:**
- Modify: `app/utils/incident_store.py`

- [ ] **Step 1: Add `cache_lookup` between `fetch_similar_incidents` and `_embed_text`**

Insert the following after the closing of `fetch_similar_incidents` and before `async def _embed_text`:

```python
async def cache_lookup(
    dsn: str,
    error_summary: str,
    threshold: float = 0.95,
) -> dict | None:
    """Return the nearest resolved incident if cosine similarity >= threshold.

    Uses the ivfflat index on resolved_incidents.embedding for O(log n) lookup.
    Returns None on empty dsn, embedding failure, no match above threshold, or any error.
    """
    if not dsn or not error_summary:
        return None

    embedding = await _embed_text(error_summary)
    if not embedding:
        return None

    sql = """
        SELECT recommended_action, error_summary,
               1 - (embedding <=> $1::vector) AS similarity
        FROM resolved_incidents
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> $1::vector
        LIMIT 1;
    """

    try:
        import asyncpg  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("asyncpg not installed — cache_lookup disabled")
        return None

    try:
        conn = await asyncpg.connect(dsn, statement_cache_size=0)
        try:
            row = await conn.fetchrow(sql, json.dumps(embedding))
            if row is None or row["similarity"] < threshold:
                return None
            return dict(row)
        finally:
            await conn.close()
    except Exception as exc:
        logger.warning("cache_lookup error (non-critical): %s", exc)
        return None
```

- [ ] **Step 2: Run tests to confirm they pass**

```bash
cd /Users/Joseph/llmops/autonomous-sre && pytest tests/test_incident_store.py -v
```

Expected output:
```
tests/test_incident_store.py::TestSaveResolvedIncident::test_no_op_for_empty_dsn PASSED
tests/test_incident_store.py::TestSaveResolvedIncident::test_no_op_for_empty_error_summary PASSED
tests/test_incident_store.py::TestSaveResolvedIncident::test_inserts_row_when_valid PASSED
tests/test_incident_store.py::TestSaveResolvedIncident::test_swallows_db_error PASSED
tests/test_incident_store.py::TestFetchSimilarIncidents::test_returns_empty_for_empty_dsn PASSED
tests/test_incident_store.py::TestFetchSimilarIncidents::test_returns_empty_for_empty_summary PASSED
tests/test_incident_store.py::TestFetchSimilarIncidents::test_returns_rows_from_db PASSED
tests/test_incident_store.py::TestFetchSimilarIncidents::test_returns_empty_on_db_error PASSED
tests/test_incident_store.py::TestCacheLookup::test_cache_lookup_hit PASSED
tests/test_incident_store.py::TestCacheLookup::test_cache_lookup_miss_below_threshold PASSED
tests/test_incident_store.py::TestCacheLookup::test_cache_lookup_empty_dsn PASSED
tests/test_incident_store.py::TestCacheLookup::test_cache_lookup_embedding_failure PASSED
tests/test_incident_store.py::TestCacheLookup::test_cache_lookup_db_error PASSED
13 passed
```

- [ ] **Step 3: Commit**

```bash
git add app/utils/incident_store.py tests/test_incident_store.py
git commit -m "feat: add cache_lookup to incident_store using pgvector ANN"
```

---

### Task 3: Update `router.py`

**Files:**
- Modify: `app/nodes/router.py`

- [ ] **Step 1: Replace the full contents of `app/nodes/router.py`**

```python
"""Router node — checks pgvector cache before processing.

If a similar past incident is found in resolved_incidents, short-circuits
directly to human_gate with the cached recommended action.
"""
from __future__ import annotations

import logging

from langsmith import traceable

from app.agents.state import SREState
from app.config import get_settings

logger = logging.getLogger(__name__)


@traceable(name="router_node", metadata={"phase": "routing"})
async def router_node(state: SREState) -> dict:
    """Check pgvector cache. On a hit, populate recommended_action and short-circuit."""
    settings = get_settings()
    error_summary = state.get("triage_summary", "")

    base_update: dict = {
        "cache_hit": False,
        "cache_key": "",
        "reasoning_log": list(state.get("reasoning_log", [])) + ["[router] cache_hit=False"],
        "current_node": "router",
    }

    if error_summary and settings.postgres_dsn:
        cache_result = await _check_cache(settings, error_summary)
        if cache_result is not None:
            recommended = cache_result.get("recommended_action", "")
            cache_key = cache_result.get("cache_key", "")
            return {
                **base_update,
                "cache_hit": True,
                "cache_key": cache_key,
                "recommended_action": recommended,
                "proposed_action": recommended,
                "reasoning_log": list(state.get("reasoning_log", [])) + [
                    f"[router] cache_hit=True | recommended_action={recommended!r}"
                ],
            }

    return base_update


async def _check_cache(settings, error_summary: str) -> dict | None:
    """Check pgvector cache and return cached entry if found."""
    try:
        from app.utils.incident_store import cache_lookup

        return await cache_lookup(
            dsn=settings.postgres_dsn,
            error_summary=error_summary,
            threshold=settings.cache_similarity_threshold,
        )
    except Exception as exc:
        logger.warning("Cache check failed (non-critical): %s", exc)
        return None
```

- [ ] **Step 2: Run router-related tests**

```bash
cd /Users/Joseph/llmops/autonomous-sre && pytest tests/ -v -k "router"
```

Expected: all router tests pass (or 0 collected — either is fine; no failures).

- [ ] **Step 3: Commit**

```bash
git add app/nodes/router.py
git commit -m "refactor: router uses incident_store.cache_lookup instead of redis"
```

---

### Task 4: Simplify `verification.py`

**Files:**
- Modify: `app/nodes/verification.py`

- [ ] **Step 1: Replace `_persist_resolution` to remove the Redis block**

Find `async def _persist_resolution` in `app/nodes/verification.py` and replace the entire function with:

```python
async def _persist_resolution(state: dict) -> None:
    """Fire-and-forget: save to incident_store. Swallows all errors."""
    dsn = os.environ.get("POSTGRES_DSN", "")

    if dsn:
        try:
            from app.utils.incident_store import save_resolved_incident
            await save_resolved_incident(dsn, state)
        except Exception as exc:
            logger.warning("Failed to save resolved incident (non-critical): %s", exc)
```

- [ ] **Step 2: Run verification tests**

```bash
cd /Users/Joseph/llmops/autonomous-sre && pytest tests/ -v -k "verification"
```

Expected: all verification tests pass.

- [ ] **Step 3: Commit**

```bash
git add app/nodes/verification.py
git commit -m "refactor: verification drops redis cache_store — save_resolved_incident covers it"
```

---

### Task 5: Clean up `config.py`

**Files:**
- Modify: `app/config.py`

- [ ] **Step 1: Remove three fields from `Settings`**

In `app/config.py`, remove the Redis block entirely:

```python
    # Redis (alert state store)
    redis_url: str = "redis://localhost:6379"
```

And replace the semantic caching block:

```python
    # Semantic caching
    cache_similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 86400
    semantic_cache_enabled: bool = False
```

with just:

```python
    # Semantic caching
    cache_similarity_threshold: float = 0.95
```

- [ ] **Step 2: Run full test suite to catch any references to removed fields**

```bash
cd /Users/Joseph/llmops/autonomous-sre && pytest tests/ -v
```

Expected: all tests pass with no `AttributeError` on `redis_url`, `semantic_cache_enabled`, or `cache_ttl_seconds`.

- [ ] **Step 3: Commit**

```bash
git add app/config.py
git commit -m "chore: remove redis_url, semantic_cache_enabled, cache_ttl_seconds from config"
```

---

### Task 6: Delete `semantic_cache.py` and `tests/test_semantic_cache.py`

**Files:**
- Delete: `app/utils/semantic_cache.py`
- Delete: `tests/test_semantic_cache.py`

- [ ] **Step 1: Delete both files**

```bash
rm /Users/Joseph/llmops/autonomous-sre/app/utils/semantic_cache.py \
   /Users/Joseph/llmops/autonomous-sre/tests/test_semantic_cache.py
```

- [ ] **Step 2: Run the full test suite to confirm no dangling imports**

```bash
cd /Users/Joseph/llmops/autonomous-sre && pytest tests/ -v
```

Expected: all tests pass, no `ImportError` or `ModuleNotFoundError`.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: delete semantic_cache.py and its tests — replaced by pgvector"
```

---

### Task 7: Remove `redis` from `requirements.txt`

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Remove the redis line from `requirements.txt`**

Delete this line:

```
redis>=7.4.0
```

- [ ] **Step 2: Confirm no remaining redis imports in app source**

```bash
grep -r "import redis" /Users/Joseph/llmops/autonomous-sre/app/
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: remove redis dependency — semantic cache now backed by pgvector"
```

---

## Self-Review

### Spec coverage

| Spec requirement | Task |
|---|---|
| Delete `semantic_cache.py` | Task 6 |
| Delete `tests/test_semantic_cache.py` | Task 6 |
| Remove `redis_url` from config | Task 5 |
| Remove `semantic_cache_enabled` from config | Task 5 |
| Remove `cache_ttl_seconds` from config | Task 5 |
| Remove `redis>=7.4.0` from requirements | Task 7 |
| Add `cache_lookup` to `incident_store.py` | Task 2 |
| `router.py` calls `incident_store.cache_lookup` | Task 3 |
| Remove `semantic_cache_enabled` guard from router | Task 3 — replaced with `postgres_dsn` guard |
| `verification.py` drops Redis `cache_store` block | Task 4 |
| Tests: hit, miss, empty dsn, embed fail, db error | Tasks 1–2 |
| Keep `cache_similarity_threshold` in config | Task 5 — explicitly preserved |

All spec requirements covered. No gaps.

### Placeholder scan

No TBD, TODO, "similar to", or vague steps. Every code step contains complete, runnable code.

### Type consistency

- `cache_lookup` signature (`dsn: str, error_summary: str, threshold: float = 0.95`) — consistent across Task 1 (test import + calls), Task 2 (implementation), Task 3 (caller in `_check_cache`).
- `settings.postgres_dsn` used in Task 3 — this field exists in `config.py` and is not removed in Task 5.
- `settings.cache_similarity_threshold` used in Task 3 — explicitly preserved in Task 5.
- `_embed_text` referenced in Task 2 implementation — existing private function in `incident_store.py`, not modified.
- Mock pattern in Task 1 matches exactly the existing `tests/test_incident_store.py` style (`patch(..., return_value=...)`, `AsyncMock` for conn methods).
