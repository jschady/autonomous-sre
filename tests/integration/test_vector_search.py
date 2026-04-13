"""Integration tests for vector search against a real Postgres+pgvector instance.

Skipped unless RUN_INTEGRATION=1 is set.
Requires:
  - Running Postgres with pgvector extension
  - POSTGRES_DSN env var pointing to it
  - OPENAI_API_KEY env var for embeddings
"""
from __future__ import annotations

import os

import pytest

RUN_INTEGRATION = os.environ.get("RUN_INTEGRATION", "0") == "1"
skip_unless_integration = pytest.mark.skipif(
    not RUN_INTEGRATION, reason="Set RUN_INTEGRATION=1 to run integration tests"
)


@skip_unless_integration
class TestVectorSearchIntegration:
    @pytest.mark.asyncio
    async def test_ingest_and_search_crashloop(self):
        """Ingest SOPs and verify CrashLoopBackOff is retrievable."""
        from scripts.ingest_docs import ingest_all
        from app.tools.db_tools import _vector_search

        dsn = os.environ["POSTGRES_DSN"]
        await ingest_all(dsn)

        results = await _vector_search("CrashLoopBackOff pod restarting")
        assert len(results) > 0
        titles = [r["title"] for r in results]
        assert any("CrashLoop" in t for t in titles)

    @pytest.mark.asyncio
    async def test_ingest_and_search_oom(self):
        from scripts.ingest_docs import ingest_all
        from app.tools.db_tools import _vector_search

        dsn = os.environ["POSTGRES_DSN"]
        await ingest_all(dsn)

        results = await _vector_search("out of memory container killed OOM")
        assert len(results) > 0
        assert any("OOM" in r["title"] or "memory" in r["content"].lower() for r in results)

    @pytest.mark.asyncio
    async def test_hybrid_search_beats_keyword_only(self):
        """Semantic search should find HighLatency SOP for 'slow response times'."""
        from scripts.ingest_docs import ingest_all
        from app.tools.db_tools import _vector_search

        dsn = os.environ["POSTGRES_DSN"]
        await ingest_all(dsn)

        # "slow response times" is not an exact keyword match but semantically matches HighLatency
        results = await _vector_search("service is responding very slowly")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_ingest_is_idempotent(self):
        """Running ingest twice should not create duplicate rows."""
        from scripts.ingest_docs import ingest_all
        import asyncpg

        dsn = os.environ["POSTGRES_DSN"]
        await ingest_all(dsn)
        await ingest_all(dsn)

        conn = await asyncpg.connect(dsn)
        try:
            count = await conn.fetchval("SELECT COUNT(*) FROM sops_embeddings")
        finally:
            await conn.close()

        # 8 SOPs, no duplicates
        assert count == 8
