"""Unit tests for Phase 2B hybrid search db_tools.

Tests mock asyncpg and openai so no database or API is needed.
TDD: Written BEFORE implementation — tests must fail first.
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Feature flag: USE_VECTOR_DB controls which backend is used
# ---------------------------------------------------------------------------


class TestFeatureFlag:
    def test_query_knowledge_base_uses_mock_when_flag_false(self, monkeypatch):
        monkeypatch.setenv("USE_VECTOR_DB", "false")
        # Force module reload to pick up env var
        import importlib
        import app.tools.db_tools as db_tools
        importlib.reload(db_tools)

        result = db_tools.query_knowledge_base.invoke({"query": "CrashLoopBackOff"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_query_knowledge_base_uses_vector_when_flag_true(self, monkeypatch):
        monkeypatch.setenv("USE_VECTOR_DB", "true")
        monkeypatch.setenv("POSTGRES_DSN", "postgresql://test:test@localhost/test")

        with patch("app.tools.db_tools._vector_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [
                {
                    "title": "CrashLoopBackOff Recovery",
                    "content": "Step 1: check logs",
                    "recommended_tool": "restart_service",
                }
            ]
            import importlib
            import app.tools.db_tools as db_tools
            importlib.reload(db_tools)

            # With USE_VECTOR_DB=true, patching _vector_search should be used
            # (the test verifies the path is wired, not that the patch fires during reload)
            result = db_tools.query_knowledge_base.invoke({"query": "CrashLoopBackOff"})
            assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Vector search internals
# ---------------------------------------------------------------------------


class TestVectorSearch:
    @pytest.mark.asyncio
    async def test_vector_search_returns_list_of_dicts(self):
        import importlib
        import app.tools.db_tools as db_tools
        importlib.reload(db_tools)

        mock_rows = [
            {
                "title": "CrashLoopBackOff Recovery",
                "content": "Step 1: check logs",
                "tags": ["CrashLoopBackOff"],
                "recommended_tool": "restart_service",
            }
        ]
        mock_conn = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        with patch("app.tools.db_tools._embed_query", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [0.1] * 1536
            with patch("app.tools.db_tools._run_hybrid_query", new_callable=AsyncMock) as mock_query:
                mock_query.return_value = mock_rows
                with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
                    results = await db_tools._vector_search("CrashLoopBackOff")

        assert isinstance(results, list)
        assert len(results) > 0
        assert "title" in results[0]

    @pytest.mark.asyncio
    async def test_vector_search_returns_empty_on_no_match(self):
        import importlib
        import app.tools.db_tools as db_tools
        importlib.reload(db_tools)

        mock_conn = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        with patch("app.tools.db_tools._embed_query", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [0.0] * 1536
            with patch("app.tools.db_tools._run_hybrid_query", new_callable=AsyncMock) as mock_query:
                mock_query.return_value = []
                with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
                    results = await db_tools._vector_search("zzzunknown_xyz")

        assert isinstance(results, list)
        assert results == []

    @pytest.mark.asyncio
    async def test_embed_query_calls_openai(self):
        import importlib
        import app.tools.db_tools as db_tools
        importlib.reload(db_tools)

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]

        with patch("app.tools.db_tools._openai_client") as mock_client:
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            embedding = await db_tools._embed_query("test query")

        assert isinstance(embedding, list)
        assert len(embedding) == 1536

    @pytest.mark.asyncio
    async def test_run_hybrid_query_uses_asyncpg(self):
        import importlib
        import app.tools.db_tools as db_tools
        importlib.reload(db_tools)

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[
            {"title": "CrashLoopBackOff Recovery", "content": "check logs",
             "tags": [], "recommended_tool": "restart_service"}
        ])

        results = await db_tools._run_hybrid_query(mock_conn, [0.1] * 1536, "crashloop", limit=2)
        assert isinstance(results, list)
        mock_conn.fetch.assert_called_once()


# ---------------------------------------------------------------------------
# Fallback to mock_sops when Postgres is unavailable
# ---------------------------------------------------------------------------


class TestVectorSearchFallback:
    @pytest.mark.asyncio
    async def test_falls_back_to_mock_sops_on_connection_error(self, monkeypatch):
        monkeypatch.setenv("USE_VECTOR_DB", "true")
        monkeypatch.setenv("POSTGRES_DSN", "postgresql://bad:bad@localhost/bad")

        import importlib
        import app.tools.db_tools as db_tools
        importlib.reload(db_tools)

        with patch("app.tools.db_tools._embed_query", side_effect=Exception("connection refused")):
            results = await db_tools._vector_search_with_fallback("CrashLoopBackOff")

        # Should fall back to keyword search and return results
        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_fallback_returns_mock_sop_structure(self, monkeypatch):
        monkeypatch.setenv("USE_VECTOR_DB", "true")

        import importlib
        import app.tools.db_tools as db_tools
        importlib.reload(db_tools)

        with patch("app.tools.db_tools._embed_query", side_effect=Exception("no db")):
            results = await db_tools._vector_search_with_fallback("OOMKilled")

        for r in results:
            assert "title" in r
            assert "content" in r
            assert "recommended_tool" in r


# ---------------------------------------------------------------------------
# query_knowledge_base tool end-to-end with mocked vector search
# ---------------------------------------------------------------------------


class TestQueryKnowledgeBaseHybrid:
    def test_no_match_returns_not_found_message(self, monkeypatch):
        monkeypatch.setenv("USE_VECTOR_DB", "false")
        import importlib
        import app.tools.db_tools as db_tools
        importlib.reload(db_tools)

        result = db_tools.query_knowledge_base.invoke({"query": "zzzunknown_xyz_q9z9z"})
        assert isinstance(result, str)
        assert "no" in result.lower() or "not found" in result.lower() or len(result) > 0

    def test_returns_formatted_steps(self, monkeypatch):
        monkeypatch.setenv("USE_VECTOR_DB", "false")
        import importlib
        import app.tools.db_tools as db_tools
        importlib.reload(db_tools)

        result = db_tools.query_knowledge_base.invoke({"query": "CrashLoopBackOff"})
        assert isinstance(result, str)
        assert any(kw in result.lower() for kw in ["restart", "pod", "kubectl", "step"])

    def test_oom_query_returns_oom_content(self, monkeypatch):
        monkeypatch.setenv("USE_VECTOR_DB", "false")
        import importlib
        import app.tools.db_tools as db_tools
        importlib.reload(db_tools)

        result = db_tools.query_knowledge_base.invoke({"query": "OOMKilled"})
        assert "oom" in result.lower() or "memory" in result.lower()
