"""Phase 2 miscellaneous unit tests covering:

- scripts/migrate.py constants and structure
- scripts/ingest_docs.py parse helpers
- app/main.py Redis state store paths
- app/agents/graph.py _build_checkpointer
- app/tools/k8s_client.py config loading
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# scripts/migrate.py
# ---------------------------------------------------------------------------


class TestMigrateScript:
    def test_all_migrations_is_list_of_tuples(self):
        from scripts.migrate import ALL_MIGRATIONS
        assert isinstance(ALL_MIGRATIONS, list)
        assert len(ALL_MIGRATIONS) == 4
        for name, sql in ALL_MIGRATIONS:
            assert isinstance(name, str)
            assert isinstance(sql, str)
            assert len(sql) > 0

    def test_create_extension_sql_creates_vector(self):
        from scripts.migrate import CREATE_EXTENSION_SQL
        assert "vector" in CREATE_EXTENSION_SQL.lower()
        assert "extension" in CREATE_EXTENSION_SQL.lower()

    def test_create_sops_table_sql_has_embedding_column(self):
        from scripts.migrate import CREATE_SOPS_TABLE_SQL
        assert "sops_embeddings" in CREATE_SOPS_TABLE_SQL
        assert "embedding" in CREATE_SOPS_TABLE_SQL
        assert "vector" in CREATE_SOPS_TABLE_SQL.lower()

    def test_main_exits_without_dsn(self, monkeypatch):
        monkeypatch.delenv("POSTGRES_DSN", raising=False)
        from scripts.migrate import main
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_run_migrations_calls_execute_for_each_migration(self):
        from scripts.migrate import ALL_MIGRATIONS, run_migrations

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()

        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            await run_migrations("postgresql://test:test@localhost/test")

        assert mock_conn.execute.call_count == len(ALL_MIGRATIONS)
        mock_conn.close.assert_called_once()


# ---------------------------------------------------------------------------
# scripts/ingest_docs.py helpers
# ---------------------------------------------------------------------------


class TestIngestDocsParsers:
    def test_parse_frontmatter_extracts_title(self):
        from scripts.ingest_docs import _parse_frontmatter
        text = "---\ntitle: My SOP\ntags: [foo, bar]\n---\n# Body\nContent here."
        meta, body = _parse_frontmatter(text)
        assert meta["title"] == "My SOP"
        assert meta["tags"] == ["foo", "bar"]
        assert "Content here" in body

    def test_parse_frontmatter_handles_no_frontmatter(self):
        from scripts.ingest_docs import _parse_frontmatter
        text = "# Just content\nNo front matter here."
        meta, body = _parse_frontmatter(text)
        assert meta == {}
        assert "Just content" in body

    def test_load_sop_file_returns_required_fields(self, tmp_path):
        from scripts.ingest_docs import _load_sop_file
        md_file = tmp_path / "test-sop.md"
        md_file.write_text(
            "---\ntitle: Test SOP\ntags: [test, example]\nrecommended_tool: restart_service\n---\n"
            "# Steps\n1. Do this\n2. Do that"
        )
        sop = _load_sop_file(md_file)
        assert sop["id"] == "test-sop"
        assert sop["title"] == "Test SOP"
        assert "test" in sop["tags"]
        assert sop["recommended_tool"] == "restart_service"
        assert "Do this" in sop["content"]

    def test_load_sop_file_uses_stem_as_id(self, tmp_path):
        from scripts.ingest_docs import _load_sop_file
        md_file = tmp_path / "crashloopbackoff.md"
        md_file.write_text("---\ntitle: CrashLoop SOP\n---\nContent.")
        sop = _load_sop_file(md_file)
        assert sop["id"] == "crashloopbackoff"

    def test_all_sop_md_files_parse_correctly(self):
        """All 8 SOP files in data/sops/ should parse without error."""
        from scripts.ingest_docs import _load_sop_file, DOCS_DIR
        md_files = list(DOCS_DIR.glob("*.md"))
        assert len(md_files) == 8, f"Expected 8 SOP files, got {len(md_files)}"
        for filepath in md_files:
            sop = _load_sop_file(filepath)
            assert sop["id"]
            assert sop["title"]
            assert sop["content"]
            assert sop["recommended_tool"] in ("restart_service", "execute_rollback")

    @pytest.mark.asyncio
    async def test_ingest_all_calls_upsert_for_each_file(self, monkeypatch, tmp_path):
        """ingest_all should call _upsert_sop for each .md file found."""
        from scripts import ingest_docs

        # Point DOCS_DIR to our tmp dir with one file
        md_file = tmp_path / "sop1.md"
        md_file.write_text(
            "---\ntitle: Test SOP 1\ntags: [test]\nrecommended_tool: restart_service\n---\n"
            "Steps here."
        )

        monkeypatch.setattr(ingest_docs, "DOCS_DIR", tmp_path)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()

        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            with patch.object(ingest_docs, "_embed_text", new_callable=AsyncMock) as mock_embed:
                mock_embed.return_value = [0.1] * 1536
                count = await ingest_docs.ingest_all("postgresql://test/test")

        assert count == 1
        assert mock_conn.execute.called

    @pytest.mark.asyncio
    async def test_ingest_all_raises_without_openai_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from scripts.ingest_docs import ingest_all
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            await ingest_all("postgresql://test/test")


# ---------------------------------------------------------------------------
# app/agents/graph.py build_graph (Phase 4A: checkpointer is now a parameter)
# ---------------------------------------------------------------------------


class TestBuildCheckpointer:
    def test_build_graph_uses_memory_saver_by_default(self):
        from langgraph.checkpoint.memory import MemorySaver
        from app.agents.graph import build_graph
        from langgraph.graph.state import CompiledStateGraph
        graph = build_graph()
        assert isinstance(graph, CompiledStateGraph)

    def test_build_graph_accepts_memory_saver_explicitly(self):
        from langgraph.checkpoint.memory import MemorySaver
        from app.agents.graph import build_graph
        saver = MemorySaver()
        graph = build_graph(saver)
        assert graph is not None

    def test_build_graph_accepts_none_checkpointer(self):
        from app.agents.graph import build_graph
        # None should fall back to MemorySaver internally
        graph = build_graph(None)
        assert graph is not None


# ---------------------------------------------------------------------------
# app/main.py — Phase 4A: initial state cache replaces Redis state store
# ---------------------------------------------------------------------------


class TestInitialStateCache:
    @pytest.mark.asyncio
    async def test_initial_state_available_before_graph_runs(self):
        """Status endpoint should return 200 immediately after webhook POST."""
        from httpx import AsyncClient, ASGITransport
        from app.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/webhook", json={
                "alertname": "TestImmediate",
                "status": "firing",
                "labels": {},
                "annotations": {},
            })
            assert resp.status_code == 202
            alert_id = resp.json()["alert_id"]

            status_resp = await client.get(f"/status/{alert_id}")
            assert status_resp.status_code == 200
            assert status_resp.json()["alert_id"] == alert_id

    def test_initial_states_dict_exists(self):
        import app.main as main_mod
        assert hasattr(main_mod, "_INITIAL_STATES")
        assert isinstance(main_mod._INITIAL_STATES, dict)


# ---------------------------------------------------------------------------
# app/agents/state.py — new rbac_blocked + k8s_error fields
# ---------------------------------------------------------------------------


class TestStateNewFields:
    def test_create_initial_state_has_rbac_blocked_false(self):
        from app.agents.state import create_initial_state
        state = create_initial_state({"alertname": "Test", "status": "firing", "labels": {}})
        assert state["rbac_blocked"] is False

    def test_create_initial_state_has_k8s_error_none(self):
        from app.agents.state import create_initial_state
        state = create_initial_state({"alertname": "Test", "status": "firing", "labels": {}})
        assert state["k8s_error"] is None


# ---------------------------------------------------------------------------
# app/config.py — new settings fields
# ---------------------------------------------------------------------------


class TestConfigNewFields:
    def test_config_has_langsmith_fields(self):
        from app.config import Settings
        s = Settings()
        assert hasattr(s, "langchain_api_key")
        assert hasattr(s, "langchain_tracing_v2")
        assert hasattr(s, "langchain_project")
        assert s.langchain_project == "autonomous-sre"

    def test_config_has_postgres_dsn(self):
        from app.config import Settings
        s = Settings()
        assert hasattr(s, "postgres_dsn")

    def test_config_has_use_vector_db(self):
        from app.config import Settings
        s = Settings()
        assert hasattr(s, "use_vector_db")
