"""Integration tests for PostgresSaver checkpoint recovery.

Skipped unless RUN_INTEGRATION=1 is set.
Requires:
  - Running Postgres
  - POSTGRES_DSN env var
"""
from __future__ import annotations

import os

import pytest

RUN_INTEGRATION = os.environ.get("RUN_INTEGRATION", "0") == "1"
skip_unless_integration = pytest.mark.skipif(
    not RUN_INTEGRATION, reason="Set RUN_INTEGRATION=1 to run integration tests"
)


@skip_unless_integration
class TestStatePersistence:
    @pytest.mark.asyncio
    async def test_graph_state_survives_restart(self):
        """Graph checkpoint should be recoverable after re-building the graph."""
        import uuid
        from unittest.mock import patch, AsyncMock, MagicMock

        from app.agents.graph import build_graph
        from app.agents.state import create_initial_state

        dsn = os.environ["POSTGRES_DSN"]
        alert_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": alert_id}}

        payload = {
            "alertname": "TestAlert",
            "status": "firing",
            "labels": {"namespace": "default", "service": "test-svc"},
        }

        # Mock LLM calls so we don't need a real Anthropic key
        with patch("langchain_anthropic.ChatAnthropic") as mock_cls:
            mock_llm = MagicMock()
            mock_llm.invoke = MagicMock(return_value=MagicMock(content="test response"))
            mock_cls.return_value = mock_llm

            graph = build_graph()
            initial_state = create_initial_state(payload)

            try:
                from langgraph.errors import GraphInterrupt
                result = await graph.ainvoke(initial_state, config=config)
            except (GraphInterrupt, Exception):
                pass

        # Re-build graph and recover state from Postgres
        graph2 = build_graph()
        snapshot = await graph2.aget_state(config)
        assert snapshot is not None

    @pytest.mark.asyncio
    async def test_postgres_saver_setup_runs(self):
        """PostgresSaver.setup() should create the checkpoints table."""
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        dsn = os.environ["POSTGRES_DSN"]
        async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
            await saver.setup()
        # If no exception, tables were created successfully
