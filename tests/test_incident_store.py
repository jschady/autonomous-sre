"""Unit tests for app.utils.incident_store — Postgres incident persistence."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.utils.incident_store import cache_lookup, fetch_similar_incidents, save_resolved_incident


def _make_state(**overrides) -> dict:
    base = {
        "alert_id": "test-alert-123",
        "alert_payload": {"alertname": "OOMKilled"},
        "metadata": {"namespace": "checkout"},
        "error_summary": "Container consumed all available memory and was killed.",
        "triage_summary": "OOMKilled — memory limit too low.",
        "recommended_action": "Increase memory limit to 1Gi.",
        "action_result": "Resolved",
        "severity": "critical",
    }
    return {**base, **overrides}


class TestSaveResolvedIncident:
    @pytest.mark.asyncio
    async def test_no_op_for_empty_dsn(self):
        # Should not raise, no side effects
        await save_resolved_incident(dsn="", state=_make_state())

    @pytest.mark.asyncio
    async def test_no_op_for_empty_error_summary(self):
        state = _make_state(error_summary="")
        await save_resolved_incident(dsn="postgresql://test", state=state)

    @pytest.mark.asyncio
    async def test_inserts_row_when_valid(self):
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.close = AsyncMock()

        with patch("app.utils.incident_store._embed_text", return_value=[0.1, 0.2, 0.3]), \
             patch("asyncpg.connect", return_value=mock_conn):
            await save_resolved_incident(dsn="postgresql://test", state=_make_state())

        mock_conn.execute.assert_called_once()
        sql_arg = mock_conn.execute.call_args.args[0]
        assert "resolved_incidents" in sql_arg

    @pytest.mark.asyncio
    async def test_swallows_db_error(self):
        with patch("app.utils.incident_store._embed_text", return_value=[0.1, 0.2]), \
             patch("asyncpg.connect", side_effect=ConnectionError("DB down")):
            # Should not raise
            await save_resolved_incident(dsn="postgresql://test", state=_make_state())


class TestFetchSimilarIncidents:
    @pytest.mark.asyncio
    async def test_returns_empty_for_empty_dsn(self):
        result = await fetch_similar_incidents(dsn="", error_summary="OOMKilled")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_for_empty_summary(self):
        result = await fetch_similar_incidents(dsn="postgresql://test", error_summary="")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_rows_from_db(self):
        mock_rows = [
            {
                "alertname": "OOMKilled",
                "namespace": "checkout",
                "triage_summary": "Memory too low",
                "recommended_action": "Increase limit",
                "action_result": "Resolved",
            }
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_conn.close = AsyncMock()

        with patch("app.utils.incident_store._embed_text", return_value=[0.1, 0.2, 0.3]), \
             patch("asyncpg.connect", return_value=mock_conn):
            result = await fetch_similar_incidents(
                dsn="postgresql://test",
                error_summary="Container OOMKilled",
                limit=3,
            )

        assert len(result) == 1
        assert result[0]["alertname"] == "OOMKilled"

    @pytest.mark.asyncio
    async def test_returns_empty_on_db_error(self):
        with patch("app.utils.incident_store._embed_text", return_value=[0.1, 0.2]), \
             patch("asyncpg.connect", side_effect=ConnectionError("DB down")):
            result = await fetch_similar_incidents(
                dsn="postgresql://test",
                error_summary="OOMKilled",
            )
        assert result == []


class TestCacheLookup:
    @pytest.mark.asyncio
    async def test_cache_lookup_hit(self):
        """Returns dict with recommended_action when nearest row similarity >= threshold."""
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
            result = await cache_lookup("postgresql://test", "OOMKilled pod crashing", threshold=0.90)

        assert result is not None
        assert result["recommended_action"] == "restart the pod"
        mock_conn.fetchrow.assert_called_once()
        sql_arg = mock_conn.fetchrow.call_args.args[0]
        assert "resolved_incidents" in sql_arg
        assert "<=>" in sql_arg

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
    async def test_cache_lookup_no_rows_in_db(self):
        """Returns None when fetchrow returns None (empty table)."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        mock_conn.close = AsyncMock()

        with patch("app.utils.incident_store._embed_text", return_value=[0.1] * 1536), \
             patch("asyncpg.connect", return_value=mock_conn):
            result = await cache_lookup("postgresql://test", "OOMKilled pod crashing")

        assert result is None
        mock_conn.fetchrow.assert_called_once()

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
        """Returns None when asyncpg raises."""
        with patch("app.utils.incident_store._embed_text", return_value=[0.1] * 1536), \
             patch("asyncpg.connect", side_effect=Exception("connection refused")):
            result = await cache_lookup("postgresql://test", "OOMKilled pod crashing")

        assert result is None
