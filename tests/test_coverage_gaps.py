"""Additional tests to cover remaining branches and improve coverage.

Targets: app/nodes/action.py, app/main.py, app/agents/state.py, app/agents/graph.py
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage

import app.tools.k8s_tools as k8s_tools
from app.tools.k8s_tools import EXECUTED_ACTIONS
from app.agents.state import create_initial_state, _resolve_max_retries
from app.nodes.action import action_node, _determine_action_tool
from app.agents.graph import route_after_triage, route_after_verification


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_mock_state():
    EXECUTED_ACTIONS.clear()
    k8s_tools.MOCK_HEALTHY = False
    yield


@pytest.fixture
def base_state():
    return create_initial_state({
        "alertname": "HighErrorRate",
        "status": "firing",
        "labels": {
            "namespace": "payments",
            "service": "payment-api",
            "region": "us-west-2",
            "env": "prod",
            "cluster_id": "k8s-prod-2",
        },
        "annotations": {"summary": "Error rate > 10%"},
    })


# ---------------------------------------------------------------------------
# action.py coverage gaps
# ---------------------------------------------------------------------------

class TestActionNodeCoverage:
    @pytest.mark.asyncio
    async def test_action_executes_rollback(self, base_state):
        """Cover the execute_rollback branch in action_node."""
        state = {
            **base_state,
            "human_approved": True,
            "recommended_action": "execute_rollback for payment-api",
            "proposed_action": "rollback payment-api to previous version",
            "sop_matches": [{"title": "HighErrorRate", "recommended_tool": "execute_rollback"}],
        }
        result = await action_node(state)
        assert isinstance(result["action_result"], str)
        assert len(result["action_result"]) > 0
        assert any("payment-api" in a for a in EXECUTED_ACTIONS)

    @pytest.mark.asyncio
    async def test_action_sop_fallback_determines_rollback(self, base_state):
        """Cover SOP fallback path in _determine_action_tool."""
        state = {
            **base_state,
            "human_approved": True,
            "recommended_action": "apply fix",  # No 'restart' or 'rollback' keyword
            "proposed_action": "apply fix",
            "sop_matches": [{"title": "HighErrorRate", "recommended_tool": "execute_rollback"}],
        }
        result = await action_node(state)
        assert isinstance(result, dict)
        assert any("payment-api" in a or "rollback" in a.lower() for a in EXECUTED_ACTIONS)

    @pytest.mark.asyncio
    async def test_action_tool_exception_sets_failed(self, base_state):
        """Cover the tool exception handler in action_node."""
        state = {
            **base_state,
            "human_approved": True,
            "recommended_action": "restart_service payment-api",
            "proposed_action": "restart_service payment-api",
            "sop_matches": [],
        }
        with patch("app.nodes.action.TOOL_REGISTRY", {"restart_service": MagicMock(invoke=MagicMock(side_effect=RuntimeError("k8s API error")))}):
            result = await action_node(state)
        assert result["status"] == "failed"
        assert len(result["error_log"]) > 0

    def test_determine_action_tool_rollback_keywords(self):
        assert _determine_action_tool("rollback checkout-api", []) == "execute_rollback"
        assert _determine_action_tool("roll back the deployment", []) == "execute_rollback"

    def test_determine_action_tool_sop_fallback(self):
        sops = [{"recommended_tool": "execute_rollback"}]
        result = _determine_action_tool("apply the fix", sops)
        assert result == "execute_rollback"

    def test_determine_action_tool_default_restart(self):
        result = _determine_action_tool("do something", [])
        assert result == "restart_service"

    def test_determine_action_tool_unknown_sop_tool_falls_back(self):
        sops = [{"recommended_tool": "unknown_tool_xyz"}]
        result = _determine_action_tool("do something", sops)
        assert result == "restart_service"


# ---------------------------------------------------------------------------
# state.py coverage gaps
# ---------------------------------------------------------------------------

class TestStateResolveCoverage:
    def test_resolve_max_retries_invalid_env(self):
        """Cover the ValueError branch in _resolve_max_retries."""
        import os
        old = os.environ.get("MAX_RETRIES")
        os.environ["MAX_RETRIES"] = "not_a_number"
        try:
            result = _resolve_max_retries()
            assert result == 3  # Falls back to default
        finally:
            if old is not None:
                os.environ["MAX_RETRIES"] = old
            else:
                del os.environ["MAX_RETRIES"]


# ---------------------------------------------------------------------------
# graph.py coverage gaps (route functions with failed status)
# ---------------------------------------------------------------------------

class TestRoutersCoverage:
    def test_route_after_triage_failed_status(self):
        """Cover the failed status branch in route_after_triage."""
        from langgraph.graph import END
        state = create_initial_state({"alertname": "Test"})
        state = {**state, "status": "failed"}
        result = route_after_triage(state)
        assert result == END

    def test_route_after_verification_escalated_status(self):
        """Cover the escalated branch in route_after_verification."""
        state = create_initial_state({"alertname": "Test"})
        state = {**state, "status": "escalated", "resolved": False, "retry_count": 0}
        result = route_after_verification(state)
        assert result == "escalate"

    def test_route_after_verification_failed_status(self):
        """Cover the failed branch in route_after_verification."""
        state = create_initial_state({"alertname": "Test"})
        state = {**state, "status": "failed", "resolved": False, "retry_count": 0}
        result = route_after_verification(state)
        assert result == "escalate"

    def test_route_after_verification_retry(self):
        """Cover the retry path in route_after_verification."""
        state = create_initial_state({"alertname": "Test"})
        state = {**state, "resolved": False, "retry_count": 0, "max_retries": 3, "status": "in_progress"}
        result = route_after_verification(state)
        assert result == "triage"


# ---------------------------------------------------------------------------
# main.py coverage gaps
# ---------------------------------------------------------------------------

class TestMainCoverage:
    @pytest.mark.asyncio
    async def test_run_graph_handles_general_exception(self):
        """Cover the generic exception handler in _run_graph."""
        from app.main import _run_graph, ALERT_STATES, ALERT_CONFIGS
        payload = {"alertname": "TestAlert"}
        state = create_initial_state(payload)
        alert_id = state["alert_id"]
        config = {"configurable": {"thread_id": alert_id}}
        ALERT_CONFIGS[alert_id] = config
        ALERT_STATES[alert_id] = state

        with patch("app.main._graph") as mock_graph:
            mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("unexpected error"))
            await _run_graph(alert_id, state)

        updated = ALERT_STATES.get(alert_id, {})
        assert updated.get("status") == "failed"
        assert any("unexpected error" in e for e in updated.get("error_log", []))

    @pytest.mark.asyncio
    async def test_slack_interactive_get_state_exception(self):
        """Cover aget_state exception handler in slack_interactive."""
        from httpx import AsyncClient, ASGITransport
        from app.main import app, ALERT_CONFIGS, ALERT_STATES
        from app.agents.state import create_initial_state

        payload = {"alertname": "TestAlert"}
        state = create_initial_state(payload)
        alert_id = state["alert_id"]
        config = {"configurable": {"thread_id": alert_id}}
        ALERT_CONFIGS[alert_id] = config
        ALERT_STATES[alert_id] = state

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            with patch("app.main._graph") as mock_graph:
                mock_graph.aget_state = AsyncMock(side_effect=Exception("checkpointer error"))
                response = await client.post(
                    "/slack/interactive",
                    json={"alert_id": alert_id, "approved": True},
                )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_slack_interactive_graph_not_interrupted(self):
        """Cover the 409 path when graph is not waiting for approval."""
        from httpx import AsyncClient, ASGITransport
        from app.main import app, ALERT_CONFIGS, ALERT_STATES
        from app.agents.state import create_initial_state

        payload = {"alertname": "TestAlert"}
        state = create_initial_state(payload)
        alert_id = state["alert_id"]
        config = {"configurable": {"thread_id": alert_id}}
        ALERT_CONFIGS[alert_id] = config
        ALERT_STATES[alert_id] = state

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            with patch("app.main._graph") as mock_graph:
                # Snapshot with no 'next' nodes = not waiting
                mock_snapshot = MagicMock()
                mock_snapshot.next = []
                mock_graph.aget_state = AsyncMock(return_value=mock_snapshot)
                response = await client.post(
                    "/slack/interactive",
                    json={"alert_id": alert_id, "approved": True},
                )
        assert response.status_code == 409
