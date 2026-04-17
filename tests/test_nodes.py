"""Tests for LangGraph node functions.

TDD: Written BEFORE implementation. All LLM calls are mocked.
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage

import app.tools.k8s_tools as k8s_tools

from app.agents.state import create_initial_state, SREState
from app.nodes.triage import triage_node
from app.nodes.processor import processor_node
from app.nodes.researcher import research_node
from app.agents.graph import route_after_action
from app.nodes.action import action_node, _extract_scale_factor, _extract_initial_limit
from app.nodes.verification import verification_node
from app.nodes.human_gate import human_gate_node, notify_slack_node


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def crashloop_state() -> SREState:
    return create_initial_state({
        "alertname": "PodCrashLooping",
        "status": "firing",
        "labels": {
            "region": "us-east-1",
            "env": "prod",
            "cluster_id": "k8s-prod-1",
            "namespace": "checkout",
            "service": "checkout-api",
            "pod": "checkout-api-7d9f8b-xkj2p",
        },
        "annotations": {
            "summary": "Pod checkout-api-7d9f8b-xkj2p is crash looping",
            "description": "Pod has restarted 8 times in the last 10 minutes",
        },
    })


@pytest.fixture
def high_error_state() -> SREState:
    return create_initial_state({
        "alertname": "HighErrorRate",
        "status": "firing",
        "labels": {
            "region": "us-west-2",
            "env": "prod",
            "namespace": "payments",
            "service": "payment-api",
        },
        "annotations": {"summary": "Error rate above 10% for 5 minutes"},
    })


@pytest.fixture
def unknown_alert_state() -> SREState:
    return create_initial_state({
        "alertname": "WeirdUnknownAlert",
        "status": "firing",
        "labels": {},
        "annotations": {"summary": "Unknown alert with no context"},
    })


# ---------------------------------------------------------------------------
# Mock LLM helpers
# ---------------------------------------------------------------------------

def _make_mock_llm_with_tool_call(tool_name: str, tool_args: dict, text: str = "") -> MagicMock:
    """Create a mock LLM that returns an AIMessage with a tool call."""
    tool_call = {
        "id": "call_123",
        "name": tool_name,
        "args": tool_args,
        "type": "tool_call",
    }
    ai_msg = AIMessage(content=text, tool_calls=[tool_call])
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = ai_msg
    mock_llm.ainvoke = AsyncMock(return_value=ai_msg)
    mock_llm.bind_tools.return_value = mock_llm
    return mock_llm


def _make_mock_llm_text_only(text: str) -> MagicMock:
    """Create a mock LLM that returns a plain text AIMessage."""
    ai_msg = AIMessage(content=text)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = ai_msg
    mock_llm.ainvoke = AsyncMock(return_value=ai_msg)
    mock_llm.bind_tools.return_value = mock_llm
    return mock_llm


# ---------------------------------------------------------------------------
# Triage Node Tests
# ---------------------------------------------------------------------------

class TestTriageNode:
    @pytest.mark.asyncio
    async def test_triage_returns_critical_severity(self, crashloop_state):
        mock_llm = _make_mock_llm_text_only(
            '{"severity": "critical", "tools_to_run": ["get_cluster_events", "fetch_container_logs"], '
            '"triage_summary": "Pod is CrashLoopBackOff, needs immediate restart"}'
        )
        with patch("app.nodes.triage.ChatAnthropic", return_value=mock_llm):
            result = await triage_node(crashloop_state)

        assert result["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_triage_selects_relevant_tools(self, crashloop_state):
        mock_llm = _make_mock_llm_text_only(
            '{"severity": "critical", "tools_to_run": ["get_cluster_events", "fetch_container_logs"], '
            '"triage_summary": "CrashLoop detected"}'
        )
        with patch("app.nodes.triage.ChatAnthropic", return_value=mock_llm):
            result = await triage_node(crashloop_state)

        assert isinstance(result["tools_to_run"], list)
        assert len(result["tools_to_run"]) > 0
        valid_tools = {
            "get_cluster_events", "fetch_container_logs",
            "get_system_metrics", "restart_service",
            "execute_rollback", "query_knowledge_base",
        }
        for tool_name in result["tools_to_run"]:
            assert tool_name in valid_tools

    @pytest.mark.asyncio
    async def test_triage_unknown_alert_escalates(self, unknown_alert_state):
        mock_llm = _make_mock_llm_text_only(
            '{"severity": "unknown", "tools_to_run": [], '
            '"triage_summary": "Cannot determine remediation — escalating"}'
        )
        with patch("app.nodes.triage.ChatAnthropic", return_value=mock_llm):
            result = await triage_node(unknown_alert_state)

        assert result["status"] == "escalated"

    @pytest.mark.asyncio
    async def test_triage_returns_immutable_update(self, crashloop_state):
        mock_llm = _make_mock_llm_text_only(
            '{"severity": "warning", "tools_to_run": ["get_system_metrics"], '
            '"triage_summary": "Minor issue detected"}'
        )
        original_severity = crashloop_state["severity"]
        with patch("app.nodes.triage.ChatAnthropic", return_value=mock_llm):
            result = await triage_node(crashloop_state)

        # Result is a dict (partial update), not the full mutated state
        assert isinstance(result, dict)
        # Original state should not be mutated
        assert crashloop_state["severity"] == original_severity

    @pytest.mark.asyncio
    async def test_triage_populates_triage_summary(self, crashloop_state):
        mock_llm = _make_mock_llm_text_only(
            '{"severity": "critical", "tools_to_run": ["get_cluster_events"], '
            '"triage_summary": "Pod is crash looping every 30 seconds"}'
        )
        with patch("app.nodes.triage.ChatAnthropic", return_value=mock_llm):
            result = await triage_node(crashloop_state)

        assert isinstance(result["triage_summary"], str)
        assert len(result["triage_summary"]) > 0

    @pytest.mark.asyncio
    async def test_triage_handles_llm_error_gracefully(self, crashloop_state):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM API error")
        mock_llm.bind_tools.return_value = mock_llm
        with patch("app.nodes.triage.ChatAnthropic", return_value=mock_llm):
            result = await triage_node(crashloop_state)

        assert result["status"] == "failed"
        assert len(result["error_log"]) > 0


# ---------------------------------------------------------------------------
# Processor Node Tests
# ---------------------------------------------------------------------------

class TestProcessorNode:
    @pytest.fixture
    def state_with_tools(self, crashloop_state) -> SREState:
        return {
            **crashloop_state,
            "severity": "critical",
            "tools_to_run": ["get_cluster_events", "fetch_container_logs"],
            "triage_summary": "Pod is CrashLoopBackOff",
        }

    @pytest.mark.asyncio
    async def test_processor_calls_tools_from_state(self, state_with_tools):
        mock_llm = _make_mock_llm_text_only("Pod crash summary.")
        with patch("app.utils.llm_factory._build_claude_client", return_value=mock_llm), \
             patch("app.utils.llm_factory.get_settings", return_value=MagicMock(
                 local_model_enabled=False, runpod_base_url="",
                 triage_model="claude-sonnet-4-6", anthropic_api_key="test",
             )):
            result = await processor_node(state_with_tools)
        assert isinstance(result["raw_logs"], str)
        assert len(result["raw_logs"]) > 0

    @pytest.mark.asyncio
    async def test_processor_returns_error_summary(self, state_with_tools):
        mock_llm = _make_mock_llm_text_only(
            "The pod is repeatedly crashing due to a database connection failure. "
            "Error: connection refused on port 5432."
        )
        with patch("app.utils.llm_factory._build_claude_client", return_value=mock_llm), \
             patch("app.utils.llm_factory.get_settings", return_value=MagicMock(
                 local_model_enabled=False, runpod_base_url="",
                 triage_model="claude-sonnet-4-6", anthropic_api_key="test",
             )):
            result = await processor_node(state_with_tools)

        assert isinstance(result["error_summary"], str)
        assert len(result["error_summary"]) > 0

    @pytest.mark.asyncio
    async def test_processor_concatenates_raw_logs(self, state_with_tools):
        mock_llm = _make_mock_llm_text_only("Pod crash loop summary.")
        with patch("app.utils.llm_factory._build_claude_client", return_value=mock_llm), \
             patch("app.utils.llm_factory.get_settings", return_value=MagicMock(
                 local_model_enabled=False, runpod_base_url="",
                 triage_model="claude-sonnet-4-6", anthropic_api_key="test",
             )):
            result = await processor_node(state_with_tools)
        assert "checkout" in result["raw_logs"].lower() or len(result["raw_logs"]) > 50

    @pytest.mark.asyncio
    async def test_processor_uses_metadata_namespace(self, state_with_tools):
        mock_llm = _make_mock_llm_text_only("Summary.")
        with patch("app.utils.llm_factory._build_claude_client", return_value=mock_llm), \
             patch("app.utils.llm_factory.get_settings", return_value=MagicMock(
                 local_model_enabled=False, runpod_base_url="",
                 triage_model="claude-sonnet-4-6", anthropic_api_key="test",
             )):
            result = await processor_node(state_with_tools)
        assert "checkout" in result["raw_logs"]

    @pytest.mark.asyncio
    async def test_processor_handles_empty_tools_list(self, crashloop_state):
        state = {**crashloop_state, "tools_to_run": []}
        mock_llm = _make_mock_llm_text_only("No tools available to run.")
        with patch("app.utils.llm_factory._build_claude_client", return_value=mock_llm), \
             patch("app.utils.llm_factory.get_settings", return_value=MagicMock(
                 local_model_enabled=False, runpod_base_url="",
                 triage_model="claude-sonnet-4-6", anthropic_api_key="test",
             )):
            result = await processor_node(state)

        assert isinstance(result, dict)
        assert "error_summary" in result

    @pytest.mark.asyncio
    async def test_processor_handles_llm_error_gracefully(self, state_with_tools):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM API error")
        with patch("app.utils.llm_factory._build_claude_client", return_value=mock_llm), \
             patch("app.utils.llm_factory.get_settings", return_value=MagicMock(
                 local_model_enabled=False, runpod_base_url="",
                 triage_model="claude-sonnet-4-6", anthropic_api_key="test",
             )):
            result = await processor_node(state_with_tools)

        assert result["status"] == "failed"
        assert len(result["error_log"]) > 0


# ---------------------------------------------------------------------------
# Researcher Node Tests
# ---------------------------------------------------------------------------

class TestResearcherNode:
    @pytest.fixture
    def state_with_error_summary(self, crashloop_state) -> SREState:
        return {
            **crashloop_state,
            "severity": "critical",
            "tools_to_run": ["get_cluster_events"],
            "triage_summary": "Pod is CrashLoopBackOff",
            "raw_logs": "ERROR: connection refused\nFATAL: application startup failed",
            "error_summary": "CrashLoopBackOff due to database connection failure",
        }

    @pytest.mark.asyncio
    async def test_researcher_finds_matching_sop(self, state_with_error_summary):
        mock_llm = _make_mock_llm_text_only(
            "Based on the CrashLoopBackOff pattern, recommended action: restart_service. "
            "Apply rolling restart to restore pod."
        )
        with patch("app.utils.llm_factory._build_claude_client", return_value=mock_llm), \
             patch("app.utils.llm_factory.get_settings", return_value=MagicMock(
                 local_model_enabled=False, runpod_base_url="",
                 triage_model="claude-sonnet-4-6", anthropic_api_key="test",
             )), \
             patch("app.utils.incident_store.fetch_similar_incidents", return_value=[]):
            result = await research_node(state_with_error_summary)

        assert isinstance(result["sop_matches"], list)
        assert len(result["sop_matches"]) >= 1

    @pytest.mark.asyncio
    async def test_researcher_extracts_recommended_action(self, state_with_error_summary):
        mock_llm = _make_mock_llm_text_only(
            "Recommended action: restart_service. "
            "Execute rolling restart for the checkout-api deployment."
        )
        with patch("app.utils.llm_factory._build_claude_client", return_value=mock_llm), \
             patch("app.utils.llm_factory.get_settings", return_value=MagicMock(
                 local_model_enabled=False, runpod_base_url="",
                 triage_model="claude-sonnet-4-6", anthropic_api_key="test",
             )), \
             patch("app.utils.incident_store.fetch_similar_incidents", return_value=[]):
            result = await research_node(state_with_error_summary)

        assert isinstance(result["recommended_action"], str)
        assert len(result["recommended_action"]) > 0

    @pytest.mark.asyncio
    async def test_researcher_no_match_graceful(self, crashloop_state):
        state = {
            **crashloop_state,
            "raw_logs": "some totally unknown zzzxyz error",
            "error_summary": "totally unknown zzzxyz failure pattern",
        }
        mock_llm = _make_mock_llm_text_only(
            "No matching SOP found. Manual investigation recommended."
        )
        with patch("app.utils.llm_factory._build_claude_client", return_value=mock_llm), \
             patch("app.utils.llm_factory.get_settings", return_value=MagicMock(
                 local_model_enabled=False, runpod_base_url="",
                 triage_model="claude-sonnet-4-6", anthropic_api_key="test",
             )), \
             patch("app.utils.incident_store.fetch_similar_incidents", return_value=[]):
            result = await research_node(state)

        assert "manual investigation" in result["recommended_action"].lower()

    @pytest.mark.asyncio
    async def test_researcher_returns_immutable_update(self, state_with_error_summary):
        mock_llm = _make_mock_llm_text_only("Restart the service.")
        original_sop_matches = state_with_error_summary["sop_matches"][:]
        with patch("app.utils.llm_factory._build_claude_client", return_value=mock_llm), \
             patch("app.utils.llm_factory.get_settings", return_value=MagicMock(
                 local_model_enabled=False, runpod_base_url="",
                 triage_model="claude-sonnet-4-6", anthropic_api_key="test",
             )), \
             patch("app.utils.incident_store.fetch_similar_incidents", return_value=[]):
            result = await research_node(state_with_error_summary)

        assert isinstance(result, dict)
        assert state_with_error_summary["sop_matches"] == original_sop_matches


# ---------------------------------------------------------------------------
# Human Gate Node Tests
# ---------------------------------------------------------------------------

class TestHumanGateNode:
    @pytest.mark.asyncio
    async def test_human_gate_raises_interrupt(self, crashloop_state):
        from langgraph.errors import GraphInterrupt
        state = {
            **crashloop_state,
            "proposed_action": "restart_service checkout-api",
            "error_summary": "CrashLoopBackOff detected",
        }
        # interrupt() raises GraphInterrupt when called within a graph context.
        # Here we patch it to simulate that behavior in unit tests.
        with patch("app.nodes.human_gate.interrupt", side_effect=GraphInterrupt(("test",))):
            with pytest.raises(GraphInterrupt):
                await human_gate_node(state)

    @pytest.mark.asyncio
    async def test_human_gate_returns_approved_true(self, crashloop_state):
        """When resumed with approved=True, human_approved should be True."""
        state = {
            **crashloop_state,
            "proposed_action": "restart checkout-api",
            "error_summary": "CrashLoopBackOff",
        }
        with patch("app.nodes.human_gate.interrupt", return_value={"approved": True}):
            result = await human_gate_node(state)
        assert result["human_approved"] is True
        assert "human_gate" in result["current_node"]

    @pytest.mark.asyncio
    async def test_human_gate_returns_approved_false(self, crashloop_state):
        """When resumed with approved=False, human_approved should be False."""
        state = {
            **crashloop_state,
            "proposed_action": "restart checkout-api",
            "error_summary": "CrashLoopBackOff",
        }
        with patch("app.nodes.human_gate.interrupt", return_value={"approved": False}):
            result = await human_gate_node(state)
        assert result["human_approved"] is False

    @pytest.mark.asyncio
    async def test_human_gate_does_not_post_slack(self, crashloop_state):
        """human_gate_node must not post or update any Slack messages."""
        mock_send = AsyncMock()
        mock_update = AsyncMock()
        state = {
            **crashloop_state,
            "proposed_action": "restart checkout-api",
            "error_summary": "CrashLoopBackOff",
        }
        with patch("app.nodes.human_gate.interrupt", return_value={"approved": True}), \
             patch("app.utils.slack_client.send_slack_message", mock_send), \
             patch("app.utils.slack_client.update_slack_message", mock_update):
            await human_gate_node(state)
        mock_send.assert_not_called()
        mock_update.assert_not_called()


# ---------------------------------------------------------------------------
# Notify Slack Node Tests
# ---------------------------------------------------------------------------

class TestNotifySlackNode:
    @pytest.mark.asyncio
    async def test_no_update_on_first_attempt(self, crashloop_state):
        """First attempt (no existing ts): update_slack_message must NOT be called."""
        state = {
            **crashloop_state,
            "proposed_action": "restart checkout-api",
            "error_summary": "CrashLoopBackOff",
            "slack_message_ts": None,
            "slack_channel": None,
        }
        mock_update = AsyncMock()
        mock_send = AsyncMock(return_value={"ok": True, "ts": "1234567890.123456", "channel": "C12345"})
        with patch("app.utils.slack_client.update_slack_message", mock_update), \
             patch("app.utils.slack_client.send_slack_message", mock_send):
            await notify_slack_node(state)
        mock_update.assert_not_called()
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_ts_list_contains_new_ts_on_first_post(self, crashloop_state):
        """First post: slack_message_ts_list should contain the new ts."""
        state = {
            **crashloop_state,
            "proposed_action": "restart checkout-api",
            "error_summary": "CrashLoopBackOff",
            "slack_message_ts": None,
            "slack_channel": None,
            "slack_message_ts_list": [],
        }
        new_ts = "1111111111.111111"
        mock_send = AsyncMock(return_value={"ok": True, "ts": new_ts, "channel": "C12345"})
        with patch("app.utils.slack_client.send_slack_message", mock_send):
            result = await notify_slack_node(state)
        assert result["slack_message_ts"] == new_ts
        assert new_ts in result["slack_message_ts_list"]

    @pytest.mark.asyncio
    async def test_ts_list_appended_on_first_post(self, crashloop_state):
        """notify_slack_node appends new ts to existing slack_message_ts_list."""
        state = {
            **crashloop_state,
            "proposed_action": "restart checkout-api",
            "error_summary": "CrashLoopBackOff",
            "slack_message_ts": None,
            "slack_channel": None,
            "slack_message_ts_list": ["existing-ts-1"],
        }
        new_ts = "new-ts-9999"
        mock_send = AsyncMock(return_value={"ok": True, "ts": new_ts, "channel": "C12345"})
        with patch("app.utils.slack_client.send_slack_message", mock_send):
            result = await notify_slack_node(state)
        assert "existing-ts-1" in result["slack_message_ts_list"]
        assert new_ts in result["slack_message_ts_list"]
        assert len(result["slack_message_ts_list"]) == 2

    @pytest.mark.asyncio
    async def test_updates_old_message_on_retry(self, crashloop_state):
        """Retry (slack_message_ts set): update old message in-place, no new post."""
        state = {
            **crashloop_state,
            "proposed_action": "restart checkout-api",
            "error_summary": "CrashLoopBackOff",
            "slack_message_ts": "1111111111.111111",
            "slack_channel": "C12345",
        }
        mock_update = AsyncMock(return_value={"ok": True})
        mock_send = AsyncMock()
        with patch("app.utils.slack_client.update_slack_message", mock_update), \
             patch("app.utils.slack_client.send_slack_message", mock_send):
            result = await notify_slack_node(state)
        mock_update.assert_called_once()
        mock_send.assert_not_called()
        call_args = mock_update.call_args
        ts_used = call_args.kwargs.get("ts") or (call_args.args[0] if call_args.args else None)
        assert ts_used == "1111111111.111111"
        # ts unchanged on in-place update
        assert result["slack_message_ts"] == "1111111111.111111"


# ---------------------------------------------------------------------------
# UpdateSlackMessage Tests
# ---------------------------------------------------------------------------

class TestUpdateSlackMessage:
    @pytest.mark.asyncio
    async def test_update_calls_chat_update_endpoint(self):
        from app.utils.slack_client import update_slack_message
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("app.utils.slack_client.get_settings", return_value=MagicMock(
            slack_bot_token="xoxb-test", slack_channel_id="C12345"
        )), patch("httpx.AsyncClient", return_value=mock_client):
            await update_slack_message(ts="123.456", channel="C12345", blocks={"blocks": [], "text": "Hi"})

        mock_client.post.assert_called_once()
        url = mock_client.post.call_args[0][0]
        assert "chat.update" in url

    @pytest.mark.asyncio
    async def test_delete_calls_chat_delete_endpoint(self):
        from app.utils.slack_client import delete_slack_message
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("app.utils.slack_client.get_settings", return_value=MagicMock(
            slack_bot_token="xoxb-test", slack_channel_id="C12345"
        )), patch("httpx.AsyncClient", return_value=mock_client):
            await delete_slack_message(ts="123.456", channel="C12345")

        mock_client.post.assert_called_once()
        url = mock_client.post.call_args[0][0]
        assert "chat.delete" in url

    @pytest.mark.asyncio
    async def test_delete_no_op_when_no_token(self):
        from app.utils.slack_client import delete_slack_message
        with patch("app.utils.slack_client.get_settings", return_value=MagicMock(
            slack_bot_token=None, slack_channel_id=None
        )):
            result = await delete_slack_message(ts="123", channel="C")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_no_op_when_no_token(self):
        from app.utils.slack_client import update_slack_message
        with patch("app.utils.slack_client.get_settings", return_value=MagicMock(
            slack_bot_token=None, slack_channel_id=None
        )):
            result = await update_slack_message(ts="123", channel="C", blocks={})
        assert result is None


# ---------------------------------------------------------------------------
# route_after_action Tests
# ---------------------------------------------------------------------------

class TestRouteAfterAction:
    def _make_state(self, status="", rbac_blocked=False, human_approved=True) -> dict:
        return {
            "status": status,
            "rbac_blocked": rbac_blocked,
            "human_approved": human_approved,
        }

    def test_rejection_routes_to_escalate(self):
        """Human rejection (status=escalated, rbac_blocked=False) must go to escalate."""
        state = self._make_state(status="escalated", rbac_blocked=False, human_approved=False)
        assert route_after_action(state) == "escalate"

    def test_rbac_blocked_routes_to_escalate(self):
        state = self._make_state(rbac_blocked=True)
        assert route_after_action(state) == "escalate"

    def test_success_routes_to_verification(self):
        state = self._make_state(status="in_progress", rbac_blocked=False, human_approved=True)
        assert route_after_action(state) == "verification"

    def test_no_escalated_status_routes_to_verification(self):
        state = self._make_state(status="", rbac_blocked=False)
        assert route_after_action(state) == "verification"


# ---------------------------------------------------------------------------
# Action Node Tests
# ---------------------------------------------------------------------------

class TestActionNode:
    @pytest.fixture
    def approved_restart_state(self, crashloop_state) -> SREState:
        return {
            **crashloop_state,
            "severity": "critical",
            "tools_to_run": ["get_cluster_events"],
            "triage_summary": "CrashLoopBackOff",
            "error_summary": "Pod is crash looping",
            "sop_matches": [{"title": "CrashLoopBackOff Recovery", "recommended_tool": "restart_service"}],
            "recommended_action": "restart_service checkout-api",
            "proposed_action": "restart_service checkout-api",
            "human_approved": True,
        }

    @pytest.fixture
    def rejected_state(self, crashloop_state) -> SREState:
        return {
            **crashloop_state,
            "proposed_action": "restart_service checkout-api",
            "human_approved": False,
            "recommended_action": "restart_service checkout-api",
        }

    @pytest.mark.asyncio
    async def test_action_executes_restart(self, approved_restart_state):
        result = await action_node(approved_restart_state)
        assert isinstance(result["action_result"], str)
        assert len(result["action_result"]) > 0

    @pytest.mark.asyncio
    async def test_action_unapproved_escalates(self, rejected_state):
        result = await action_node(rejected_state)
        assert result["status"] == "escalated"

    @pytest.mark.asyncio
    async def test_action_returns_dict(self, approved_restart_state):
        result = await action_node(approved_restart_state)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_action_scale_memory_passes_factor(self, crashloop_state):
        """scale_memory tool should receive the factor parsed from proposed_action."""
        state = {
            **crashloop_state,
            "severity": "critical",
            "human_approved": True,
            "proposed_action": "Scale memory with factor=1.5 to provide headroom.",
            "recommended_action": "Scale memory with factor=1.5 to provide headroom.",
            "sop_matches": [],
        }
        with patch("app.nodes.action.TOOL_REGISTRY") as mock_registry:
            mock_tool = MagicMock()
            mock_tool.invoke.return_value = "Scaled memory limit: 128Mi -> 192Mi"
            mock_registry.__contains__ = lambda self, key: key == "scale_memory"
            mock_registry.__getitem__ = lambda self, key: mock_tool
            mock_registry.get = lambda key, default=None: mock_tool if key == "scale_memory" else default
            result = await action_node(state)
        mock_tool.invoke.assert_called_once()
        call_kwargs = mock_tool.invoke.call_args[0][0]
        assert call_kwargs["factor"] == pytest.approx(1.5)

    @pytest.mark.asyncio
    async def test_action_scale_memory_defaults_factor_when_absent(self, crashloop_state):
        """scale_memory should default to factor=2.0 when proposed_action has no factor."""
        state = {
            **crashloop_state,
            "severity": "critical",
            "human_approved": True,
            "proposed_action": "Increase memory limit for OOM recovery.",
            "recommended_action": "Increase memory limit for OOM recovery.",
            "sop_matches": [],
        }
        with patch("app.nodes.action.TOOL_REGISTRY") as mock_registry:
            mock_tool = MagicMock()
            mock_tool.invoke.return_value = "Scaled memory limit: 128Mi -> 256Mi"
            mock_registry.__contains__ = lambda self, key: key == "scale_memory"
            mock_registry.__getitem__ = lambda self, key: mock_tool
            mock_registry.get = lambda key, default=None: mock_tool if key == "scale_memory" else default
            result = await action_node(state)
        call_kwargs = mock_tool.invoke.call_args[0][0]
        assert call_kwargs["factor"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _extract_scale_factor unit tests
# ---------------------------------------------------------------------------

class TestExtractScaleFactor:
    def test_extracts_factor_equals(self):
        assert _extract_scale_factor("Scale memory with factor=1.5") == pytest.approx(1.5)

    def test_extracts_factor_space(self):
        assert _extract_scale_factor("Scale memory factor 1.3 for headroom") == pytest.approx(1.3)

    def test_clamps_minimum(self):
        assert _extract_scale_factor("factor=0.5") == pytest.approx(1.1)

    def test_clamps_maximum(self):
        assert _extract_scale_factor("factor=10.0") == pytest.approx(4.0)

    def test_defaults_when_missing(self):
        assert _extract_scale_factor("restart the service") == pytest.approx(2.0)

    def test_case_insensitive(self):
        assert _extract_scale_factor("FACTOR=1.8") == pytest.approx(1.8)

    def test_integer_factor(self):
        assert _extract_scale_factor("factor=2") == pytest.approx(2.0)


class TestExtractInitialLimit:
    def test_extracts_equals_syntax(self):
        assert _extract_initial_limit("Set initial_limit=192Mi based on usage") == "192Mi"

    def test_extracts_space_syntax(self):
        assert _extract_initial_limit("initial limit 256Mi") == "256Mi"

    def test_extracts_gi_unit(self):
        assert _extract_initial_limit("initial_limit=1Gi") == "1Gi"

    def test_defaults_when_missing(self):
        assert _extract_initial_limit("Scale memory with factor=1.5") == "256Mi"

    def test_case_insensitive(self):
        assert _extract_initial_limit("INITIAL_LIMIT=128Mi") == "128Mi"


# ---------------------------------------------------------------------------
# Verification Node Tests
# ---------------------------------------------------------------------------

_HEALTHY_METRICS = "service=checkout-api error_rate=0.2% latency_p99=45ms cpu_usage=22.0% status=healthy"
_DEGRADED_METRICS = "service=checkout-api error_rate=12.5% latency_p99=4500ms cpu_usage=92.0% status=degraded"


class TestVerificationNode:
    @pytest.fixture
    def post_action_state(self, crashloop_state) -> SREState:
        return {
            **crashloop_state,
            "severity": "critical",
            "proposed_action": "restart_service checkout-api",
            "action_result": "Rolling restart initiated",
            "human_approved": True,
        }

    @pytest.mark.asyncio
    async def test_verification_healthy_resolves(self, post_action_state):
        with patch("app.nodes.verification.get_system_metrics") as mock_metrics:
            mock_metrics.invoke.return_value = _HEALTHY_METRICS
            result = await verification_node(post_action_state)
        assert result["resolved"] is True
        assert result["status"] == "resolved"

    @pytest.mark.asyncio
    async def test_verification_unhealthy_increments(self, post_action_state):
        with patch("app.nodes.verification.get_system_metrics") as mock_metrics:
            mock_metrics.invoke.return_value = _DEGRADED_METRICS
            result = await verification_node(post_action_state)
        assert result["retry_count"] == post_action_state["retry_count"] + 1

    @pytest.mark.asyncio
    async def test_verification_uses_metadata_namespace(self, post_action_state):
        with patch("app.nodes.verification.get_system_metrics") as mock_metrics:
            mock_metrics.invoke.return_value = _HEALTHY_METRICS
            result = await verification_node(post_action_state)
        assert isinstance(result, dict)
        assert "resolved" in result

    @pytest.mark.asyncio
    async def test_verification_sets_metrics_healthy(self, post_action_state):
        with patch("app.nodes.verification.get_system_metrics") as mock_metrics:
            mock_metrics.invoke.return_value = _HEALTHY_METRICS
            result = await verification_node(post_action_state)
        assert result["metrics_healthy"] is True

    @pytest.mark.asyncio
    async def test_verification_returns_dict(self, post_action_state):
        with patch("app.nodes.verification.get_system_metrics") as mock_metrics:
            mock_metrics.invoke.return_value = _DEGRADED_METRICS
            result = await verification_node(post_action_state)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Verification Node — OOM path
# ---------------------------------------------------------------------------

class TestVerificationNodeOOM:
    @pytest.fixture
    def oom_state(self, crashloop_state) -> SREState:
        return {
            **crashloop_state,
            "alert_payload": {
                "alertname": "OOMKilled",
                "labels": {
                    "namespace": "checkout",
                    "service": "checkout-api",
                    "pod": "checkout-api-7d9f8b-xkj2p",
                },
            },
            "metadata": {"namespace": "checkout", "service": "checkout-api"},
            "workload": "checkout-api",
            "proposed_action": "Scale memory with factor=2.0 to fix OOMKilled pods.",
            "recommended_action": "scale_memory to increase container memory limit",
            "action_result": "Memory limit scaled",
            "human_approved": True,
        }

    def _running_pod(self, name: str, container_statuses=None):
        pod = MagicMock()
        pod.metadata.name = name
        pod.status.phase = "Running"
        pod.status.container_statuses = container_statuses or []
        return pod

    @pytest.mark.asyncio
    async def test_oom_verification_running_pod_no_recent_oom_resolves(self, oom_state, mock_k8s_apis):
        """Running pod with no recent OOMKill in lastState → resolved."""
        _, mock_core = mock_k8s_apis
        mock_core.list_namespaced_pod.return_value = MagicMock(
            items=[self._running_pod("checkout-api-abc-xyz")]
        )
        result = await verification_node(oom_state)
        assert result["resolved"] is True
        assert result["metrics_healthy"] is True

    @pytest.mark.asyncio
    async def test_oom_verification_pod_with_recent_oomkill_not_resolved(self, oom_state, mock_k8s_apis):
        """Running pod whose lastState shows OOMKilled → not resolved yet."""
        _, mock_core = mock_k8s_apis
        cs = MagicMock()
        cs.last_state.terminated.reason = "OOMKilled"
        pod = self._running_pod("checkout-api-abc-xyz", container_statuses=[cs])
        mock_core.list_namespaced_pod.return_value = MagicMock(items=[pod])
        result = await verification_node(oom_state)
        assert result["resolved"] is False

    @pytest.mark.asyncio
    async def test_oom_verification_no_pods_not_resolved(self, oom_state, mock_k8s_apis):
        """No pods found for workload → not resolved."""
        _, mock_core = mock_k8s_apis
        mock_core.list_namespaced_pod.return_value = MagicMock(items=[])
        result = await verification_node(oom_state)
        assert result["resolved"] is False

    @pytest.mark.asyncio
    async def test_oom_verification_pending_pod_not_resolved(self, oom_state, mock_k8s_apis):
        """Pod still Pending (rollout in progress) → not resolved, retry increments."""
        _, mock_core = mock_k8s_apis
        pod = MagicMock()
        pod.metadata.name = "checkout-api-abc-xyz"
        pod.status.phase = "Pending"
        pod.status.container_statuses = []
        mock_core.list_namespaced_pod.return_value = MagicMock(items=[pod])
        result = await verification_node(oom_state)
        assert result["resolved"] is False
        assert result["retry_count"] == oom_state["retry_count"] + 1

    @pytest.mark.asyncio
    async def test_oom_verification_does_not_call_prometheus(self, oom_state, mock_k8s_apis):
        """OOM verification must use K8s API, not Prometheus metrics."""
        _, mock_core = mock_k8s_apis
        mock_core.list_namespaced_pod.return_value = MagicMock(
            items=[self._running_pod("checkout-api-abc-xyz")]
        )
        with patch("app.nodes.verification.get_system_metrics") as mock_prom:
            await verification_node(oom_state)
        mock_prom.invoke.assert_not_called()
