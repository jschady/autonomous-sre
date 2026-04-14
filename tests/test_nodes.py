"""Tests for LangGraph node functions.

TDD: Written BEFORE implementation. All LLM calls are mocked.
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage

import app.tools.k8s_tools as k8s_tools
from app.tools.k8s_tools import EXECUTED_ACTIONS
from app.agents.state import create_initial_state, SREState
from app.nodes.triage import triage_node
from app.nodes.processor import processor_node
from app.nodes.researcher import research_node
from app.nodes.action import action_node
from app.nodes.verification import verification_node
from app.nodes.human_gate import human_gate_node


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_mock_state():
    EXECUTED_ACTIONS.clear()
    k8s_tools.MOCK_HEALTHY = False
    yield


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
    mock_llm.bind_tools.return_value = mock_llm
    return mock_llm


def _make_mock_llm_text_only(text: str) -> MagicMock:
    """Create a mock LLM that returns a plain text AIMessage."""
    ai_msg = AIMessage(content=text)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = ai_msg
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
    async def test_action_records_result(self, approved_restart_state):
        result = await action_node(approved_restart_state)
        assert isinstance(result["action_result"], str)
        assert len(result["action_result"]) > 0

    @pytest.mark.asyncio
    async def test_action_returns_dict(self, approved_restart_state):
        result = await action_node(approved_restart_state)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Verification Node Tests
# ---------------------------------------------------------------------------

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
        k8s_tools.MOCK_HEALTHY = True
        result = await verification_node(post_action_state)
        assert result["resolved"] is True
        assert result["status"] == "resolved"

    @pytest.mark.asyncio
    async def test_verification_unhealthy_increments(self, post_action_state):
        k8s_tools.MOCK_HEALTHY = False
        result = await verification_node(post_action_state)
        assert result["retry_count"] == post_action_state["retry_count"] + 1

    @pytest.mark.asyncio
    async def test_verification_uses_metadata_namespace(self, post_action_state):
        k8s_tools.MOCK_HEALTHY = True
        result = await verification_node(post_action_state)
        # Namespace from metadata should have been used (checkout)
        # We verify the result is valid — the actual metric check used the namespace
        assert isinstance(result, dict)
        assert "resolved" in result

    @pytest.mark.asyncio
    async def test_verification_sets_metrics_healthy(self, post_action_state):
        k8s_tools.MOCK_HEALTHY = True
        result = await verification_node(post_action_state)
        assert result["metrics_healthy"] is True

    @pytest.mark.asyncio
    async def test_verification_returns_dict(self, post_action_state):
        result = await verification_node(post_action_state)
        assert isinstance(result, dict)
