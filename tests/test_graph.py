"""Tests for the LangGraph agent graph.

TDD: Written BEFORE implementation.
Graph tests mock LLM calls and use MOCK_HEALTHY to control verification outcomes.
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage

import app.tools.k8s_tools as k8s_tools
from app.tools.k8s_tools import EXECUTED_ACTIONS
from app.agents.state import create_initial_state
from app.agents.graph import build_graph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_mock_state():
    EXECUTED_ACTIONS.clear()
    k8s_tools.MOCK_HEALTHY = False
    yield


@pytest.fixture
def crashloop_payload():
    return {
        "alertname": "PodCrashLooping",
        "status": "firing",
        "labels": {
            "region": "us-east-1",
            "env": "prod",
            "cluster_id": "k8s-prod-1",
            "namespace": "checkout",
            "service": "checkout-api",
            "pod": "crash-api-7d9f8b-xkj2p",
        },
        "annotations": {
            "summary": "Pod is crash looping",
            "description": "Pod has restarted 8 times in the last 10 minutes",
        },
    }


@pytest.fixture
def unknown_payload():
    return {
        "alertname": "WeirdUnknownAlert",
        "status": "firing",
        "labels": {},
        "annotations": {"summary": "Unknown alert with no context"},
    }


def _make_mock_llm(responses: list[str]) -> MagicMock:
    """Mock ChatAnthropic that returns responses in sequence."""
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    side_effects = [AIMessage(content=r) for r in responses]
    mock_llm.invoke.side_effect = side_effects
    return mock_llm


# ---------------------------------------------------------------------------
# Graph construction tests
# ---------------------------------------------------------------------------

class TestGraphBuilds:
    def test_graph_builds_without_error(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        graph = build_graph()
        node_names = set(graph.get_graph().nodes.keys())
        expected = {"triage", "processor", "researcher", "human_gate", "action", "verification", "escalate"}
        assert expected.issubset(node_names)

    def test_graph_is_compiled(self):
        graph = build_graph()
        # A compiled graph should have an invoke or stream method
        assert hasattr(graph, "invoke") or hasattr(graph, "astream")


# ---------------------------------------------------------------------------
# Happy path integration test
# ---------------------------------------------------------------------------

class TestHappyPath:
    @pytest.mark.asyncio
    async def test_happy_path_resolves(self, crashloop_payload):
        """Full graph run with mocked LLM + MOCK_HEALTHY=True + auto-approve → resolved."""
        k8s_tools.MOCK_HEALTHY = True

        triage_response = (
            '{"severity": "critical", "tools_to_run": ["get_cluster_events"], '
            '"triage_summary": "CrashLoopBackOff detected"}'
        )
        processor_response = "Pod is crash looping due to DB connection failure."
        researcher_response = "Restart the service to resolve the CrashLoopBackOff."

        mock_llm = _make_mock_llm([triage_response, processor_response, researcher_response])

        graph = build_graph()
        initial_state = create_initial_state(crashloop_payload)
        config = {"configurable": {"thread_id": initial_state["alert_id"]}}

        with patch("app.nodes.triage.ChatAnthropic", return_value=mock_llm), \
             patch("app.nodes.processor.ChatAnthropic", return_value=mock_llm), \
             patch("app.nodes.researcher.ChatAnthropic", return_value=mock_llm):

            # Run up to human gate (will interrupt)
            from langgraph.errors import GraphInterrupt
            try:
                await graph.ainvoke(initial_state, config=config)
            except GraphInterrupt:
                pass

            # Resume with approval
            from langgraph.types import Command
            try:
                final_state = await graph.ainvoke(
                    Command(resume={"approved": True}),
                    config=config,
                )
            except GraphInterrupt:
                # If another interrupt occurs, resume again
                final_state = await graph.ainvoke(
                    Command(resume={"approved": True}),
                    config=config,
                )

        assert final_state["status"] == "resolved"
        assert final_state["resolved"] is True


# ---------------------------------------------------------------------------
# Max retries escalation test
# ---------------------------------------------------------------------------

class TestMaxRetriesEscalation:
    @pytest.mark.asyncio
    async def test_max_retries_escalates(self, crashloop_payload):
        """When verification always fails, graph should escalate after max_retries."""
        k8s_tools.MOCK_HEALTHY = False

        triage_response = (
            '{"severity": "critical", "tools_to_run": ["get_cluster_events"], '
            '"triage_summary": "CrashLoopBackOff detected"}'
        )
        processor_response = "Crash loop due to DB connection failure."
        researcher_response = "Restart the service."

        # Provide enough responses for multiple triage cycles (up to max_retries)
        responses = []
        for _ in range(10):
            responses.extend([triage_response, processor_response, researcher_response])

        mock_llm = _make_mock_llm(responses)

        # Use max_retries=1 to speed up test
        payload = {**crashloop_payload, "labels": {**crashloop_payload["labels"]}}
        graph = build_graph()
        initial_state = create_initial_state(payload)
        initial_state = {**initial_state, "max_retries": 1}
        config = {"configurable": {"thread_id": initial_state["alert_id"]}}

        from langgraph.errors import GraphInterrupt
        from langgraph.types import Command

        with patch("app.nodes.triage.ChatAnthropic", return_value=mock_llm), \
             patch("app.nodes.processor.ChatAnthropic", return_value=mock_llm), \
             patch("app.nodes.researcher.ChatAnthropic", return_value=mock_llm):

            # Run until first interrupt
            try:
                await graph.ainvoke(initial_state, config=config)
            except GraphInterrupt:
                pass

            # Auto-approve
            final_state = await graph.ainvoke(
                Command(resume={"approved": True}),
                config=config,
            )

        assert final_state["status"] in ("escalated", "failed")


# ---------------------------------------------------------------------------
# Unknown alert escalation test
# ---------------------------------------------------------------------------

class TestUnknownAlertEscalation:
    @pytest.mark.asyncio
    async def test_unknown_alert_escalates_before_human_gate(self, unknown_payload):
        """Unknown alert with no tools should escalate at triage."""
        # Triage returns unknown severity with empty tools
        triage_response = (
            '{"severity": "unknown", "tools_to_run": [], '
            '"triage_summary": "Cannot determine issue — escalating"}'
        )
        mock_llm = _make_mock_llm([triage_response])

        graph = build_graph()
        initial_state = create_initial_state(unknown_payload)
        config = {"configurable": {"thread_id": initial_state["alert_id"]}}

        with patch("app.nodes.triage.ChatAnthropic", return_value=mock_llm):
            final_state = await graph.ainvoke(initial_state, config=config)

        assert final_state["status"] == "escalated"
