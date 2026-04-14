"""Integration tests for LLM fallback and kill switch behavior.

These tests verify:
  1. Kill switch: LOCAL_MODEL_ENABLED=false always routes to Claude
  2. RunPod unreachable: factory falls back to Claude with warning
  3. Graph structure: router node is present and conditional edges work
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.utils.llm_factory import get_llm_client


class TestKillSwitch:
    """When LOCAL_MODEL_ENABLED is false, all traffic goes to Claude."""

    def test_returns_claude_when_kill_switch_off(self):
        mock_claude = MagicMock()
        with patch("app.utils.llm_factory.get_settings") as mock_settings, \
             patch("app.utils.llm_factory._build_claude_client", return_value=mock_claude):
            mock_settings.return_value = MagicMock(
                local_model_enabled=False,
                runpod_base_url="https://pod-12345-8000.proxy.runpod.net/v1",
                triage_model="claude-sonnet-4-6",
                anthropic_api_key="sk-test",
            )
            result = get_llm_client(provider="local")
        assert result is mock_claude

    def test_returns_claude_even_with_explicit_local_provider(self):
        mock_claude = MagicMock()
        with patch("app.utils.llm_factory.get_settings") as mock_settings, \
             patch("app.utils.llm_factory._build_claude_client", return_value=mock_claude), \
             patch("app.utils.llm_factory._build_local_client") as mock_local:
            mock_settings.return_value = MagicMock(
                local_model_enabled=False,
                runpod_base_url="https://pod-12345-8000.proxy.runpod.net/v1",
                triage_model="claude-sonnet-4-6",
                anthropic_api_key="sk-test",
            )
            get_llm_client(provider="local")
            mock_local.assert_not_called()


class TestRunPodFallback:
    """When RunPod endpoint returns errors, factory falls back to Claude."""

    def test_falls_back_when_local_build_returns_none(self):
        mock_claude = MagicMock()
        with patch("app.utils.llm_factory.get_settings") as mock_settings, \
             patch("app.utils.llm_factory._build_local_client", return_value=None), \
             patch("app.utils.llm_factory._build_claude_client", return_value=mock_claude):
            mock_settings.return_value = MagicMock(
                local_model_enabled=True,
                runpod_base_url="https://pod-12345-8000.proxy.runpod.net/v1",
                triage_model="claude-sonnet-4-6",
                anthropic_api_key="sk-test",
            )
            result = get_llm_client(provider="local")
        assert result is mock_claude

    def test_falls_back_when_runpod_url_empty(self):
        mock_claude = MagicMock()
        with patch("app.utils.llm_factory.get_settings") as mock_settings, \
             patch("app.utils.llm_factory._build_claude_client", return_value=mock_claude):
            mock_settings.return_value = MagicMock(
                local_model_enabled=True,
                runpod_base_url="",  # empty URL = no local model
                triage_model="claude-sonnet-4-6",
                anthropic_api_key="sk-test",
            )
            result = get_llm_client(provider="local")
        assert result is mock_claude


class TestGraphStructure:
    """Verify the graph has the expected Phase 3 node structure."""

    def test_router_node_registered(self):
        from app.agents.graph import build_graph
        from langgraph.checkpoint.memory import MemorySaver

        with patch("app.agents.graph._build_checkpointer", return_value=MemorySaver()):
            graph = build_graph()

        node_names = list(graph.nodes.keys())
        assert "router" in node_names
        assert "triage" in node_names
        assert "processor" in node_names
        assert "researcher" in node_names
        assert "human_gate" in node_names
        assert "action" in node_names
        assert "verification" in node_names
        assert "escalate" in node_names

    def test_triage_routes_to_router(self):
        """After triage, the graph should go to router (not processor directly)."""
        from app.agents.graph import route_after_triage

        normal_state = {"status": "in_progress"}
        assert route_after_triage(normal_state) == "router"  # type: ignore[arg-type]

        escalated_state = {"status": "escalated"}
        from langgraph.graph import END
        assert route_after_triage(escalated_state) == END  # type: ignore[arg-type]

    def test_router_routes_cache_hit_to_human_gate(self):
        from app.agents.graph import route_after_router

        cache_hit_state = {"cache_hit": True}
        assert route_after_router(cache_hit_state) == "human_gate"  # type: ignore[arg-type]

        cache_miss_state = {"cache_hit": False}
        assert route_after_router(cache_miss_state) == "processor"  # type: ignore[arg-type]
