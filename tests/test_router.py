"""Unit tests for app.nodes.router — LLM provider selection + cache check."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.nodes.router import _select_provider, router_node


def _make_state(**overrides) -> dict:
    base = {
        "alert_payload": {"alertname": "PodCrashLooping"},
        "metadata": {"namespace": "checkout"},
        "task_complexity": "moderate",
        "llm_provider": "claude",
        "cache_hit": False,
        "cache_key": "",
        "error_summary": "",
        "reasoning_log": [],
        "current_node": "triage",
    }
    return {**base, **overrides}


class TestSelectProvider:
    """_select_provider — pure routing logic."""

    def test_returns_claude_when_local_disabled(self):
        settings = MagicMock(local_model_enabled=False)
        assert _select_provider("simple", settings) == "claude"
        assert _select_provider("complex", settings) == "claude"

    def test_returns_local_for_simple_complexity(self):
        settings = MagicMock(local_model_enabled=True)
        assert _select_provider("simple", settings) == "local"

    def test_returns_local_for_moderate_complexity(self):
        settings = MagicMock(local_model_enabled=True)
        assert _select_provider("moderate", settings) == "local"

    def test_returns_claude_for_complex_tasks(self):
        settings = MagicMock(local_model_enabled=True)
        assert _select_provider("complex", settings) == "claude"

    def test_returns_claude_for_unknown_complexity(self):
        settings = MagicMock(local_model_enabled=True)
        # Unknown complexity falls through to claude (not in _LOCAL_COMPLEXITIES)
        assert _select_provider("unknown_value", settings) == "claude"


class TestRouterNode:
    """router_node — async integration of provider selection + cache check."""

    @pytest.mark.asyncio
    async def test_sets_llm_provider_claude_for_complex(self):
        state = _make_state(task_complexity="complex")
        with patch("app.nodes.router.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                local_model_enabled=True,
                semantic_cache_enabled=False,
            )
            result = await router_node(state)

        assert result["llm_provider"] == "claude"
        assert result["cache_hit"] is False
        assert result["current_node"] == "router"

    @pytest.mark.asyncio
    async def test_sets_llm_provider_local_for_simple(self):
        state = _make_state(task_complexity="simple")
        with patch("app.nodes.router.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                local_model_enabled=True,
                semantic_cache_enabled=False,
            )
            result = await router_node(state)

        assert result["llm_provider"] == "local"

    @pytest.mark.asyncio
    async def test_cache_hit_returns_recommended_action(self):
        state = _make_state(
            task_complexity="simple",
            error_summary="OOMKilled in checkout",
        )
        cached_action = "Increase memory limit to 1Gi"

        with patch("app.nodes.router.get_settings") as mock_settings, \
             patch("app.nodes.router._check_cache", return_value={
                 "recommended_action": cached_action,
                 "cache_key": "cache:incident:abc",
             }):
            mock_settings.return_value = MagicMock(
                local_model_enabled=True,
                semantic_cache_enabled=True,
                redis_url="redis://localhost:6379",
                cache_similarity_threshold=0.95,
            )
            result = await router_node(state)

        assert result["cache_hit"] is True
        assert result["recommended_action"] == cached_action
        assert result["proposed_action"] == cached_action

    @pytest.mark.asyncio
    async def test_cache_miss_does_not_set_recommended_action(self):
        state = _make_state(
            task_complexity="moderate",
            error_summary="Pod crash looping",
        )
        with patch("app.nodes.router.get_settings") as mock_settings, \
             patch("app.nodes.router._check_cache", return_value=None):
            mock_settings.return_value = MagicMock(
                local_model_enabled=False,
                semantic_cache_enabled=True,
                redis_url="redis://localhost:6379",
                cache_similarity_threshold=0.95,
            )
            result = await router_node(state)

        assert result["cache_hit"] is False
        assert "recommended_action" not in result

    @pytest.mark.asyncio
    async def test_adds_reasoning_log_entry(self):
        state = _make_state(task_complexity="moderate", reasoning_log=["[triage] test"])
        with patch("app.nodes.router.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                local_model_enabled=False,
                semantic_cache_enabled=False,
            )
            result = await router_node(state)

        assert len(result["reasoning_log"]) == 2
        assert "[router]" in result["reasoning_log"][-1]
