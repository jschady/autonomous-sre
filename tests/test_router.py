"""Unit tests for app.nodes.router — semantic cache check."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.nodes.router import router_node


def _make_state(**overrides) -> dict:
    base = {
        "alert_payload": {"alertname": "PodCrashLooping"},
        "metadata": {"namespace": "checkout"},
        "cache_hit": False,
        "cache_key": "",
        "error_summary": "",
        "reasoning_log": [],
        "current_node": "triage",
    }
    return {**base, **overrides}


class TestRouterNode:
    """router_node — cache miss/hit paths."""

    @pytest.mark.asyncio
    async def test_cache_disabled_returns_base_update(self):
        state = _make_state(error_summary="OOMKilled in checkout")
        with patch("app.nodes.router.get_settings") as mock_settings:
            mock_settings.return_value = type("S", (), {"semantic_cache_enabled": False})()
            result = await router_node(state)

        assert result["cache_hit"] is False
        assert result["current_node"] == "router"

    @pytest.mark.asyncio
    async def test_cache_hit_returns_recommended_action(self):
        state = _make_state(error_summary="OOMKilled in checkout")
        cached_action = "Increase memory limit to 1Gi"

        with patch("app.nodes.router.get_settings") as mock_settings, \
             patch("app.nodes.router._check_cache", return_value={
                 "recommended_action": cached_action,
                 "cache_key": "cache:incident:abc",
             }):
            mock_settings.return_value = type("S", (), {
                "semantic_cache_enabled": True,
                "redis_url": "redis://localhost:6379",
                "cache_similarity_threshold": 0.95,
            })()
            result = await router_node(state)

        assert result["cache_hit"] is True
        assert result["recommended_action"] == cached_action
        assert result["proposed_action"] == cached_action
        assert result["cache_key"] == "cache:incident:abc"

    @pytest.mark.asyncio
    async def test_cache_miss_does_not_set_recommended_action(self):
        state = _make_state(error_summary="Pod crash looping")
        with patch("app.nodes.router.get_settings") as mock_settings, \
             patch("app.nodes.router._check_cache", return_value=None):
            mock_settings.return_value = type("S", (), {
                "semantic_cache_enabled": True,
                "redis_url": "redis://localhost:6379",
                "cache_similarity_threshold": 0.95,
            })()
            result = await router_node(state)

        assert result["cache_hit"] is False
        assert "recommended_action" not in result

    @pytest.mark.asyncio
    async def test_adds_reasoning_log_entry(self):
        state = _make_state(reasoning_log=["[triage] test"])
        with patch("app.nodes.router.get_settings") as mock_settings:
            mock_settings.return_value = type("S", (), {"semantic_cache_enabled": False})()
            result = await router_node(state)

        assert len(result["reasoning_log"]) == 2
        assert "[router]" in result["reasoning_log"][-1]

    @pytest.mark.asyncio
    async def test_no_cache_check_when_error_summary_empty(self):
        """Cache is not queried when error_summary is empty."""
        state = _make_state(error_summary="")
        with patch("app.nodes.router.get_settings") as mock_settings, \
             patch("app.nodes.router._check_cache") as mock_check:
            mock_settings.return_value = type("S", (), {"semantic_cache_enabled": True})()
            await router_node(state)

        mock_check.assert_not_called()
