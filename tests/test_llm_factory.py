"""Unit tests for app.utils.llm_factory."""
from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from app.utils.llm_factory import get_llm_client, ainvoke_with_fallback


@pytest.fixture(autouse=True)
def clear_settings():
    from app.config import clear_settings_cache
    clear_settings_cache()
    yield
    clear_settings_cache()


class TestGetLlmClient:
    def test_returns_claude_client(self):
        mock_client = MagicMock()
        with patch("app.utils.llm_factory.get_settings") as mock_settings, \
             patch("app.utils.llm_factory._build_claude_client", return_value=mock_client):
            mock_settings.return_value = MagicMock(
                triage_model="claude-sonnet-4-6",
                anthropic_api_key="test-key",
            )
            result = get_llm_client()
            assert result is mock_client

    def test_passes_model_override(self):
        with patch("app.utils.llm_factory.get_settings") as mock_settings, \
             patch("app.utils.llm_factory._build_claude_client") as mock_build:
            mock_settings.return_value = MagicMock(
                triage_model="claude-sonnet-4-6",
                anthropic_api_key="test-key",
            )
            mock_build.return_value = MagicMock()
            get_llm_client(model_override="claude-haiku-4-5-20251001")
            _, kwargs_or_args = mock_build.call_args
            # model_override is the second positional arg
            assert mock_build.called


class TestAinvokeWithFallback:
    @pytest.mark.asyncio
    async def test_invokes_claude_regardless_of_provider(self):
        mock_response = MagicMock()
        mock_client = MagicMock()
        mock_client.ainvoke = AsyncMock(return_value=mock_response)

        with patch("app.utils.llm_factory.get_settings") as mock_settings, \
             patch("app.utils.llm_factory._build_claude_client", return_value=mock_client):
            mock_settings.return_value = MagicMock(
                triage_model="claude-sonnet-4-6",
                anthropic_api_key="test-key",
            )
            result = await ainvoke_with_fallback(provider="local", messages=["hi"])

        assert result is mock_response
        mock_client.ainvoke.assert_called_once_with(["hi"])

    @pytest.mark.asyncio
    async def test_provider_arg_ignored(self):
        """provider="claude" and provider="local" both call Claude."""
        mock_response = MagicMock()
        mock_client = MagicMock()
        mock_client.ainvoke = AsyncMock(return_value=mock_response)

        with patch("app.utils.llm_factory.get_settings") as mock_settings, \
             patch("app.utils.llm_factory._build_claude_client", return_value=mock_client):
            mock_settings.return_value = MagicMock(
                triage_model="claude-sonnet-4-6",
                anthropic_api_key="test-key",
            )
            for provider in ("claude", "local", "unknown"):
                result = await ainvoke_with_fallback(provider=provider, messages=["msg"])
                assert result is mock_response
