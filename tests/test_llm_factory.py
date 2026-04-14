"""Unit tests for app.utils.llm_factory — LLM client factory."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.utils.llm_factory import get_llm_client


@pytest.fixture(autouse=True)
def clear_settings():
    from app.config import clear_settings_cache
    clear_settings_cache()
    yield
    clear_settings_cache()


class TestGetLlmClientClaude:
    """get_llm_client always returns Claude when local model is disabled."""

    def test_returns_claude_when_local_disabled(self):
        with patch("app.utils.llm_factory.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                local_model_enabled=False,
                runpod_base_url="https://example.runpod.net/v1",
                triage_model="claude-sonnet-4-6",
                anthropic_api_key="test-key",
            )
            with patch("app.utils.llm_factory._build_claude_client") as mock_claude:
                mock_claude.return_value = MagicMock()
                get_llm_client(provider="local")
                mock_claude.assert_called_once()

    def test_returns_claude_when_no_runpod_url(self):
        with patch("app.utils.llm_factory.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                local_model_enabled=True,
                runpod_base_url="",
                triage_model="claude-sonnet-4-6",
                anthropic_api_key="test-key",
            )
            with patch("app.utils.llm_factory._build_claude_client") as mock_claude:
                mock_claude.return_value = MagicMock()
                get_llm_client(provider="local")
                mock_claude.assert_called_once()

    def test_returns_claude_for_unknown_provider(self):
        with patch("app.utils.llm_factory.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                local_model_enabled=True,
                runpod_base_url="https://example.runpod.net/v1",
                triage_model="claude-sonnet-4-6",
                anthropic_api_key="test-key",
            )
            with patch("app.utils.llm_factory._build_claude_client") as mock_claude:
                mock_claude.return_value = MagicMock()
                get_llm_client(provider="claude")
                mock_claude.assert_called_once()


class TestGetLlmClientLocal:
    """get_llm_client returns local model when enabled and URL is set."""

    def test_returns_local_when_enabled(self):
        mock_local = MagicMock()
        with patch("app.utils.llm_factory.get_settings") as mock_settings, \
             patch("app.utils.llm_factory._build_local_client", return_value=mock_local):
            mock_settings.return_value = MagicMock(
                local_model_enabled=True,
                runpod_base_url="https://example.runpod.net/v1",
                triage_model="claude-sonnet-4-6",
                anthropic_api_key="test-key",
            )
            result = get_llm_client(provider="local")
            assert result is mock_local

    def test_falls_back_to_claude_when_local_build_fails(self):
        mock_claude = MagicMock()
        with patch("app.utils.llm_factory.get_settings") as mock_settings, \
             patch("app.utils.llm_factory._build_local_client", return_value=None), \
             patch("app.utils.llm_factory._build_claude_client", return_value=mock_claude):
            mock_settings.return_value = MagicMock(
                local_model_enabled=True,
                runpod_base_url="https://example.runpod.net/v1",
                triage_model="claude-sonnet-4-6",
                anthropic_api_key="test-key",
            )
            result = get_llm_client(provider="local")
            assert result is mock_claude


class TestBuildLocalClient:
    """_build_local_client handles import errors and connection failures."""

    def test_returns_none_when_langchain_openai_missing(self):
        from app.utils.llm_factory import _build_local_client

        mock_settings = MagicMock(
            runpod_base_url="https://example.runpod.net/v1",
            runpod_api_key="test-key",
            local_model_name="meta-llama/Llama-3.1-8B-Instruct",
        )
        with patch.dict("sys.modules", {"langchain_openai": None}):
            result = _build_local_client(mock_settings)
            assert result is None

    def test_returns_none_on_exception(self):
        from app.utils.llm_factory import _build_local_client

        mock_settings = MagicMock(
            runpod_base_url="https://example.runpod.net/v1",
            runpod_api_key="test-key",
            local_model_name="meta-llama/Llama-3.1-8B-Instruct",
        )
        with patch("langchain_openai.ChatOpenAI", side_effect=RuntimeError("boom")):
            result = _build_local_client(mock_settings)
            assert result is None
