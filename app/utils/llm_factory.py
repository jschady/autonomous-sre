"""LLM client factory for Claude (Anthropic)."""
from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from app.config import get_settings

logger = logging.getLogger(__name__)


def get_llm_client(model_override: str | None = None) -> BaseChatModel:
    """Return a ChatAnthropic client."""
    return _build_claude_client(get_settings(), model_override)


def _build_claude_client(settings, model_override: str | None) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    model = model_override or settings.triage_model
    return ChatAnthropic(model=model, api_key=settings.anthropic_api_key)


async def ainvoke_with_fallback(
    provider: str,
    messages: list,
    model_override: str | None = None,
):
    """Invoke Claude. The provider argument is ignored (kept for call-site compatibility)."""
    return await _build_claude_client(get_settings(), model_override).ainvoke(messages)
