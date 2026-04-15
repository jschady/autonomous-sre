"""LLM client factory — returns ChatAnthropic or ChatOpenAI (for vLLM/RunPod).

Reads provider selection and config from Settings. Falls back to Claude
if the local endpoint is unavailable or local_model_enabled is False.
"""
from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from app.config import get_settings

logger = logging.getLogger(__name__)


def get_llm_client(
    provider: str,
    model_override: str | None = None,
) -> BaseChatModel:
    """Return an LLM client for the requested provider.

    Args:
        provider:       "local" or "claude". Any other value falls back to Claude.
        model_override: Override the default model name (only applies to Claude).

    Returns:
        A LangChain BaseChatModel instance. Never raises — falls back to Claude.
    """
    settings = get_settings()

    if provider == "local" and settings.local_model_enabled:
        if settings.runpod_serverless_enabled and settings.runpod_serverless_endpoint_id:
            client = _build_serverless_client(settings)
        elif settings.runpod_base_url:
            client = _build_local_client(settings)
        else:
            client = None
        return client or _build_claude_client(settings, model_override)

    return _build_claude_client(settings, model_override)


def _build_serverless_client(settings) -> BaseChatModel | None:
    """Build a RunPodServerlessChat client for the configured serverless endpoint.

    Returns None on any failure so the caller can fall back to Claude.
    """
    try:
        from app.utils.runpod_client import RunPodServerlessChat

        client = RunPodServerlessChat(
            endpoint_id=settings.runpod_serverless_endpoint_id,
            api_key=settings.runpod_api_key or "",
            model_name=settings.local_model_name,
            timeout=settings.runpod_cold_start_timeout,
        )
        logger.info(
            "LLM factory: using RunPod Serverless endpoint %s",
            settings.runpod_serverless_endpoint_id,
        )
        return client
    except Exception as exc:
        logger.warning(
            "Failed to build RunPod Serverless client (%s) — falling back to Claude", exc
        )
        return None


def _build_local_client(settings) -> BaseChatModel | None:  # type: ignore[return]
    """Try to build a ChatOpenAI client pointing at the RunPod always-on vLLM endpoint.

    Returns None on any failure so the caller can fall back to Claude.
    """
    try:
        from langchain_openai import ChatOpenAI  # type: ignore[import-untyped]

        client = ChatOpenAI(
            base_url=settings.runpod_base_url,
            api_key=settings.runpod_api_key or "EMPTY",
            model=settings.local_model_name,
            timeout=settings.runpod_cold_start_timeout,
            max_retries=0,
        )
        logger.info(
            "LLM factory: using local model %s at %s",
            settings.local_model_name,
            settings.runpod_base_url,
        )
        return client
    except ImportError:
        logger.warning(
            "langchain-openai not installed — cannot use local model, falling back to Claude"
        )
        return None
    except Exception as exc:
        logger.warning(
            "Failed to build local LLM client (%s) — falling back to Claude", exc
        )
        return None


def _build_claude_client(settings, model_override: str | None) -> BaseChatModel:
    """Build a ChatAnthropic client."""
    from langchain_anthropic import ChatAnthropic

    model = model_override or settings.triage_model
    logger.debug("LLM factory: using Claude model %s", model)
    return ChatAnthropic(
        model=model,
        api_key=settings.anthropic_api_key,
    )


async def ainvoke_with_fallback(
    provider: str,
    messages: list,
    model_override: str | None = None,
):
    """Async LLM invoke with fallback to Claude. Use this in async node functions.

    If the call fails (e.g. RunPod endpoint unreachable), retries once with
    Claude and logs a warning rather than propagating the error.

    Returns the LangChain response object.
    Raises only if the Claude fallback also fails.
    """
    settings = get_settings()
    client = get_llm_client(provider=provider, model_override=model_override)

    if provider == "local" and settings.local_model_enabled:
        try:
            return await client.ainvoke(messages)
        except Exception as exc:
            logger.warning(
                "Local LLM ainvoke failed (%s) — falling back to Claude", exc
            )
            claude = _build_claude_client(settings, model_override)
            return await claude.ainvoke(messages)

    return await client.ainvoke(messages)
