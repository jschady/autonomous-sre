"""Router node — selects LLM provider and checks semantic cache.

Inserted between triage and processor. Responsibilities:
  1. Check semantic cache: if a similar past incident exists, short-circuit
     and skip directly to human_gate with the cached recommended action.
  2. Select the LLM provider (Claude or local Llama) based on task_complexity
     set by triage.

Routing rules:
  - local_model_enabled=False → always "claude"
  - task_complexity "simple" or "moderate" → "local" (cost-saving)
  - task_complexity "complex" → "claude" (reasoning-heavy)
"""
from __future__ import annotations

import logging
import os

from langsmith import traceable

from app.agents.state import SREState
from app.config import get_settings

logger = logging.getLogger(__name__)

# Complexities that benefit from local inference (cheaper, fast enough)
_LOCAL_COMPLEXITIES = {"simple", "moderate"}


@traceable(name="router_node", metadata={"phase": "routing"})
async def router_node(state: SREState) -> dict:
    """Select LLM provider + check semantic cache.

    Returns a dict with:
      - llm_provider: "claude" or "local"
      - cache_hit: True if a cached resolution was found
      - cache_key: Redis key of the matched cache entry
      - recommended_action: populated if cache_hit is True
      - proposed_action: populated if cache_hit is True
      - current_node: "router"
    """
    settings = get_settings()
    complexity = state.get("task_complexity", "moderate")
    error_summary = state.get("error_summary", "")

    # --- LLM provider selection ---
    provider = _select_provider(complexity, settings)

    reasoning_entry = (
        f"[router] task_complexity={complexity} | llm_provider={provider}"
    )

    base_update: dict = {
        "llm_provider": provider,
        "cache_hit": False,
        "cache_key": "",
        "reasoning_log": list(state.get("reasoning_log", [])) + [reasoning_entry],
        "current_node": "router",
    }

    # --- Semantic cache check ---
    if settings.semantic_cache_enabled and error_summary:
        cache_result = await _check_cache(settings, error_summary)
        if cache_result is not None:
            recommended = cache_result.get("recommended_action", "")
            cache_key = cache_result.get("cache_key", "")
            cache_reasoning = (
                f"[router] cache_hit=True | recommended_action={recommended!r}"
            )
            return {
                **base_update,
                "cache_hit": True,
                "cache_key": cache_key,
                "recommended_action": recommended,
                "proposed_action": recommended,
                "reasoning_log": list(state.get("reasoning_log", []))
                + [reasoning_entry, cache_reasoning],
            }

    return base_update


def _select_provider(complexity: str, settings) -> str:  # type: ignore[return]
    """Determine which LLM provider to use based on complexity and settings."""
    if not settings.local_model_enabled:
        return "claude"

    if complexity in _LOCAL_COMPLEXITIES:
        logger.debug("router: routing to local model (complexity=%s)", complexity)
        return "local"

    logger.debug("router: routing to Claude (complexity=%s)", complexity)
    return "claude"


async def _check_cache(settings, error_summary: str) -> dict | None:
    """Check semantic cache and return cached entry if found."""
    try:
        from app.utils.semantic_cache import cache_lookup

        result = await cache_lookup(
            redis_url=settings.redis_url,
            error_summary=error_summary,
            threshold=settings.cache_similarity_threshold,
        )
        return result
    except Exception as exc:
        logger.warning("Cache check failed (non-critical): %s", exc)
        return None
