"""Router node — checks semantic cache before processing.

If a similar past incident is found in the cache, short-circuits directly
to human_gate with the cached recommended action, skipping processor and
researcher.
"""
from __future__ import annotations

import logging

from langsmith import traceable

from app.agents.state import SREState
from app.config import get_settings

logger = logging.getLogger(__name__)


@traceable(name="router_node", metadata={"phase": "routing"})
async def router_node(state: SREState) -> dict:
    """Check semantic cache. On a hit, populate recommended_action and short-circuit."""
    settings = get_settings()
    error_summary = state.get("error_summary", "")

    base_update: dict = {
        "cache_hit": False,
        "cache_key": "",
        "reasoning_log": list(state.get("reasoning_log", [])) + ["[router] cache_hit=False"],
        "current_node": "router",
    }

    if settings.semantic_cache_enabled and error_summary:
        cache_result = await _check_cache(settings, error_summary)
        if cache_result is not None:
            recommended = cache_result.get("recommended_action", "")
            cache_key = cache_result.get("cache_key", "")
            return {
                **base_update,
                "cache_hit": True,
                "cache_key": cache_key,
                "recommended_action": recommended,
                "proposed_action": recommended,
                "reasoning_log": list(state.get("reasoning_log", [])) + [
                    f"[router] cache_hit=True | recommended_action={recommended!r}"
                ],
            }

    return base_update


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
