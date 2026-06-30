"""Router node — checks pgvector cache before processing.

If a similar past incident is found in resolved_incidents, short-circuits
directly to human_gate with the cached recommended action.
"""
from __future__ import annotations

import logging

from langsmith import traceable

from app.agents.state import SREState
from app.config import get_settings

logger = logging.getLogger(__name__)


@traceable(name="router_node", metadata={"phase": "routing"})
async def router_node(state: SREState) -> dict:
    """Check pgvector cache. On a hit, populate recommended_action and short-circuit."""
    settings = get_settings()
    error_summary = state.get("triage_summary", "")

    base_update: dict = {
        "cache_hit": False,
        "cache_key": "",
        "reasoning_log": list(state.get("reasoning_log", [])) + ["[router] cache_hit=False"],
        "current_node": "router",
    }

    if error_summary and settings.postgres_dsn:
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
    try:
        from app.utils.incident_store import cache_lookup

        return await cache_lookup(
            dsn=settings.postgres_dsn,
            error_summary=error_summary,
            threshold=settings.cache_similarity_threshold,
        )
    except Exception as exc:
        logger.warning("Cache check failed (non-critical): %s", exc)
        return None
