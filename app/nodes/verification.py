"""Verification node — checks system metrics to confirm resolution.

Calls get_system_metrics using the service from metadata.
If metrics are healthy, marks resolved=True, persists the incident to
the resolved_incidents table (for few-shot retrieval), and stores it
in the semantic cache (for future cache hits).

Otherwise, increments retry_count.
"""
from __future__ import annotations

import logging
import os
import re

from langsmith import traceable

from app.agents.state import SREState
from app.tools.k8s_tools import get_system_metrics

logger = logging.getLogger(__name__)

_ERROR_RATE_THRESHOLD = 5.0  # percent


@traceable(name="verification_node", metadata={"phase": "verification"})
async def verification_node(state: SREState) -> dict:
    """Check metrics after remediation action and update resolved/retry state."""
    metadata = state["metadata"]
    payload = state["alert_payload"]
    service = metadata.get("service", payload.get("alertname", "unknown"))

    try:
        metrics_str = get_system_metrics.invoke({"service_name": service})
        healthy = _is_healthy(metrics_str)

    except Exception as exc:
        return {
            "metrics_healthy": False,
            "resolved": False,
            "retry_count": state.get("retry_count", 0) + 1,
            "error_log": state.get("error_log", []) + [f"verification_node error: {exc}"],
            "current_node": "verification",
        }

    retry_count = state.get("retry_count", 0)

    if healthy:
        reasoning_entry = f"[verification] metrics=healthy | resolved=True | service={service}"

        # Persist the successful resolution for future few-shot + cache use
        await _persist_resolution(state)

        return {
            "metrics_healthy": True,
            "resolved": True,
            "status": "resolved",
            "reasoning_log": list(state.get("reasoning_log", [])) + [reasoning_entry],
            "current_node": "verification",
        }

    reasoning_entry = (
        f"[verification] metrics=degraded | resolved=False | "
        f"retry={retry_count + 1} | service={service}"
    )
    return {
        "metrics_healthy": False,
        "resolved": False,
        "retry_count": retry_count + 1,
        "reasoning_log": list(state.get("reasoning_log", [])) + [reasoning_entry],
        "current_node": "verification",
    }


async def _persist_resolution(state: dict) -> None:
    """Fire-and-forget: save to incident_store + semantic cache. Swallows all errors."""
    dsn = os.environ.get("POSTGRES_DSN", "")
    redis_url = os.environ.get("REDIS_URL", "")

    error_summary = state.get("error_summary", "")
    recommended_action = state.get("recommended_action", "")

    # Save to Postgres resolved_incidents table
    if dsn:
        try:
            from app.utils.incident_store import save_resolved_incident
            await save_resolved_incident(dsn, state)
        except Exception as exc:
            logger.warning("Failed to save resolved incident (non-critical): %s", exc)

    # Store in semantic cache for future fast-path
    if redis_url and error_summary and recommended_action:
        try:
            from app.config import get_settings
            from app.utils.semantic_cache import cache_store

            settings = get_settings()
            if settings.semantic_cache_enabled:
                await cache_store(
                    redis_url=redis_url,
                    error_summary=error_summary,
                    recommended_action=recommended_action,
                    ttl_seconds=settings.cache_ttl_seconds,
                )
        except Exception as exc:
            logger.warning("Failed to store in semantic cache (non-critical): %s", exc)


def _is_healthy(metrics_str: str) -> bool:
    """Parse metrics string and determine if system is healthy."""
    match = re.search(r"error_rate=([\d.]+)", metrics_str)
    if match:
        error_rate = float(match.group(1))
        return error_rate < _ERROR_RATE_THRESHOLD

    return "status=healthy" in metrics_str.lower()
