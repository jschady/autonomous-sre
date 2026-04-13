"""Verification node — checks system metrics to confirm resolution.

Calls get_system_metrics using the service from metadata.
If metrics are healthy, marks resolved=True. Otherwise, increments retry_count.
"""
from __future__ import annotations

import re

from langsmith import traceable

from app.agents.state import SREState
from app.tools.k8s_tools import get_system_metrics


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


def _is_healthy(metrics_str: str) -> bool:
    """Parse metrics string and determine if system is healthy."""
    match = re.search(r"error_rate=([\d.]+)", metrics_str)
    if match:
        error_rate = float(match.group(1))
        return error_rate < _ERROR_RATE_THRESHOLD

    # If we can't parse error rate, check status field
    return "status=healthy" in metrics_str.lower()
