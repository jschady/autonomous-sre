"""Verification node — checks system health to confirm resolution.

For OOM alerts: pod stability (phase + lastState) via K8s API.
For other alerts: service error_rate via Prometheus / mock metrics.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re

from kubernetes.client.exceptions import ApiException
from langsmith import traceable

from app.agents.state import SREState
from app.tools.k8s_helpers import get_core_v1
from app.tools.k8s_tools import get_system_metrics

logger = logging.getLogger(__name__)

_ERROR_RATE_THRESHOLD = 5.0  # percent
_OOM_KEYWORDS = frozenset({"oom", "scale memory", "increase memory", "memory limit"})

# Seconds to wait after action before checking health.
# Gives K8s rolling updates time to create new pods and
# Prometheus time to scrape fresh metrics.
_DEFAULT_VERIFICATION_DELAY = 30
_VERIFICATION_DELAY = int(os.environ.get("VERIFICATION_DELAY_SECONDS", str(_DEFAULT_VERIFICATION_DELAY)))


@traceable(name="verification_node", metadata={"phase": "verification"})
async def verification_node(state: SREState) -> dict:
    """Check pod stability (OOM) or error_rate (other alerts) after remediation."""
    metadata = state["metadata"]
    payload = state["alert_payload"]
    workload = (
        state.get("workload")
        or metadata.get("service")
        or payload.get("alertname", "unknown")
    )
    namespace = metadata.get("namespace", "default")

    # Wait for K8s rolling updates / Prometheus scrapes to reflect the action.
    if _VERIFICATION_DELAY > 0:
        logger.info(
            "Waiting %ds before verification for workload=%s namespace=%s",
            _VERIFICATION_DELAY, workload, namespace,
        )
        await asyncio.sleep(_VERIFICATION_DELAY)

    try:
        if _is_oom_alert(state):
            healthy, reason = _check_pod_stable(workload, namespace)
        else:
            metrics_str = get_system_metrics.invoke({"service_name": workload, "namespace": namespace})
            healthy = _is_healthy(metrics_str)
            reason = metrics_str

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
        reasoning_entry = f"[verification] healthy=True | resolved=True | workload={workload} | {reason}"
        await _persist_resolution(state)
        return {
            "metrics_healthy": True,
            "resolved": True,
            "status": "resolved",
            "reasoning_log": list(state.get("reasoning_log", [])) + [reasoning_entry],
            "current_node": "verification",
        }

    reasoning_entry = (
        f"[verification] healthy=False | resolved=False | "
        f"retry={retry_count + 1} | workload={workload} | {reason}"
    )
    return {
        "metrics_healthy": False,
        "resolved": False,
        "retry_count": retry_count + 1,
        "reasoning_log": list(state.get("reasoning_log", [])) + [reasoning_entry],
        "current_node": "verification",
    }


def _is_oom_alert(state: SREState) -> bool:
    """Return True if this alert is OOMKilled or the action involves memory scaling."""
    alertname = state["alert_payload"].get("alertname", "").lower()
    if "oom" in alertname:
        return True
    text = (state.get("recommended_action", "") + " " + state.get("proposed_action", "")).lower()
    return any(kw in text for kw in _OOM_KEYWORDS)


def _check_pod_stable(workload: str, namespace: str) -> tuple[bool, str]:
    """Check pod stability after scale_memory via K8s API.

    Stable = at least one Running pod with no recent OOMKill in lastState.
    Returns (is_stable, reason_string).
    """
    try:
        core = get_core_v1()
        pod_list = core.list_namespaced_pod(namespace=namespace)
        pods = [
            p for p in (pod_list.items or [])
            if (p.metadata.name or "").startswith(workload)
        ]
    except ApiException as exc:
        return False, f"k8s API error: {exc}"
    except Exception as exc:
        return False, f"unexpected error: {exc}"

    if not pods:
        return False, f"no pods found for workload={workload} in namespace={namespace}"

    for pod in pods:
        phase = pod.status.phase or "Unknown"
        if phase != "Running":
            continue
        recently_oom = any(
            (cs.last_state and cs.last_state.terminated
             and cs.last_state.terminated.reason == "OOMKilled")
            for cs in (pod.status.container_statuses or [])
        )
        if not recently_oom:
            return True, f"pod={pod.metadata.name} phase=Running no_recent_oom=True"

    phases = [(p.metadata.name, p.status.phase or "Unknown") for p in pods]
    return False, f"no stable Running pods; phases={phases}"


async def _persist_resolution(state: dict) -> None:
    """Fire-and-forget: save to incident_store + semantic cache. Swallows all errors."""
    dsn = os.environ.get("POSTGRES_DSN", "")
    redis_url = os.environ.get("REDIS_URL", "")

    error_summary = state.get("triage_summary", "")
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
