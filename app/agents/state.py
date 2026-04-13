"""SREState schema and factory for the Autonomous SRE agent graph."""
import os
import uuid
from typing import Optional, TypedDict


class SREState(TypedDict):
    # --- Input ---
    alert_payload: dict
    alert_id: str
    metadata: dict  # region, env, cluster_id, namespace, etc.

    # --- Triage Output ---
    severity: str
    tools_to_run: list[str]
    triage_summary: str

    # --- Processor Output ---
    raw_logs: str
    error_summary: str

    # --- Research Output ---
    sop_matches: list[dict]
    recommended_action: str

    # --- Action Output ---
    proposed_action: str
    human_approved: bool
    action_result: str
    rbac_blocked: bool  # True if k8s action was denied due to RBAC
    k8s_error: Optional[str]  # Raw k8s API error for reasoning log

    # --- Verification Output ---
    metrics_healthy: bool
    resolved: bool

    # --- LLMOps: Token & Cost Tracking ---
    token_usage: list[dict]      # List of NodeUsage dicts, one per LLM call
    cost_estimate_usd: float     # Running total cost across all LLM calls

    # --- Control Flow ---
    retry_count: int
    max_retries: int
    status: str
    error_log: list[str]
    reasoning_log: list[str]
    current_node: str


_MAX_RETRIES_HARD_CAP = 5
_DEFAULT_MAX_RETRIES = 3


def _extract_metadata(alert_payload: dict) -> dict:
    """Extract structured metadata from webhook labels."""
    labels = alert_payload.get("labels", {})
    standard_keys = {"region", "env", "cluster_id", "namespace"}
    metadata = {
        "region": labels.get("region", "unknown"),
        "env": labels.get("env", "unknown"),
        "cluster_id": labels.get("cluster_id", "unknown"),
        "namespace": labels.get("namespace", "default"),
        **{k: v for k, v in labels.items() if k not in standard_keys},
    }
    return metadata


def _resolve_max_retries() -> int:
    """Read MAX_RETRIES from environment, capped at hard limit of 5."""
    raw = os.environ.get("MAX_RETRIES", str(_DEFAULT_MAX_RETRIES))
    try:
        value = int(raw)
    except ValueError:
        value = _DEFAULT_MAX_RETRIES
    return min(value, _MAX_RETRIES_HARD_CAP)


def create_initial_state(alert_payload: dict) -> SREState:
    """Create a fresh SREState from a raw webhook payload.

    Returns a new dict each call — callers must not rely on shared state.
    """
    return SREState(
        alert_payload=dict(alert_payload),
        alert_id=str(uuid.uuid4()),
        metadata=_extract_metadata(alert_payload),
        # Triage
        severity="unknown",
        tools_to_run=[],
        triage_summary="",
        # Processor
        raw_logs="",
        error_summary="",
        # Research
        sop_matches=[],
        recommended_action="",
        # Action
        proposed_action="",
        human_approved=False,
        action_result="",
        rbac_blocked=False,
        k8s_error=None,
        # Verification
        metrics_healthy=False,
        resolved=False,
        # LLMOps
        token_usage=[],
        cost_estimate_usd=0.0,
        # Control flow
        retry_count=0,
        max_retries=_resolve_max_retries(),
        status="in_progress",
        error_log=[],
        reasoning_log=[],
        current_node="",
    )
