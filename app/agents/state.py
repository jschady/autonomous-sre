"""SREState schema and factory for the Autonomous SRE agent graph."""
import logging
import os
import uuid
from typing import Optional, TypedDict

logger = logging.getLogger(__name__)


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

    # --- Processor Output (also used by action/verification) ---
    workload: str  # Deployment/workload name derived from pod label (e.g. "chaos-app")

    # --- Action Output ---
    proposed_action: str
    human_approved: bool
    action_result: str
    rbac_blocked: bool  # True if k8s action was denied due to RBAC
    k8s_error: Optional[str]  # Raw k8s API error for reasoning log

    # --- Slack Thread Tracking ---
    slack_message_ts: Optional[str]   # ts of the last posted approval message
    slack_channel: Optional[str]      # channel of the last posted approval message
    slack_message_ts_list: list[str]  # all approval message ts values (for bulk delete on resolution)

    # --- Verification Output ---
    metrics_healthy: bool
    resolved: bool

    # --- Phase 3: Routing & Caching ---
    task_complexity: str   # "simple", "moderate", "complex" (set by triage)
    llm_provider: str      # "claude" or "local" (set by router)
    cache_hit: bool        # True if semantic cache matched
    cache_key: str         # Redis key of the cache hit

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


_PROMETHEUS_INTERNAL_LABELS = {"job", "instance"}


def _extract_metadata(alert_payload: dict) -> dict:
    """Extract structured metadata from webhook labels.

    Prometheus-internal labels (job, instance) are excluded — they describe
    the scrape target, not the alerting workload, and confuse LLM reasoning.
    """
    labels = alert_payload.get("labels", {})
    standard_keys = {"region", "env", "cluster_id", "namespace"}
    excluded = standard_keys | _PROMETHEUS_INTERNAL_LABELS
    namespace = labels.get("namespace")
    if namespace is None:
        logger.warning(
            "alert_payload missing 'namespace' label (alertname=%r); "
            "defaulting to 'default' — all K8s tool calls will target the default namespace",
            alert_payload.get("alertname", "unknown"),
        )
        namespace = "default"
    metadata = {
        "region": labels.get("region", "unknown"),
        "env": labels.get("env", "unknown"),
        "cluster_id": labels.get("cluster_id", "unknown"),
        "namespace": namespace,
        **{k: v for k, v in labels.items() if k not in excluded},
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
        workload="",
        # Research
        sop_matches=[],
        recommended_action="",
        # Action
        proposed_action="",
        human_approved=False,
        action_result="",
        rbac_blocked=False,
        k8s_error=None,
        # Slack thread tracking
        slack_message_ts=None,
        slack_channel=None,
        slack_message_ts_list=[],
        # Verification
        metrics_healthy=False,
        resolved=False,
        # Phase 3: Routing & Caching
        task_complexity="moderate",
        llm_provider="claude",
        cache_hit=False,
        cache_key="",
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
