"""Human gate node — pauses graph execution pending human approval.

Uses LangGraph's interrupt() mechanism to surface the proposed action
to a human operator. Sends a rich Slack Block Kit message (when configured)
before pausing so operators receive an actionable notification immediately.

Resumes only after explicit approval/rejection via /slack/interactive.
"""
from __future__ import annotations

import logging

from langgraph.types import interrupt

from app.agents.state import SREState

logger = logging.getLogger(__name__)


async def human_gate_node(state: SREState) -> dict:
    """Interrupt graph execution and wait for human approval.

    1. Sends a Slack Block Kit approval message (best-effort; never blocks).
    2. Calls interrupt() to pause the graph — state is checkpointed at this point.
    3. On resume, the caller must provide {"approved": True/False}.
    """
    alert_id = state["alert_id"]
    proposed_action = state.get("proposed_action", "")
    alertname = state.get("metadata", {}).get("alertname", "Unknown Alert")
    severity = state.get("severity", "unknown")
    error_summary = state.get("error_summary", "")
    triage_summary = state.get("triage_summary", "")

    # Send Slack notification (best-effort; failure must not prevent the interrupt)
    await _notify_slack(
        alert_id=alert_id,
        alertname=alertname,
        severity=severity,
        error_summary=error_summary,
        triage_summary=triage_summary,
        proposed_action=proposed_action,
    )

    approval = interrupt({
        "proposed_action": proposed_action,
        "alert_id": alert_id,
        "metadata": state["metadata"],
        "error_summary": error_summary,
        "severity": severity,
        "triage_summary": triage_summary,
    })

    approved = approval.get("approved", False) if isinstance(approval, dict) else False

    reasoning_entry = (
        f"[human_gate] decision={'approved' if approved else 'rejected'} | "
        f"proposed_action={proposed_action!r}"
    )

    return {
        "human_approved": approved,
        "reasoning_log": list(state.get("reasoning_log", [])) + [reasoning_entry],
        "current_node": "human_gate",
    }


async def _notify_slack(
    alert_id: str,
    alertname: str,
    severity: str,
    error_summary: str,
    triage_summary: str,
    proposed_action: str,
) -> None:
    """Send Slack Block Kit approval message. Silently logs on failure."""
    try:
        from app.utils.slack_blocks import build_approval_message
        from app.utils.slack_client import send_slack_message

        blocks = build_approval_message(
            alert_id=alert_id,
            alertname=alertname,
            severity=severity,
            error_summary=error_summary,
            triage_summary=triage_summary,
            proposed_action=proposed_action,
        )
        await send_slack_message(blocks)
    except Exception as exc:
        logger.warning("Slack notification failed for alert %s: %s", alert_id, exc)
