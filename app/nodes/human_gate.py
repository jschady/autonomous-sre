"""Human gate node — pauses graph execution pending human approval.

Uses LangGraph's interrupt() mechanism to surface the proposed action
to a human operator.

The Slack notification is intentionally split into a SEPARATE preceding
node (notify_slack_node) so that slack_message_ts is checkpointed before
interrupt() is ever called.  LangGraph re-runs a node from the top when
resuming from interrupt(), so any code before interrupt() would execute
twice.  By moving the Slack post into notify_slack_node (which completes
and is checkpointed before human_gate_node starts), the notification fires
exactly once regardless of how many times human_gate_node is resumed.

On retry, notify_slack_node updates the existing approval message in-place
with the new approval request and prior attempt context — no separate message.

Resumes only after explicit approval/rejection via /slack/interactive.
"""
from __future__ import annotations

import logging

from langgraph.types import interrupt

from app.agents.state import SREState

logger = logging.getLogger(__name__)


async def notify_slack_node(state: SREState) -> dict:
    """Post or update the Slack approval message and checkpoint the ts.

    Runs immediately before human_gate_node.  Because this node completes
    and returns before human_gate_node calls interrupt(), its state updates
    (slack_message_ts, slack_channel, slack_message_ts_list) are persisted
    in the checkpoint.  On retry paths the existing message is updated
    in-place rather than re-posted.
    """
    alert_id = state["alert_id"]
    proposed_action = state.get("proposed_action", "")
    alertname = state.get("alert_payload", {}).get("alertname", "Unknown Alert")
    severity = state.get("severity", "unknown")
    error_summary = state.get("error_summary", "")
    triage_summary = state.get("triage_summary", "")
    action_result = state.get("action_result", "")  # populated on retries

    old_ts = state.get("slack_message_ts")
    old_channel = state.get("slack_channel")

    new_ts, new_channel, is_new_message = await _notify_slack(
        alert_id=alert_id,
        alertname=alertname,
        severity=severity,
        error_summary=error_summary,
        triage_summary=triage_summary,
        proposed_action=proposed_action,
        prior_attempt=action_result or None,
        old_ts=old_ts,
        old_channel=old_channel,
    )

    existing_ts_list = list(state.get("slack_message_ts_list", []))
    updated_ts_list = existing_ts_list + ([new_ts] if is_new_message and new_ts else [])

    return {
        "slack_message_ts": new_ts,
        "slack_channel": new_channel,
        "slack_message_ts_list": updated_ts_list,
        "current_node": "notify_slack",
    }


async def human_gate_node(state: SREState) -> dict:
    """Interrupt graph execution and wait for human approval.

    notify_slack_node has already posted/updated the Slack approval message
    and checkpointed slack_message_ts before this node runs.  This node
    only calls interrupt() and processes the resume value — no Slack I/O.

    On resume, LangGraph re-runs this node from the top.  Since there are
    no Slack calls here, that re-execution is safe and idempotent.
    """
    alert_id = state["alert_id"]
    proposed_action = state.get("proposed_action", "")
    error_summary = state.get("error_summary", "")
    severity = state.get("severity", "unknown")
    triage_summary = state.get("triage_summary", "")

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
    prior_attempt: str | None = None,
    old_ts: str | None = None,
    old_channel: str | None = None,
) -> tuple[str | None, str | None, bool]:
    """Send or update Slack approval message.

    On retry (old_ts + old_channel present), updates the existing message
    in-place with new approval content and prior attempt context.
    On first run, posts a new message.

    Returns:
        (ts, channel, is_new_message)
        is_new_message=True when a new message was posted (ts should be tracked).
        is_new_message=False when an existing message was updated in-place.
    """
    try:
        from app.utils.slack_blocks import build_approval_message
        from app.utils.slack_client import send_slack_message, update_slack_message

        blocks = build_approval_message(
            alert_id=alert_id,
            alertname=alertname,
            severity=severity,
            error_summary=error_summary,
            triage_summary=triage_summary,
            proposed_action=proposed_action,
            prior_attempt=prior_attempt,
        )

        if old_ts and old_channel:
            # Retry: update the existing message in-place (preserves the same ts)
            await update_slack_message(ts=old_ts, channel=old_channel, blocks=blocks)
            return (old_ts, old_channel, False)

        # First run: post a new message
        resp = await send_slack_message(blocks)
        new_ts = (resp or {}).get("ts")
        new_channel = (resp or {}).get("channel")
        return (new_ts, new_channel, True)
    except Exception as exc:
        logger.warning("Slack notification failed for alert %s: %s", alert_id, exc)
        return (None, None, False)
