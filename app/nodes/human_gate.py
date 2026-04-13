"""Human gate node — pauses graph execution pending human approval.

Uses LangGraph's interrupt() mechanism to surface the proposed action
to a human operator. Resumes only after explicit approval/rejection.
"""
from __future__ import annotations

from langgraph.types import interrupt

from app.agents.state import SREState


async def human_gate_node(state: SREState) -> dict:
    """Interrupt graph execution and wait for human approval.

    The interrupt payload is surfaced via the LangGraph API.
    On resume, the caller must provide {"approved": True/False}.
    """
    approval = interrupt({
        "proposed_action": state.get("proposed_action", ""),
        "alert_id": state["alert_id"],
        "metadata": state["metadata"],
        "error_summary": state.get("error_summary", ""),
        "severity": state.get("severity", "unknown"),
        "triage_summary": state.get("triage_summary", ""),
    })

    approved = approval.get("approved", False) if isinstance(approval, dict) else False

    reasoning_entry = (
        f"[human_gate] decision={'approved' if approved else 'rejected'} | "
        f"proposed_action={state.get('proposed_action', '')!r}"
    )

    return {
        "human_approved": approved,
        "reasoning_log": list(state.get("reasoning_log", [])) + [reasoning_entry],
        "current_node": "human_gate",
    }
