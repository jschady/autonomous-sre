"""Action node — executes the approved remediation action.

Reads human_approved flag from state. If approved, runs the recommended
tool (restart_service or execute_rollback). If not approved, escalates.
"""
from __future__ import annotations

from langsmith import traceable

from app.agents.state import SREState
from app.tools import TOOL_REGISTRY

_VALID_TOOLS = frozenset({"restart_service", "execute_rollback"})


@traceable(name="action_node", metadata={"phase": "action"})
async def action_node(state: SREState) -> dict:
    """Execute approved remediation or escalate if rejected."""
    if not state.get("human_approved", False):
        reasoning_entry = "[action] human rejected — escalating to on-call team"
        return {
            "status": "escalated",
            "action_result": "Action rejected by human operator — escalating to on-call team.",
            "reasoning_log": list(state.get("reasoning_log", [])) + [reasoning_entry],
            "current_node": "action",
        }

    recommended_action = state.get("recommended_action", "")
    proposed_action = state.get("proposed_action", recommended_action)
    sop_matches = state.get("sop_matches", [])
    metadata = state["metadata"]

    service = metadata.get("service", state["alert_payload"].get("alertname", "unknown"))
    deployment = metadata.get("service", service)
    namespace = metadata.get("namespace", "default")

    # Determine which tool to call
    tool_name = _determine_action_tool(proposed_action, sop_matches)

    if tool_name not in TOOL_REGISTRY:
        return {
            "action_result": (
                f"Could not determine action tool from: '{proposed_action}'. "
                "Manual intervention required."
            ),
            "current_node": "action",
        }

    tool = TOOL_REGISTRY[tool_name]
    try:
        if tool_name == "restart_service":
            result = tool.invoke({"service_id": service, "namespace": namespace})
        elif tool_name == "execute_rollback":
            result = tool.invoke({"deployment_name": deployment, "namespace": namespace})
        else:
            result = tool.invoke({"service_id": service, "namespace": namespace})

        # Detect RBAC denial surfaced as a string prefix in the tool result
        rbac_blocked = isinstance(result, str) and result.startswith("[RBAC_DENIED]")

        reasoning_entry = f"[action] executed tool={tool_name} | result={result!r}"
        return {
            "action_result": result,
            "rbac_blocked": rbac_blocked,
            "reasoning_log": list(state.get("reasoning_log", [])) + [reasoning_entry],
            "current_node": "action",
        }

    except Exception as exc:
        return {
            "action_result": f"Action execution failed: {exc}",
            "rbac_blocked": False,
            "status": "failed",
            "error_log": list(state.get("error_log", [])) + [f"action_node error: {exc}"],
            "current_node": "action",
        }


def _determine_action_tool(proposed_action: str, sop_matches: list[dict]) -> str:
    """Determine which tool to call based on proposed_action text and SOPs."""
    text_lower = proposed_action.lower()

    if "restart" in text_lower:
        return "restart_service"
    if "rollback" in text_lower or "roll back" in text_lower:
        return "execute_rollback"

    # Fall back to SOP recommendation
    for sop in sop_matches:
        tool = sop.get("recommended_tool", "")
        if tool in _VALID_TOOLS:
            return tool

    return "restart_service"
