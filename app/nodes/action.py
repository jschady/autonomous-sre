"""Action node — executes the approved remediation action.

Reads human_approved flag from state. If approved, runs the recommended
tool (restart_service or execute_rollback). If not approved, escalates.
"""
from __future__ import annotations

import re

from langsmith import traceable

from app.agents.state import SREState
from app.tools import TOOL_REGISTRY

_VALID_TOOLS = frozenset({"restart_service", "execute_rollback", "scale_memory"})


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

    namespace = metadata.get("namespace", "default")
    # Use the workload name derived by processor (deployment/statefulset name stripped of pod/RS hashes).
    # Fall back to service label or alertname if processor didn't run yet.
    workload = (
        state.get("workload")
        or metadata.get("service")
        or state["alert_payload"].get("alertname", "unknown")
    )

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
        if tool_name == "scale_memory":
            factor = _extract_scale_factor(proposed_action)
            initial_limit = _extract_initial_limit(proposed_action)
            result = tool.invoke({"workload_name": workload, "namespace": namespace, "factor": factor, "initial_limit": initial_limit})
        elif tool_name == "restart_service":
            result = tool.invoke({"service_id": workload, "namespace": namespace})
        elif tool_name == "execute_rollback":
            result = tool.invoke({"deployment_name": workload, "namespace": namespace})
        else:
            result = tool.invoke({"service_id": workload, "namespace": namespace})

        # Detect RBAC denial surfaced as a string prefix in the tool result
        rbac_blocked = isinstance(result, str) and result.startswith("[RBAC_DENIED]")

        reasoning_entry = f"[action] executed tool={tool_name} workload={workload} namespace={namespace} | result={result!r}"
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
    text_lower = proposed_action.lower()

    # OOM / memory-related → scale memory limits then restart
    if any(kw in text_lower for kw in ("scale memory", "increase memory", "memory limit", "oom")):
        return "scale_memory"
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


_MEMORY_LIMIT_PATTERN = re.compile(
    r"\binitial[_-]?limit[=\s]+([0-9]+(?:\.[0-9]+)?(?:Gi|Mi|Ki|G|M|K)i?)\b",
    re.IGNORECASE,
)
_DEFAULT_INITIAL_LIMIT = "256Mi"


def _extract_initial_limit(text: str) -> str:
    """Parse `initial_limit=XXXMi` from proposed_action; defaults to '256Mi'."""
    match = _MEMORY_LIMIT_PATTERN.search(text)
    return match.group(1) if match else _DEFAULT_INITIAL_LIMIT


def _extract_scale_factor(text: str) -> float:
    """Parse `factor=X.X` from a proposed_action string; clamp to [1.1, 4.0].

    Falls back to 2.0 if no factor is found.
    """
    match = re.search(r"\bfactor[=\s]+([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if match:
        return max(1.1, min(float(match.group(1)), 4.0))
    return 2.0
