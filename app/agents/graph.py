"""LangGraph agent graph for the Autonomous SRE system.

Defines the stateful workflow:
  triage → router → processor → researcher → human_gate → action → verification
  with conditional routing for:
    - escalation / failure after triage
    - semantic cache hits (router → human_gate, skipping processor + researcher)
    - RBAC blocking after action
    - retries after failed verification

Checkpointing:
  - Accepts a checkpointer instance (AsyncPostgresSaver or MemorySaver).
  - The caller (FastAPI lifespan) is responsible for initialising the checkpointer.
  - Falls back to MemorySaver when no checkpointer is provided.
"""
from __future__ import annotations

import logging

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.state import SREState
from app.nodes.action import action_node
from app.nodes.human_gate import human_gate_node, notify_slack_node
from app.nodes.processor import processor_node
from app.nodes.researcher import research_node
from app.nodes.router import router_node
from app.nodes.triage import triage_node
from app.nodes.verification import verification_node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Terminal node
# ---------------------------------------------------------------------------

async def escalate_node(state: SREState) -> dict:
    return {
        "status": "escalated",
        "resolved": False,
        "current_node": "escalate",
    }


# ---------------------------------------------------------------------------
# Router functions
# ---------------------------------------------------------------------------

def route_after_triage(state: SREState) -> str:
    """Route to router if triage succeeded, otherwise end (escalated/failed)."""
    if state.get("status") == "escalated":
        return END
    if state.get("status") == "failed":
        return END
    return "router"


def route_after_router(state: SREState) -> str:
    """Route based on semantic cache result.

    - cache_hit=True  → notify_slack (skip processor + researcher)
    - cache_hit=False → processor  (full analysis path)
    """
    if state.get("cache_hit"):
        return "notify_slack"
    return "processor"


def route_after_action(state: SREState) -> str:
    """Route to escalate if RBAC blocked or human rejected, otherwise to verification."""
    if state.get("rbac_blocked"):
        return "escalate"
    if state.get("status") == "escalated":
        return "escalate"
    return "verification"


def route_after_verification(state: SREState) -> str:
    """Route to END if resolved, escalate if retries exhausted, else retry triage."""
    if state.get("resolved"):
        return END
    if state.get("status") in ("escalated", "failed"):
        return "escalate"
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    if retry_count >= max_retries:
        return "escalate"
    return "triage"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Build and compile the SRE agent graph.

    Args:
        checkpointer: An already-initialised checkpointer (e.g. AsyncPostgresSaver
                      entered via its async context manager, or MemorySaver).
                      Defaults to a new MemorySaver when None is provided.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()
        logger.info("build_graph: using MemorySaver (no checkpointer provided)")
    else:
        logger.info("build_graph: using %s", type(checkpointer).__name__)

    workflow = StateGraph(SREState)

    workflow.add_node("triage", triage_node)
    workflow.add_node("router", router_node)
    workflow.add_node("processor", processor_node)
    workflow.add_node("researcher", research_node)
    workflow.add_node("notify_slack", notify_slack_node)
    workflow.add_node("human_gate", human_gate_node)
    workflow.add_node("action", action_node)
    workflow.add_node("verification", verification_node)
    workflow.add_node("escalate", escalate_node)

    workflow.set_entry_point("triage")

    # triage → router (or END on failure)
    workflow.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "router": "router",
            END: END,
        },
    )

    # router → processor (full path) or notify_slack (cache hit, skips processor+researcher)
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "processor": "processor",
            "notify_slack": "notify_slack",
        },
    )

    workflow.add_edge("processor", "researcher")
    workflow.add_edge("researcher", "notify_slack")
    workflow.add_edge("notify_slack", "human_gate")
    workflow.add_edge("human_gate", "action")

    workflow.add_conditional_edges(
        "action",
        route_after_action,
        {
            "verification": "verification",
            "escalate": "escalate",
        },
    )
    workflow.add_conditional_edges(
        "verification",
        route_after_verification,
        {
            "triage": "triage",
            "escalate": "escalate",
            END: END,
        },
    )
    workflow.add_edge("escalate", END)

    return workflow.compile(checkpointer=checkpointer)
