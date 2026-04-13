"""LangGraph agent graph for the Autonomous SRE system.

Defines the stateful workflow:
  triage → processor → researcher → human_gate → action → verification
  with conditional routing for escalation, RBAC blocking, and retries.

Checkpointing:
  - Uses PostgresSaver when POSTGRES_DSN is set (Phase 2).
  - Falls back to MemorySaver when POSTGRES_DSN is absent (Phase 1 compat).
"""
from __future__ import annotations

import logging
import os

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.state import SREState
from app.nodes.action import action_node
from app.nodes.human_gate import human_gate_node
from app.nodes.processor import processor_node
from app.nodes.researcher import research_node
from app.nodes.triage import triage_node
from app.nodes.verification import verification_node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Terminal node
# ---------------------------------------------------------------------------

async def escalate_node(state: SREState) -> dict:
    """Terminal escalation node — marks alert as escalated and ends graph."""
    return {
        "status": "escalated",
        "resolved": False,
        "current_node": "escalate",
    }


# ---------------------------------------------------------------------------
# Router functions
# ---------------------------------------------------------------------------

def route_after_triage(state: SREState) -> str:
    """Route to processor if triage succeeded, otherwise end (escalated)."""
    if state.get("status") == "escalated":
        return END
    if state.get("status") == "failed":
        return END
    return "processor"


def route_after_action(state: SREState) -> str:
    """Route to escalate if RBAC blocked the action, otherwise to verification."""
    if state.get("rbac_blocked"):
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
# Checkpointer factory
# ---------------------------------------------------------------------------

def _build_checkpointer():
    """Return PostgresSaver if POSTGRES_DSN is set, otherwise MemorySaver.

    Note: AsyncPostgresSaver.from_conn_string() returns an async context manager
    in modern langgraph versions. build_graph() is synchronous, so we fall back
    to MemorySaver when the result cannot be used directly as a checkpointer.
    """
    dsn = os.environ.get("POSTGRES_DSN", "")
    if dsn:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            result = AsyncPostgresSaver.from_conn_string(dsn)
            if hasattr(result, "__aenter__"):
                logger.warning(
                    "AsyncPostgresSaver.from_conn_string returned a context manager; "
                    "build_graph() is synchronous — falling back to MemorySaver"
                )
            else:
                logger.info("Using PostgresSaver for checkpointing (DSN configured)")
                return result
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to create PostgresSaver, falling back to MemorySaver: %s", exc
            )
    logger.info("Using MemorySaver for checkpointing (no POSTGRES_DSN)")
    return MemorySaver()


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> CompiledStateGraph:
    """Build and compile the SRE agent graph.

    Uses PostgresSaver when POSTGRES_DSN is configured, MemorySaver otherwise.
    """
    workflow = StateGraph(SREState)

    # Register nodes
    workflow.add_node("triage", triage_node)
    workflow.add_node("processor", processor_node)
    workflow.add_node("researcher", research_node)
    workflow.add_node("human_gate", human_gate_node)
    workflow.add_node("action", action_node)
    workflow.add_node("verification", verification_node)
    workflow.add_node("escalate", escalate_node)

    # Entry point
    workflow.set_entry_point("triage")

    # Edges
    workflow.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "processor": "processor",
            END: END,
        },
    )
    workflow.add_edge("processor", "researcher")
    workflow.add_edge("researcher", "human_gate")
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

    checkpointer = _build_checkpointer()
    return workflow.compile(checkpointer=checkpointer)
