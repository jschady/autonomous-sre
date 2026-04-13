"""FastAPI application for the Autonomous SRE webhook API.

Endpoints:
  POST /webhook           — Receive Alertmanager webhook, start graph execution
  GET  /status/{alert_id} — Retrieve current graph state for an alert
  POST /slack/interactive  — Receive Slack approval/rejection to resume graph

State persistence:
  - Alert states stored in Redis when REDIS_URL is configured (Phase 2).
  - Falls back to in-memory dict when Redis is unavailable (Phase 1 compat).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException

from app.agents.graph import build_graph
from app.agents.state import SREState, create_initial_state
from app.models import AlertStatusResponse, AlertWebhook, SlackInteraction
from app.utils.cost_store import save_alert_cost

logger = logging.getLogger(__name__)

app = FastAPI(title="Autonomous SRE", version="0.2.0")

# ---------------------------------------------------------------------------
# In-memory fallback store (used when Redis is unavailable)
# ---------------------------------------------------------------------------

_ALERT_STATES_MEMORY: dict[str, SREState] = {}
_ALERT_CONFIGS_MEMORY: dict[str, dict] = {}

# Maps alert_id → graph config (thread_id for checkpointer)
ALERT_CONFIGS: dict[str, dict] = _ALERT_CONFIGS_MEMORY

# Single compiled graph instance (reused across requests)
_graph = build_graph()


# ---------------------------------------------------------------------------
# Redis state store (lazy initialisation)
# ---------------------------------------------------------------------------

_redis_client: Any = None


def _get_redis_url() -> str:
    return os.environ.get("REDIS_URL", "")


async def _get_redis() -> Any:
    """Return a Redis client, or None if Redis is not configured/available."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    url = _get_redis_url()
    if not url:
        return None

    try:
        import redis.asyncio as aioredis
        _redis_client = aioredis.from_url(url, decode_responses=True)
        await _redis_client.ping()
        logger.info("Redis connected at %s", url)
        return _redis_client
    except Exception as exc:
        logger.warning("Redis unavailable (%s), using in-memory state store", exc)
        _redis_client = None
        return None


async def _state_set(alert_id: str, state: SREState) -> None:
    """Persist alert state to Redis (or fall back to memory)."""
    r = await _get_redis()
    if r is not None:
        try:
            await r.set(f"alert:{alert_id}", json.dumps(state), ex=86400)
            return
        except Exception as exc:
            logger.warning("Redis set failed for alert %s: %s", alert_id, exc)
    _ALERT_STATES_MEMORY[alert_id] = state


async def _state_get(alert_id: str) -> SREState | None:
    """Retrieve alert state from Redis (or fall back to memory)."""
    r = await _get_redis()
    if r is not None:
        try:
            raw = await r.get(f"alert:{alert_id}")
            if raw is not None:
                return json.loads(raw)
        except Exception as exc:
            logger.warning("Redis get failed for alert %s: %s", alert_id, exc)
    return _ALERT_STATES_MEMORY.get(alert_id)


# Expose ALERT_STATES as a proxy for backward-compat in tests
class _AlertStatesProxy:
    """Synchronous dict-like proxy for backward-compat with Phase 1 tests."""

    def get(self, alert_id: str, default=None):  # noqa: ANN001
        state = _ALERT_STATES_MEMORY.get(alert_id, default)
        return state

    def __setitem__(self, alert_id: str, value: SREState) -> None:
        _ALERT_STATES_MEMORY[alert_id] = value

    def __getitem__(self, alert_id: str) -> SREState:
        return _ALERT_STATES_MEMORY[alert_id]

    def __contains__(self, alert_id: str) -> bool:
        return alert_id in _ALERT_STATES_MEMORY


ALERT_STATES = _AlertStatesProxy()


# ---------------------------------------------------------------------------
# Background task: run graph until interrupt or completion
# ---------------------------------------------------------------------------

async def _run_graph(alert_id: str, initial_state: SREState) -> None:
    """Run the graph from initial state, persisting snapshots to state store."""
    config = ALERT_CONFIGS[alert_id]
    try:
        from langgraph.errors import GraphInterrupt
        result = await _graph.ainvoke(initial_state, config=config)
        await _state_set(alert_id, result)
        await save_alert_cost(os.environ.get("POSTGRES_DSN", ""), result)
    except GraphInterrupt:
        # Graph paused at human_gate — snapshot current state from checkpointer
        snapshot = await _graph.aget_state(config)
        if snapshot and snapshot.values:
            await _state_set(alert_id, snapshot.values)  # type: ignore[arg-type]
    except Exception as exc:
        current = await _state_get(alert_id) or {}
        await _state_set(alert_id, {  # type: ignore[arg-type]
            **current,
            "status": "failed",
            "error_log": list(current.get("error_log", [])) + [str(exc)],
        })


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/webhook", status_code=202)
async def receive_webhook(
    alert: AlertWebhook,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """Accept a Prometheus/Alertmanager webhook and start remediation graph."""
    payload = alert.model_dump()
    initial_state = create_initial_state(payload)
    alert_id = initial_state["alert_id"]

    config = {"configurable": {"thread_id": alert_id}}
    ALERT_CONFIGS[alert_id] = config
    await _state_set(alert_id, initial_state)

    background_tasks.add_task(_run_graph, alert_id, initial_state)

    return {"alert_id": alert_id, "status": "accepted"}


@app.get("/status/{alert_id}", response_model=AlertStatusResponse)
async def get_alert_status(alert_id: str) -> AlertStatusResponse:
    """Retrieve the current execution state for an alert."""
    state = await _state_get(alert_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")

    return AlertStatusResponse(
        alert_id=alert_id,
        status=state.get("status", "unknown"),
        current_node=state.get("current_node", ""),
        retry_count=state.get("retry_count", 0),
        error_log=list(state.get("error_log", [])),
        reasoning_log=list(state.get("reasoning_log", [])),
        metadata=dict(state.get("metadata", {})),
        token_usage=list(state.get("token_usage", [])),
        cost_estimate_usd=float(state.get("cost_estimate_usd", 0.0)),
    )


@app.post("/slack/interactive")
async def slack_interactive(interaction: SlackInteraction) -> dict[str, Any]:
    """Receive Slack approval/rejection and resume the paused graph."""
    alert_id = interaction.alert_id
    config = ALERT_CONFIGS.get(alert_id)

    if config is None:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")

    # Verify the graph is actually waiting at human_gate
    try:
        snapshot = await _graph.aget_state(config)
    except Exception:
        raise HTTPException(status_code=404, detail=f"No active graph state for '{alert_id}'")

    if snapshot is None or not snapshot.next:
        raise HTTPException(
            status_code=409,
            detail=f"Alert '{alert_id}' is not waiting for approval",
        )

    try:
        from langgraph.errors import GraphInterrupt
        from langgraph.types import Command

        result = await _graph.ainvoke(
            Command(resume={"approved": interaction.approved}),
            config=config,
        )
        await _state_set(alert_id, result)
        return {"alert_id": alert_id, "status": result.get("status", "unknown")}

    except GraphInterrupt:
        # Graph paused again (retry cycle)
        snapshot = await _graph.aget_state(config)
        if snapshot and snapshot.values:
            await _state_set(alert_id, snapshot.values)  # type: ignore[arg-type]
        return {"alert_id": alert_id, "status": "waiting_for_approval"}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
