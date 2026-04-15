"""FastAPI application for the Autonomous SRE webhook API.

Endpoints:
  POST /webhook            — Receive Alertmanager webhook, start graph execution
  GET  /status/{alert_id}  — Retrieve current graph state for an alert
  POST /slack/interactive  — Receive Slack approval/rejection to resume graph
  GET  /cost-report        — Aggregate cost breakdown across all alerts

State persistence:
  - When POSTGRES_DSN is set: AsyncPostgresSaver (Supabase / Postgres) provides
    durable, resumable checkpointing across restarts and long pauses.
  - Falls back to MemorySaver when POSTGRES_DSN is absent (local dev / tests).

Lifespan:
  The FastAPI lifespan context manager initialises the async checkpointer and
  compiles the graph exactly once at startup, keeping the connection pool open
  for the lifetime of the process.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response

from app.config import get_settings as _get_settings
_get_settings()  # export LANGCHAIN_* to os.environ before any LangChain import

from app.agents.graph import build_graph
from app.agents.state import SREState, create_initial_state
from app.middleware.slack_verify import verify_slack_signature
from app.models import AlertStatusResponse, AlertWebhook, SlackInteraction, SlackInteractionPayload

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-process initial-state cache
# Stores the submitted initial state so that /status/{alert_id} can return a
# result immediately (before the background graph run populates the checkpointer).
# Entries are removed once the graph run completes.
# ---------------------------------------------------------------------------

_INITIAL_STATES: dict[str, SREState] = {}


# ---------------------------------------------------------------------------
# Lifespan: initialise checkpointer + graph
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = _get_settings()
    if settings.postgres_dsn:
        try:
            import psycopg
            from psycopg.rows import dict_row
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            # prepare_threshold=None disables psycopg3 prepared statements entirely.
            # NOTE: prepare_threshold=0 means "prepare on first use" (NOT disabled) —
            # that causes "__pg3_N does not exist" errors with PgBouncer transaction
            # mode (Supabase port 6543) because prepared statements are connection-scoped
            # and PgBouncer can route each transaction to a different backend connection.
            async with await psycopg.AsyncConnection.connect(
                settings.postgres_dsn,
                autocommit=True,
                prepare_threshold=None,
                row_factory=dict_row,
            ) as conn:
                saver = AsyncPostgresSaver(conn)
                await saver.setup()
                app.state.graph = build_graph(saver)
                logger.info("Graph initialised with AsyncPostgresSaver (Supabase)")
                yield
        except Exception as exc:
            logger.warning(
                "AsyncPostgresSaver init failed (%s) — falling back to MemorySaver", exc
            )
            app.state.graph = build_graph()
            yield
    else:
        app.state.graph = build_graph()
        logger.info("Graph initialised with MemorySaver (no POSTGRES_DSN)")
        yield


app = FastAPI(title="Autonomous SRE", version="0.4.0", lifespan=lifespan)


def _graph():
    """Return the compiled graph from app state, or build a fallback for tests."""
    try:
        return app.state.graph
    except AttributeError:
        # Fallback for test contexts where lifespan is not running
        if not hasattr(app, "_fallback_graph"):
            app._fallback_graph = build_graph()
        return app._fallback_graph


def _make_config(alert_id: str) -> dict:
    """Return the LangGraph config for a given alert_id."""
    return {"configurable": {"thread_id": alert_id}}


# ---------------------------------------------------------------------------
# Background task: run graph until interrupt or completion
# ---------------------------------------------------------------------------

async def _run_graph(alert_id: str, initial_state: SREState) -> None:
    """Run the graph from initial state. State is persisted via the checkpointer."""
    config = _make_config(alert_id)
    try:
        from langgraph.errors import GraphInterrupt
        await _graph().ainvoke(initial_state, config=config)
    except GraphInterrupt:
        # Graph paused at human_gate — state saved in checkpointer
        logger.info("Graph paused at human_gate for alert %s", alert_id)
    except Exception as exc:
        logger.error("Graph execution failed for alert %s: %s", alert_id, exc)
    finally:
        _INITIAL_STATES.pop(alert_id, None)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/webhook", status_code=202)
async def receive_webhook(
    alert: AlertWebhook,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Accept a Prometheus/Alertmanager webhook and start remediation graph.

    Handles two formats:
    - Alertmanager v4 envelope (``alerts`` array present): spawns one graph
      run per individual alert; returns ``alert_ids`` list plus ``alert_id``
      (first entry) for backward compatibility.
    - Legacy flat format (tests / direct callers): single run, returns
      ``alert_id``.
    """
    if alert.alerts:
        # Alertmanager v4 envelope — process each alert individually
        alert_ids: list[str] = []
        for am_alert in alert.alerts:
            labels = am_alert.get("labels", {})
            alertname = (
                labels.get("alertname")
                or alert.groupLabels.get("alertname")
                or alert.commonLabels.get("alertname")
                or "unknown"
            )
            payload: dict[str, Any] = {
                "alertname": alertname,
                "status": am_alert.get("status", alert.status),
                "labels": labels,
                "annotations": am_alert.get("annotations", {}),
                "startsAt": am_alert.get("startsAt", ""),
                "endsAt": am_alert.get("endsAt", ""),
                "generatorURL": am_alert.get("generatorURL", ""),
            }
            initial_state = create_initial_state(payload)
            alert_id = initial_state["alert_id"]
            _INITIAL_STATES[alert_id] = initial_state
            background_tasks.add_task(_run_graph, alert_id, initial_state)
            alert_ids.append(alert_id)
        first_id = alert_ids[0] if alert_ids else ""
        return {"alert_id": first_id, "alert_ids": alert_ids, "status": "accepted"}

    # Legacy flat format
    if not alert.alertname:
        raise HTTPException(status_code=422, detail="alertname is required")
    payload = alert.model_dump(exclude={"alerts", "groupLabels", "commonLabels"})
    initial_state = create_initial_state(payload)
    alert_id = initial_state["alert_id"]
    _INITIAL_STATES[alert_id] = initial_state
    background_tasks.add_task(_run_graph, alert_id, initial_state)
    return {"alert_id": alert_id, "status": "accepted"}


@app.get("/status/{alert_id}", response_model=AlertStatusResponse)
async def get_alert_status(alert_id: str) -> AlertStatusResponse:
    """Retrieve the current execution state for an alert."""
    config = _make_config(alert_id)

    # Prefer checkpointer state (most recent after graph runs)
    try:
        snapshot = await _graph().aget_state(config)
        if snapshot and snapshot.values:
            state: SREState = snapshot.values  # type: ignore[assignment]
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
    except Exception as exc:
        logger.debug("aget_state failed for %s: %s", alert_id, exc)

    # Fall back to initial state cache (alert submitted but graph not yet run)
    initial = _INITIAL_STATES.get(alert_id)
    if initial:
        return AlertStatusResponse(
            alert_id=alert_id,
            status=initial.get("status", "accepted"),
            current_node=initial.get("current_node", ""),
            retry_count=initial.get("retry_count", 0),
            error_log=list(initial.get("error_log", [])),
            reasoning_log=list(initial.get("reasoning_log", [])),
            metadata=dict(initial.get("metadata", {})),
            token_usage=list(initial.get("token_usage", [])),
            cost_estimate_usd=float(initial.get("cost_estimate_usd", 0.0)),
        )

    raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")


@app.post("/slack/interactive", dependencies=[Depends(verify_slack_signature)])
async def slack_interactive(request: Request) -> Any:
    """Receive Slack approval/rejection and resume the paused graph.

    Accepts two formats:
      1. Real Slack interactive payload: application/x-www-form-urlencoded
         with a ``payload`` field containing JSON (Block Kit button click).
      2. Legacy JSON body: {"alert_id": "...", "approved": true/false}
         (used by tests and direct API calls).

    HMAC verification (when SLACK_SIGNING_SECRET is configured) is handled by
    the slack_verify dependency on this route.
    """
    settings = _get_settings()
    content_type = request.headers.get("content-type", "")

    alert_id: str
    approved: bool

    if "application/x-www-form-urlencoded" in content_type:
        # Real Slack interactive payload
        form = await request.form()
        raw_payload = form.get("payload", "")
        if not raw_payload:
            raise HTTPException(status_code=400, detail="Missing payload field")
        try:
            slack_data = json.loads(raw_payload)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid payload JSON")

        try:
            interaction = SlackInteractionPayload.model_validate(slack_data)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        action = interaction.actions[0] if interaction.actions else None
        if not action:
            raise HTTPException(status_code=400, detail="No actions in payload")

        # Button value encodes the alert_id; action_id encodes approve/reject
        alert_id = action.value
        approved = action.action_id.startswith("approve_")

        # Acknowledge Slack immediately (must respond within 3s)
        # The actual result will be posted via response_url asynchronously
        response_url: str | None = interaction.response_url

        async def _resume_and_notify():
            result_status = await _resume_graph(alert_id, approved)
            if response_url:
                await _post_slack_response_url(response_url, alert_id, result_status)

        asyncio.create_task(_resume_and_notify())
        return Response(
            content=json.dumps({"text": f"Processing {'approval' if approved else 'rejection'}..."}),
            media_type="application/json",
        )
    else:
        # Legacy JSON body (tests + direct API)
        try:
            body = await request.json()
            interaction_legacy = SlackInteraction.model_validate(body)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        alert_id = interaction_legacy.alert_id
        approved = interaction_legacy.approved

        result_status = await _resume_graph(alert_id, approved)
        return {"alert_id": alert_id, "status": result_status}


async def _resume_graph(alert_id: str, approved: bool) -> str:
    """Resume the paused graph with the given approval decision.

    Returns the resulting status string.
    Raises HTTPException on 404/409/500.
    """
    config = _make_config(alert_id)

    try:
        snapshot = await _graph().aget_state(config)
    except Exception:
        raise HTTPException(status_code=404, detail=f"No active graph state for '{alert_id}'")

    # Empty values + no next = thread never ran (alert not found)
    if snapshot is None or (not snapshot.values and not snapshot.next):
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")

    if not snapshot.next:
        raise HTTPException(
            status_code=409,
            detail=f"Alert '{alert_id}' is not waiting for approval",
        )

    try:
        from langgraph.errors import GraphInterrupt
        from langgraph.types import Command

        result = await _graph().ainvoke(
            Command(resume={"approved": approved}),
            config=config,
        )
        return result.get("status", "unknown")

    except GraphInterrupt:
        # Graph paused again (retry cycle)
        return "waiting_for_approval"

    except Exception as exc:
        logger.error("Graph resume failed for alert %s: %s", alert_id, exc)
        raise HTTPException(status_code=500, detail="Internal server error")


_SLACK_RESPONSE_URL_PREFIX = "https://hooks.slack.com/"


async def _post_slack_response_url(response_url: str, alert_id: str, status: str) -> None:
    """Post the final result back to Slack via the response_url."""
    if not response_url.startswith(_SLACK_RESPONSE_URL_PREFIX):
        logger.warning(
            "Rejecting response_url with unexpected domain for alert %s: %s",
            alert_id,
            response_url,
        )
        return
    try:
        import httpx
        message = {
            "replace_original": True,
            "text": f"Alert `{alert_id}` — status: *{status}*",
        }
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(response_url, json=message)
    except Exception as exc:
        logger.warning("Failed to post Slack response_url update: %s", exc)


# ---------------------------------------------------------------------------
# Cost report endpoint
# ---------------------------------------------------------------------------

@app.get("/cost-report")
async def get_cost_report() -> dict:
    """Return aggregate cost breakdown across all processed alerts."""
    from app.utils.cost_store import get_cost_report
    return await get_cost_report()
