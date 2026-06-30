"""FastAPI application for the Autonomous SRE webhook API.

Endpoints:
  POST /webhook            — Receive Alertmanager webhook, start graph execution
  GET  /status/{alert_id}  — Retrieve current graph state for an alert
  POST /slack/interactive  — Receive Slack approval/rejection to resume graph

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
import collections
import hmac
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response

from app.config import get_settings as _get_settings
_get_settings()  # export LANGCHAIN_* to os.environ before any LangChain import

from app.agents.graph import build_graph
from app.agents.state import SREState, create_initial_state
from app.middleware.slack_verify import verify_slack_signature
from app.models import AlertStatusResponse, AlertWebhook, SlackInteraction, SlackInteractionPayload

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-process initial-state cache
# Stores the submitted initial state so that /status/{alert_id} can return a
# result immediately (before the background graph run populates the checkpointer).
# Entries are removed once the graph run completes.
# ---------------------------------------------------------------------------

_INITIAL_STATES: dict[str, SREState] = {}

# ---------------------------------------------------------------------------
# In-process workload deduplication registry
# Maps workload_key (namespace/workload_name) → alert_id currently remediating
# that workload.  Prevents concurrent graph runs from mutating the same K8s
# resource simultaneously.
# ---------------------------------------------------------------------------

_ACTIVE_WORKLOADS: dict[str, str] = {}
_ALERT_TO_WORKLOAD: dict[str, str] = {}

_WORKLOAD_COOLDOWNS: dict[str, float] = {}
_COOLDOWN_SECONDS = 300  # 5 minutes

# Regex: strip ReplicaSet (7-12 hex chars) + pod hash (5 chars) from pod names
# so all pods of the same Deployment share a single dedup key.
#   e.g. "chaos-app-7c949f9b88-txqcz" → "chaos-app"
_POD_HASH_RE = re.compile(r"^(.+)-[a-z0-9]{7,12}-[a-z0-9]{5}$")


def _workload_key(labels: dict) -> str:
    """Derive a stable deduplication key from alert labels.

    Priority order for workload identification:
      deployment > statefulset > daemonset > pod (normalized)

    When falling back to pod name, ReplicaSet and pod hash suffixes are stripped
    so that all pods of the same Deployment map to a single key.

    Returns ``"{namespace}/{workload_name}"`` where both components fall back
    to ``"default"`` / ``"unknown"`` when the corresponding label is absent.
    """
    namespace = labels.get("namespace", "default") or "default"
    workload = (
        labels.get("deployment")
        or labels.get("statefulset")
        or labels.get("daemonset")
    )
    if not workload:
        pod = labels.get("pod", "unknown")
        match = _POD_HASH_RE.match(pod)
        workload = match.group(1) if match else pod
    return f"{namespace}/{workload}"


def _is_in_cooldown(wk: str) -> bool:
    """Return True if *wk* is within its post-resolution cooldown window."""
    expires = _WORKLOAD_COOLDOWNS.get(wk)
    if expires is None:
        return False
    if time.monotonic() < expires:
        return True
    _WORKLOAD_COOLDOWNS.pop(wk, None)
    return False


def _set_cooldown(wk: str) -> None:
    """Record a cooldown for *wk* starting now."""
    _WORKLOAD_COOLDOWNS[wk] = time.monotonic() + _COOLDOWN_SECONDS


# ---------------------------------------------------------------------------
# Webhook rate limiter — sliding window, configurable via WEBHOOK_RATE_LIMIT_PER_MINUTE
# ---------------------------------------------------------------------------

class _WebhookRateLimiter:
    def __init__(self, max_per_minute: int) -> None:
        self._max = max_per_minute
        self._timestamps: collections.deque[float] = collections.deque()
        self._lock = asyncio.Lock()

    async def check(self) -> bool:
        """Return True if the request is within the rate limit, False if it should be rejected."""
        if self._max <= 0:
            return True
        async with self._lock:
            now = time.monotonic()
            cutoff = now - 60.0
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()
            if len(self._timestamps) >= self._max:
                return False
            self._timestamps.append(now)
            return True


_rate_limiter: _WebhookRateLimiter | None = None

_SLACK_RESPONSE_URL_PREFIX = "https://hooks.slack.com/"


# ---------------------------------------------------------------------------
# Lifespan: initialise checkpointer + graph
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rate_limiter
    settings = _get_settings()
    _rate_limiter = _WebhookRateLimiter(settings.webhook_rate_limit_per_minute)
    if settings.postgres_dsn:
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import AsyncConnectionPool
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            async with AsyncConnectionPool(
                conninfo=settings.postgres_dsn,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": None,
                    "row_factory": dict_row,
                },
                min_size=1,
                max_size=5,
            ) as pool:
                saver = AsyncPostgresSaver(pool)
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

async def _run_graph(alert_id: str, initial_state: SREState, workload_key: str = "") -> None:
    """Run the graph from initial state. State is persisted via the checkpointer.

    Args:
        alert_id: Unique identifier for this alert run.
        initial_state: The initial SREState to feed into the graph.
        workload_key: Deduplication key (namespace/workload).  Cleared from
            _ACTIVE_WORKLOADS and _ALERT_TO_WORKLOAD only when the graph fully
            completes or fails — NOT when it pauses at human_gate (GraphInterrupt).
            The lock is released by _resume_and_notify once the Slack approval
            cycle finishes.
    """
    from langgraph.errors import GraphInterrupt

    config = _make_config(alert_id)
    try:
        await _graph().ainvoke(initial_state, config=config)
    except GraphInterrupt:
        # Graph paused at human_gate — state saved in checkpointer.
        # Keep workload in _ACTIVE_WORKLOADS: still waiting for human approval.
        # _resume_and_notify will release the lock after the Slack cycle ends.
        _INITIAL_STATES.pop(alert_id, None)
        logger.info("Graph paused at human_gate for alert %s", alert_id)
        return
    except Exception as exc:
        logger.error("Graph execution failed for alert %s: %s", alert_id, exc)

    # Reached only on successful completion or unrecoverable exception.
    _INITIAL_STATES.pop(alert_id, None)
    if workload_key:
        _ACTIVE_WORKLOADS.pop(workload_key, None)
        _ALERT_TO_WORKLOAD.pop(alert_id, None)
        _set_cooldown(workload_key)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/webhook", status_code=202)
async def receive_webhook(
    alert: AlertWebhook,
    background_tasks: BackgroundTasks,
    request: Request,
) -> dict[str, Any]:
    """Accept a Prometheus/Alertmanager webhook and start remediation graph.

    Handles two formats:
    - Alertmanager v4 envelope (``alerts`` array present): spawns one graph
      run per individual alert; returns ``alert_ids`` list plus ``alert_id``
      (first entry) for backward compatibility.
    - Legacy flat format (tests / direct callers): single run, returns
      ``alert_id``.
    """
    settings = _get_settings()

    # Bearer token authentication (when WEBHOOK_AUTH_TOKEN is configured)
    if settings.webhook_auth_token:
        auth_header = request.headers.get("Authorization", "")
        provided = auth_header.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(provided, settings.webhook_auth_token):
            raise HTTPException(status_code=401, detail="Unauthorized")

    # Sliding-window rate limit
    limiter = _rate_limiter or _WebhookRateLimiter(settings.webhook_rate_limit_per_minute)
    if not await limiter.check():
        raise HTTPException(status_code=429, detail="Too many requests")

    if alert.alerts:
        # Alertmanager v4 envelope — process each alert individually.
        # Duplicate workloads are silently suppressed (200) rather than rejected (409),
        # so Alertmanager does not interpret the response as a delivery failure.
        alert_ids: list[str] = []
        suppressed: list[dict] = []

        for am_alert in alert.alerts:
            # Skip resolved alerts — Alertmanager is telling us the alert
            # stopped firing; no remediation needed.
            if am_alert.get("status") == "resolved":
                continue
            labels = am_alert.get("labels", {})
            # Merge commonLabels so that labels promoted out of individual alerts
            # (e.g. namespace when all alerts in the group share one) are preserved.
            # Individual alert labels take precedence over commonLabels.
            merged_labels = {**alert.commonLabels, **labels}
            alertname = (
                merged_labels.get("alertname")
                or alert.groupLabels.get("alertname")
                or "unknown"
            )
            payload: dict[str, Any] = {
                "alertname": alertname,
                "status": am_alert.get("status", alert.status),
                "labels": merged_labels,
                "annotations": am_alert.get("annotations", {}),
                "startsAt": am_alert.get("startsAt", ""),
                "endsAt": am_alert.get("endsAt", ""),
                "generatorURL": am_alert.get("generatorURL", ""),
            }
            initial_state = create_initial_state(payload)
            alert_id = initial_state["alert_id"]
            workload_key = _workload_key(merged_labels)

            existing_alert_id = _ACTIVE_WORKLOADS.get(workload_key)
            if existing_alert_id is not None:
                suppressed.append({"workload_key": workload_key, "existing_alert_id": existing_alert_id})
                continue

            if _is_in_cooldown(workload_key):
                suppressed.append({"workload_key": workload_key, "existing_alert_id": "(cooldown)"})
                continue

            _ACTIVE_WORKLOADS[workload_key] = alert_id
            _ALERT_TO_WORKLOAD[alert_id] = workload_key
            _INITIAL_STATES[alert_id] = initial_state
            background_tasks.add_task(_run_graph, alert_id, initial_state, workload_key)
            alert_ids.append(alert_id)

        if not alert_ids:
            if not suppressed:
                # Every alert in the envelope was resolved — nothing to do
                return {"status": "ignored", "detail": "All alerts are resolved"}
            first_suppressed = suppressed[0]
            return Response(
                content=json.dumps({
                    "status": "suppressed",
                    "suppressed_count": len(suppressed),
                    "detail": (
                        f"Workload '{first_suppressed['workload_key']}' is already being remediated "
                        f"(alert_id={first_suppressed['existing_alert_id']})"
                    ),
                }),
                status_code=200,
                media_type="application/json",
            )

        first_id = alert_ids[0] if alert_ids else ""
        return {
            "alert_id": first_id,
            "alert_ids": alert_ids,
            "status": "accepted",
            "suppressed_count": len(suppressed),
        }

    # Legacy flat format
    if alert.status == "resolved":
        return {"status": "ignored", "detail": "Resolved alerts do not require remediation"}
    if not alert.alertname:
        raise HTTPException(status_code=422, detail="alertname is required")
    payload = alert.model_dump(exclude={"alerts", "groupLabels", "commonLabels"})
    workload_key = _workload_key(payload.get("labels") or {})

    existing_alert_id = _ACTIVE_WORKLOADS.get(workload_key)
    if existing_alert_id is not None:
        return Response(
            content=json.dumps({
                "status": "suppressed",
                "alert_id": existing_alert_id,
                "detail": (
                    f"Workload '{workload_key}' is already being remediated "
                    f"(alert_id={existing_alert_id})"
                ),
            }),
            status_code=200,
            media_type="application/json",
        )

    if _is_in_cooldown(workload_key):
        return Response(
            content=json.dumps({
                "status": "suppressed",
                "detail": f"Workload '{workload_key}' is in post-resolution cooldown",
            }),
            status_code=200,
            media_type="application/json",
        )

    initial_state = create_initial_state(payload)
    alert_id = initial_state["alert_id"]
    _ACTIVE_WORKLOADS[workload_key] = alert_id
    _ALERT_TO_WORKLOAD[alert_id] = workload_key
    _INITIAL_STATES[alert_id] = initial_state
    background_tasks.add_task(_run_graph, alert_id, initial_state, workload_key)
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
async def slack_interactive(request: Request, background_tasks: BackgroundTasks) -> Any:
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
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid Slack interactive payload")

        action = interaction.actions[0] if interaction.actions else None
        if not action:
            raise HTTPException(status_code=400, detail="No actions in payload")

        # Button value encodes the alert_id; action_id encodes approve/reject
        alert_id = action.value
        approved = action.action_id.startswith("approve_")

        # Acknowledge Slack immediately (must respond within 3s).
        # The in-progress and final updates are handled by the background task
        # via chat.update (bot token) so Slack sees them reliably.
        response_url: str | None = interaction.response_url

        background_tasks.add_task(_resume_and_notify, alert_id, approved, response_url)
        return Response(status_code=200)
    else:
        # Legacy JSON body (tests + direct API)
        try:
            body = await request.json()
            interaction_legacy = SlackInteraction.model_validate(body)
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid interaction payload")

        alert_id = interaction_legacy.alert_id
        approved = interaction_legacy.approved

        result = await _resume_graph(alert_id, approved)
        result_status = result["status"]
        if result_status not in ("waiting_for_approval",):
            try:
                from app.utils.slack_blocks import build_resolution_message
                from app.utils.slack_client import send_slack_message
                await send_slack_message(
                    build_resolution_message(alert_id, result_status, result.get("action_result", ""))
                )
            except Exception as exc:
                logger.warning("Slack notification failed for alert %s: %s", alert_id, exc)
        return {"alert_id": alert_id, "status": result_status}


async def _resume_and_notify(alert_id: str, approved: bool, response_url: str | None) -> None:
    """Background task: resume the graph then update Slack based on outcome.

    Flow:
      1. Capture ts/channel from current graph state (before resuming).
      2. Immediately update the Slack message to in-progress so the user sees
         feedback right away without waiting for the graph to finish.
      3. Resume the graph.
      4. Replace the in-progress message with the final outcome.
    """
    from app.utils.slack_blocks import build_working_message
    from app.utils.slack_client import update_slack_message

    # Step 1: capture ts/channel before the graph run (result's except paths don't have them)
    config = _make_config(alert_id)
    ts: str | None = None
    channel: str | None = None
    try:
        snap = await _graph().aget_state(config)
        if snap and snap.values:
            ts = snap.values.get("slack_message_ts")
            channel = snap.values.get("slack_channel")
    except Exception as exc:
        logger.warning("Could not fetch graph state for in-progress update (alert %s): %s", alert_id, exc)

    # Step 2: show in-progress immediately
    working_msg = build_working_message(alert_id, approved)
    if ts and channel:
        await update_slack_message(ts=ts, channel=channel, blocks=working_msg)
    elif response_url and response_url.startswith(_SLACK_RESPONSE_URL_PREFIX):
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(response_url, json=working_msg)
        except Exception as exc:
            logger.warning("Failed to post in-progress message for alert %s: %s", alert_id, exc)

    # Step 3: resume the graph
    try:
        result = await _resume_graph(alert_id, approved)
    except HTTPException as exc:
        logger.error("Graph resume failed for alert %s: %s", alert_id, exc.detail)
        result = {"status": "error", "action_result": "", "human_approved": approved,
                  "alertname": "Unknown Alert", "error_summary": "",
                  "slack_message_ts_list": [], "slack_channel": None}
    except Exception as exc:
        logger.error("Unexpected error resuming alert %s: %s", alert_id, exc)
        result = {"status": "error", "action_result": "", "human_approved": approved,
                  "alertname": "Unknown Alert", "error_summary": "",
                  "slack_message_ts_list": [], "slack_channel": None}

    status = result["status"]

    # Release the workload lock when the graph is no longer paused at human_gate.
    # "waiting_for_approval" means the graph interrupted again (retry cycle) — keep lock.
    # All other statuses (resolved, escalated, error) mean the run has ended or is stuck.
    if status != "waiting_for_approval":
        workload_key = _ALERT_TO_WORKLOAD.pop(alert_id, None)
        if workload_key:
            _ACTIVE_WORKLOADS.pop(workload_key, None)
            _set_cooldown(workload_key)

    # ts/channel from graph result (may be updated after resume); fall back to captured values
    result_ts = result.get("slack_message_ts") or ts
    result_channel = result.get("slack_channel") or channel

    # Step 4: replace the in-progress message with the final outcome
    if status == "resolved":
        await _post_slack_success(
            alert_id=alert_id,
            alertname=result.get("alertname", "Unknown Alert"),
            action_result=result.get("action_result", ""),
            ts=result_ts,
            channel=result_channel,
        )
    elif status == "escalated" and not result.get("human_approved", False):
        # Human rejection: update the message in-place via bot token
        from app.utils.slack_blocks import build_rejected_message
        rejected_blocks = build_rejected_message(
            alert_id=alert_id,
            alertname=result.get("alertname", "Unknown Alert"),
            error_summary=result.get("error_summary", ""),
        )
        if result_ts and result_channel:
            await update_slack_message(ts=result_ts, channel=result_channel, blocks=rejected_blocks)
        elif response_url:
            await _post_slack_rejection(
                response_url=response_url,
                alert_id=alert_id,
                alertname=result.get("alertname", "Unknown Alert"),
                error_summary=result.get("error_summary", ""),
            )
    elif status == "waiting_for_approval":
        # Retry cycle: notify_slack_node already posted/updated the approval message
        # with prior-attempt context before human_gate re-interrupted.
        # Do NOT overwrite it — the user needs those buttons to respond.
        logger.info("Graph re-interrupted at human_gate for alert %s (retry cycle)", alert_id)
    else:
        # Escalated-after-approval, RBAC blocked, error — show what was attempted
        from app.utils.slack_blocks import build_failed_message
        failed_blocks = build_failed_message(
            alert_id=alert_id,
            alertname=result.get("alertname", "Unknown Alert"),
            error_summary=result.get("error_summary", ""),
            action_result=result.get("action_result", ""),
        )
        if result_ts and result_channel:
            await update_slack_message(ts=result_ts, channel=result_channel, blocks=failed_blocks)
        elif response_url:
            await _post_slack_response_url(
                response_url, alert_id, status, result.get("action_result", "")
            )


async def _resume_graph(alert_id: str, approved: bool) -> dict:
    """Resume the paused graph with the given approval decision.

    Returns a dict with keys: status, action_result, alertname, error_summary,
    slack_message_ts_list, slack_channel, human_approved.
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
        return {
            "status": result.get("status", "unknown"),
            "action_result": result.get("action_result", ""),
            "alertname": result.get("alert_payload", {}).get("alertname", "Unknown Alert"),
            "error_summary": result.get("error_summary", ""),
            "slack_message_ts": result.get("slack_message_ts"),
            "slack_message_ts_list": list(result.get("slack_message_ts_list", [])),
            "slack_channel": result.get("slack_channel"),
            "human_approved": result.get("human_approved", False),
        }

    except GraphInterrupt:
        # Graph paused again (retry cycle) — fetch state to get context
        try:
            snap = await _graph().aget_state(config)
            vals = snap.values if snap else {}
        except Exception:
            vals = {}
        return {
            "status": "waiting_for_approval",
            "action_result": "",
            "alertname": vals.get("alert_payload", {}).get("alertname", "Unknown Alert"),
            "error_summary": vals.get("error_summary", ""),
            "slack_message_ts": vals.get("slack_message_ts"),
            "slack_message_ts_list": list(vals.get("slack_message_ts_list", [])),
            "slack_channel": vals.get("slack_channel"),
            "human_approved": approved,
        }

    except Exception as exc:
        logger.error("Graph resume failed for alert %s: %s", alert_id, exc)
        raise HTTPException(status_code=500, detail="Internal server error")


async def _post_slack_rejection(
    response_url: str,
    alert_id: str,
    alertname: str,
    error_summary: str,
) -> None:
    """Replace the approval message with a rejected notice via response_url."""
    if not response_url.startswith(_SLACK_RESPONSE_URL_PREFIX):
        logger.warning("Rejecting response_url with unexpected domain for alert %s", alert_id)
        return
    try:
        import httpx
        from app.utils.slack_blocks import build_rejected_message
        message = build_rejected_message(alert_id, alertname, error_summary)
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(response_url, json=message)
    except Exception as exc:
        logger.warning("Failed to post Slack rejection update: %s", exc)


async def _post_slack_success(
    alert_id: str,
    alertname: str,
    action_result: str,
    ts: str | None,
    channel: str | None,
) -> None:
    """Update the approval message in-place with resolution, or post remediation message if update fails."""
    from app.utils.slack_blocks import build_resolution_message
    from app.utils.slack_client import update_slack_message, send_slack_message

    resolution_blocks = build_resolution_message(alert_id, "resolved", action_result)

    # Try to update the existing message (the approval/working message)
    if ts and channel:
        try:
            await update_slack_message(ts=ts, channel=channel, blocks=resolution_blocks)
            logger.info("Updated Slack message to resolved for alert %s", alert_id)
            return
        except Exception as exc:
            logger.warning(
                "Failed to update Slack message ts=%s for alert %s: %s (will post remediation message)",
                ts, alert_id, exc
            )

    # Fallback: post a new remediation message if update failed or ts/channel unavailable
    try:
        from app.utils.slack_blocks import build_remediation_message
        remediation = build_remediation_message(
            alert_id=alert_id,
            alertname=alertname,
            action_result=action_result,
            error_msg="Failed to update message in Slack",
        )
        await send_slack_message(remediation)
        logger.info(
            "Posted remediation message for alert %s (message update failed, but action succeeded)",
            alert_id,
        )
    except Exception as exc:
        logger.error(
            "Failed to post remediation message for alert %s: %s (action succeeded but couldn't notify)",
            alert_id, exc
        )


async def _post_slack_response_url(
    response_url: str, alert_id: str, status: str, action_result: str = ""
) -> None:
    """Post the final Block Kit result back to Slack via the response_url."""
    if not response_url.startswith(_SLACK_RESPONSE_URL_PREFIX):
        logger.warning(
            "Rejecting response_url with unexpected domain for alert %s: %s",
            alert_id,
            response_url,
        )
        return
    try:
        import httpx
        from app.utils.slack_blocks import build_resolution_message
        message = build_resolution_message(alert_id, status, action_result)
        message["replace_original"] = True
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(response_url, json=message)
    except Exception as exc:
        logger.warning("Failed to post Slack response_url update: %s", exc)

