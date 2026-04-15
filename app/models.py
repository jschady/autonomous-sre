"""Pydantic models for the Autonomous SRE API."""
from __future__ import annotations

from pydantic import BaseModel


class AlertWebhook(BaseModel):
    """Accepts either the legacy flat format OR an Alertmanager v4 webhook envelope.

    Flat (legacy / tests):
        {"alertname": "...", "status": "firing", "labels": {...}, ...}

    Alertmanager v4 envelope:
        {"receiver": "...", "status": "firing", "alerts": [{...}],
         "groupLabels": {"alertname": "..."}, ...}
    """
    # Flat format fields
    alertname: str = ""
    status: str = "firing"
    labels: dict[str, str] = {}
    annotations: dict[str, str] = {}
    generatorURL: str = ""
    startsAt: str = ""
    endsAt: str = ""
    # Alertmanager v4 envelope fields
    alerts: list[dict] = []
    groupLabels: dict[str, str] = {}
    commonLabels: dict[str, str] = {}

    model_config = {"extra": "ignore"}


class SlackInteraction(BaseModel):
    """Legacy JSON body format used by tests and direct API callers."""
    alert_id: str
    approved: bool


# ---------------------------------------------------------------------------
# Real Slack interactive payload models (Block Kit button clicks)
# ---------------------------------------------------------------------------

class SlackAction(BaseModel):
    """A single action from a Slack Block Kit interactive message."""
    action_id: str = ""
    value: str = ""
    type: str = ""

    model_config = {"extra": "ignore"}


class SlackUser(BaseModel):
    id: str = ""
    username: str = ""

    model_config = {"extra": "ignore"}


class SlackInteractionPayload(BaseModel):
    """Top-level payload sent by Slack for interactive component actions."""
    type: str = ""
    actions: list[SlackAction] = []
    user: SlackUser = SlackUser()
    response_url: str | None = None
    trigger_id: str = ""

    model_config = {"extra": "ignore"}


class AlertStatusResponse(BaseModel):
    alert_id: str
    status: str
    current_node: str
    retry_count: int
    error_log: list[str]
    reasoning_log: list[str]
    metadata: dict
    token_usage: list[dict] = []
    cost_estimate_usd: float = 0.0
