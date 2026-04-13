"""Pydantic models for the Autonomous SRE API."""
from __future__ import annotations

from pydantic import BaseModel


class AlertWebhook(BaseModel):
    alertname: str
    status: str = "firing"
    labels: dict[str, str] = {}
    annotations: dict[str, str] = {}
    generatorURL: str = ""
    startsAt: str = ""
    endsAt: str = ""


class SlackInteraction(BaseModel):
    alert_id: str
    approved: bool


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
