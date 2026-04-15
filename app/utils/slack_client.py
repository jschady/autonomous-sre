"""Async Slack API client for posting Block Kit messages.

Uses httpx (already a dependency) rather than the slack_sdk to keep
the production Docker image lean.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

_SLACK_POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"


async def send_slack_message(
    blocks: dict[str, Any],
    channel_id: str | None = None,
    text_fallback: str = "SRE Bot notification",
) -> dict[str, Any] | None:
    """Post a Block Kit message to Slack.

    Args:
        blocks:        The full Block Kit payload dict (from slack_blocks module).
        channel_id:    Override the channel from settings.
        text_fallback: Plain-text fallback for notifications.

    Returns:
        The Slack API response dict, or None on failure.
    """
    settings = get_settings()
    token = settings.slack_bot_token
    channel = channel_id or settings.slack_channel_id

    if not token or not channel:
        logger.debug("Slack notifications disabled (no SLACK_BOT_TOKEN or SLACK_CHANNEL_ID)")
        return None

    payload = {
        "channel": channel,
        "text": blocks.get("text", text_fallback),
        "blocks": blocks.get("blocks", []),
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                _SLACK_POST_MESSAGE_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data: dict = response.json()
            if not data.get("ok"):
                logger.warning("Slack API error: %s", data.get("error", "unknown"))
            return data
    except httpx.HTTPStatusError as exc:
        logger.warning("Slack HTTP error %s: %s", exc.response.status_code, exc)
        return None
    except Exception as exc:
        logger.warning("Slack send failed: %s", exc)
        return None
