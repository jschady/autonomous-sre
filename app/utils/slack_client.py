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
_SLACK_UPDATE_MESSAGE_URL = "https://slack.com/api/chat.update"
_SLACK_DELETE_MESSAGE_URL = "https://slack.com/api/chat.delete"


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


async def delete_slack_message(
    ts: str,
    channel: str,
) -> dict[str, Any] | None:
    """Delete an existing Slack message via chat.delete.

    Args:
        ts:      Timestamp of the message to delete.
        channel: Channel ID where the message was posted.

    Returns:
        The Slack API response dict, or None on failure / Slack disabled.
    """
    settings = get_settings()
    token = settings.slack_bot_token

    if not token:
        logger.debug("Slack delete disabled (no SLACK_BOT_TOKEN)")
        return None

    payload = {"channel": channel, "ts": ts}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                _SLACK_DELETE_MESSAGE_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data: dict = response.json()
            if not data.get("ok"):
                logger.warning("Slack chat.delete error: %s", data.get("error", "unknown"))
            return data
    except httpx.HTTPStatusError as exc:
        logger.warning("Slack delete HTTP error %s: %s", exc.response.status_code, exc)
        return None
    except Exception as exc:
        logger.warning("Slack delete failed: %s", exc)
        return None


async def update_slack_message(
    ts: str,
    channel: str,
    blocks: dict[str, Any],
) -> dict[str, Any] | None:
    """Update an existing Slack message in-place via chat.update.

    Args:
        ts:      Timestamp of the message to update (from original post response).
        channel: Channel ID where the original message was posted.
        blocks:  Replacement Block Kit payload dict.

    Returns:
        The Slack API response dict, or None on failure / Slack disabled.
    """
    settings = get_settings()
    token = settings.slack_bot_token

    if not token:
        logger.debug("Slack update disabled (no SLACK_BOT_TOKEN)")
        return None

    payload = {
        "channel": channel,
        "ts": ts,
        "text": blocks.get("text", ""),
        "blocks": blocks.get("blocks", []),
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                _SLACK_UPDATE_MESSAGE_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data: dict = response.json()
            if not data.get("ok"):
                logger.warning("Slack chat.update error: %s", data.get("error", "unknown"))
            return data
    except httpx.HTTPStatusError as exc:
        logger.warning("Slack update HTTP error %s: %s", exc.response.status_code, exc)
        return None
    except Exception as exc:
        logger.warning("Slack update failed: %s", exc)
        return None
