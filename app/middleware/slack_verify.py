"""Slack request signature verification.

Implements the Slack signing secret HMAC-SHA256 verification protocol:
https://api.slack.com/authentication/verifying-requests-from-slack

Use as a FastAPI dependency on the /slack/interactive route:

    from app.middleware.slack_verify import verify_slack_signature

    @app.post("/slack/interactive", dependencies=[Depends(verify_slack_signature)])
    async def slack_interactive(...): ...
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import time

from fastapi import Depends, HTTPException, Request

from app.config import get_settings

logger = logging.getLogger(__name__)

_SLACK_SIGNATURE_VERSION = "v0"
_MAX_TIMESTAMP_AGE_SECONDS = 300  # 5 minutes


async def verify_slack_signature(request: Request) -> None:
    """FastAPI dependency that verifies Slack's HMAC request signature.

    Rejects with 403 if:
    - SLACK_SIGNING_SECRET is configured and the signature is invalid
    - The request timestamp is more than 5 minutes old (replay attack prevention)

    When SLACK_SIGNING_SECRET is not configured, verification is skipped
    (allows local development without Slack credentials).
    """
    settings = get_settings()
    signing_secret = settings.slack_signing_secret
    if not signing_secret:
        return  # Verification disabled in local/test environments

    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    slack_signature = request.headers.get("X-Slack-Signature", "")

    if not timestamp or not slack_signature:
        raise HTTPException(status_code=403, detail="Missing Slack signature headers")

    try:
        ts = int(timestamp)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid timestamp header")

    if abs(time.time() - ts) > _MAX_TIMESTAMP_AGE_SECONDS:
        raise HTTPException(status_code=403, detail="Request timestamp too old")

    body = await request.body()
    base_string = f"{_SLACK_SIGNATURE_VERSION}:{timestamp}:{body.decode('utf-8')}"

    expected = (
        _SLACK_SIGNATURE_VERSION
        + "="
        + hmac.new(
            signing_secret.encode("utf-8"),
            base_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
    )

    if not hmac.compare_digest(expected, slack_signature):
        logger.warning("Slack signature verification failed")
        raise HTTPException(status_code=403, detail="Invalid Slack signature")
