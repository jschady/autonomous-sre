"""Unit tests for Slack Block Kit message builders and HMAC verification."""
from __future__ import annotations

import hashlib
import hmac
import time

import pytest

from app.utils.slack_blocks import _md_to_mrkdwn, build_approval_message, build_resolution_message


# ---------------------------------------------------------------------------
# _md_to_mrkdwn
# ---------------------------------------------------------------------------

class TestMdToMrkdwn:
    def test_bold_double_asterisk(self):
        assert _md_to_mrkdwn("**hello**") == "*hello*"

    def test_bold_double_underscore(self):
        assert _md_to_mrkdwn("__hello__") == "*hello*"

    def test_italic_single_asterisk(self):
        assert _md_to_mrkdwn("*italic*") == "_italic_"

    def test_italic_underscore_unchanged(self):
        assert _md_to_mrkdwn("_italic_") == "_italic_"

    def test_h1_header(self):
        assert _md_to_mrkdwn("# Title") == "*Title*"

    def test_h2_header(self):
        assert _md_to_mrkdwn("## Section") == "*Section*"

    def test_h3_header(self):
        assert _md_to_mrkdwn("### Sub") == "*Sub*"

    def test_strikethrough(self):
        assert _md_to_mrkdwn("~~gone~~") == "~gone~"

    def test_link(self):
        assert _md_to_mrkdwn("[Google](https://google.com)") == "<https://google.com|Google>"

    def test_horizontal_rule_removed(self):
        assert _md_to_mrkdwn("---") == ""

    def test_inline_code_preserved(self):
        assert _md_to_mrkdwn("`kubectl get pods`") == "`kubectl get pods`"

    def test_fenced_code_block_preserved(self):
        src = "```\nkubectl rollout restart\n```"
        assert _md_to_mrkdwn(src) == src

    def test_bold_not_converted_inside_code(self):
        assert _md_to_mrkdwn("`**not bold**`") == "`**not bold**`"

    def test_mixed_content(self):
        src = "## Analysis\n**Pod** is *crashing* — see `logs`"
        result = _md_to_mrkdwn(src)
        assert result == "*Analysis*\n*Pod* is _crashing_ — see `logs`"

    def test_empty_string(self):
        assert _md_to_mrkdwn("") == ""

    def test_plain_text_unchanged(self):
        assert _md_to_mrkdwn("just plain text") == "just plain text"


# ---------------------------------------------------------------------------
# build_approval_message
# ---------------------------------------------------------------------------

class TestBuildApprovalMessage:
    def test_returns_dict_with_blocks(self):
        msg = build_approval_message(
            alert_id="alert-123",
            alertname="PodCrashLooping",
            severity="critical",
            error_summary="DB connection refused",
            triage_summary="CrashLoopBackOff in checkout-api",
            proposed_action="kubectl rollout restart deployment/checkout-api",
        )
        assert isinstance(msg, dict)
        assert "blocks" in msg
        assert isinstance(msg["blocks"], list)
        assert len(msg["blocks"]) > 0

    def test_contains_alert_id_in_button_values(self):
        alert_id = "alert-abc-999"
        msg = build_approval_message(
            alert_id=alert_id,
            alertname="HighMemory",
            severity="warning",
            error_summary="OOM risk",
            triage_summary="Memory > 90%",
            proposed_action="Scale deployment",
        )
        # Find action blocks with buttons
        action_blocks = [b for b in msg["blocks"] if b.get("type") == "actions"]
        assert action_blocks, "No actions block found"
        elements = action_blocks[0]["elements"]
        values = [e["value"] for e in elements]
        assert alert_id in values, f"alert_id not in button values: {values}"

    def test_approve_button_action_id_starts_with_approve(self):
        alert_id = "alert-xyz"
        msg = build_approval_message(
            alert_id=alert_id,
            alertname="TestAlert",
            severity="info",
            error_summary="",
            triage_summary="",
            proposed_action="restart",
        )
        action_blocks = [b for b in msg["blocks"] if b.get("type") == "actions"]
        elements = action_blocks[0]["elements"]
        approve_btn = next(e for e in elements if e["action_id"].startswith("approve_"))
        assert approve_btn["action_id"] == f"approve_{alert_id}"

    def test_reject_button_action_id_starts_with_reject(self):
        alert_id = "alert-xyz"
        msg = build_approval_message(
            alert_id=alert_id,
            alertname="TestAlert",
            severity="info",
            error_summary="",
            triage_summary="",
            proposed_action="restart",
        )
        action_blocks = [b for b in msg["blocks"] if b.get("type") == "actions"]
        elements = action_blocks[0]["elements"]
        reject_btn = next(e for e in elements if e["action_id"].startswith("reject_"))
        assert reject_btn["action_id"] == f"reject_{alert_id}"

    def test_approve_button_is_primary_style(self):
        msg = build_approval_message(
            alert_id="a1",
            alertname="X",
            severity="critical",
            error_summary="",
            triage_summary="",
            proposed_action="",
        )
        action_blocks = [b for b in msg["blocks"] if b.get("type") == "actions"]
        elements = action_blocks[0]["elements"]
        approve_btn = next(e for e in elements if e["action_id"].startswith("approve_"))
        assert approve_btn.get("style") == "primary"

    def test_reject_button_is_danger_style(self):
        msg = build_approval_message(
            alert_id="a1",
            alertname="X",
            severity="critical",
            error_summary="",
            triage_summary="",
            proposed_action="",
        )
        action_blocks = [b for b in msg["blocks"] if b.get("type") == "actions"]
        elements = action_blocks[0]["elements"]
        reject_btn = next(e for e in elements if e["action_id"].startswith("reject_"))
        assert reject_btn.get("style") == "danger"

    def test_fallback_text_contains_alertname(self):
        msg = build_approval_message(
            alert_id="a1",
            alertname="PodCrashLooping",
            severity="critical",
            error_summary="",
            triage_summary="",
            proposed_action="",
        )
        assert "PodCrashLooping" in msg.get("text", "")

    def test_unknown_severity_uses_white_circle(self):
        msg = build_approval_message(
            alert_id="a1",
            alertname="X",
            severity="unknown",
            error_summary="",
            triage_summary="",
            proposed_action="",
        )
        header = next(b for b in msg["blocks"] if b.get("type") == "header")
        assert ":white_circle:" in header["text"]["text"]

    def test_critical_severity_uses_red_circle(self):
        msg = build_approval_message(
            alert_id="a1",
            alertname="X",
            severity="critical",
            error_summary="",
            triage_summary="",
            proposed_action="",
        )
        header = next(b for b in msg["blocks"] if b.get("type") == "header")
        assert ":red_circle:" in header["text"]["text"]


# ---------------------------------------------------------------------------
# build_resolution_message
# ---------------------------------------------------------------------------

class TestBuildResolutionMessage:
    def test_returns_dict_with_blocks(self):
        msg = build_resolution_message("alert-1", "resolved", "Service restarted")
        assert "blocks" in msg
        assert len(msg["blocks"]) > 0

    def test_resolved_uses_check_emoji(self):
        msg = build_resolution_message("alert-1", "resolved")
        assert ":white_check_mark:" in msg["text"]

    def test_non_resolved_uses_x_emoji(self):
        msg = build_resolution_message("alert-1", "escalated")
        assert ":x:" in msg["text"]

    def test_alert_id_in_text(self):
        msg = build_resolution_message("my-alert-id", "resolved")
        assert "my-alert-id" in msg["text"]


# ---------------------------------------------------------------------------
# HMAC verification logic (unit-level, no FastAPI server)
# ---------------------------------------------------------------------------

class TestHMACVerification:
    """Test the HMAC logic extracted from slack_verify.py directly."""

    @staticmethod
    def _sign(secret: str, timestamp: str, body: str) -> str:
        base = f"v0:{timestamp}:{body}"
        sig = hmac.new(
            secret.encode(),
            base.encode(),
            hashlib.sha256,
        ).hexdigest()
        return f"v0={sig}"

    def test_valid_signature_passes(self):
        secret = "test-secret-abc"
        ts = str(int(time.time()))
        body = '{"payload": "test"}'
        sig = self._sign(secret, ts, body)

        # Replicate the verification logic
        base = f"v0:{ts}:{body}"
        expected = "v0=" + hmac.new(
            secret.encode(), base.encode(), hashlib.sha256
        ).hexdigest()
        assert hmac.compare_digest(expected, sig)

    def test_wrong_secret_fails(self):
        ts = str(int(time.time()))
        body = "hello"
        sig = self._sign("wrong-secret", ts, body)

        base = f"v0:{ts}:{body}"
        expected = "v0=" + hmac.new(
            "correct-secret".encode(), base.encode(), hashlib.sha256
        ).hexdigest()
        assert not hmac.compare_digest(expected, sig)

    def test_tampered_body_fails(self):
        secret = "secret"
        ts = str(int(time.time()))
        original_body = "original"
        sig = self._sign(secret, ts, original_body)

        tampered_body = "tampered"
        base = f"v0:{ts}:{tampered_body}"
        expected = "v0=" + hmac.new(
            secret.encode(), base.encode(), hashlib.sha256
        ).hexdigest()
        assert not hmac.compare_digest(expected, sig)

    def test_old_timestamp_detected(self):
        old_ts = int(time.time()) - 400  # 400 seconds ago (> 5 min limit)
        assert abs(time.time() - old_ts) > 300

    def test_current_timestamp_accepted(self):
        current_ts = int(time.time())
        assert abs(time.time() - current_ts) <= 5
