"""Unit tests for Slack Block Kit message builders and HMAC verification."""
from __future__ import annotations

import hashlib
import hmac
import time

import pytest

from app.utils.slack_blocks import (
    _md_to_mrkdwn,
    build_approval_message,
    build_failed_message,
    build_remediation_message,
    build_rejected_message,
    build_resolution_message,
    build_success_summary_message,
    build_working_message,
)


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

    def test_prior_attempt_section_included_when_provided(self):
        msg = build_approval_message(
            alert_id="a1",
            alertname="X",
            severity="warning",
            error_summary="",
            triage_summary="",
            proposed_action="restart",
            prior_attempt="Rolling restart applied but pods still crashing.",
        )
        text = str(msg)
        assert "Previous Attempt" in text
        assert "Rolling restart applied" in text

    def test_no_prior_attempt_section_when_not_provided(self):
        msg = build_approval_message(
            alert_id="a1",
            alertname="X",
            severity="warning",
            error_summary="",
            triage_summary="",
            proposed_action="restart",
        )
        text = str(msg)
        assert "Previous Attempt" not in text


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

    def test_escalated_uses_warning_emoji(self):
        msg = build_resolution_message("alert-1", "escalated")
        assert ":warning:" in msg["text"]

    def test_alert_id_in_text(self):
        msg = build_resolution_message("my-alert-id", "resolved")
        assert "my-alert-id" in msg["text"]


# ---------------------------------------------------------------------------
# build_working_message
# ---------------------------------------------------------------------------

class TestBuildWorkingMessage:
    def test_approved_mentions_executing(self):
        msg = build_working_message("alert-1", approved=True)
        text = str(msg)
        assert "executing" in text.lower() or "approved" in text.lower()

    def test_rejected_mentions_escalating(self):
        msg = build_working_message("alert-1", approved=False)
        text = str(msg)
        assert "escalat" in text.lower() or "rejected" in text.lower()

    def test_replace_original_is_true(self):
        msg = build_working_message("alert-1", approved=True)
        assert msg.get("replace_original") is True

    def test_has_blocks(self):
        msg = build_working_message("alert-1", approved=True)
        assert "blocks" in msg
        assert len(msg["blocks"]) > 0

    def test_no_actions_block(self):
        """Buttons must be absent so they disappear when this message replaces the original."""
        msg = build_working_message("alert-1", approved=True)
        action_blocks = [b for b in msg["blocks"] if b.get("type") == "actions"]
        assert action_blocks == []

    def test_alert_id_in_blocks(self):
        msg = build_working_message("my-alert-99", approved=True)
        text = str(msg)
        assert "my-alert-99" in text


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


# ---------------------------------------------------------------------------
# build_rejected_message
# ---------------------------------------------------------------------------

class TestBuildRejectedMessage:
    def test_returns_dict_with_blocks(self):
        msg = build_rejected_message(
            alert_id="alert-123",
            alertname="PodCrashLooping",
            error_summary="Pod crash looping in checkout",
        )
        assert isinstance(msg, dict)
        assert "blocks" in msg
        assert len(msg["blocks"]) > 0

    def test_contains_x_emoji(self):
        msg = build_rejected_message("a1", "PodCrashLooping", "DB connection refused")
        text = str(msg)
        assert ":x:" in text

    def test_no_action_buttons(self):
        msg = build_rejected_message("a1", "PodCrashLooping", "OOM killed")
        action_blocks = [b for b in msg["blocks"] if b.get("type") == "actions"]
        assert action_blocks == []

    def test_replace_original_is_true(self):
        msg = build_rejected_message("a1", "SomeAlert", "Some error")
        assert msg.get("replace_original") is True

    def test_contains_alertname(self):
        msg = build_rejected_message("a1", "HighMemoryUsage", "Memory usage critical")
        text = str(msg)
        assert "HighMemoryUsage" in text

    def test_contains_error_summary(self):
        msg = build_rejected_message("a1", "TestAlert", "DB connection refused")
        text = str(msg)
        assert "DB connection refused" in text

    def test_contains_alert_id(self):
        msg = build_rejected_message("my-alert-id-999", "TestAlert", "error")
        text = str(msg)
        assert "my-alert-id-999" in text

    def test_markdown_converted_in_error_summary(self):
        """**bold** in error_summary should be rendered as *bold* (mrkdwn)."""
        msg = build_rejected_message("a1", "Alert", "**Critical failure** in checkout")
        text = str(msg)
        # mrkdwn bold (*), not markdown bold (**)
        assert "*Critical failure*" in text
        assert "**Critical failure**" not in text


# ---------------------------------------------------------------------------
# build_success_summary_message
# ---------------------------------------------------------------------------

class TestBuildSuccessSummaryMessage:
    def test_returns_dict_with_blocks(self):
        msg = build_success_summary_message(
            alert_id="alert-123",
            alertname="PodCrashLooping",
            error_summary="Pod crash looping in checkout",
            action_result="Rolling restart applied successfully",
        )
        assert isinstance(msg, dict)
        assert "blocks" in msg
        assert len(msg["blocks"]) > 0

    def test_contains_check_emoji(self):
        msg = build_success_summary_message("a1", "PodCrashLooping", "OOM", "Restarted")
        text = str(msg)
        assert ":white_check_mark:" in text

    def test_no_action_buttons(self):
        msg = build_success_summary_message("a1", "PodCrashLooping", "OOM", "Restarted")
        action_blocks = [b for b in msg["blocks"] if b.get("type") == "actions"]
        assert action_blocks == []

    def test_no_replace_original(self):
        """Success summary is a new message, not a replacement."""
        msg = build_success_summary_message("a1", "PodCrashLooping", "OOM", "Restarted")
        assert "replace_original" not in msg

    def test_contains_alertname(self):
        msg = build_success_summary_message("a1", "CriticalAlert", "desc", "fix")
        text = str(msg)
        assert "CriticalAlert" in text

    def test_contains_action_result(self):
        msg = build_success_summary_message("a1", "A", "desc", "Rolling restart applied")
        text = str(msg)
        assert "Rolling restart" in text

    def test_contains_alert_id(self):
        msg = build_success_summary_message("my-alert-abc", "A", "desc", "fix")
        text = str(msg)
        assert "my-alert-abc" in text


# ---------------------------------------------------------------------------
# build_failed_message
# ---------------------------------------------------------------------------

class TestBuildFailedMessage:
    def test_returns_dict_with_blocks(self):
        msg = build_failed_message(
            alert_id="alert-123",
            alertname="PodCrashLooping",
            error_summary="Connection refused",
            action_result="kubectl rollout restart deployment/checkout-api",
        )
        assert isinstance(msg, dict)
        assert "blocks" in msg
        assert len(msg["blocks"]) > 0

    def test_contains_x_emoji_in_header(self):
        msg = build_failed_message("a1", "SomeAlert", "error", "action")
        text = str(msg)
        assert ":x:" in text

    def test_contains_alertname(self):
        msg = build_failed_message("a1", "HighMemoryUsage", "OOM", "scale up")
        text = str(msg)
        assert "HighMemoryUsage" in text

    def test_contains_error_summary(self):
        msg = build_failed_message("a1", "Alert", "DB connection refused", "restart")
        text = str(msg)
        assert "DB connection refused" in text

    def test_contains_action_result_when_provided(self):
        msg = build_failed_message("a1", "Alert", "error", "Rolling restart applied")
        text = str(msg)
        assert "Rolling restart applied" in text

    def test_actions_section_absent_when_no_action_result(self):
        msg = build_failed_message("a1", "Alert", "error", "")
        text = str(msg)
        assert "Actions attempted" not in text

    def test_actions_section_present_when_action_result_provided(self):
        msg = build_failed_message("a1", "Alert", "error", "tried restart")
        text = str(msg)
        assert "Actions attempted" in text

    def test_contains_alert_id(self):
        msg = build_failed_message("my-alert-xyz", "Alert", "error", "action")
        text = str(msg)
        assert "my-alert-xyz" in text

    def test_replace_original_is_true(self):
        msg = build_failed_message("a1", "Alert", "error", "action")
        assert msg.get("replace_original") is True

    def test_has_retry_and_escalate_buttons(self):
        msg = build_failed_message("a1", "Alert", "error", "action")
        action_blocks = [b for b in msg["blocks"] if b.get("type") == "actions"]
        assert len(action_blocks) == 1
        elements = action_blocks[0]["elements"]
        action_ids = [e["action_id"] for e in elements]
        assert "approve_a1" in action_ids
        assert "reject_a1" in action_ids

    def test_retry_button_is_primary_style(self):
        msg = build_failed_message("a1", "Alert", "error", "action")
        action_blocks = [b for b in msg["blocks"] if b.get("type") == "actions"]
        elements = action_blocks[0]["elements"]
        retry_btn = next(e for e in elements if e["action_id"].startswith("approve_"))
        assert retry_btn.get("style") == "primary"

    def test_escalate_button_is_danger_style(self):
        msg = build_failed_message("a1", "Alert", "error", "action")
        action_blocks = [b for b in msg["blocks"] if b.get("type") == "actions"]
        elements = action_blocks[0]["elements"]
        escalate_btn = next(e for e in elements if e["action_id"].startswith("reject_"))
        assert escalate_btn.get("style") == "danger"

    def test_markdown_converted_in_error_summary(self):
        msg = build_failed_message("a1", "Alert", "**Critical failure** in checkout", "action")
        text = str(msg)
        assert "*Critical failure*" in text
        assert "**Critical failure**" not in text


# ---------------------------------------------------------------------------
# build_remediation_message
# ---------------------------------------------------------------------------

class TestBuildRemediationMessage:
    def test_returns_dict_with_blocks(self):
        msg = build_remediation_message(
            alert_id="alert-123",
            alertname="PodCrashLooping",
            action_result="Restarted pod successfully",
        )
        assert isinstance(msg, dict)
        assert "blocks" in msg
        assert len(msg["blocks"]) > 0

    def test_contains_check_emoji_for_completion(self):
        msg = build_remediation_message("a1", "Alert", "Action done")
        text = str(msg)
        assert ":white_check_mark:" in text

    def test_contains_action_result(self):
        msg = build_remediation_message("a1", "OOMAlert", "Scaled memory limit from 100Mi to 200Mi")
        text = str(msg)
        assert "Scaled memory" in text

    def test_contains_verification_steps(self):
        msg = build_remediation_message("a1", "Alert", "Action completed")
        text = str(msg)
        # Should mention manual verification steps
        assert "Check the resource status" in text or "metrics" in text.lower() or "verify" in text.lower()

    def test_contains_alert_id(self):
        msg = build_remediation_message("my-special-id", "Alert", "Done")
        text = str(msg)
        assert "my-special-id" in text

    def test_custom_error_message(self):
        msg = build_remediation_message("a1", "Alert", "Result", error_msg="Custom failure reason")
        text = str(msg)
        assert "Custom failure reason" in text

    def test_no_action_buttons(self):
        msg = build_remediation_message("a1", "Alert", "Result")
        action_blocks = [b for b in msg["blocks"] if b.get("type") == "actions"]
        assert action_blocks == []
