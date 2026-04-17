"""Slack Block Kit message builders for the Autonomous SRE bot.

Produces rich interactive messages with context sections and
[Approve] / [Reject] buttons for the human approval workflow.
"""
from __future__ import annotations

import re

_HEADER_MAX = 150
_SECTION_MAX = 3000


def _trunc(text: str, max_len: int) -> str:
    if len(text) > max_len:
        return text[: max_len - 1] + "\u2026"
    return text


def _md_to_mrkdwn(text: str) -> str:
    """Convert standard Markdown to Slack mrkdwn format.

    Key differences handled:
      - **bold** / __bold__  →  *bold*
      - *italic*             →  _italic_
      - # Heading            →  *Heading*
      - ~~strike~~           →  ~strike~
      - [label](url)         →  <url|label>
      - ``` / ` code `       →  unchanged (Slack renders these natively)
      - --- horizontal rule  →  removed
    """
    # 1. Protect code spans and fenced blocks from further conversion
    protected: list[str] = []

    def _stash(m: re.Match) -> str:
        protected.append(m.group(0))
        return f"\x00{len(protected) - 1}\x00"

    text = re.sub(r"```[\s\S]*?```", _stash, text)
    text = re.sub(r"`[^`\n]+`", _stash, text)

    # Sentinel strings for bold — must not appear in real text
    _BOLD_OPEN = "\x00BOLD\x00"
    _BOLD_CLOSE = "\x00/BOLD\x00"

    # 2. ATX headers → bold heading (via sentinel so italic step won't touch them)
    text = re.sub(r"^#{1,6}\s+(.+)$", lambda m: f"{_BOLD_OPEN}{m.group(1)}{_BOLD_CLOSE}", text, flags=re.MULTILINE)

    # 3. Bold: **…** / __…__ — also stash with sentinel
    text = re.sub(r"\*\*(.+?)\*\*", lambda m: f"{_BOLD_OPEN}{m.group(1)}{_BOLD_CLOSE}", text, flags=re.DOTALL)
    text = re.sub(r"__(.+?)__", lambda m: f"{_BOLD_OPEN}{m.group(1)}{_BOLD_CLOSE}", text, flags=re.DOTALL)

    # 4. Italic: *…* → _…_  (only single stars remain now)
    text = re.sub(r"\*([^*\n]+?)\*", lambda m: f"_{m.group(1)}_", text)

    # 5. Restore bold sentinels as Slack bold
    text = text.replace(_BOLD_OPEN, "*").replace(_BOLD_CLOSE, "*")

    # 6. Strikethrough: ~~…~~ → ~…~
    text = re.sub(r"~~(.+?)~~", r"~\1~", text, flags=re.DOTALL)

    # 7. Links: [label](url) → <url|label>
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)

    # 8. Horizontal rules → blank line
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # 9. Restore protected code spans/blocks
    for i, snippet in enumerate(protected):
        text = text.replace(f"\x00{i}\x00", snippet)

    return text.strip()


def build_approval_message(
    alert_id: str,
    alertname: str,
    severity: str,
    error_summary: str,
    triage_summary: str,
    proposed_action: str,
    prior_attempt: str | None = None,
) -> dict:
    """Return a Slack Block Kit payload for the human approval gate.

    The [Approve] button has action_id ``approve_<alert_id>`` and the
    [Reject] button has action_id ``reject_<alert_id>``.  The ``value``
    field on both buttons contains the ``alert_id`` so the
    ``/slack/interactive`` handler can look up the correct graph thread.
    """
    severity_emoji = {
        "critical": ":red_circle:",
        "warning": ":large_yellow_circle:",
        "info": ":large_blue_circle:",
    }.get(severity.lower(), ":white_circle:")

    analysis = _md_to_mrkdwn(error_summary or triage_summary or "_No analysis available_")
    action = _md_to_mrkdwn(proposed_action or "_No action proposed_")
    # Plain truncated text for the confirm dialog (plain_text type, no mrkdwn overhead)
    confirm_action = _trunc(proposed_action or "No action proposed", 260)

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": _trunc(f"{severity_emoji}  Alert: {alertname}", _HEADER_MAX),
                "emoji": True,
            },
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Severity:*\n{severity.upper()}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Alert ID:*\n`{alert_id}`",
                },
            ],
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": _trunc(f"*:mag: Analysis*\n{analysis}", _SECTION_MAX),
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": _trunc(f"*:wrench: Recommended Action*\n{action}", _SECTION_MAX),
            },
        },
    ]

    if prior_attempt:
        prior_text = _md_to_mrkdwn(prior_attempt)
        blocks.append({"type": "divider"})
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": _trunc(f"*:warning: Previous Attempt Result*\n{prior_text}", _SECTION_MAX),
                },
            }
        )

    blocks.append({"type": "divider"})
    blocks.append(
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Approve", "emoji": True},
                    "style": "primary",
                    "action_id": f"approve_{alert_id}",
                    "value": alert_id,
                    "confirm": {
                        "title": {"type": "plain_text", "text": "Confirm Approval"},
                        "text": {
                            "type": "plain_text",
                            "text": f"Execute: {confirm_action}?",
                        },
                        "confirm": {"type": "plain_text", "text": "Yes, execute"},
                        "deny": {"type": "plain_text", "text": "Cancel"},
                    },
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Reject", "emoji": True},
                    "style": "danger",
                    "action_id": f"reject_{alert_id}",
                    "value": alert_id,
                },
            ],
        }
    )

    return {
        "text": f"[{severity.upper()}] {alertname} — approval required",
        "blocks": blocks,
    }


def build_working_message(alert_id: str, approved: bool) -> dict:
    """Return a Slack Block Kit payload that replaces the approval buttons immediately.

    Sent as the synchronous HTTP response to the button click so Slack removes
    the [Approve]/[Reject] buttons before the background graph run finishes.
    Must include ``replace_original: true`` so Slack swaps the original message.
    """
    if approved:
        status_line = ":hourglass_flowing_sand:  *Approved — agent is executing the action...*"
    else:
        status_line = ":hourglass_flowing_sand:  *Rejected — escalating to on-call team...*"

    return {
        "replace_original": True,
        "text": "Agent is working on it...",
        "blocks": [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": status_line},
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"Alert ID: `{alert_id}`"},
                ],
            },
        ],
    }


def build_rejected_message(alert_id: str, alertname: str, error_summary: str) -> dict:
    """Return a Slack Block Kit payload for a human-rejected alert.

    Replaces the approval message with a concise ❌ notice and removes buttons.
    Must include ``replace_original: true`` so Slack swaps the original message.
    """
    summary = _trunc(_md_to_mrkdwn(error_summary or "No details available"), _SECTION_MAX)
    return {
        "replace_original": True,
        "text": f":x: {alertname} — rejected",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f":x:  *{alertname}* — rejected by operator\n{summary}",
                },
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"Alert ID: `{alert_id}`"},
                ],
            },
        ],
    }


def build_success_summary_message(
    alert_id: str,
    alertname: str,
    error_summary: str,
    action_result: str,
) -> dict:
    """Return a Slack Block Kit payload summarising a resolved incident.

    Posted as a *new* message after all incident-related messages are deleted.
    Does NOT include ``replace_original`` — this is a fresh post, not a replacement.
    """
    summary = _trunc(error_summary or "Alert resolved", _SECTION_MAX)
    result_text = _trunc(_md_to_mrkdwn(action_result) if action_result else "No details", _SECTION_MAX)
    return {
        "text": f":white_check_mark: Resolved: {alertname}",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": _trunc(f":white_check_mark:  Resolved: {alertname}", _HEADER_MAX),
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Issue:* {summary}"},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Steps taken:*\n{result_text}"},
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"Alert ID: `{alert_id}`"},
                ],
            },
        ],
    }


def build_failed_message(
    alert_id: str,
    alertname: str,
    error_summary: str,
    action_result: str,
    proposed_action: str = "",
) -> dict:
    """Return a Slack Block Kit payload for a failed remediation attempt.

    Shows the error, any actions that were attempted, and Approve/Reject buttons
    so the operator can retry or escalate.
    """
    error_text = _trunc(_md_to_mrkdwn(error_summary or "Remediation failed"), _SECTION_MAX)

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": _trunc(f":x:  Failed: {alertname}", _HEADER_MAX),
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Error:*\n{error_text}"},
        },
    ]

    if action_result:
        prior_text = _trunc(_md_to_mrkdwn(action_result), _SECTION_MAX)
        blocks.append({"type": "divider"})
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Actions attempted:*\n{prior_text}",
                },
            }
        )

    blocks.append({"type": "divider"})
    blocks.append(
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"Alert ID: `{alert_id}`"},
            ],
        }
    )

    confirm_action = _trunc(proposed_action or action_result or "Retry the action", 260)
    blocks.append(
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Retry", "emoji": True},
                    "style": "primary",
                    "action_id": f"approve_{alert_id}",
                    "value": alert_id,
                    "confirm": {
                        "title": {"type": "plain_text", "text": "Confirm Retry"},
                        "text": {
                            "type": "plain_text",
                            "text": f"Retry: {confirm_action}?",
                        },
                        "confirm": {"type": "plain_text", "text": "Yes, retry"},
                        "deny": {"type": "plain_text", "text": "Cancel"},
                    },
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Escalate", "emoji": True},
                    "style": "danger",
                    "action_id": f"reject_{alert_id}",
                    "value": alert_id,
                },
            ],
        }
    )

    return {
        "replace_original": True,
        "text": f":x: Failed: {alertname}",
        "blocks": blocks,
    }


def build_resolution_message(alert_id: str, status: str, action_result: str = "") -> dict:
    """Return a Slack Block Kit payload announcing the resolution outcome."""
    if status == "resolved":
        status_emoji = ":white_check_mark:"
    elif status == "rejected":
        status_emoji = ":warning:"
    elif status == "waiting_for_approval":
        status_emoji = ":arrows_counterclockwise:"
    elif status == "escalated":
        status_emoji = ":warning:"
    elif status == "in_progress":
        status_emoji = ":hourglass_flowing_sand:"
    else:
        status_emoji = ":warning:"

    result_text = _md_to_mrkdwn(action_result) if action_result else ""

    blocks: list[dict] = [
        {"type": "divider"},
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": _trunc(f"{status_emoji}  Alert {alert_id} — {status.upper()}", _HEADER_MAX),
                "emoji": True,
            },
        },
    ]

    if result_text:
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": _trunc(result_text, _SECTION_MAX)},
            }
        )

    return {
        "text": f"{status_emoji} Alert `{alert_id}` — {status}",
        "blocks": blocks,
    }


def build_remediation_message(
    alert_id: str,
    alertname: str,
    action_result: str,
    error_msg: str = "Failed to update Slack message",
) -> dict:
    """Return a Slack message explaining what was done and how to verify.

    Used when the automated update fails but the action succeeded.
    Shows what was executed and provides manual verification steps.
    """
    result_text = _md_to_mrkdwn(action_result) if action_result else "No action details available"

    return {
        "text": f":warning: {alertname} — Action completed (message update failed)",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": ":white_check_mark: Action Completed",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Alert:* {alertname}\n*Status:* ✅ Remediation applied successfully",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*What was done:*\n{_trunc(result_text, _SECTION_MAX)}",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Next steps:*\n1. Check the resource status in your cluster\n2. Verify metrics are healthy (error rate, latency, pod restarts)\n3. Close this alert if the issue is resolved",
                },
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"Alert ID: `{alert_id}` | {error_msg}"},
                ],
            },
        ],
    }
