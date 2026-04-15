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

    return {
        "text": f"[{severity.upper()}] {alertname} — approval required",
        "blocks": [
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
            {"type": "divider"},
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
                                "type": "mrkdwn",
                                "text": _trunc(f"Execute: _{action}_?", 300),
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
            },
        ],
    }


def build_resolution_message(alert_id: str, status: str, action_result: str = "") -> dict:
    """Return a Slack Block Kit payload announcing the resolution outcome."""
    status_emoji = ":white_check_mark:" if status == "resolved" else ":x:"
    result_text = _md_to_mrkdwn(action_result) if action_result else ""
    return {
        "text": f"{status_emoji} Alert `{alert_id}` — {status}",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"{status_emoji} *Alert `{alert_id}` {status}*"
                        + (f"\n{result_text}" if result_text else "")
                    ),
                },
            }
        ],
    }
