"""Triage node — classifies alert severity and selects diagnostic tools.

Receives raw alert payload + metadata, returns severity, tools_to_run, triage_summary.
"""
from __future__ import annotations

import json
import re

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langsmith import traceable

from app.agents.state import SREState
from app.config import get_settings
from app.tools import ALL_TOOLS, TOOL_REGISTRY
from app.utils.llm_cost import accumulate_cost, extract_usage

_TRIAGE_PROMPT = """\
You are an expert SRE. Analyze the following Kubernetes alert and determine:
1. Severity: "critical", "warning", or "unknown"
2. Which diagnostic tools to run (choose from the available tools)
3. A concise triage summary

Alert: {alertname}
Status: {status}
Summary: {summary}
Metadata: region={region}, env={env}, namespace={namespace}, cluster={cluster_id}

Available tools: {tool_names}

Respond with ONLY valid JSON in this exact format:
{{"severity": "<critical|warning|unknown>", "tools_to_run": ["tool1", "tool2"], "triage_summary": "<summary>"}}

If the alert is unrecognizable or you cannot determine any remediation, set severity to "unknown" and tools_to_run to [].
"""


@traceable(name="triage_node", metadata={"phase": "triage"})
async def triage_node(state: SREState) -> dict:
    """Classify the alert and select tools for investigation."""
    settings = get_settings()
    payload = state["alert_payload"]
    metadata = state["metadata"]
    annotations = payload.get("annotations", {})

    try:
        llm = ChatAnthropic(
            model=settings.triage_model,
            api_key=settings.anthropic_api_key,
        )
        llm.bind_tools(ALL_TOOLS)

        prompt = _TRIAGE_PROMPT.format(
            alertname=payload.get("alertname", "unknown"),
            status=payload.get("status", "firing"),
            summary=annotations.get("summary", annotations.get("description", "")),
            region=metadata.get("region", "unknown"),
            env=metadata.get("env", "unknown"),
            namespace=metadata.get("namespace", "default"),
            cluster_id=metadata.get("cluster_id", "unknown"),
            tool_names=", ".join(TOOL_REGISTRY.keys()),
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = _parse_triage_response(response.content)

        severity = parsed.get("severity", "unknown")
        tools_to_run = _validate_tools(parsed.get("tools_to_run", []))
        triage_summary = parsed.get("triage_summary", "")

        status = "escalated" if severity == "unknown" and not tools_to_run else state["status"]

        usage = extract_usage("triage", settings.triage_model, response)
        updated_token_usage = list(state.get("token_usage", [])) + [dict(usage)]

        reasoning_entry = (
            f"[triage] severity={severity} | "
            f"tools={tools_to_run} | "
            f"summary={triage_summary!r} | "
            f"tokens={usage['total_tokens']} cost=${usage['cost_usd']:.6f}"
        )

        return {
            "severity": severity,
            "tools_to_run": tools_to_run,
            "triage_summary": triage_summary,
            "status": status,
            "token_usage": updated_token_usage,
            "cost_estimate_usd": accumulate_cost(updated_token_usage),
            "reasoning_log": list(state.get("reasoning_log", [])) + [reasoning_entry],
            "current_node": "triage",
        }

    except Exception as exc:
        return {
            "status": "failed",
            "error_log": state.get("error_log", []) + [f"triage_node error: {exc}"],
            "current_node": "triage",
        }


def _parse_triage_response(content: str) -> dict:
    """Extract JSON from LLM response, handling markdown code fences."""
    text = content.strip()
    # Strip markdown code blocks
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("```").strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # Attempt to extract JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


def _validate_tools(tool_names: list) -> list[str]:
    """Filter tool names to only include registered tools."""
    return [name for name in tool_names if name in TOOL_REGISTRY]
