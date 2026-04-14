"""Triage node — classifies alert severity, selects diagnostic tools, and scores task complexity.

Receives raw alert payload + metadata, returns severity, tools_to_run,
triage_summary, and task_complexity (used by router for LLM selection).

Triage always uses Claude (it runs before the router sets llm_provider).
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
from app.utils.prompt_loader import load_prompt, render_prompt


@traceable(name="triage_node", metadata={"phase": "triage"})
async def triage_node(state: SREState) -> dict:
    """Classify the alert, select tools, and score task complexity."""
    settings = get_settings()
    payload = state["alert_payload"]
    metadata = state["metadata"]
    annotations = payload.get("annotations", {})

    try:
        # Triage always uses Claude — runs before the router decides the provider
        llm = ChatAnthropic(
            model=settings.triage_model,
            api_key=settings.anthropic_api_key,
        )
        llm.bind_tools(ALL_TOOLS)

        prompt_config = load_prompt("triage", settings.prompt_dir)
        prompt = render_prompt(
            prompt_config,
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
        task_complexity = _validate_complexity(parsed.get("task_complexity", "moderate"))

        status = "escalated" if severity == "unknown" and not tools_to_run else state["status"]

        usage = extract_usage("triage", settings.triage_model, response)
        updated_token_usage = list(state.get("token_usage", [])) + [dict(usage)]

        reasoning_entry = (
            f"[triage] severity={severity} | "
            f"tools={tools_to_run} | "
            f"complexity={task_complexity} | "
            f"summary={triage_summary!r} | "
            f"tokens={usage['total_tokens']} cost=${usage['cost_usd']:.6f}"
        )

        return {
            "severity": severity,
            "tools_to_run": tools_to_run,
            "triage_summary": triage_summary,
            "task_complexity": task_complexity,
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
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("```").strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
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


def _validate_complexity(value: str) -> str:
    """Ensure complexity is one of the valid values; default to 'moderate'."""
    valid = {"simple", "moderate", "complex"}
    return value if value in valid else "moderate"
