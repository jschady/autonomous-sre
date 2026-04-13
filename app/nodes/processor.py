"""Processor node — executes diagnostic tools and summarises findings.

Takes tools_to_run from state, invokes each tool, aggregates raw output,
then uses LLM to produce a concise error_summary.
"""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langsmith import traceable

from app.agents.state import SREState
from app.config import get_settings
from app.tools import TOOL_REGISTRY
from app.utils.llm_cost import accumulate_cost, extract_usage

_SUMMARISE_PROMPT = """\
You are an expert SRE. The following diagnostic data was collected from Kubernetes tools.
Provide a concise technical summary of the root cause and current state.
Focus on specific errors, patterns, and actionable findings.

Alert: {alertname}
Namespace: {namespace}
Service: {service}

--- Diagnostic Data ---
{raw_data}
--- End Diagnostic Data ---

Summarise the key findings in 2-4 sentences. Be specific about errors found.
"""


@traceable(name="processor_node", metadata={"phase": "processing"})
async def processor_node(state: SREState) -> dict:
    """Run diagnostic tools and produce an error summary."""
    settings = get_settings()
    metadata = state["metadata"]
    payload = state["alert_payload"]
    tools_to_run = state.get("tools_to_run", [])

    raw_parts: list[str] = []
    namespace = metadata.get("namespace", "default")
    service = metadata.get("service", payload.get("alertname", "unknown"))
    pod = metadata.get("pod", f"{service}-pod")

    # Execute each requested tool
    for tool_name in tools_to_run:
        tool = TOOL_REGISTRY.get(tool_name)
        if tool is None:
            continue
        try:
            output = _invoke_tool(tool_name, tool, namespace, service, pod)
            raw_parts.append(f"[{tool_name}]\n{output}")
        except Exception as exc:
            raw_parts.append(f"[{tool_name}] ERROR: {exc}")

    raw_logs = "\n\n".join(raw_parts) if raw_parts else "No diagnostic data collected."

    try:
        llm = ChatAnthropic(
            model=settings.processor_model,
            api_key=settings.anthropic_api_key,
        )
        prompt = _SUMMARISE_PROMPT.format(
            alertname=payload.get("alertname", "unknown"),
            namespace=namespace,
            service=service,
            raw_data=raw_logs,
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        error_summary = response.content.strip()

    except Exception as exc:
        return {
            "raw_logs": raw_logs,
            "status": "failed",
            "error_log": state.get("error_log", []) + [f"processor_node LLM error: {exc}"],
            "current_node": "processor",
        }

    usage = extract_usage("processor", settings.processor_model, response)
    updated_token_usage = list(state.get("token_usage", [])) + [dict(usage)]

    reasoning_entry = (
        f"[processor] tools_run={tools_to_run} | "
        f"summary={error_summary!r} | "
        f"tokens={usage['total_tokens']} cost=${usage['cost_usd']:.6f}"
    )

    return {
        "raw_logs": raw_logs,
        "error_summary": error_summary,
        "token_usage": updated_token_usage,
        "cost_estimate_usd": accumulate_cost(updated_token_usage),
        "reasoning_log": list(state.get("reasoning_log", [])) + [reasoning_entry],
        "current_node": "processor",
    }


def _invoke_tool(tool_name: str, tool, namespace: str, service: str, pod: str) -> str:
    """Invoke a tool with appropriate arguments based on its name."""
    if tool_name == "get_cluster_events":
        return tool.invoke({"namespace": namespace, "service": service})
    if tool_name == "fetch_container_logs":
        return tool.invoke({"pod_id": pod, "container": service})
    if tool_name == "get_system_metrics":
        return tool.invoke({"service_name": service})
    if tool_name == "query_knowledge_base":
        return tool.invoke({"query": service})
    # For action tools (restart/rollback), just describe what would be done
    return tool.invoke({"service_id": service}) if tool_name == "restart_service" else str(tool_name)
