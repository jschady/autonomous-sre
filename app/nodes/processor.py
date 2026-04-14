"""Processor node — executes diagnostic tools and summarises findings.

Takes tools_to_run from state, invokes each tool, aggregates raw output,
then uses an LLM to produce a concise error_summary.

Key Phase 3 changes:
  - Uses Haiku (via settings.processor_model) to summarize logs before they
    reach the researcher. This compresses 10MB log files into ~200-word summaries
    that fit comfortably in Llama 8B's context window.
  - Uses LLM factory (respects state["llm_provider"] set by router node).
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage
from langsmith import traceable

from app.agents.state import SREState
from app.config import get_settings
from app.tools import TOOL_REGISTRY
from app.utils.llm_cost import accumulate_cost, extract_usage
from app.utils.llm_factory import ainvoke_with_fallback
from app.utils.prompt_loader import load_prompt, render_prompt

# Character budget for raw_logs passed to the summary LLM.
# Prevents OOM / context overflow when logs are very large.
_MAX_RAW_LOG_CHARS = 50_000


@traceable(name="processor_node", metadata={"phase": "processing"})
async def processor_node(state: SREState) -> dict:
    """Run diagnostic tools and produce a compact error summary.

    Always uses Haiku (processor_model) for log summarization regardless of
    the router's llm_provider decision — Haiku is extremely cost-effective for
    this compression task and produces output sized for Llama 8B's context.
    """
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

    # Truncate before sending to LLM to avoid context overflow
    raw_data_for_llm = raw_logs[:_MAX_RAW_LOG_CHARS]
    if len(raw_logs) > _MAX_RAW_LOG_CHARS:
        raw_data_for_llm += "\n[... truncated for context window ...]"

    try:
        # Processor ALWAYS uses Haiku for log summarization.
        # This is a deliberate design choice: Haiku compresses large diagnostic
        # data into a tight summary that fits in Llama 8B's 32K context window.
        # The researcher node then uses the router's llm_provider selection.
        prompt_config = load_prompt("processor", settings.prompt_dir)
        prompt = render_prompt(
            prompt_config,
            alertname=payload.get("alertname", "unknown"),
            namespace=namespace,
            service=service,
            raw_data=raw_data_for_llm,
        )
        response = await ainvoke_with_fallback(
            provider="claude",
            messages=[HumanMessage(content=prompt)],
            model_override=settings.processor_model,
        )
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
    if tool_name == "restart_service":
        return tool.invoke({"service_id": service, "namespace": namespace})
    if tool_name == "execute_rollback":
        return tool.invoke({"deployment_name": service, "namespace": namespace})
    raise ValueError(f"No argument mapping defined for tool: {tool_name!r}")
