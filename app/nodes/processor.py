"""Processor node — executes diagnostic tools and summarises findings.

Takes tools_to_run from state, invokes each tool, aggregates raw output,
then uses Haiku (processor_model) to produce a concise error_summary.
"""
from __future__ import annotations

import re

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
    service = (
        metadata.get("service")
        or metadata.get("container")
        or payload.get("alertname", "unknown")
    )
    pod = metadata.get("pod", f"{service}-pod")
    # "container" label is set on pod-level alerts (KubePodCrashLooping etc.)
    # and is the correct container name; "service" is the scrape target, not the pod's container.
    container_hint = metadata.get("container", service)
    # Derive the actual workload name (deployment/statefulset) from the pod name.
    # For Deployment pods: chaos-app-7c949f9b88-txqcz → chaos-app
    # For StatefulSet/bare pods the name is returned as-is.
    # Used for restart/rollback and list_pods fallback — avoids using the scrape-target service name.
    workload = _derive_workload_name(pod)

    # Execute each requested tool
    for tool_name in tools_to_run:
        tool = TOOL_REGISTRY.get(tool_name)
        if tool is None:
            continue
        try:
            output = _invoke_tool(tool_name, tool, namespace, service, pod, container_hint, workload)
            raw_parts.append(f"[{tool_name}]\n{output}")
        except Exception as exc:
            raw_parts.append(f"[{tool_name}] ERROR: {exc}")

    raw_logs = "\n\n".join(raw_parts) if raw_parts else "No diagnostic data collected."

    # Truncate before sending to LLM to avoid context overflow
    raw_data_for_llm = raw_logs[:_MAX_RAW_LOG_CHARS]
    if len(raw_logs) > _MAX_RAW_LOG_CHARS:
        raw_data_for_llm += "\n[... truncated for context window ...]"

    try:
        prompt_config = load_prompt("processor", settings.prompt_dir)
        prompt = render_prompt(
            prompt_config,
            alertname=payload.get("alertname", "unknown"),
            namespace=namespace,
            service=workload,
            raw_data=raw_data_for_llm,
        )
        response = await ainvoke_with_fallback(
            provider="claude",
            messages=[HumanMessage(content=prompt)],
            model_override=settings.processor_model,
        )
        error_summary = response.content.strip()

        # Deterministically append memory_limit= tag when get_pod_resources ran.
        # The researcher prompt uses this to choose factor= vs initial_limit=.
        # We do this programmatically so it is never lost due to LLM paraphrasing.
        if "get_pod_resources" in tools_to_run and "memory_limit=" not in error_summary:
            tag = _extract_memory_limit_tag(raw_parts)
            if tag:
                error_summary += f"\n{tag}"

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
        "workload": workload,
        "token_usage": updated_token_usage,
        "cost_estimate_usd": accumulate_cost(updated_token_usage),
        "reasoning_log": list(state.get("reasoning_log", [])) + [reasoning_entry],
        "current_node": "processor",
    }


def _derive_workload_name(pod_name: str) -> str:
    """Extract deployment/workload name from a pod name by stripping RS and pod hash suffixes.

    Deployment pod format: {deployment}-{rs_hash}-{pod_hash}
      e.g. chaos-app-7c949f9b88-txqcz  →  chaos-app
    StatefulSet / bare pod names (no two trailing hashes) are returned unchanged.
    """
    match = re.match(r"^(.+)-[a-z0-9]{7,12}-[a-z0-9]{5}$", pod_name)
    return match.group(1) if match else pod_name


def _resolve_container_name(pod_id: str, fallback: str) -> str:
    """Return the first container name in pod_id, or fallback on any error."""
    list_tool = TOOL_REGISTRY.get("list_pod_containers")
    if list_tool is None:
        return fallback
    try:
        result = list_tool.invoke({"pod_id": pod_id})
        # result format: "[Containers in pod=...]\ncontainers: foo, bar"
        for line in result.splitlines():
            if line.startswith("containers:"):
                names = line.removeprefix("containers:").strip()
                if names and names != "(none)":
                    return names.split(",")[0].strip()
    except Exception:
        pass
    return fallback


def _find_current_pod(namespace: str, workload: str) -> str | None:
    """Find the first currently running/crashing pod for a workload by name prefix."""
    list_tool = TOOL_REGISTRY.get("list_pods")
    if list_tool is None:
        return None
    try:
        result = list_tool.invoke({"namespace": namespace, "name_prefix": workload})
        for line in result.splitlines():
            # Lines look like: "chaos-app-7c949f9b88-txqcz   CrashLoopBackOff   5"
            # Skip the header line "[Pods in ...]" and the column header "NAME  STATUS  RESTARTS"
            parts = line.split()
            if parts and not parts[0].startswith("[") and parts[0] != "NAME":
                return parts[0]
    except Exception:
        pass
    return None


def _extract_memory_limit_tag(raw_parts: list[str]) -> str | None:
    """Parse get_pod_resources output and return 'memory_limit=VALUE' for the first container.

    Returns None if no resource data is available.
    The returned tag is appended to error_summary so the researcher prompt
    can reliably distinguish existing-limit (use factor=) from no-limit (use initial_limit=).
    """
    for part in raw_parts:
        if "[get_pod_resources]" not in part:
            continue
        match = re.search(r"memory_limit:\s*(\S+)", part)
        if match:
            return f"memory_limit={match.group(1)}"
    return None


def _invoke_tool(
    tool_name: str, tool, namespace: str, service: str, pod: str, container_hint: str, workload: str
) -> str:
    """Invoke a tool with appropriate arguments based on its name."""
    # Always qualify pod with namespace so _parse_pod_ref doesn't default to "default"
    qualified_pod = f"{namespace}/{pod}" if "/" not in pod else pod
    if tool_name == "get_cluster_events":
        return tool.invoke({"namespace": namespace, "service": workload})
    if tool_name == "list_pods":
        return tool.invoke({"namespace": namespace, "name_prefix": workload})
    if tool_name == "list_pod_containers":
        result = tool.invoke({"pod_id": qualified_pod})
        if "[ERROR] Pod not found" in result:
            current = _find_current_pod(namespace, workload)
            if current:
                return tool.invoke({"pod_id": f"{namespace}/{current}"})
        return result
    if tool_name == "fetch_container_logs":
        # If the alerted pod is gone, find the replacement.
        # _resolve_container_name calls list_pod_containers — reuse the result
        # to avoid calling it twice with the same pod_id.
        effective_pod = qualified_pod
        container = _resolve_container_name(qualified_pod, fallback="")
        if not container:
            current = _find_current_pod(namespace, workload)
            if current:
                effective_pod = f"{namespace}/{current}"
                container = _resolve_container_name(effective_pod, fallback=container_hint)
            else:
                container = container_hint
        return tool.invoke({"pod_id": effective_pod, "container": container})
    if tool_name == "get_pod_resources":
        result = tool.invoke({"pod_id": qualified_pod})
        if "[ERROR] Pod not found" in result:
            current = _find_current_pod(namespace, workload)
            if current:
                return tool.invoke({"pod_id": f"{namespace}/{current}"})
        return result
    if tool_name == "get_system_metrics":
        return tool.invoke({"service_name": workload, "namespace": namespace})
    if tool_name == "restart_service":
        return tool.invoke({"service_id": workload, "namespace": namespace})
    if tool_name == "execute_rollback":
        return tool.invoke({"deployment_name": workload, "namespace": namespace})
    if tool_name == "scale_memory":
        return tool.invoke({"workload_name": workload, "namespace": namespace})
    raise ValueError(f"No argument mapping defined for tool: {tool_name!r}")
