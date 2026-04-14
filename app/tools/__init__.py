"""Tool registry for the Autonomous SRE system."""
from app.tools.k8s_tools import (
    get_cluster_events,
    fetch_container_logs,
    get_system_metrics,
    restart_service,
    execute_rollback,
)

# query_knowledge_base is intentionally excluded from the registry.
# The researcher node calls search_sops() directly with full alert context,
# which produces far better matches than the bare service-name query the
# processor would issue. Keeping it in the registry caused triage to select
# it for the processor, resulting in misleading "No matching SOPs" output.

ALL_TOOLS = [
    get_cluster_events,
    fetch_container_logs,
    get_system_metrics,
    restart_service,
    execute_rollback,
]

TOOL_REGISTRY = {tool.name: tool for tool in ALL_TOOLS}
