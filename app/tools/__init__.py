"""Tool registry for the Autonomous SRE system."""
from app.tools.k8s_tools import (
    get_cluster_events,
    fetch_container_logs,
    get_system_metrics,
    restart_service,
    execute_rollback,
)
from app.tools.db_tools import query_knowledge_base

ALL_TOOLS = [
    get_cluster_events,
    fetch_container_logs,
    get_system_metrics,
    restart_service,
    execute_rollback,
    query_knowledge_base,
]

TOOL_REGISTRY = {tool.name: tool for tool in ALL_TOOLS}
