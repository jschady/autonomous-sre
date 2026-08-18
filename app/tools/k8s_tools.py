"""Kubernetes tools — re-export facade.

All tools and helpers live in k8s_helpers, k8s_read_tools, and k8s_action_tools.
This module re-exports the public API so existing imports continue to work.
"""
# Helpers & shared state (used by tests/conftest)
from app.tools.k8s_helpers import (  # noqa: F401
    EXECUTED_ACTIONS,
    ExecuteRollbackInput,
    FetchContainerLogsInput,
    GetClusterEventsInput,
    GetSystemMetricsInput,
    RestartServiceInput,
    ScaleMemoryInput,
    is_mock_healthy,
    set_mock_healthy,
)

# Read-only tools
from app.tools.k8s_read_tools import (  # noqa: F401
    fetch_container_logs,
    get_cluster_events,
    get_pod_resources,
    get_system_metrics,
    list_pod_containers,
    list_pods,
)

# Mutating tools
from app.tools.k8s_action_tools import (  # noqa: F401
    execute_rollback,
    restart_service,
    scale_memory,
)
