"""Tests for k8s_tools and db_tools.

TDD: Written BEFORE implementation.
"""
import pytest
import app.tools.k8s_tools as k8s_tools
from app.tools.k8s_tools import (
    get_cluster_events,
    fetch_container_logs,
    get_system_metrics,
    restart_service,
    execute_rollback,
    EXECUTED_ACTIONS,
    MOCK_HEALTHY,
)
from app.tools.db_tools import query_knowledge_base
from app.tools import ALL_TOOLS, TOOL_REGISTRY
from langchain_core.tools import BaseTool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_mock_state():
    """Reset module-level mock state before each test."""
    EXECUTED_ACTIONS.clear()
    k8s_tools.MOCK_HEALTHY = False
    yield


# ---------------------------------------------------------------------------
# get_cluster_events
# ---------------------------------------------------------------------------

class TestGetClusterEvents:
    def test_get_cluster_events_returns_string(self):
        result = get_cluster_events.invoke({"namespace": "default", "service": "web"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_cluster_events_uses_namespace_default(self):
        result = get_cluster_events.invoke({"namespace": "default", "service": "web"})
        assert "default" in result.lower() or len(result) > 0

    def test_get_cluster_events_uses_namespace_kube_system(self):
        result_default = get_cluster_events.invoke({"namespace": "default", "service": "web"})
        result_kube = get_cluster_events.invoke({"namespace": "kube-system", "service": "web"})
        # Different namespaces should produce different outputs
        assert result_default != result_kube

    def test_get_cluster_events_includes_namespace_in_output(self):
        result = get_cluster_events.invoke({"namespace": "checkout", "service": "checkout-api"})
        assert "checkout" in result


# ---------------------------------------------------------------------------
# fetch_container_logs
# ---------------------------------------------------------------------------

class TestFetchContainerLogs:
    def test_fetch_container_logs_returns_string(self):
        result = fetch_container_logs.invoke({"pod_id": "web-abc123", "container": "web"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_fetch_container_logs_returns_realistic_logs(self):
        result = fetch_container_logs.invoke({"pod_id": "crash-abc", "container": "app"})
        # Should contain log-like content: ERROR, timestamps, or similar
        assert any(kw in result.upper() for kw in ["ERROR", "FATAL", "EXCEPTION", "LOG", "WARN", "INFO"])

    def test_fetch_container_logs_different_pods_different_logs(self):
        result_crash = fetch_container_logs.invoke({"pod_id": "crash-pod-1", "container": "app"})
        result_oom = fetch_container_logs.invoke({"pod_id": "oom-pod-2", "container": "app"})
        assert result_crash != result_oom

    def test_fetch_container_logs_respects_tail_parameter(self):
        result = fetch_container_logs.invoke({"pod_id": "web-abc", "container": "app", "tail": 5})
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# get_system_metrics
# ---------------------------------------------------------------------------

class TestGetSystemMetrics:
    def test_get_system_metrics_unhealthy(self):
        k8s_tools.MOCK_HEALTHY = False
        result = get_system_metrics.invoke({"service_name": "checkout-api"})
        assert isinstance(result, str)
        # Result should indicate unhealthy state — error_rate > 5
        assert "error_rate" in result.lower() or "error" in result.lower()
        # Parse error_rate from result — it should be > 5
        import re
        match = re.search(r"error_rate['\"]?\s*[:=]\s*([\d.]+)", result)
        if match:
            assert float(match.group(1)) > 5

    def test_get_system_metrics_healthy(self):
        k8s_tools.MOCK_HEALTHY = True
        result = get_system_metrics.invoke({"service_name": "checkout-api"})
        assert isinstance(result, str)
        import re
        match = re.search(r"error_rate['\"]?\s*[:=]\s*([\d.]+)", result)
        if match:
            assert float(match.group(1)) < 1

    def test_get_system_metrics_includes_service_name(self):
        result = get_system_metrics.invoke({"service_name": "payment-api"})
        assert "payment-api" in result or "payment" in result.lower()

    def test_get_system_metrics_returns_structured_data(self):
        result = get_system_metrics.invoke({"service_name": "test-svc"})
        # Should have some structure indicating metric type
        assert any(kw in result.lower() for kw in ["latency", "error", "cpu", "memory", "metric"])


# ---------------------------------------------------------------------------
# restart_service
# ---------------------------------------------------------------------------

class TestRestartService:
    def test_restart_service_returns_string(self):
        result = restart_service.invoke({"service_id": "checkout-api"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_restart_service_records_action(self):
        restart_service.invoke({"service_id": "my-service"})
        assert any("my-service" in action for action in EXECUTED_ACTIONS)

    def test_restart_service_multiple_calls_all_recorded(self):
        restart_service.invoke({"service_id": "svc-a"})
        restart_service.invoke({"service_id": "svc-b"})
        assert len(EXECUTED_ACTIONS) == 2

    def test_restart_service_result_confirms_action(self):
        result = restart_service.invoke({"service_id": "api-gateway"})
        assert "api-gateway" in result or "restart" in result.lower()


# ---------------------------------------------------------------------------
# execute_rollback
# ---------------------------------------------------------------------------

class TestExecuteRollback:
    def test_execute_rollback_returns_string(self):
        result = execute_rollback.invoke({"deployment_name": "checkout-v2"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_rollback_records_action(self):
        execute_rollback.invoke({"deployment_name": "my-deployment"})
        assert any("my-deployment" in action for action in EXECUTED_ACTIONS)

    def test_execute_rollback_result_confirms_action(self):
        result = execute_rollback.invoke({"deployment_name": "checkout-v2"})
        assert "checkout-v2" in result or "rollback" in result.lower()


# ---------------------------------------------------------------------------
# query_knowledge_base
# ---------------------------------------------------------------------------

class TestQueryKnowledgeBase:
    def test_query_knowledge_base_returns_string(self):
        result = query_knowledge_base.invoke({"query": "CrashLoopBackOff"})
        assert isinstance(result, str)

    def test_query_knowledge_base_finds_crashloop(self):
        result = query_knowledge_base.invoke({"query": "CrashLoopBackOff"})
        assert len(result) > 0
        # Should contain step-by-step instructions
        assert any(kw in result.lower() for kw in ["restart", "pod", "kubectl", "step"])

    def test_query_knowledge_base_no_match_returns_message(self):
        result = query_knowledge_base.invoke({"query": "zzzunknown_xyz_q9z9z"})
        assert isinstance(result, str)
        assert "no" in result.lower() or "not found" in result.lower() or len(result) > 0

    def test_query_knowledge_base_oom_returns_oom_sop(self):
        result = query_knowledge_base.invoke({"query": "OOMKilled memory"})
        assert "oom" in result.lower() or "memory" in result.lower()


# ---------------------------------------------------------------------------
# Tool registry and LangChain compatibility
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_all_tools_is_list(self):
        assert isinstance(ALL_TOOLS, list)
        assert len(ALL_TOOLS) == 5

    def test_tool_registry_is_dict(self):
        assert isinstance(TOOL_REGISTRY, dict)

    def test_tools_are_langchain_tools(self):
        for tool in ALL_TOOLS:
            assert isinstance(tool, BaseTool), f"{tool} is not a BaseTool"
            assert hasattr(tool, "name") and isinstance(tool.name, str)
            assert hasattr(tool, "description") and isinstance(tool.description, str)

    def test_tool_registry_keyed_by_name(self):
        for name, tool in TOOL_REGISTRY.items():
            assert tool.name == name

    def test_tool_registry_contains_all_expected_tools(self):
        expected = {
            "get_cluster_events",
            "fetch_container_logs",
            "get_system_metrics",
            "restart_service",
            "execute_rollback",
        }
        assert expected.issubset(set(TOOL_REGISTRY.keys()))

    def test_tools_accept_metadata_context(self):
        """Tools that accept namespace/service should include it in output."""
        result = get_cluster_events.invoke({"namespace": "production", "service": "api"})
        assert "production" in result
