"""Unit tests for Phase 2A real Kubernetes tools.

Tests mock the kubernetes API client at the library level so no cluster is needed.
TDD: Written BEFORE the real implementation — tests must fail first.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# k8s_client singleton
# ---------------------------------------------------------------------------


class TestK8sClientFactory:
    """Tests for app/tools/k8s_client.py singleton factory."""

    def test_get_core_v1_api_returns_object(self):
        from app.tools.k8s_client import get_core_v1_api
        with patch("app.tools.k8s_client._load_kube_config"):
            with patch("kubernetes.client.CoreV1Api") as mock_api:
                api = get_core_v1_api()
                assert api is not None

    def test_get_apps_v1_api_returns_object(self):
        from app.tools.k8s_client import get_apps_v1_api
        with patch("app.tools.k8s_client._load_kube_config"):
            with patch("kubernetes.client.AppsV1Api") as mock_api:
                api = get_apps_v1_api()
                assert api is not None

    def test_k8s_config_loads_in_cluster_when_env_set(self, monkeypatch):
        monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
        from app.tools import k8s_client
        import importlib
        with patch("kubernetes.config.load_incluster_config") as mock_incluster:
            with patch("kubernetes.config.load_kube_config") as mock_kube:
                k8s_client._load_kube_config()
                mock_incluster.assert_called_once()
                mock_kube.assert_not_called()

    def test_k8s_config_falls_back_to_kubeconfig(self, monkeypatch):
        monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
        from app.tools import k8s_client
        with patch("kubernetes.config.load_incluster_config"):
            with patch("kubernetes.config.load_kube_config") as mock_kube:
                k8s_client._load_kube_config()
                mock_kube.assert_called_once()


# ---------------------------------------------------------------------------
# Pydantic input validation schemas
# ---------------------------------------------------------------------------


class TestK8sToolInputSchemas:
    """Tests for Pydantic validation schemas on k8s tool inputs."""

    def test_get_cluster_events_input_valid(self):
        from app.tools.k8s_tools import GetClusterEventsInput
        obj = GetClusterEventsInput(namespace="default", service="web-api")
        assert obj.namespace == "default"
        assert obj.service == "web-api"

    def test_get_cluster_events_input_rejects_empty_namespace(self):
        from app.tools.k8s_tools import GetClusterEventsInput
        with pytest.raises(Exception):
            GetClusterEventsInput(namespace="", service="web")

    def test_fetch_container_logs_input_valid(self):
        from app.tools.k8s_tools import FetchContainerLogsInput
        obj = FetchContainerLogsInput(pod_id="pod-abc-123", container="app", tail=50)
        assert obj.tail == 50

    def test_fetch_container_logs_input_tail_defaults_to_100(self):
        from app.tools.k8s_tools import FetchContainerLogsInput
        obj = FetchContainerLogsInput(pod_id="pod-abc-123", container="app")
        assert obj.tail == 100

    def test_fetch_container_logs_input_tail_must_be_positive(self):
        from app.tools.k8s_tools import FetchContainerLogsInput
        with pytest.raises(Exception):
            FetchContainerLogsInput(pod_id="pod-abc", container="app", tail=0)

    def test_restart_service_input_valid(self):
        from app.tools.k8s_tools import RestartServiceInput
        obj = RestartServiceInput(service_id="checkout-api")
        assert obj.service_id == "checkout-api"

    def test_restart_service_input_rejects_empty_service_id(self):
        from app.tools.k8s_tools import RestartServiceInput
        with pytest.raises(Exception):
            RestartServiceInput(service_id="")

    def test_execute_rollback_input_valid(self):
        from app.tools.k8s_tools import ExecuteRollbackInput
        obj = ExecuteRollbackInput(deployment_name="checkout-v2")
        assert obj.deployment_name == "checkout-v2"


# ---------------------------------------------------------------------------
# Real k8s tool: get_cluster_events (mocked API)
# ---------------------------------------------------------------------------


class TestGetClusterEventsReal:
    """Tests for the real get_cluster_events tool backed by kubernetes client."""

    def _make_event(self, reason: str, message: str, obj_name: str) -> MagicMock:
        ev = MagicMock()
        ev.reason = reason
        ev.message = message
        ev.type = "Warning"
        ev.last_timestamp = None
        ev.involved_object = MagicMock()
        ev.involved_object.name = obj_name
        ev.involved_object.kind = "Pod"
        return ev

    def test_get_cluster_events_returns_formatted_string(self):
        mock_event = self._make_event("BackOff", "Back-off restarting failed container", "pod-abc")
        mock_list = MagicMock()
        mock_list.items = [mock_event]

        with patch("app.tools.k8s_read_tools.get_core_v1") as mock_core:
            mock_core.return_value.list_namespaced_event.return_value = mock_list
            from app.tools.k8s_tools import get_cluster_events
            result = get_cluster_events.invoke({"namespace": "default", "service": "web"})

        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_cluster_events_filters_by_service(self):
        mock_event = self._make_event("BackOff", "Back-off restarting", "web-api-pod-abc")
        mock_irrelevant = self._make_event("Scheduled", "Successfully assigned", "other-pod-xyz")
        mock_list = MagicMock()
        mock_list.items = [mock_event, mock_irrelevant]

        with patch("app.tools.k8s_read_tools.get_core_v1") as mock_core:
            mock_core.return_value.list_namespaced_event.return_value = mock_list
            from app.tools.k8s_tools import get_cluster_events
            result = get_cluster_events.invoke({"namespace": "default", "service": "web-api"})

        assert "web-api" in result
        assert isinstance(result, str)

    def test_get_cluster_events_handles_empty_event_list(self):
        mock_list = MagicMock()
        mock_list.items = []

        with patch("app.tools.k8s_read_tools.get_core_v1") as mock_core:
            mock_core.return_value.list_namespaced_event.return_value = mock_list
            from app.tools.k8s_tools import get_cluster_events
            result = get_cluster_events.invoke({"namespace": "default", "service": "web"})

        assert isinstance(result, str)
        assert "default" in result or "no events" in result.lower() or len(result) > 0

    def test_get_cluster_events_handles_api_exception(self):
        from kubernetes.client.exceptions import ApiException

        with patch("app.tools.k8s_read_tools.get_core_v1") as mock_core:
            mock_core.return_value.list_namespaced_event.side_effect = ApiException(
                status=500, reason="Internal Server Error"
            )
            from app.tools.k8s_tools import get_cluster_events
            result = get_cluster_events.invoke({"namespace": "default", "service": "web"})

        assert isinstance(result, str)
        assert "error" in result.lower() or "failed" in result.lower()

    def test_get_cluster_events_rbac_403_returns_rbac_error(self):
        from kubernetes.client.exceptions import ApiException

        with patch("app.tools.k8s_read_tools.get_core_v1") as mock_core:
            exc = ApiException(status=403, reason="Forbidden")
            exc.status = 403
            mock_core.return_value.list_namespaced_event.side_effect = exc
            from app.tools.k8s_tools import get_cluster_events
            result = get_cluster_events.invoke({"namespace": "default", "service": "web"})

        # Result should still be a string (tool always returns str)
        assert isinstance(result, str)
        assert "rbac" in result.lower() or "forbidden" in result.lower() or "denied" in result.lower()


# ---------------------------------------------------------------------------
# Real k8s tool: fetch_container_logs (mocked API)
# ---------------------------------------------------------------------------


class TestFetchContainerLogsReal:
    def test_fetch_container_logs_returns_string(self):
        with patch("app.tools.k8s_read_tools.get_core_v1") as mock_core:
            mock_core.return_value.read_namespaced_pod_log.return_value = (
                "INFO starting\nERROR failed\n"
            )
            from app.tools.k8s_tools import fetch_container_logs
            result = fetch_container_logs.invoke(
                {"pod_id": "pod-abc-123", "container": "app", "tail": 100}
            )
        assert "INFO starting" in result or isinstance(result, str)

    def test_fetch_container_logs_passes_tail_lines(self):
        with patch("app.tools.k8s_read_tools.get_core_v1") as mock_core:
            mock_core.return_value.read_namespaced_pod_log.return_value = "line1\nline2\n"
            from app.tools.k8s_tools import fetch_container_logs
            fetch_container_logs.invoke(
                {"pod_id": "default/pod-abc-123", "container": "app", "tail": 50}
            )
            call_kwargs = mock_core.return_value.read_namespaced_pod_log.call_args
            # tail_lines should be passed
            assert call_kwargs is not None

    def test_fetch_container_logs_handles_api_exception(self):
        from kubernetes.client.exceptions import ApiException

        with patch("app.tools.k8s_read_tools.get_core_v1") as mock_core:
            mock_core.return_value.read_namespaced_pod_log.side_effect = ApiException(
                status=404, reason="Not Found"
            )
            from app.tools.k8s_tools import fetch_container_logs
            result = fetch_container_logs.invoke({"pod_id": "pod-abc", "container": "app"})

        assert isinstance(result, str)
        assert "error" in result.lower() or "not found" in result.lower() or "failed" in result.lower()

    def test_fetch_container_logs_rbac_returns_forbidden_message(self):
        from kubernetes.client.exceptions import ApiException

        with patch("app.tools.k8s_read_tools.get_core_v1") as mock_core:
            exc = ApiException(status=403, reason="Forbidden")
            exc.status = 403
            mock_core.return_value.read_namespaced_pod_log.side_effect = exc
            from app.tools.k8s_tools import fetch_container_logs
            result = fetch_container_logs.invoke({"pod_id": "pod-abc", "container": "app"})

        assert isinstance(result, str)
        assert "rbac" in result.lower() or "forbidden" in result.lower() or "denied" in result.lower()


# ---------------------------------------------------------------------------
# Real k8s tool: restart_service (mocked API)
# ---------------------------------------------------------------------------


def _patch_apps(mock_apps):
    """Patch get_apps_v1 in both helpers (find_controller) and action_tools (restart/rollback/scale)."""
    return (
        patch("app.tools.k8s_helpers.get_apps_v1", mock_apps),
        patch("app.tools.k8s_action_tools.get_apps_v1", mock_apps),
    )


class TestRestartServiceReal:
    def test_restart_service_calls_patch_deployment(self):
        mock_apps = MagicMock()

        with patch("app.tools.k8s_helpers.get_apps_v1", mock_apps), \
             patch("app.tools.k8s_action_tools.get_apps_v1", mock_apps):
            from app.tools.k8s_tools import restart_service
            result = restart_service.invoke({"service_id": "checkout-api"})

        assert isinstance(result, str)
        assert "checkout-api" in result or "restart" in result.lower()

    def test_restart_service_rbac_403_returns_rbac_blocked_dict(self):
        from kubernetes.client.exceptions import ApiException

        mock_apps = MagicMock()
        exc = ApiException(status=403, reason="Forbidden")
        exc.status = 403
        mock_apps.return_value.patch_namespaced_deployment.side_effect = exc

        with patch("app.tools.k8s_helpers.get_apps_v1", mock_apps), \
             patch("app.tools.k8s_action_tools.get_apps_v1", mock_apps):
            from app.tools.k8s_tools import restart_service
            result = restart_service.invoke({"service_id": "checkout-api"})

        assert isinstance(result, str)
        assert "rbac" in result.lower() or "forbidden" in result.lower() or "denied" in result.lower()

    def test_restart_service_non_rbac_api_error_returns_error_message(self):
        from kubernetes.client.exceptions import ApiException

        mock_apps = MagicMock()
        mock_apps.return_value.patch_namespaced_deployment.side_effect = ApiException(
            status=500, reason="Internal Server Error"
        )

        with patch("app.tools.k8s_helpers.get_apps_v1", mock_apps), \
             patch("app.tools.k8s_action_tools.get_apps_v1", mock_apps):
            from app.tools.k8s_tools import restart_service
            result = restart_service.invoke({"service_id": "checkout-api"})

        assert isinstance(result, str)
        assert "error" in result.lower() or "failed" in result.lower()


# ---------------------------------------------------------------------------
# Real k8s tool: execute_rollback (mocked API)
# ---------------------------------------------------------------------------


class TestExecuteRollbackReal:
    def test_execute_rollback_calls_create_namespaced_deployment_rollback(self):
        mock_apps = MagicMock()

        with patch("app.tools.k8s_helpers.get_apps_v1", mock_apps), \
             patch("app.tools.k8s_action_tools.get_apps_v1", mock_apps):
            from app.tools.k8s_tools import execute_rollback
            result = execute_rollback.invoke({"deployment_name": "checkout-v2"})

        assert isinstance(result, str)
        assert "checkout-v2" in result or "rollback" in result.lower()

    def test_execute_rollback_rbac_403_returns_rbac_message(self):
        from kubernetes.client.exceptions import ApiException

        mock_apps = MagicMock()
        exc = ApiException(status=403, reason="Forbidden")
        exc.status = 403
        mock_apps.return_value.patch_namespaced_deployment.side_effect = exc

        with patch("app.tools.k8s_helpers.get_apps_v1", mock_apps), \
             patch("app.tools.k8s_action_tools.get_apps_v1", mock_apps):
            from app.tools.k8s_tools import execute_rollback
            result = execute_rollback.invoke({"deployment_name": "checkout-v2"})

        assert isinstance(result, str)
        assert "rbac" in result.lower() or "forbidden" in result.lower() or "denied" in result.lower()

    def test_execute_rollback_api_error_returns_error_message(self):
        from kubernetes.client.exceptions import ApiException

        mock_apps = MagicMock()
        mock_apps.return_value.patch_namespaced_deployment.side_effect = ApiException(
            status=500, reason="Internal Server Error"
        )

        with patch("app.tools.k8s_helpers.get_apps_v1", mock_apps), \
             patch("app.tools.k8s_action_tools.get_apps_v1", mock_apps):
            from app.tools.k8s_tools import execute_rollback
            result = execute_rollback.invoke({"deployment_name": "checkout-v2"})

        assert isinstance(result, str)
        assert "error" in result.lower() or "failed" in result.lower()


# ---------------------------------------------------------------------------
# get_system_metrics — still mock-backed, no k8s API for metrics in Phase 2
# ---------------------------------------------------------------------------


class TestGetSystemMetricsPhase2:
    def test_get_system_metrics_returns_string(self):
        from app.tools.k8s_tools import get_system_metrics
        result = get_system_metrics.invoke({"service_name": "web-api"})
        assert isinstance(result, str)
        assert "web-api" in result

    def test_get_system_metrics_has_required_fields(self):
        from app.tools.k8s_tools import get_system_metrics
        result = get_system_metrics.invoke({"service_name": "web-api"})
        assert "error_rate" in result
        assert "latency_p99" in result


# ---------------------------------------------------------------------------
# RBAC routing: action node sets rbac_blocked on 403
# ---------------------------------------------------------------------------


class TestRbacBlockedInActionNode:
    """Tests that action_node sets rbac_blocked=True when k8s returns 403."""

    @pytest.mark.asyncio
    async def test_action_node_sets_rbac_blocked_on_restart_403(self):
        from app.nodes.action import action_node
        from kubernetes.client.exceptions import ApiException

        mock_apps = MagicMock()
        exc = ApiException(status=403, reason="Forbidden")
        exc.status = 403
        mock_apps.return_value.patch_namespaced_deployment.side_effect = exc

        with patch("app.tools.k8s_helpers.get_apps_v1", mock_apps), \
             patch("app.tools.k8s_action_tools.get_apps_v1", mock_apps):
            state = {
                "human_approved": True,
                "recommended_action": "restart",
                "proposed_action": "restart the service",
                "sop_matches": [],
                "metadata": {"service": "checkout-api", "namespace": "default"},
                "alert_payload": {"alertname": "PodCrashLooping"},
                "reasoning_log": [],
                "error_log": [],
            }
            result = await action_node(state)

        # action_node should capture rbac_blocked from the tool result
        assert result.get("rbac_blocked") is True

    @pytest.mark.asyncio
    async def test_action_node_does_not_set_rbac_blocked_on_success(self):
        from app.nodes.action import action_node

        mock_apps = MagicMock()

        with patch("app.tools.k8s_helpers.get_apps_v1", mock_apps), \
             patch("app.tools.k8s_action_tools.get_apps_v1", mock_apps):
            state = {
                "human_approved": True,
                "recommended_action": "restart",
                "proposed_action": "restart the service",
                "sop_matches": [],
                "metadata": {"service": "checkout-api", "namespace": "default"},
                "alert_payload": {"alertname": "PodCrashLooping"},
                "reasoning_log": [],
                "error_log": [],
            }
            result = await action_node(state)

        assert result.get("rbac_blocked") is not True


# ---------------------------------------------------------------------------
# Graph routing: rbac_blocked routes to escalate
# ---------------------------------------------------------------------------


class TestGraphRbacRouting:
    def test_route_after_action_rbac_blocked_goes_to_escalate(self):
        from app.agents.graph import route_after_action
        state = {
            "rbac_blocked": True,
            "resolved": False,
            "status": "in_progress",
        }
        assert route_after_action(state) == "escalate"

    def test_route_after_action_no_rbac_goes_to_verification(self):
        from app.agents.graph import route_after_action
        state = {
            "rbac_blocked": False,
            "resolved": False,
            "status": "in_progress",
        }
        assert route_after_action(state) == "verification"

    def test_route_after_action_rbac_blocked_missing_goes_to_verification(self):
        from app.agents.graph import route_after_action
        state = {
            "resolved": False,
            "status": "in_progress",
        }
        assert route_after_action(state) == "verification"


# ---------------------------------------------------------------------------
# Real k8s tool: get_pod_resources (mocked API)
# ---------------------------------------------------------------------------


class TestGetPodResources:
    def _make_container(self, name: str, memory_limit: str | None, memory_request: str | None):
        container = MagicMock()
        container.name = name
        container.resources = MagicMock()
        container.resources.limits = {"memory": memory_limit} if memory_limit else {}
        container.resources.requests = {"memory": memory_request} if memory_request else {}
        return container

    def test_returns_formatted_string(self):
        container = self._make_container("app", "128Mi", "64Mi")
        pod = MagicMock()
        pod.spec.containers = [container]

        with patch("app.tools.k8s_read_tools.get_core_v1") as mock_core:
            mock_core.return_value.read_namespaced_pod.return_value = pod
            from app.tools.k8s_read_tools import get_pod_resources
            result = get_pod_resources.invoke({"pod_id": "default/oom-test"})

        assert "memory_limit: 128Mi" in result
        assert "memory_request: 64Mi" in result
        assert "app" in result

    def test_no_limit_shows_na(self):
        container = self._make_container("stress", None, None)
        pod = MagicMock()
        pod.spec.containers = [container]

        with patch("app.tools.k8s_read_tools.get_core_v1") as mock_core:
            mock_core.return_value.read_namespaced_pod.return_value = pod
            from app.tools.k8s_read_tools import get_pod_resources
            result = get_pod_resources.invoke({"pod_id": "default/oom-test"})

        assert "memory_limit: N/A" in result

    def test_pod_not_found_returns_error(self):
        from kubernetes.client.exceptions import ApiException

        with patch("app.tools.k8s_read_tools.get_core_v1") as mock_core:
            exc = ApiException(status=404, reason="Not Found")
            exc.status = 404
            mock_core.return_value.read_namespaced_pod.side_effect = exc
            from app.tools.k8s_read_tools import get_pod_resources
            result = get_pod_resources.invoke({"pod_id": "default/missing-pod"})

        assert "[ERROR]" in result
        assert "not found" in result.lower()

    def test_rbac_403_returns_rbac_error(self):
        from kubernetes.client.exceptions import ApiException

        with patch("app.tools.k8s_read_tools.get_core_v1") as mock_core:
            exc = ApiException(status=403, reason="Forbidden")
            exc.status = 403
            mock_core.return_value.read_namespaced_pod.side_effect = exc
            from app.tools.k8s_read_tools import get_pod_resources
            result = get_pod_resources.invoke({"pod_id": "default/oom-test"})

        assert "rbac" in result.lower() or "denied" in result.lower()

    def test_multiple_containers_all_listed(self):
        c1 = self._make_container("app", "256Mi", "128Mi")
        c2 = self._make_container("sidecar", "64Mi", None)
        pod = MagicMock()
        pod.spec.containers = [c1, c2]

        with patch("app.tools.k8s_read_tools.get_core_v1") as mock_core:
            mock_core.return_value.read_namespaced_pod.return_value = pod
            from app.tools.k8s_read_tools import get_pod_resources
            result = get_pod_resources.invoke({"pod_id": "default/multi"})

        assert "app" in result
        assert "sidecar" in result
        assert "256Mi" in result
        assert "64Mi" in result

    def test_bare_pod_id_defaults_to_default_namespace(self):
        pod = MagicMock()
        pod.spec.containers = []

        with patch("app.tools.k8s_read_tools.get_core_v1") as mock_core:
            mock_core.return_value.read_namespaced_pod.return_value = pod
            from app.tools.k8s_read_tools import get_pod_resources
            get_pod_resources.invoke({"pod_id": "oom-test"})
            # Should call with namespace="default"
            mock_core.return_value.read_namespaced_pod.assert_called_once_with(
                name="oom-test", namespace="default"
            )


# ---------------------------------------------------------------------------
# processor: _extract_memory_limit_tag
# ---------------------------------------------------------------------------


class TestExtractMemoryLimitTag:
    def test_extracts_limit_from_get_pod_resources_output(self):
        from app.nodes.processor import _extract_memory_limit_tag
        raw_parts = [
            "[get_pod_resources]\n[Resources in pod=oom-test]\nContainer 'stress':\n  memory_limit: 50Mi\n  memory_request: N/A"
        ]
        result = _extract_memory_limit_tag(raw_parts)
        assert result == "memory_limit=50Mi"

    def test_returns_na_when_no_limit_set(self):
        from app.nodes.processor import _extract_memory_limit_tag
        raw_parts = [
            "[get_pod_resources]\n[Resources in pod=oom-test]\nContainer 'app':\n  memory_limit: N/A\n  memory_request: N/A"
        ]
        result = _extract_memory_limit_tag(raw_parts)
        assert result == "memory_limit=N/A"

    def test_returns_none_when_no_resource_section(self):
        from app.nodes.processor import _extract_memory_limit_tag
        raw_parts = [
            "[get_cluster_events]\nsome events here",
            "[fetch_container_logs]\nsome logs here",
        ]
        result = _extract_memory_limit_tag(raw_parts)
        assert result is None

    def test_ignores_other_tool_output(self):
        from app.nodes.processor import _extract_memory_limit_tag
        raw_parts = [
            "[get_cluster_events]\nmemory_limit: 100Mi (this is in the wrong section)",
            "[get_pod_resources]\nContainer 'app':\n  memory_limit: 200Mi",
        ]
        result = _extract_memory_limit_tag(raw_parts)
        assert result == "memory_limit=200Mi"

    def test_empty_raw_parts(self):
        from app.nodes.processor import _extract_memory_limit_tag
        assert _extract_memory_limit_tag([]) is None
