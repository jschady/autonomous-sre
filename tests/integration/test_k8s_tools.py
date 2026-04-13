"""Integration tests for Kubernetes tools against a real cluster.

Skipped unless RUN_INTEGRATION=1 is set.
Requires:
  - A reachable Kubernetes cluster (KUBECONFIG or in-cluster)
  - Sufficient RBAC permissions
"""
from __future__ import annotations

import os

import pytest

RUN_INTEGRATION = os.environ.get("RUN_INTEGRATION", "0") == "1"
skip_unless_integration = pytest.mark.skipif(
    not RUN_INTEGRATION, reason="Set RUN_INTEGRATION=1 to run integration tests"
)


@skip_unless_integration
class TestK8sToolsIntegration:
    def test_get_cluster_events_default_namespace(self):
        from app.tools.k8s_tools import get_cluster_events
        result = get_cluster_events.invoke({"namespace": "default", "service": "kubernetes"})
        assert isinstance(result, str)
        # Should return events or "no events" — either is valid
        assert len(result) > 0

    def test_get_cluster_events_kube_system(self):
        from app.tools.k8s_tools import get_cluster_events
        result = get_cluster_events.invoke({"namespace": "kube-system", "service": "coredns"})
        assert isinstance(result, str)

    def test_fetch_container_logs_nonexistent_pod_returns_error(self):
        """Fetching logs for a non-existent pod should return a helpful error message."""
        from app.tools.k8s_tools import fetch_container_logs
        result = fetch_container_logs.invoke(
            {"pod_id": "default/nonexistent-pod-xyz-123", "container": "app"}
        )
        assert isinstance(result, str)
        assert "error" in result.lower() or "not found" in result.lower()
