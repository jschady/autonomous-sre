"""Shared pytest fixtures for the Autonomous SRE test suite."""
from __future__ import annotations

import os
import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock
from httpx import AsyncClient, ASGITransport

# Disable LangSmith tracing globally in tests to prevent network hangs
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")

import app.tools.k8s_tools as k8s_tools
from app.tools.k8s_tools import EXECUTED_ACTIONS

SAMPLE_ALERT = {
    "alertname": "PodCrashLooping",
    "status": "firing",
    "labels": {
        "region": "us-east-1",
        "env": "prod",
        "cluster_id": "k8s-prod-1",
        "namespace": "checkout",
        "service": "checkout-api",
        "pod": "checkout-api-7d9f8b-xkj2p",
    },
    "annotations": {
        "summary": "Pod checkout-api-7d9f8b-xkj2p is crash looping",
        "description": "Pod has restarted 8 times in the last 10 minutes",
    },
}


@pytest.fixture(autouse=True)
def reset_mock_state():
    """Reset module-level mock state before each test."""
    EXECUTED_ACTIONS.clear()
    k8s_tools.MOCK_HEALTHY = False
    yield


@pytest.fixture(autouse=True)
def mock_k8s_apis():
    """Auto-mock the Kubernetes API clients to prevent real network calls.

    Tests that need specific behaviour should override _get_core_v1 or
    _get_apps_v1 with their own patches inside the test body.
    """
    mock_apps = MagicMock()
    mock_apps.patch_namespaced_deployment.return_value = MagicMock()

    def _log_side_effect(name="", namespace="default", container="", tail_lines=100, **kwargs):
        """Return different log content based on pod name for realistic test isolation."""
        pod_lower = name.lower()
        if pod_lower.startswith("crash"):
            return "ERROR Failed to connect\nFATAL Application startup failed\n"
        if pod_lower.startswith("oom"):
            return "ERROR OutOfMemoryError: Java heap space\nFATAL Container OOMKilled\n"
        return f"INFO mock log for pod={name} container={container}\n"

    mock_core = MagicMock()
    mock_core.list_namespaced_event.return_value = MagicMock(items=[])
    mock_core.read_namespaced_pod_log.side_effect = _log_side_effect

    with patch("app.tools.k8s_tools._get_apps_v1", return_value=mock_apps):
        with patch("app.tools.k8s_tools._get_core_v1", return_value=mock_core):
            yield mock_apps, mock_core


@pytest.fixture
def sample_alert():
    return SAMPLE_ALERT.copy()


@pytest.fixture
def mock_llm():
    """Patch ChatAnthropic to avoid real API calls."""
    with patch("langchain_anthropic.ChatAnthropic") as mock:
        yield mock


@pytest_asyncio.fixture
async def test_client():
    from app.main import app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client
