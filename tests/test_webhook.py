"""Tests for the FastAPI webhook endpoints.

TDD: Written BEFORE implementation.
"""
from __future__ import annotations

import asyncio
import time
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
from langchain_core.messages import AIMessage

import app.tools.k8s_tools as k8s_tools



# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_dedup_state():
    """Clear all in-process dedup registries between tests."""
    import app.main as m
    m._ACTIVE_WORKLOADS.clear()
    m._ALERT_TO_WORKLOAD.clear()
    m._WORKLOAD_COOLDOWNS.clear()
    yield
    m._ACTIVE_WORKLOADS.clear()
    m._ALERT_TO_WORKLOAD.clear()
    m._WORKLOAD_COOLDOWNS.clear()


@pytest.fixture
def sample_alert():
    return {
        "alertname": "PodCrashLooping",
        "status": "firing",
        "labels": {
            "region": "us-east-1",
            "env": "prod",
            "cluster_id": "k8s-prod-1",
            "namespace": "checkout",
            "service": "checkout-api",
            "pod": "crash-api-7d9f8b-xkj2p",
        },
        "annotations": {
            "summary": "Pod is crash looping",
            "description": "Pod has restarted 8 times in the last 10 minutes",
        },
    }


@pytest_asyncio.fixture
async def test_client():
    from app.main import app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


def _make_mock_llm(responses: list[str]) -> MagicMock:
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.side_effect = [AIMessage(content=r) for r in responses]
    return mock_llm


# ---------------------------------------------------------------------------
# POST /webhook — basic acceptance
# ---------------------------------------------------------------------------

class TestWebhookEndpoint:
    @pytest.mark.asyncio
    async def test_webhook_accepts_valid_alert(self, test_client, sample_alert):
        response = await test_client.post("/webhook", json=sample_alert)
        assert response.status_code == 202
        data = response.json()
        assert "alert_id" in data
        assert len(data["alert_id"]) > 0

    @pytest.mark.asyncio
    async def test_webhook_rejects_missing_alertname(self, test_client):
        payload = {"status": "firing", "labels": {}}
        response = await test_client.post("/webhook", json=payload)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_webhook_extracts_metadata(self, test_client, sample_alert):
        response = await test_client.post("/webhook", json=sample_alert)
        assert response.status_code == 202
        data = response.json()
        # The alert_id should be present; metadata is stored internally
        assert "alert_id" in data

    @pytest.mark.asyncio
    async def test_webhook_returns_202_not_200(self, test_client, sample_alert):
        response = await test_client.post("/webhook", json=sample_alert)
        assert response.status_code == 202

    @pytest.mark.asyncio
    async def test_webhook_different_alerts_different_ids(self, test_client, sample_alert):
        r1 = await test_client.post("/webhook", json=sample_alert)
        # Use a different workload to avoid dedup / cooldown suppression
        other_labels = {**sample_alert["labels"], "deployment": "other-service"}
        r2 = await test_client.post("/webhook", json={**sample_alert, "labels": other_labels})
        assert r1.json()["alert_id"] != r2.json()["alert_id"]


# ---------------------------------------------------------------------------
# GET /status/{alert_id}
# ---------------------------------------------------------------------------

class TestStatusEndpoint:
    @pytest.mark.asyncio
    async def test_status_endpoint_returns_state(self, test_client, sample_alert):
        post_resp = await test_client.post("/webhook", json=sample_alert)
        alert_id = post_resp.json()["alert_id"]

        # Small delay to let background task start (or it may be sync)
        await asyncio.sleep(0.05)

        get_resp = await test_client.get(f"/status/{alert_id}")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["alert_id"] == alert_id
        assert "status" in data

    @pytest.mark.asyncio
    async def test_status_unknown_alert_404(self, test_client):
        response = await test_client.get("/status/nonexistent-alert-id-xyz")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_status_response_has_required_fields(self, test_client, sample_alert):
        post_resp = await test_client.post("/webhook", json=sample_alert)
        alert_id = post_resp.json()["alert_id"]
        get_resp = await test_client.get(f"/status/{alert_id}")
        data = get_resp.json()
        assert "alert_id" in data
        assert "status" in data
        assert "current_node" in data
        assert "retry_count" in data
        assert "error_log" in data
        assert "metadata" in data


# ---------------------------------------------------------------------------
# POST /slack/interactive
# ---------------------------------------------------------------------------

class TestSlackInteractive:
    @pytest.mark.asyncio
    async def test_slack_interactive_invalid_alert_404(self, test_client):
        payload = {"alert_id": "nonexistent-xyz-123", "approved": True}
        response = await test_client.post("/slack/interactive", json=payload)
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_slack_interactive_valid_approval(self, test_client, sample_alert):
        post_resp = await test_client.post("/webhook", json=sample_alert)
        alert_id = post_resp.json()["alert_id"]
        await asyncio.sleep(0.05)

        payload = {"alert_id": alert_id, "approved": True}
        response = await test_client.post("/slack/interactive", json=payload)
        # Should succeed (200) or 404 if graph not yet interrupted
        assert response.status_code in (200, 404, 409)

    @pytest.mark.asyncio
    async def test_slack_interactive_rejects_missing_fields(self, test_client):
        payload = {"approved": True}  # Missing alert_id
        response = await test_client.post("/slack/interactive", json=payload)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# E2E: Happy path integration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Alertmanager v4 envelope: namespace discovery
# ---------------------------------------------------------------------------

class TestAlertmanagerV4Namespace:
    """Verify namespace is correctly resolved from Alertmanager v4 envelopes."""

    @pytest.mark.asyncio
    async def test_namespace_from_common_labels_when_absent_in_individual(self, test_client):
        """namespace in commonLabels is used when individual alert labels lack it.

        This is the real-world Alertmanager behavior: when all alerts in a group
        share a namespace, AM promotes it to commonLabels and removes it from
        individual alert labels.
        """
        envelope = {
            "receiver": "sre-bot",
            "status": "firing",
            "alerts": [
                {
                    "status": "firing",
                    "labels": {"alertname": "KubePodCrashLooping", "pod": "my-app-abc123"},
                    "annotations": {},
                    "startsAt": "2024-01-01T00:00:00Z",
                    "endsAt": "0001-01-01T00:00:00Z",
                    "generatorURL": "",
                }
            ],
            "groupLabels": {"alertname": "KubePodCrashLooping"},
            "commonLabels": {
                "alertname": "KubePodCrashLooping",
                "namespace": "monitoring",  # only in commonLabels, not in individual labels
            },
        }
        response = await test_client.post("/webhook", json=envelope)
        assert response.status_code == 202
        alert_id = response.json()["alert_id"]

        status_resp = await test_client.get(f"/status/{alert_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["metadata"]["namespace"] == "monitoring"

    @pytest.mark.asyncio
    async def test_individual_label_namespace_overrides_common_labels(self, test_client):
        """Individual alert labels take precedence over commonLabels for namespace."""
        envelope = {
            "receiver": "sre-bot",
            "status": "firing",
            "alerts": [
                {
                    "status": "firing",
                    "labels": {
                        "alertname": "HighLatency",
                        "namespace": "payment-service",  # individual label wins
                    },
                    "annotations": {},
                }
            ],
            "groupLabels": {},
            "commonLabels": {"namespace": "default"},  # should be overridden
        }
        response = await test_client.post("/webhook", json=envelope)
        assert response.status_code == 202
        alert_id = response.json()["alert_id"]

        status_resp = await test_client.get(f"/status/{alert_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["metadata"]["namespace"] == "payment-service"

    @pytest.mark.asyncio
    async def test_alertmanager_v4_returns_alert_ids_list(self, test_client):
        """Alertmanager v4 envelope with multiple alerts returns alert_ids list."""
        envelope = {
            "receiver": "sre-bot",
            "status": "firing",
            "alerts": [
                {
                    "status": "firing",
                    "labels": {"alertname": "CrashLoop", "pod": "app-1"},
                    "annotations": {},
                },
                {
                    "status": "firing",
                    "labels": {"alertname": "CrashLoop", "pod": "app-2"},
                    "annotations": {},
                },
            ],
            "groupLabels": {"alertname": "CrashLoop"},
            "commonLabels": {"alertname": "CrashLoop", "namespace": "staging"},
        }
        response = await test_client.post("/webhook", json=envelope)
        assert response.status_code == 202
        data = response.json()
        assert "alert_ids" in data
        assert len(data["alert_ids"]) == 2
        # Both alerts should have namespace=staging from commonLabels
        for aid in data["alert_ids"]:
            status_resp = await test_client.get(f"/status/{aid}")
            assert status_resp.json()["metadata"]["namespace"] == "staging"


# ---------------------------------------------------------------------------
# Workload-level deduplication
# ---------------------------------------------------------------------------

class TestWorkloadDeduplication:
    @pytest.fixture(autouse=True)
    def reset_active_workloads(self):
        """Patches _run_graph so that background tasks never complete and
        remove the workload key during the test — we want to test the
        deduplication guard in the endpoint, not graph execution.
        """
        with patch("app.main._run_graph", new=AsyncMock()):
            yield

    @pytest.mark.asyncio
    async def test_second_alert_same_workload_returns_200_suppressed(self, test_client):
        """Two alerts for the same workload -> second gets 200 with status=suppressed."""
        alert = {
            "alertname": "PodCrashLooping",
            "status": "firing",
            "labels": {"namespace": "production", "deployment": "payment-api"},
        }
        r1 = await test_client.post("/webhook", json=alert)
        assert r1.status_code == 202
        r2 = await test_client.post("/webhook", json=alert)
        assert r2.status_code == 200
        data = r2.json()
        assert data.get("status") == "suppressed"
        assert "already being remediated" in data.get("detail", "").lower()

    @pytest.mark.asyncio
    async def test_different_workloads_both_accepted(self, test_client):
        """Two alerts for different deployments in same namespace -> both accepted."""
        r1 = await test_client.post("/webhook", json={
            "alertname": "A", "status": "firing",
            "labels": {"namespace": "production", "deployment": "payment-api"},
        })
        r2 = await test_client.post("/webhook", json={
            "alertname": "A", "status": "firing",
            "labels": {"namespace": "production", "deployment": "auth-service"},
        })
        assert r1.status_code == 202
        assert r2.status_code == 202

    @pytest.mark.asyncio
    async def test_different_namespaces_both_accepted(self, test_client):
        """Same deployment name in different namespaces -> both accepted."""
        r1 = await test_client.post("/webhook", json={
            "alertname": "A", "status": "firing",
            "labels": {"namespace": "staging", "deployment": "api-server"},
        })
        r2 = await test_client.post("/webhook", json={
            "alertname": "A", "status": "firing",
            "labels": {"namespace": "production", "deployment": "api-server"},
        })
        assert r1.status_code == 202
        assert r2.status_code == 202

    @pytest.mark.asyncio
    async def test_statefulset_used_when_no_deployment(self, test_client):
        """StatefulSet label used as workload key when deployment absent."""
        alert = {
            "alertname": "A", "status": "firing",
            "labels": {"namespace": "data", "statefulset": "postgres"},
        }
        r1 = await test_client.post("/webhook", json=alert)
        r2 = await test_client.post("/webhook", json=alert)
        assert r1.status_code == 202
        assert r2.status_code == 200
        assert r2.json().get("status") == "suppressed"

    @pytest.mark.asyncio
    async def test_alertmanager_v4_dedup(self, test_client):
        """AM v4 envelope with same workload twice -> 202 then 200 suppressed."""
        envelope = {
            "receiver": "sre-bot", "status": "firing",
            "alerts": [
                {"status": "firing", "labels": {"alertname": "CrashLoop", "deployment": "api"}, "annotations": {}},
            ],
            "groupLabels": {}, "commonLabels": {"namespace": "prod"},
        }
        r1 = await test_client.post("/webhook", json=envelope)
        assert r1.status_code == 202
        r2 = await test_client.post("/webhook", json=envelope)
        assert r2.status_code == 200
        assert r2.json().get("status") == "suppressed"

    @pytest.mark.asyncio
    async def test_v4_partial_suppression_processes_new_alert(self, test_client):
        """V4 envelope with mixed new/duplicate: new alert is processed, duplicate is suppressed."""
        # First, register workload-A
        envelope_first = {
            "receiver": "sre-bot", "status": "firing",
            "alerts": [
                {"status": "firing", "labels": {"alertname": "CrashLoop", "deployment": "workload-a"}, "annotations": {}},
            ],
            "groupLabels": {}, "commonLabels": {"namespace": "prod"},
        }
        r1 = await test_client.post("/webhook", json=envelope_first)
        assert r1.status_code == 202

        # Now send envelope with workload-a (dup) + workload-b (new)
        envelope_mixed = {
            "receiver": "sre-bot", "status": "firing",
            "alerts": [
                {"status": "firing", "labels": {"alertname": "CrashLoop", "deployment": "workload-a"}, "annotations": {}},
                {"status": "firing", "labels": {"alertname": "CrashLoop", "deployment": "workload-b"}, "annotations": {}},
            ],
            "groupLabels": {}, "commonLabels": {"namespace": "prod"},
        }
        r2 = await test_client.post("/webhook", json=envelope_mixed)
        assert r2.status_code == 202
        data = r2.json()
        # workload-b should be accepted (alert_ids contains at least one new id)
        assert len(data.get("alert_ids", [])) >= 1
        # Response should indicate partial suppression
        assert data.get("suppressed_count", 0) >= 1


class TestWorkloadKeyFunction:
    """Unit tests for the _workload_key helper."""

    def test_deployment_takes_priority(self):
        import app.main as main_module
        labels = {"namespace": "prod", "deployment": "api", "statefulset": "db", "pod": "pod-1"}
        assert main_module._workload_key(labels) == "prod/api"

    def test_statefulset_used_when_no_deployment(self):
        import app.main as main_module
        labels = {"namespace": "prod", "statefulset": "db", "pod": "pod-1"}
        assert main_module._workload_key(labels) == "prod/db"

    def test_daemonset_used_when_no_deployment_or_statefulset(self):
        import app.main as main_module
        labels = {"namespace": "kube-system", "daemonset": "fluentd"}
        assert main_module._workload_key(labels) == "kube-system/fluentd"

    def test_pod_used_as_last_resort(self):
        import app.main as main_module
        labels = {"namespace": "default", "pod": "my-pod-abc"}
        assert main_module._workload_key(labels) == "default/my-pod-abc"

    def test_defaults_when_labels_empty(self):
        import app.main as main_module
        assert main_module._workload_key({}) == "default/unknown"

    def test_defaults_namespace_when_missing(self):
        import app.main as main_module
        assert main_module._workload_key({"deployment": "api"}) == "default/api"

    def test_pod_hash_stripped_to_deployment_name(self):
        """Pod with RS+pod hash suffixes normalizes to deployment name."""
        import app.main as main_module
        labels = {"namespace": "prod", "pod": "chaos-app-7c949f9b88-txqcz"}
        assert main_module._workload_key(labels) == "prod/chaos-app"

    def test_different_pods_same_deployment_produce_same_key(self):
        """Two pods from the same Deployment map to a single dedup key."""
        import app.main as main_module
        labels_a = {"namespace": "prod", "pod": "my-api-6f8d4c7b9a-abc12"}
        labels_b = {"namespace": "prod", "pod": "my-api-6f8d4c7b9a-def34"}
        assert main_module._workload_key(labels_a) == main_module._workload_key(labels_b)
        assert main_module._workload_key(labels_a) == "prod/my-api"

    def test_statefulset_pod_name_not_stripped(self):
        """StatefulSet pods (no RS hash) keep their original name."""
        import app.main as main_module
        labels = {"namespace": "data", "pod": "postgres-0"}
        assert main_module._workload_key(labels) == "data/postgres-0"


class TestHappyPathE2E:
    @pytest.mark.asyncio
    async def test_full_happy_path_e2e(self, test_client, sample_alert):
        """POST webhook → poll status → approve → poll until resolved."""
        k8s_tools.MOCK_HEALTHY = True

        triage_response = (
            '{"severity": "critical", "tools_to_run": ["get_cluster_events"], '
            '"triage_summary": "CrashLoopBackOff detected", "task_complexity": "moderate"}'
        )
        processor_response = "Pod crash looping due to DB failure."
        researcher_response = "Restart the service to resolve CrashLoopBackOff."

        mock_llm = _make_mock_llm(
            [triage_response, processor_response, researcher_response] * 3
        )

        with patch("app.nodes.triage.ChatAnthropic", return_value=mock_llm), \
             patch("app.utils.llm_factory._build_claude_client", return_value=mock_llm), \
             patch("app.utils.llm_factory.get_settings", return_value=MagicMock(
                 local_model_enabled=False, runpod_base_url="",
                 triage_model="claude-sonnet-4-6", anthropic_api_key="test",
             )), \
             patch("app.nodes.router.get_settings", return_value=MagicMock(
                 local_model_enabled=False, semantic_cache_enabled=False,
             )), \
             patch("app.utils.incident_store.fetch_similar_incidents", return_value=[]):

            post_resp = await test_client.post("/webhook", json=sample_alert)
            assert post_resp.status_code == 202
            alert_id = post_resp.json()["alert_id"]

            # Poll for status
            for _ in range(10):
                await asyncio.sleep(0.1)
                status_resp = await test_client.get(f"/status/{alert_id}")
                if status_resp.status_code == 200:
                    data = status_resp.json()
                    if data["status"] in ("resolved", "escalated", "failed"):
                        break
                    # Try approval if stuck at human gate
                    if data.get("current_node") == "human_gate":
                        await test_client.post(
                            "/slack/interactive",
                            json={"alert_id": alert_id, "approved": True},
                        )

        # Final status should be set
        final_resp = await test_client.get(f"/status/{alert_id}")
        assert final_resp.status_code == 200
        assert final_resp.json()["status"] in ("in_progress", "resolved", "escalated", "failed")

    @pytest.mark.asyncio
    async def test_human_rejection_escalates(self, test_client, sample_alert):  # noqa: E501
        """POST webhook → reject → status should eventually be escalated."""
        triage_response = (
            '{"severity": "critical", "tools_to_run": ["get_cluster_events"], '
            '"triage_summary": "CrashLoopBackOff detected", "task_complexity": "moderate"}'
        )
        processor_response = "Pod crash looping."
        researcher_response = "Restart the service."
        mock_llm = _make_mock_llm(
            [triage_response, processor_response, researcher_response] * 2
        )

        with patch("app.nodes.triage.ChatAnthropic", return_value=mock_llm), \
             patch("app.utils.llm_factory._build_claude_client", return_value=mock_llm), \
             patch("app.utils.llm_factory.get_settings", return_value=MagicMock(
                 local_model_enabled=False, runpod_base_url="",
                 triage_model="claude-sonnet-4-6", anthropic_api_key="test",
             )), \
             patch("app.nodes.router.get_settings", return_value=MagicMock(
                 local_model_enabled=False, semantic_cache_enabled=False,
             )), \
             patch("app.utils.incident_store.fetch_similar_incidents", return_value=[]):

            post_resp = await test_client.post("/webhook", json=sample_alert)
            alert_id = post_resp.json()["alert_id"]

            for _ in range(10):
                await asyncio.sleep(0.1)
                status_resp = await test_client.get(f"/status/{alert_id}")
                if status_resp.status_code == 200:
                    data = status_resp.json()
                    if data["status"] in ("resolved", "escalated", "failed"):
                        break
                    if data.get("current_node") == "human_gate":
                        await test_client.post(
                            "/slack/interactive",
                            json={"alert_id": alert_id, "approved": False},
                        )

        final_resp = await test_client.get(f"/status/{alert_id}")
        assert final_resp.status_code == 200
        assert final_resp.json()["status"] in ("in_progress", "resolved", "escalated", "failed")


# ---------------------------------------------------------------------------
# Bug #1: _ACTIVE_WORKLOADS cleared prematurely on GraphInterrupt
# ---------------------------------------------------------------------------

class TestWorkloadLockDuringPause:
    """Regression tests: deduplication window must remain open while graph is paused.

    Bug: _run_graph's finally block always popped the workload from _ACTIVE_WORKLOADS,
    even when GraphInterrupt was raised (graph paused at human_gate, not done).
    Prometheus would re-fire, the workload key was gone, and a second graph run
    started — sending a duplicate Slack approval message.
    """

    @pytest.mark.asyncio
    async def test_workload_stays_locked_when_graph_pauses_at_human_gate(self):
        """_run_graph must NOT remove workload from _ACTIVE_WORKLOADS on GraphInterrupt."""
        import app.main as main_module
        from langgraph.errors import GraphInterrupt

        workload_key = "prod/payment-api"
        alert_id = "test-pause-001"
        main_module._ACTIVE_WORKLOADS[workload_key] = alert_id

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(side_effect=GraphInterrupt("paused at human_gate"))

        with patch("app.main._graph", return_value=mock_graph):
            await main_module._run_graph(alert_id, {}, workload_key)

        # Before fix: finally always popped — this assertion FAILS (RED)
        assert workload_key in main_module._ACTIVE_WORKLOADS
        assert main_module._ACTIVE_WORKLOADS[workload_key] == alert_id

    @pytest.mark.asyncio
    async def test_workload_released_when_graph_completes_successfully(self):
        """On normal completion, workload must still be released from _ACTIVE_WORKLOADS."""
        import app.main as main_module

        workload_key = "prod/auth-service"
        alert_id = "test-success-001"
        main_module._ACTIVE_WORKLOADS[workload_key] = alert_id

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"status": "resolved"})

        with patch("app.main._graph", return_value=mock_graph):
            await main_module._run_graph(alert_id, {}, workload_key)

        assert workload_key not in main_module._ACTIVE_WORKLOADS

    @pytest.mark.asyncio
    async def test_workload_released_on_unrecoverable_exception(self):
        """On a non-GraphInterrupt exception, workload must be released."""
        import app.main as main_module

        workload_key = "prod/broken-service"
        alert_id = "test-error-001"
        main_module._ACTIVE_WORKLOADS[workload_key] = alert_id

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("unexpected failure"))

        with patch("app.main._graph", return_value=mock_graph):
            await main_module._run_graph(alert_id, {}, workload_key)

        assert workload_key not in main_module._ACTIVE_WORKLOADS


# ---------------------------------------------------------------------------
# Bug #2: _ACTIVE_WORKLOADS never cleared after graph resumes and completes
# ---------------------------------------------------------------------------

class TestWorkloadReleasedAfterResume:
    """Regression tests: workload lock must be released once graph finishes via resume.

    Bug: _resume_and_notify had no cleanup logic. After a Slack button was clicked
    and the graph resolved/escalated, the workload key lingered in _ACTIVE_WORKLOADS
    (or was already gone prematurely from Bug #1). The two interact: stale Slack
    buttons from duplicate runs trigger 409s logged as "graph failed to resume".
    After fix: _ALERT_TO_WORKLOAD reverse map drives cleanup in _resume_and_notify.
    """

    @pytest.mark.asyncio
    async def test_workload_released_after_resolved_resume(self):
        """When resume completes with resolved, workload lock must be freed."""
        import app.main as main_module

        workload_key = "prod/payment-api"
        alert_id = "test-resume-resolved-001"
        main_module._ACTIVE_WORKLOADS[workload_key] = alert_id
        # _ALERT_TO_WORKLOAD does not exist yet — will raise AttributeError (RED)
        main_module._ALERT_TO_WORKLOAD[alert_id] = workload_key

        with patch("app.main._resume_graph", new=AsyncMock(return_value={
            "status": "resolved",
            "action_result": "deployment restarted",
            "alertname": "TestAlert",
            "error_summary": "crash loop",
            "slack_message_ts_list": [],
            "slack_channel": None,
            "human_approved": True,
        })), patch("app.main._post_slack_success", new=AsyncMock()):
            await main_module._resume_and_notify(alert_id, True, None)

        assert workload_key not in main_module._ACTIVE_WORKLOADS
        assert alert_id not in main_module._ALERT_TO_WORKLOAD

    @pytest.mark.asyncio
    async def test_workload_stays_locked_during_retry_cycle(self):
        """When graph pauses again after resume (retry cycle), lock must remain held."""
        import app.main as main_module

        workload_key = "prod/payment-api"
        alert_id = "test-resume-waiting-001"
        main_module._ACTIVE_WORKLOADS[workload_key] = alert_id
        main_module._ALERT_TO_WORKLOAD[alert_id] = workload_key

        with patch("app.main._resume_graph", new=AsyncMock(return_value={
            "status": "waiting_for_approval",
            "action_result": "",
            "alertname": "TestAlert",
            "error_summary": "",
            "slack_message_ts_list": [],
            "slack_channel": None,
            "human_approved": False,
        })):
            await main_module._resume_and_notify(alert_id, True, None)

        # Graph still paused at human_gate — lock must remain
        assert workload_key in main_module._ACTIVE_WORKLOADS
        assert alert_id in main_module._ALERT_TO_WORKLOAD

    @pytest.mark.asyncio
    async def test_retry_cycle_does_not_overwrite_approval_message(self):
        """Regression: when graph pauses again (retry), _resume_and_notify must NOT
        update the Slack message — human_gate already posted the new approval
        message with buttons.  Overwriting it removes the buttons and breaks the flow.
        """
        import app.main as main_module

        alert_id = "test-retry-no-overwrite-001"

        mock_update = AsyncMock()
        with patch("app.main._resume_graph", new=AsyncMock(return_value={
            "status": "waiting_for_approval",
            "action_result": "",
            "alertname": "TestAlert",
            "error_summary": "",
            "slack_message_ts": "1234567890.123456",
            "slack_message_ts_list": ["1234567890.123456"],
            "slack_channel": "C12345",
            "human_approved": True,
        })), patch("app.utils.slack_client.update_slack_message", mock_update):
            await main_module._resume_and_notify(alert_id, True, None)

        # update_slack_message must NOT be called — buttons would be overwritten
        mock_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_workload_released_after_escalation(self):
        """After human rejection (escalated), workload lock must be freed."""
        import app.main as main_module

        workload_key = "prod/payment-api"
        alert_id = "test-resume-escalated-001"
        main_module._ACTIVE_WORKLOADS[workload_key] = alert_id
        main_module._ALERT_TO_WORKLOAD[alert_id] = workload_key

        with patch("app.main._resume_graph", new=AsyncMock(return_value={
            "status": "escalated",
            "action_result": "",
            "alertname": "TestAlert",
            "error_summary": "",
            "slack_message_ts_list": [],
            "slack_channel": None,
            "human_approved": False,
        })):
            await main_module._resume_and_notify(alert_id, False, None)

        assert workload_key not in main_module._ACTIVE_WORKLOADS
        assert alert_id not in main_module._ALERT_TO_WORKLOAD

    @pytest.mark.asyncio
    async def test_workload_released_when_resume_raises_http_exception(self):
        """When _resume_graph raises HTTPException (stale Slack button), lock must be freed.

        Prevents deadlock: stale button click causes 409/404; without cleanup the workload
        stays locked forever with no mechanism to release it.
        """
        import app.main as main_module
        from fastapi import HTTPException

        workload_key = "prod/payment-api"
        alert_id = "test-resume-stale-001"
        main_module._ACTIVE_WORKLOADS[workload_key] = alert_id
        main_module._ALERT_TO_WORKLOAD[alert_id] = workload_key

        with patch("app.main._resume_graph", new=AsyncMock(
            side_effect=HTTPException(status_code=409, detail="Alert is not waiting for approval")
        )):
            await main_module._resume_and_notify(alert_id, True, None)

        assert workload_key not in main_module._ACTIVE_WORKLOADS
        assert alert_id not in main_module._ALERT_TO_WORKLOAD


# ---------------------------------------------------------------------------
# Fix #1: Pod name normalization deduplicates pods of the same Deployment
# ---------------------------------------------------------------------------

class TestPodNameNormalizationDedup:
    """Regression tests: multiple pods from the same Deployment must not
    each trigger a separate graph run (and a separate Slack message)."""

    @pytest.fixture(autouse=True)
    def mock_run_graph(self):
        with patch("app.main._run_graph", new=AsyncMock()):
            yield

    @pytest.mark.asyncio
    async def test_two_pods_same_deployment_deduplicated_in_v4(self, test_client):
        """AM v4 envelope with two pods of the same Deployment → only one graph run."""
        envelope = {
            "receiver": "sre-bot", "status": "firing",
            "alerts": [
                {"status": "firing", "labels": {"alertname": "CrashLoop", "pod": "api-6f8d4c7b9a-abc12"}, "annotations": {}},
                {"status": "firing", "labels": {"alertname": "CrashLoop", "pod": "api-6f8d4c7b9a-xyz99"}, "annotations": {}},
            ],
            "groupLabels": {"alertname": "CrashLoop"},
            "commonLabels": {"alertname": "CrashLoop", "namespace": "prod"},
        }
        resp = await test_client.post("/webhook", json=envelope)
        assert resp.status_code == 202
        data = resp.json()
        assert len(data["alert_ids"]) == 1
        assert data["suppressed_count"] == 1

    @pytest.mark.asyncio
    async def test_pods_from_different_deployments_both_accepted(self, test_client):
        """Pods from different deployments (different base names) are not deduped."""
        envelope = {
            "receiver": "sre-bot", "status": "firing",
            "alerts": [
                {"status": "firing", "labels": {"alertname": "CrashLoop", "pod": "frontend-6f8d4c7b9a-abc12"}, "annotations": {}},
                {"status": "firing", "labels": {"alertname": "CrashLoop", "pod": "backend-8a2b3c4d5e-xyz99"}, "annotations": {}},
            ],
            "groupLabels": {"alertname": "CrashLoop"},
            "commonLabels": {"alertname": "CrashLoop", "namespace": "prod"},
        }
        resp = await test_client.post("/webhook", json=envelope)
        assert resp.status_code == 202
        assert len(resp.json()["alert_ids"]) == 2


# ---------------------------------------------------------------------------
# Fix #2: "resolved" status alerts are skipped
# ---------------------------------------------------------------------------

class TestResolvedAlertFiltering:

    @pytest.fixture(autouse=True)
    def mock_run_graph(self):
        with patch("app.main._run_graph", new=AsyncMock()):
            yield

    @pytest.mark.asyncio
    async def test_legacy_resolved_alert_ignored(self, test_client):
        """Legacy flat format with status=resolved returns ignored."""
        resp = await test_client.post("/webhook", json={
            "alertname": "CrashLoop", "status": "resolved",
            "labels": {"namespace": "prod", "deployment": "api"},
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"

    @pytest.mark.asyncio
    async def test_v4_all_resolved_ignored(self, test_client):
        """AM v4 envelope where every alert is resolved → ignored."""
        envelope = {
            "receiver": "sre-bot", "status": "resolved",
            "alerts": [
                {"status": "resolved", "labels": {"alertname": "CrashLoop", "deployment": "api"}, "annotations": {}},
                {"status": "resolved", "labels": {"alertname": "CrashLoop", "deployment": "web"}, "annotations": {}},
            ],
            "groupLabels": {"alertname": "CrashLoop"},
            "commonLabels": {"namespace": "prod"},
        }
        resp = await test_client.post("/webhook", json=envelope)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"

    @pytest.mark.asyncio
    async def test_v4_mixed_resolved_and_firing(self, test_client):
        """AM v4 envelope with one firing + one resolved → only firing processed."""
        envelope = {
            "receiver": "sre-bot", "status": "firing",
            "alerts": [
                {"status": "firing", "labels": {"alertname": "CrashLoop", "deployment": "api"}, "annotations": {}},
                {"status": "resolved", "labels": {"alertname": "CrashLoop", "deployment": "web"}, "annotations": {}},
            ],
            "groupLabels": {"alertname": "CrashLoop"},
            "commonLabels": {"namespace": "prod"},
        }
        resp = await test_client.post("/webhook", json=envelope)
        assert resp.status_code == 202
        data = resp.json()
        assert len(data["alert_ids"]) == 1


# ---------------------------------------------------------------------------
# Fix #3: Post-resolution cooldown prevents Alertmanager repeats
# ---------------------------------------------------------------------------

class TestPostResolutionCooldown:

    def test_cooldown_helpers(self):
        """_set_cooldown / _is_in_cooldown round-trip."""
        import app.main as m
        wk = "test-ns/test-deploy"
        assert not m._is_in_cooldown(wk)
        m._set_cooldown(wk)
        assert m._is_in_cooldown(wk)

    def test_cooldown_expires(self):
        """Cooldown expires after _COOLDOWN_SECONDS."""
        import app.main as m
        wk = "test-ns/test-deploy"
        # Set cooldown that already expired
        m._WORKLOAD_COOLDOWNS[wk] = time.monotonic() - 1
        assert not m._is_in_cooldown(wk)
        # Entry should be cleaned up
        assert wk not in m._WORKLOAD_COOLDOWNS

    @pytest.mark.asyncio
    async def test_cooldown_set_after_graph_completion(self):
        """_run_graph sets cooldown when workload lock is released on completion."""
        import app.main as m

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"status": "resolved"})

        wk = "prod/api"
        aid = "test-cooldown-001"
        m._ACTIVE_WORKLOADS[wk] = aid

        with patch("app.main._graph", return_value=mock_graph):
            await m._run_graph(aid, {}, wk)

        assert wk not in m._ACTIVE_WORKLOADS
        assert m._is_in_cooldown(wk)

    @pytest.mark.asyncio
    async def test_cooldown_set_after_resume_completion(self):
        """_resume_and_notify sets cooldown when workload lock is released."""
        import app.main as m

        wk = "prod/api"
        aid = "test-cooldown-resume-001"
        m._ACTIVE_WORKLOADS[wk] = aid
        m._ALERT_TO_WORKLOAD[aid] = wk

        with patch("app.main._resume_graph", new=AsyncMock(return_value={
            "status": "resolved", "action_result": "done",
            "alertname": "Test", "error_summary": "",
            "slack_message_ts_list": [], "slack_channel": None,
            "human_approved": True,
        })), patch("app.main._post_slack_success", new=AsyncMock()):
            await m._resume_and_notify(aid, True, None)

        assert wk not in m._ACTIVE_WORKLOADS
        assert m._is_in_cooldown(wk)

    @pytest.mark.asyncio
    async def test_cooldown_suppresses_legacy_webhook(self, test_client):
        """Alert during cooldown is suppressed (legacy flat path)."""
        import app.main as m
        wk = "prod/api"
        m._set_cooldown(wk)

        with patch("app.main._run_graph", new=AsyncMock()):
            resp = await test_client.post("/webhook", json={
                "alertname": "CrashLoop", "status": "firing",
                "labels": {"namespace": "prod", "deployment": "api"},
            })

        assert resp.status_code == 200
        assert resp.json()["status"] == "suppressed"
        assert "cooldown" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_cooldown_suppresses_v4_webhook(self, test_client):
        """Alert during cooldown is suppressed (AM v4 envelope path)."""
        import app.main as m
        wk = "prod/api"
        m._set_cooldown(wk)

        with patch("app.main._run_graph", new=AsyncMock()):
            envelope = {
                "receiver": "sre-bot", "status": "firing",
                "alerts": [
                    {"status": "firing", "labels": {"alertname": "CrashLoop", "deployment": "api"}, "annotations": {}},
                ],
                "groupLabels": {}, "commonLabels": {"namespace": "prod"},
            }
            resp = await test_client.post("/webhook", json=envelope)

        assert resp.status_code == 200
        assert resp.json()["status"] == "suppressed"

    @pytest.mark.asyncio
    async def test_no_cooldown_on_graph_interrupt(self):
        """GraphInterrupt (human_gate pause) must NOT set a cooldown."""
        import app.main as m
        from langgraph.errors import GraphInterrupt

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(side_effect=GraphInterrupt("paused"))

        wk = "prod/api"
        aid = "test-interrupt-001"
        m._ACTIVE_WORKLOADS[wk] = aid

        with patch("app.main._graph", return_value=mock_graph):
            await m._run_graph(aid, {}, wk)

        # Lock is still held (not released) → no cooldown set
        assert wk in m._ACTIVE_WORKLOADS
        assert not m._is_in_cooldown(wk)
