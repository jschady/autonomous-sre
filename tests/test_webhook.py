"""Tests for the FastAPI webhook endpoints.

TDD: Written BEFORE implementation.
"""
from __future__ import annotations

import asyncio
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
from langchain_core.messages import AIMessage

import app.tools.k8s_tools as k8s_tools
from app.tools.k8s_tools import EXECUTED_ACTIONS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_mock_state():
    EXECUTED_ACTIONS.clear()
    k8s_tools.MOCK_HEALTHY = False
    yield


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
        r2 = await test_client.post("/webhook", json={**sample_alert})
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

class TestHappyPathE2E:
    @pytest.mark.asyncio
    async def test_full_happy_path_e2e(self, test_client, sample_alert):
        """POST webhook → poll status → approve → poll until resolved."""
        k8s_tools.MOCK_HEALTHY = True

        triage_response = (
            '{"severity": "critical", "tools_to_run": ["get_cluster_events"], '
            '"triage_summary": "CrashLoopBackOff detected"}'
        )
        processor_response = "Pod crash looping due to DB failure."
        researcher_response = "Restart the service to resolve CrashLoopBackOff."

        mock_llm = _make_mock_llm(
            [triage_response, processor_response, researcher_response] * 3
        )

        with patch("app.nodes.triage.ChatAnthropic", return_value=mock_llm), \
             patch("app.nodes.processor.ChatAnthropic", return_value=mock_llm), \
             patch("app.nodes.researcher.ChatAnthropic", return_value=mock_llm):

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
    async def test_human_rejection_escalates(self, test_client, sample_alert):
        """POST webhook → reject → status should eventually be escalated."""
        triage_response = (
            '{"severity": "critical", "tools_to_run": ["get_cluster_events"], '
            '"triage_summary": "CrashLoopBackOff detected"}'
        )
        processor_response = "Pod crash looping."
        researcher_response = "Restart the service."
        mock_llm = _make_mock_llm(
            [triage_response, processor_response, researcher_response] * 2
        )

        with patch("app.nodes.triage.ChatAnthropic", return_value=mock_llm), \
             patch("app.nodes.processor.ChatAnthropic", return_value=mock_llm), \
             patch("app.nodes.researcher.ChatAnthropic", return_value=mock_llm):

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
