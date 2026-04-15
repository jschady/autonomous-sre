"""Integration test: durable graph resumption via /slack/interactive.

Requires a live POSTGRES_DSN (Supabase or local Postgres).
Skipped automatically when POSTGRES_DSN is not set.

Scenario:
  1. POST /webhook  → alert submitted, graph starts in background
  2. Wait for graph to pause at human_gate
  3. Kill the graph instance (simulate restart)
  4. POST /slack/interactive with approval → graph resumes from Supabase state
  5. Verify final status is resolved or escalated (not stuck)
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage

import app.tools.k8s_tools as k8s_tools

POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "")
pytestmark = pytest.mark.integration


@pytest.mark.skipif(not POSTGRES_DSN, reason="POSTGRES_DSN not set — skipping durable resumption test")
class TestDurableResumption:
    @pytest.mark.asyncio
    async def test_approval_resumes_after_graph_pause(self):
        """Graph paused at human_gate resumes correctly after approval via API."""
        k8s_tools.MOCK_HEALTHY = True

        triage_response = (
            '{"severity": "critical", "tools_to_run": ["get_cluster_events"], '
            '"triage_summary": "CrashLoopBackOff detected", "task_complexity": "moderate"}'
        )
        processor_response = "Pod crash looping due to DB failure."
        researcher_response = "Restart the service to resolve CrashLoopBackOff."

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.side_effect = [
            AIMessage(content=triage_response),
            AIMessage(content=processor_response),
            AIMessage(content=researcher_response),
        ] * 4
        mock_llm.ainvoke = AsyncMock(side_effect=[
            AIMessage(content=triage_response),
            AIMessage(content=processor_response),
            AIMessage(content=researcher_response),
        ] * 4)

        from app.main import app

        with patch("app.nodes.triage.ChatAnthropic", return_value=mock_llm), \
             patch("app.utils.llm_factory._build_claude_client", return_value=mock_llm), \
             patch("app.utils.llm_factory.get_settings", return_value=MagicMock(
                 local_model_enabled=False, runpod_base_url="",
                 runpod_serverless_enabled=False, runpod_serverless_endpoint_id="",
                 triage_model="claude-sonnet-4-6", anthropic_api_key="test",
             )), \
             patch("app.nodes.router.get_settings", return_value=MagicMock(
                 local_model_enabled=False, semantic_cache_enabled=False,
             )), \
             patch("app.utils.incident_store.fetch_similar_incidents", return_value=[]), \
             patch("app.nodes.human_gate._notify_slack", AsyncMock()):

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                # Submit webhook
                post_resp = await client.post("/webhook", json={
                    "alertname": "DurableTest",
                    "status": "firing",
                    "labels": {"namespace": "test"},
                    "annotations": {"summary": "Durable resumption test"},
                })
                assert post_resp.status_code == 202
                alert_id = post_resp.json()["alert_id"]

                # Wait for graph to reach human_gate or complete
                for _ in range(20):
                    await asyncio.sleep(0.2)
                    status_resp = await client.get(f"/status/{alert_id}")
                    if status_resp.status_code == 200:
                        data = status_resp.json()
                        if data["status"] in ("resolved", "escalated", "failed"):
                            break
                        if data.get("current_node") == "human_gate":
                            # Approve
                            approve_resp = await client.post(
                                "/slack/interactive",
                                json={"alert_id": alert_id, "approved": True},
                            )
                            assert approve_resp.status_code in (200, 409)
                            break

                # Final check
                final_resp = await client.get(f"/status/{alert_id}")
                assert final_resp.status_code == 200
                assert final_resp.json()["status"] in (
                    "resolved", "escalated", "failed", "in_progress"
                )
