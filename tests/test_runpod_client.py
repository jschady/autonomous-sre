"""Unit tests for the RunPod Serverless polling client."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.utils.runpod_client import RunPodServerlessChat


@pytest.fixture
def client():
    return RunPodServerlessChat(
        endpoint_id="test-endpoint-123",
        api_key="rp_test_key",
        timeout=10,
        poll_interval=0.1,
    )


def _mock_response(status_code: int = 200, json_data: dict = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    return resp


class TestRunPodServerlessChat:
    def test_llm_type(self, client):
        assert client._llm_type == "runpod_serverless"

    def test_messages_to_prompt_human(self, client):
        msgs = [HumanMessage(content="hello")]
        prompt = client._messages_to_prompt(msgs)
        assert prompt == [{"role": "user", "content": "hello"}]

    def test_run_url(self, client):
        assert client._run_url().endswith("/test-endpoint-123/run")

    def test_status_url(self, client):
        assert client._status_url("job-1").endswith("/test-endpoint-123/status/job-1")

    @pytest.mark.asyncio
    async def test_happy_path_submit_and_poll(self, client):
        submit_resp = _mock_response(json_data={"id": "job-abc"})
        poll_resp = _mock_response(json_data={
            "status": "COMPLETED",
            "output": "Service restarted successfully.",
        })

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=submit_resp)
        mock_client.get = AsyncMock(return_value=poll_resp)

        with patch("app.utils.runpod_client.httpx.AsyncClient", return_value=mock_client):
            result = await client._agenerate([HumanMessage(content="restart service")])

        assert result.generations[0].message.content == "Service restarted successfully."

    @pytest.mark.asyncio
    async def test_cold_start_polls_multiple_times_before_complete(self, client):
        """Simulate IN_QUEUE → IN_PROGRESS → COMPLETED sequence."""
        submit_resp = _mock_response(json_data={"id": "job-cold"})
        poll_responses = [
            _mock_response(json_data={"status": "IN_QUEUE"}),
            _mock_response(json_data={"status": "IN_PROGRESS"}),
            _mock_response(json_data={"status": "IN_PROGRESS"}),
            _mock_response(json_data={"status": "COMPLETED", "output": "done"}),
        ]

        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            resp = poll_responses[min(call_count, len(poll_responses) - 1)]
            call_count += 1
            return resp

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=submit_resp)
        mock_client.get = AsyncMock(side_effect=mock_get)

        with patch("app.utils.runpod_client.httpx.AsyncClient", return_value=mock_client):
            result = await client._agenerate([HumanMessage(content="test")])

        assert result.generations[0].message.content == "done"
        assert call_count >= 3

    @pytest.mark.asyncio
    async def test_timeout_raises_timeout_error(self):
        """Client should raise TimeoutError when job takes longer than timeout."""
        fast_client = RunPodServerlessChat(
            endpoint_id="ep",
            api_key="key",
            timeout=0.1,  # Very short timeout
            poll_interval=0.05,
        )

        submit_resp = _mock_response(json_data={"id": "job-slow"})
        poll_resp = _mock_response(json_data={"status": "IN_QUEUE"})

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=submit_resp)
        mock_client.get = AsyncMock(return_value=poll_resp)

        with patch("app.utils.runpod_client.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(TimeoutError):
                await fast_client._agenerate([HumanMessage(content="test")])

    @pytest.mark.asyncio
    async def test_failed_job_raises_runtime_error(self, client):
        submit_resp = _mock_response(json_data={"id": "job-fail"})
        poll_resp = _mock_response(json_data={
            "status": "FAILED",
            "error": "CUDA out of memory",
        })

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=submit_resp)
        mock_client.get = AsyncMock(return_value=poll_resp)

        with patch("app.utils.runpod_client.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(RuntimeError, match="FAILED"):
                await client._agenerate([HumanMessage(content="test")])

    @pytest.mark.asyncio
    async def test_openai_choices_output_format(self, client):
        """Support the OpenAI-compatible output format from vLLM."""
        submit_resp = _mock_response(json_data={"id": "job-oai"})
        poll_resp = _mock_response(json_data={
            "status": "COMPLETED",
            "output": {
                "choices": [
                    {"message": {"role": "assistant", "content": "OpenAI-style response"}}
                ]
            },
        })

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=submit_resp)
        mock_client.get = AsyncMock(return_value=poll_resp)

        with patch("app.utils.runpod_client.httpx.AsyncClient", return_value=mock_client):
            result = await client._agenerate([HumanMessage(content="test")])

        assert result.generations[0].message.content == "OpenAI-style response"

    @pytest.mark.asyncio
    async def test_submit_raises_if_no_job_id(self, client):
        submit_resp = _mock_response(json_data={"error": "bad request"})

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=submit_resp)

        with patch("app.utils.runpod_client.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(RuntimeError, match="no job id"):
                await client._async_submit([HumanMessage(content="test")])
