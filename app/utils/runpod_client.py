"""RunPod Serverless inference client.

Implements the async /run + /status polling pattern used by RunPod Serverless
endpoints. Wraps the result as a LangChain BaseChatModel so it integrates with
the existing graph nodes without modification.

Cold-start resilience: RunPod Serverless can take ~20 seconds to wake a GPU.
The client polls /status with exponential backoff up to `cold_start_timeout`
seconds before giving up.

Usage:
    client = RunPodServerlessChat(
        endpoint_id="abc123",
        api_key="rp_...",
        timeout=60,
    )
    response = await client.ainvoke(messages)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Iterator

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import field_validator

logger = logging.getLogger(__name__)

_RUNPOD_API_BASE = "https://api.runpod.ai/v2"

# RunPod Serverless job statuses
_TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
_SUCCESS_STATUS = "COMPLETED"


class RunPodServerlessChat(BaseChatModel):
    """LangChain BaseChatModel backed by a RunPod Serverless endpoint.

    Submits requests via POST /{endpoint_id}/run and polls
    GET /{endpoint_id}/status/{job_id} until the job completes.
    """

    endpoint_id: str
    api_key: str
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    timeout: float = 60.0
    poll_interval: float = 2.0

    @field_validator("endpoint_id")
    @classmethod
    def _strip_base_url(cls, v: str) -> str:
        """Accept a bare endpoint ID or a full RunPod URL; normalise to ID only."""
        for base in ("https://api.runpod.ai/v2/", "https://api.runpod.io/v2/"):
            if v.startswith(base):
                return v[len(base):].rstrip("/")
        return v

    @property
    def _llm_type(self) -> str:
        return "runpod_serverless"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _messages_to_prompt(self, messages: list[BaseMessage]) -> list[dict]:
        """Convert LangChain messages to OpenAI-style chat format."""
        result = []
        for msg in messages:
            role = getattr(msg, "type", "user")
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            result.append({"role": role, "content": msg.content})
        return result

    def _run_url(self) -> str:
        return f"{_RUNPOD_API_BASE}/{self.endpoint_id}/run"

    def _status_url(self, job_id: str) -> str:
        return f"{_RUNPOD_API_BASE}/{self.endpoint_id}/status/{job_id}"

    async def _async_submit(self, messages: list[BaseMessage]) -> str:
        """Submit inference request and return job_id."""
        payload = {
            "input": {
                "messages": self._messages_to_prompt(messages),
                "model": self.model_name,
                "max_tokens": 4096,
            }
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                self._run_url(),
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            job_id = data.get("id")
            if not job_id:
                raise RuntimeError(f"RunPod /run returned no job id: {data}")
            logger.info("RunPod Serverless job submitted: %s", job_id)
            return job_id

    async def _async_poll(self, job_id: str) -> str:
        """Poll /status/{job_id} until complete or timeout. Returns output text."""
        deadline = asyncio.get_event_loop().time() + self.timeout
        interval = self.poll_interval

        async with httpx.AsyncClient(timeout=15) as client:
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise TimeoutError(
                        f"RunPod job {job_id} timed out after {self.timeout}s "
                        "(GPU cold start may be taking longer than expected)"
                    )

                resp = await client.get(
                    self._status_url(job_id),
                    headers=self._headers(),
                )
                resp.raise_for_status()
                data = resp.json()
                status = data.get("status", "")

                if status in _TERMINAL_STATUSES:
                    if status != _SUCCESS_STATUS:
                        error = data.get("error", "unknown error")
                        raise RuntimeError(f"RunPod job {job_id} failed: {status} — {error}")

                    output = data.get("output", {})
                    # Support both plain string output and OpenAI-style choices
                    if isinstance(output, str):
                        return output
                    if isinstance(output, dict):
                        choices = output.get("choices", [])
                        if choices:
                            return choices[0].get("message", {}).get("content", "")
                        return output.get("text", str(output))
                    return str(output)

                logger.debug("RunPod job %s status: %s (polling in %.1fs)", job_id, status, interval)
                await asyncio.sleep(interval)
                # Exponential backoff up to 10s
                interval = min(interval * 1.5, 10.0)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        job_id = await self._async_submit(messages)
        text = await self._async_poll(job_id)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return asyncio.get_event_loop().run_until_complete(
            self._agenerate(messages, stop=stop, **kwargs)
        )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        # RunPod Serverless does not support streaming; emit a single chunk
        result = await self._agenerate(messages, stop=stop, **kwargs)
        yield result.generations[0]

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        result = self._generate(messages, stop=stop, **kwargs)
        yield result.generations[0]
