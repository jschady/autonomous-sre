"""LLM token usage extraction and cost estimation utilities.

Pricing reference (per 1M tokens, as of 2025):
  claude-sonnet-4-6:        $3.00 input / $15.00 output
  claude-haiku-4-5-*:       $0.80 input /  $4.00 output
  text-embedding-3-small:   $0.02 input /  $0.00 output
"""
from __future__ import annotations

from typing import TypedDict

# ---------------------------------------------------------------------------
# Pricing table — USD per 1M tokens
# ---------------------------------------------------------------------------

_PRICE_PER_M: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},
}
_DEFAULT_PRICE: dict[str, float] = {"input": 3.00, "output": 15.00}


class NodeUsage(TypedDict):
    node: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float


def extract_usage(node: str, model: str, response_obj) -> NodeUsage:
    """Pull usage_metadata from a LangChain AIMessage and compute cost.

    Works with any LangChain ChatModel response that populates
    ``response.usage_metadata`` (Anthropic, OpenAI, etc.).

    Args:
        node:         Name of the calling node (e.g. "triage").
        model:        Model ID string used for the call.
        response_obj: AIMessage returned by ``llm.invoke()``.

    Returns:
        NodeUsage dict ready to append to ``state["token_usage"]``.
    """
    meta = getattr(response_obj, "usage_metadata", None) or {}
    input_tokens: int = meta.get("input_tokens", 0)
    output_tokens: int = meta.get("output_tokens", 0)
    total_tokens: int = meta.get("total_tokens", input_tokens + output_tokens)

    cost_usd = _compute_cost(model, input_tokens, output_tokens)

    return NodeUsage(
        node=node,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=round(cost_usd, 8),
    )


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost from token counts using the pricing table."""
    # Match on prefix so model variants (e.g. -20251001 suffix) resolve correctly
    prices = _DEFAULT_PRICE
    for key, val in _PRICE_PER_M.items():
        if model.startswith(key):
            prices = val
            break

    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


def accumulate_cost(existing: list[dict]) -> float:
    """Sum cost_usd across all NodeUsage entries."""
    return round(sum(entry.get("cost_usd", 0.0) for entry in existing), 8)
