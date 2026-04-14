"""LLM token usage extraction and cost estimation utilities.

Pricing reference (per 1M tokens, as of 2025):
  claude-sonnet-4-6:        $3.00 input / $15.00 output
                            cache_write: $3.75 / cache_read: $0.30
  claude-haiku-4-5-*:       $0.80 input /  $4.00 output
                            cache_write: $1.00 / cache_read: $0.08
  text-embedding-3-small:   $0.02 input /  $0.00 output
"""
from __future__ import annotations

from typing import TypedDict

# ---------------------------------------------------------------------------
# Pricing table — USD per 1M tokens
# Keys: input, output, cache_write (creation), cache_read
# ---------------------------------------------------------------------------

_PRICE_PER_M: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-haiku-4-5-20251001": {
        "input": 0.80,
        "output": 4.00,
        "cache_write": 1.00,
        "cache_read": 0.08,
    },
    "text-embedding-3-small": {
        "input": 0.02,
        "output": 0.00,
        "cache_write": 0.00,
        "cache_read": 0.00,
    },
    # Local / self-hosted models — compute-cost estimate, not API billing
    "meta-llama/Llama-3.1-8B-Instruct": {
        "input": 0.10,
        "output": 0.10,
        "cache_write": 0.00,
        "cache_read": 0.00,
    },
    "llama": {
        "input": 0.10,
        "output": 0.10,
        "cache_write": 0.00,
        "cache_read": 0.00,
    },
}
_DEFAULT_PRICE: dict[str, float] = {
    "input": 3.00,
    "output": 15.00,
    "cache_write": 3.75,
    "cache_read": 0.30,
}


class NodeUsage(TypedDict):
    node: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    cost_usd: float


def extract_usage(node: str, model: str, response_obj) -> NodeUsage:
    """Pull usage_metadata from a LangChain AIMessage and compute cost.

    Works with any LangChain ChatModel response that populates
    ``response.usage_metadata`` (Anthropic, OpenAI, etc.).

    Accounts for Anthropic prompt-cache tokens (cache_creation_input_tokens
    and cache_read_input_tokens) which are billed at different rates and are
    the primary cause of discrepancies with LangSmith cost reports.

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
    # Anthropic prompt-cache fields (zero for non-Anthropic models)
    cache_write_tokens: int = meta.get("cache_creation_input_tokens", 0)
    cache_read_tokens: int = meta.get("cache_read_input_tokens", 0)
    total_tokens: int = meta.get(
        "total_tokens",
        input_tokens + output_tokens + cache_write_tokens + cache_read_tokens,
    )

    cost_usd = _compute_cost(
        model, input_tokens, output_tokens, cache_write_tokens, cache_read_tokens
    )

    return NodeUsage(
        node=node,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cache_write_tokens=cache_write_tokens,
        cache_read_tokens=cache_read_tokens,
        cost_usd=round(cost_usd, 8),
    )


def _compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Compute USD cost from token counts using the pricing table."""
    # Match on prefix so model variants (e.g. -20251001 suffix) resolve correctly
    prices = _DEFAULT_PRICE
    for key, val in _PRICE_PER_M.items():
        if model.startswith(key):
            prices = val
            break

    return (
        input_tokens * prices["input"]
        + output_tokens * prices["output"]
        + cache_write_tokens * prices["cache_write"]
        + cache_read_tokens * prices["cache_read"]
    ) / 1_000_000


def accumulate_cost(existing: list[dict]) -> float:
    """Sum cost_usd across all NodeUsage entries."""
    return round(sum(entry.get("cost_usd", 0.0) for entry in existing), 8)


def compute_cost_saved(
    node_usage: list[dict],
    reference_model: str = "claude-sonnet-4-6",
) -> float:
    """Compute savings vs running everything on the reference model.

    Args:
        node_usage:      List of NodeUsage dicts from state["token_usage"].
        reference_model: Model to compare against (default: claude-sonnet-4-6).

    Returns:
        USD savings (positive = saved money, negative = spent more).
    """
    actual_cost = accumulate_cost(node_usage)
    reference_cost = sum(
        _compute_cost(
            reference_model,
            entry.get("input_tokens", 0),
            entry.get("output_tokens", 0),
            entry.get("cache_write_tokens", 0),
            entry.get("cache_read_tokens", 0),
        )
        for entry in node_usage
    )
    return round(reference_cost - actual_cost, 8)
