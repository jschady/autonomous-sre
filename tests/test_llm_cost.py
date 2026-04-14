"""Tests for LLM cost extraction and estimation utilities."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.utils.llm_cost import NodeUsage, accumulate_cost, extract_usage, _compute_cost


# ---------------------------------------------------------------------------
# _compute_cost
# ---------------------------------------------------------------------------

def test_compute_cost_sonnet():
    # 1000 input + 500 output on claude-sonnet-4-6
    # input: (1000 / 1_000_000) * 3.00 = 0.003
    # output: (500 / 1_000_000) * 15.00 = 0.0075
    cost = _compute_cost("claude-sonnet-4-6", 1000, 500)
    assert abs(cost - 0.0105) < 1e-9


def test_compute_cost_sonnet_with_cache_tokens():
    # cache_write: (200 / 1_000_000) * 3.75 = 0.00075
    # cache_read:  (400 / 1_000_000) * 0.30 = 0.00012
    cost = _compute_cost("claude-sonnet-4-6", 0, 0, cache_write_tokens=200, cache_read_tokens=400)
    assert abs(cost - (0.00075 + 0.00012)) < 1e-9


def test_compute_cost_haiku_with_cache_tokens():
    # cache_write: (100 / 1_000_000) * 1.00 = 0.0001
    # cache_read:  (300 / 1_000_000) * 0.08 = 0.000024
    cost = _compute_cost("claude-haiku-4-5-20251001", 0, 0, cache_write_tokens=100, cache_read_tokens=300)
    assert abs(cost - (0.0001 + 0.000024)) < 1e-9


def test_compute_cost_haiku():
    cost = _compute_cost("claude-haiku-4-5-20251001", 1000, 500)
    # input: (1000 / 1_000_000) * 0.80 = 0.0008
    # output: (500 / 1_000_000) * 4.00 = 0.002
    assert abs(cost - 0.0028) < 1e-9


def test_compute_cost_embedding_no_output_cost():
    cost = _compute_cost("text-embedding-3-small", 1000, 0)
    assert cost > 0  # Input has cost
    cost_with_output = _compute_cost("text-embedding-3-small", 1000, 500)
    assert cost == cost_with_output  # Output tokens are free for embeddings


def test_compute_cost_unknown_model_uses_default():
    # Unknown model should use the default (sonnet) pricing
    cost_unknown = _compute_cost("some-future-model", 1000, 500)
    cost_default = _compute_cost("claude-sonnet-4-6", 1000, 500)
    assert cost_unknown == cost_default


def test_compute_cost_zero_tokens():
    assert _compute_cost("claude-sonnet-4-6", 0, 0) == 0.0


# ---------------------------------------------------------------------------
# extract_usage
# ---------------------------------------------------------------------------

def _make_response(
    input_tokens=100,
    output_tokens=50,
    total_tokens=150,
    cache_creation_input_tokens=0,
    cache_read_input_tokens=0,
):
    response = MagicMock()
    response.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
    }
    return response


def test_extract_usage_returns_node_usage():
    response = _make_response(input_tokens=200, output_tokens=100, total_tokens=300)
    result = extract_usage("triage", "claude-sonnet-4-6", response)

    assert result["node"] == "triage"
    assert result["model"] == "claude-sonnet-4-6"
    assert result["input_tokens"] == 200
    assert result["output_tokens"] == 100
    assert result["total_tokens"] == 300
    assert result["cache_write_tokens"] == 0
    assert result["cache_read_tokens"] == 0
    assert result["cost_usd"] > 0


def test_extract_usage_with_cache_tokens():
    response = _make_response(
        input_tokens=100,
        output_tokens=50,
        total_tokens=450,
        cache_creation_input_tokens=200,
        cache_read_input_tokens=100,
    )
    result = extract_usage("triage", "claude-sonnet-4-6", response)

    assert result["cache_write_tokens"] == 200
    assert result["cache_read_tokens"] == 100
    # Cost must be higher than without cache tokens
    base = extract_usage("triage", "claude-sonnet-4-6", _make_response(100, 50, 150))
    assert result["cost_usd"] > base["cost_usd"]


def test_extract_usage_handles_missing_metadata():
    response = MagicMock()
    response.usage_metadata = None
    result = extract_usage("processor", "claude-haiku-4-5-20251001", response)

    assert result["input_tokens"] == 0
    assert result["output_tokens"] == 0
    assert result["total_tokens"] == 0
    assert result["cost_usd"] == 0.0


def test_extract_usage_infers_total_from_input_output():
    response = MagicMock()
    response.usage_metadata = {"input_tokens": 80, "output_tokens": 20}
    result = extract_usage("researcher", "claude-haiku-4-5-20251001", response)
    assert result["total_tokens"] == 100


def test_extract_usage_infers_total_including_cache_tokens():
    response = MagicMock()
    response.usage_metadata = {
        "input_tokens": 80,
        "output_tokens": 20,
        "cache_creation_input_tokens": 50,
        "cache_read_input_tokens": 30,
    }
    result = extract_usage("researcher", "claude-haiku-4-5-20251001", response)
    assert result["total_tokens"] == 180


def test_extract_usage_cost_rounded_to_8_decimals():
    response = _make_response(1, 1, 2)
    result = extract_usage("triage", "claude-sonnet-4-6", response)
    # Verify rounding doesn't exceed 8 decimal places
    assert len(str(result["cost_usd"]).split(".")[-1]) <= 8


# ---------------------------------------------------------------------------
# accumulate_cost
# ---------------------------------------------------------------------------

def test_accumulate_cost_sums_entries():
    entries = [
        {"cost_usd": 0.001},
        {"cost_usd": 0.002},
        {"cost_usd": 0.0005},
    ]
    assert abs(accumulate_cost(entries) - 0.0035) < 1e-9


def test_accumulate_cost_empty_list():
    assert accumulate_cost([]) == 0.0


def test_accumulate_cost_missing_cost_key():
    entries = [{"node": "triage"}, {"cost_usd": 0.001}]
    assert abs(accumulate_cost(entries) - 0.001) < 1e-9


def test_accumulate_cost_rounds_result():
    # Result should not exceed 8 decimal places
    entries = [{"cost_usd": 0.000000001}] * 3
    result = accumulate_cost(entries)
    assert len(str(result).split(".")[-1]) <= 8


# ---------------------------------------------------------------------------
# Integration: state accumulation pattern
# ---------------------------------------------------------------------------

def test_full_accumulation_across_nodes():
    """Simulate triage + processor + researcher token accumulation."""
    token_usage: list[dict] = []

    for node, model, inp, out in [
        ("triage", "claude-sonnet-4-6", 500, 100),
        ("processor", "claude-haiku-4-5-20251001", 800, 200),
        ("researcher", "claude-haiku-4-5-20251001", 600, 150),
    ]:
        response = _make_response(inp, out, inp + out)
        usage = extract_usage(node, model, response)
        token_usage = token_usage + [dict(usage)]

    total_cost = accumulate_cost(token_usage)
    assert total_cost > 0
    assert len(token_usage) == 3
    assert token_usage[0]["node"] == "triage"
    assert token_usage[1]["node"] == "processor"
    assert token_usage[2]["node"] == "researcher"
