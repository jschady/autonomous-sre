"""Persist per-alert token usage and cost to the alert_costs Postgres table.

Phase 3 additions:
  - llm_provider: which provider was used (claude/local/mixed)
  - cost_saved_usd: savings vs running all calls on Claude Sonnet
  - cache_hit: whether the semantic cache short-circuited the analysis
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


async def save_alert_cost(dsn: str, state: dict) -> None:
    """Upsert cost data for a completed alert run.

    Safe to call multiple times (ON CONFLICT DO UPDATE).
    No-ops silently if dsn is empty or asyncpg is unavailable.
    """
    if not dsn:
        return

    try:
        import asyncpg  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("asyncpg not installed — skipping cost persistence")
        return

    from app.utils.llm_cost import compute_cost_saved

    token_usage: list[dict] = state.get("token_usage", [])
    total_input = sum(e.get("input_tokens", 0) for e in token_usage)
    total_output = sum(e.get("output_tokens", 0) for e in token_usage)
    total_tokens = sum(e.get("total_tokens", 0) for e in token_usage)
    cost_usd = state.get("cost_estimate_usd", 0.0)
    cost_saved_usd = compute_cost_saved(token_usage)
    alert_id = state.get("alert_id", "")
    alertname = state.get("alert_payload", {}).get("alertname", "unknown")
    resolved = state.get("resolved", False)
    llm_provider = state.get("llm_provider", "claude")
    cache_hit = state.get("cache_hit", False)

    sql = """
        INSERT INTO alert_costs
            (alert_id, alertname, total_tokens, input_tokens, output_tokens,
             cost_usd, cost_saved_usd, node_breakdown, resolved,
             llm_provider, cache_hit, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11, NOW())
        ON CONFLICT (alert_id) DO UPDATE SET
            total_tokens   = EXCLUDED.total_tokens,
            input_tokens   = EXCLUDED.input_tokens,
            output_tokens  = EXCLUDED.output_tokens,
            cost_usd       = EXCLUDED.cost_usd,
            cost_saved_usd = EXCLUDED.cost_saved_usd,
            node_breakdown = EXCLUDED.node_breakdown,
            resolved       = EXCLUDED.resolved,
            llm_provider   = EXCLUDED.llm_provider,
            cache_hit      = EXCLUDED.cache_hit,
            updated_at     = NOW();
    """

    try:
        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute(
                sql,
                alert_id,
                alertname,
                total_tokens,
                total_input,
                total_output,
                cost_usd,
                cost_saved_usd,
                json.dumps(token_usage),
                resolved,
                llm_provider,
                cache_hit,
            )
            logger.info(
                "Saved cost record alert_id=%s tokens=%d cost=$%.6f saved=$%.6f provider=%s",
                alert_id,
                total_tokens,
                cost_usd,
                cost_saved_usd,
                llm_provider,
            )
        finally:
            await conn.close()
    except Exception as exc:
        # Cost persistence is non-critical — log and continue
        logger.error("Failed to persist alert cost: %s", exc)


async def get_cost_report(dsn: str) -> dict:
    """Aggregate cost metrics for the /cost-report endpoint.

    Returns zeroed report if dsn empty or asyncpg unavailable.
    """
    if not dsn:
        return _empty_report()

    try:
        import asyncpg  # type: ignore[import-untyped]
    except ImportError:
        return _empty_report()

    sql = """
        SELECT
            COUNT(*)                                   AS total_alerts,
            COALESCE(SUM(cost_usd), 0)                AS total_cost_usd,
            COALESCE(SUM(cost_saved_usd), 0)          AS total_saved_usd,
            COALESCE(SUM(CASE WHEN llm_provider = 'claude'
                              THEN cost_usd END), 0)  AS claude_cost_usd,
            COALESCE(SUM(CASE WHEN llm_provider = 'local'
                              THEN cost_usd END), 0)  AS local_cost_usd,
            COUNT(*) FILTER (WHERE cache_hit = TRUE)  AS cache_hits,
            COUNT(*) FILTER (WHERE resolved = TRUE)   AS resolved_count
        FROM alert_costs;
    """

    try:
        conn = await asyncpg.connect(dsn)
        try:
            row = await conn.fetchrow(sql)
            if row is None:
                return _empty_report()
            data = dict(row)
            total = data["total_alerts"] or 1  # avoid division by zero
            return {
                "total_alerts": data["total_alerts"],
                "resolved_count": data["resolved_count"],
                "total_cost_usd": float(data["total_cost_usd"]),
                "total_saved_usd": float(data["total_saved_usd"]),
                "claude_cost_usd": float(data["claude_cost_usd"]),
                "local_cost_usd": float(data["local_cost_usd"]),
                "cache_hit_count": data["cache_hits"],
                "cache_hit_rate_pct": round(data["cache_hits"] / total * 100, 1),
            }
        finally:
            await conn.close()
    except Exception as exc:
        logger.error("Failed to fetch cost report: %s", exc)
        return _empty_report()


def aggregate_from_states(states: list[dict]) -> dict:
    """Compute a cost report from a list of in-memory SREState dicts.

    Used as a fallback when Postgres is not configured.
    """
    from app.utils.llm_cost import accumulate_cost, compute_cost_saved

    total_alerts = len(states)
    resolved_count = sum(1 for s in states if s.get("resolved"))
    cache_hits = sum(1 for s in states if s.get("cache_hit"))

    total_cost = 0.0
    total_saved = 0.0
    claude_cost = 0.0
    local_cost = 0.0

    _LOCAL_PREFIXES = ("meta-llama", "llama")

    for state in states:
        token_usage: list[dict] = state.get("token_usage", [])
        total_cost += accumulate_cost(token_usage)
        total_saved += compute_cost_saved(token_usage)

        for entry in token_usage:
            model = entry.get("model", "")
            entry_cost = entry.get("cost_usd", 0.0)
            if any(model.startswith(p) for p in _LOCAL_PREFIXES):
                local_cost += entry_cost
            else:
                claude_cost += entry_cost

    total = total_alerts or 1
    return {
        "total_alerts": total_alerts,
        "resolved_count": resolved_count,
        "total_cost_usd": round(total_cost, 6),
        "total_saved_usd": round(total_saved, 6),
        "claude_cost_usd": round(claude_cost, 6),
        "local_cost_usd": round(local_cost, 6),
        "cache_hit_count": cache_hits,
        "cache_hit_rate_pct": round(cache_hits / total * 100, 1),
    }


def _empty_report() -> dict:
    return {
        "total_alerts": 0,
        "resolved_count": 0,
        "total_cost_usd": 0.0,
        "total_saved_usd": 0.0,
        "claude_cost_usd": 0.0,
        "local_cost_usd": 0.0,
        "cache_hit_count": 0,
        "cache_hit_rate_pct": 0.0,
    }
