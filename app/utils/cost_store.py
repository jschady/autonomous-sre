"""Persist per-alert token usage and cost to the alert_costs Postgres table."""
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

    token_usage: list[dict] = state.get("token_usage", [])
    total_input = sum(e.get("input_tokens", 0) for e in token_usage)
    total_output = sum(e.get("output_tokens", 0) for e in token_usage)
    total_tokens = sum(e.get("total_tokens", 0) for e in token_usage)
    cost_usd = state.get("cost_estimate_usd", 0.0)
    alert_id = state.get("alert_id", "")
    alertname = state.get("alert_payload", {}).get("alertname", "unknown")
    resolved = state.get("resolved", False)

    sql = """
        INSERT INTO alert_costs
            (alert_id, alertname, total_tokens, input_tokens, output_tokens,
             cost_usd, node_breakdown, resolved, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, NOW())
        ON CONFLICT (alert_id) DO UPDATE SET
            total_tokens   = EXCLUDED.total_tokens,
            input_tokens   = EXCLUDED.input_tokens,
            output_tokens  = EXCLUDED.output_tokens,
            cost_usd       = EXCLUDED.cost_usd,
            node_breakdown = EXCLUDED.node_breakdown,
            resolved       = EXCLUDED.resolved,
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
                json.dumps(token_usage),
                resolved,
            )
            logger.info(
                "Saved cost record alert_id=%s tokens=%d cost=$%.6f",
                alert_id,
                total_tokens,
                cost_usd,
            )
        finally:
            await conn.close()
    except Exception as exc:
        # Cost persistence is non-critical — log and continue
        logger.error("Failed to persist alert cost: %s", exc)
