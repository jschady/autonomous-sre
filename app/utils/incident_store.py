"""Persist and retrieve resolved incidents for few-shot prompting.

Stores successful remediations in Postgres (resolved_incidents table)
with pgvector embeddings for similarity-based retrieval.
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


async def save_resolved_incident(dsn: str, state: dict) -> None:
    """Persist a resolved incident to Postgres for future few-shot retrieval.

    Non-critical: logs and swallows all errors.

    Args:
        dsn:   Postgres connection string. No-ops if empty.
        state: Completed SREState dict with resolved=True.
    """
    if not dsn:
        return

    try:
        import asyncpg  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("asyncpg not installed — skipping incident persistence")
        return

    alert_id = state.get("alert_id", "")
    alertname = state.get("alert_payload", {}).get("alertname", "unknown")
    namespace = state.get("metadata", {}).get("namespace", "default")
    error_summary = state.get("error_summary", "")
    triage_summary = state.get("triage_summary", "")
    recommended_action = state.get("recommended_action", "")
    action_result = state.get("action_result", "Resolved")
    severity = state.get("severity", "unknown")

    if not error_summary or not recommended_action:
        logger.debug(
            "Skipping incident persistence — error_summary or recommended_action empty"
        )
        return

    embedding = await _embed_text(error_summary)

    sql = """
        INSERT INTO resolved_incidents
            (alert_id, alertname, namespace, error_summary, triage_summary,
             recommended_action, action_result, severity, embedding)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::vector)
        ON CONFLICT (alert_id) DO NOTHING;
    """

    try:
        conn = await asyncpg.connect(dsn, statement_cache_size=0)
        try:
            await conn.execute(
                sql,
                alert_id,
                alertname,
                namespace,
                error_summary,
                triage_summary,
                recommended_action,
                action_result,
                severity,
                json.dumps(embedding) if embedding else None,
            )
            logger.info("Saved resolved incident alert_id=%s alertname=%s", alert_id, alertname)
        finally:
            await conn.close()
    except Exception as exc:
        logger.error("Failed to save resolved incident: %s", exc)


async def fetch_similar_incidents(
    dsn: str,
    error_summary: str,
    limit: int = 3,
) -> list[dict]:
    """Retrieve the most similar past resolved incidents for few-shot prompting.

    Args:
        dsn:           Postgres connection string. Returns [] if empty.
        error_summary: Error text to embed and search against.
        limit:         Max number of incidents to return.

    Returns:
        List of dicts with keys: alertname, namespace, triage_summary,
        recommended_action, action_result. Empty list on any error.
    """
    if not dsn or not error_summary:
        return []

    try:
        import asyncpg  # type: ignore[import-untyped]
    except ImportError:
        return []

    embedding = await _embed_text(error_summary)
    if not embedding:
        return []

    sql = """
        SELECT alertname, namespace, triage_summary, recommended_action, action_result
        FROM resolved_incidents
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> $1::vector
        LIMIT $2;
    """

    try:
        conn = await asyncpg.connect(dsn, statement_cache_size=0)
        try:
            rows = await conn.fetch(sql, json.dumps(embedding), limit)
            return [dict(row) for row in rows]
        finally:
            await conn.close()
    except Exception as exc:
        logger.error("Failed to fetch similar incidents: %s", exc)
        return []


async def cache_lookup(
    dsn: str,
    error_summary: str,
    threshold: float = 0.95,
) -> dict | None:
    """Return the nearest resolved incident if cosine similarity >= threshold.

    Uses the ivfflat index on resolved_incidents.embedding for O(log n) lookup.
    Returns None on empty dsn, embedding failure, no match above threshold, or any error.
    """
    if not dsn or not error_summary:
        return None

    embedding = await _embed_text(error_summary)
    if not embedding:
        return None

    sql = """
        SELECT recommended_action, error_summary,
               1 - (embedding <=> $1::vector) AS similarity
        FROM resolved_incidents
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> $1::vector
        LIMIT 1;
    """

    try:
        import asyncpg  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("asyncpg not installed — cache_lookup disabled")
        return None

    try:
        conn = await asyncpg.connect(dsn, statement_cache_size=0)
        try:
            row = await conn.fetchrow(sql, json.dumps(embedding))
            if row is None or row["similarity"] < threshold:
                return None
            return dict(row)
        finally:
            await conn.close()
    except Exception as exc:
        logger.warning("cache_lookup error (non-critical): %s", exc)
        return None


async def _embed_text(text: str) -> list[float]:
    """Embed text using OpenAI text-embedding-3-small. Returns [] on failure."""
    if not text:
        return []
    try:
        from openai import AsyncOpenAI  # type: ignore[import-untyped]

        client = AsyncOpenAI()
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding
    except Exception as exc:
        logger.warning("Embedding failed (non-critical): %s", exc)
        return []
