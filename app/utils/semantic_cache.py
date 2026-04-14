"""Semantic cache using Redis + OpenAI embeddings for recurring incident short-circuiting.

Workflow:
  1. On alert: vectorize error_summary → check Redis for high-similarity match
  2. If found (>= threshold): return cached recommended_action, skip to human_gate
  3. On resolution: store embedding + resolution in Redis with TTL

Redis key format: cache:incident:{sha256_hex[:16]}
Value: JSON with keys: error_summary, recommended_action, embedding (list[float]), cached_at
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_CACHE_KEY_PREFIX = "cache:incident:"
_SCAN_BATCH_SIZE = 100


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 if either vector is all-zeros or lengths differ.
    Uses math.fsum for numerical precision.
    """
    if len(a) != len(b) or not a:
        return 0.0

    dot = math.fsum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(math.fsum(x * x for x in a))
    mag_b = math.sqrt(math.fsum(x * x for x in b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)


async def compute_embedding(text: str) -> list[float]:
    """Embed text using OpenAI text-embedding-3-small.

    Raises:
        RuntimeError: if openai package is not installed.
        Exception:    propagates OpenAI API errors.
    """
    try:
        from openai import AsyncOpenAI  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("openai package is required for semantic caching") from exc

    client = AsyncOpenAI()
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def _make_cache_key(embedding: list[float]) -> str:
    """Derive a stable Redis key from an embedding vector."""
    raw = json.dumps(embedding, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return f"{_CACHE_KEY_PREFIX}{digest}"


async def cache_lookup(
    redis_url: str,
    error_summary: str,
    threshold: float = 0.95,
) -> dict | None:
    """Search Redis for a semantically similar past resolution.

    Args:
        redis_url:     Redis connection URL. Returns None if empty.
        error_summary: Current incident error summary to match.
        threshold:     Minimum cosine similarity to count as a hit.

    Returns:
        Cached dict with recommended_action and error_summary, or None.
    """
    if not redis_url or not error_summary:
        return None

    try:
        import redis.asyncio as aioredis  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("redis package not installed — semantic cache disabled")
        return None

    try:
        query_embedding = await compute_embedding(error_summary)
    except Exception as exc:
        logger.debug("compute_embedding failed during lookup: %s", exc)
        return None

    try:
        r = aioredis.from_url(redis_url, decode_responses=True)
        cursor = 0
        best_similarity = 0.0
        best_entry: dict | None = None

        while True:
            cursor, keys = await r.scan(
                cursor=cursor,
                match=f"{_CACHE_KEY_PREFIX}*",
                count=_SCAN_BATCH_SIZE,
            )
            for key in keys:
                raw = await r.get(key)
                if raw is None:
                    continue
                try:
                    entry = json.loads(raw)
                    stored_embedding: list[float] = entry.get("embedding", [])
                    sim = cosine_similarity(query_embedding, stored_embedding)
                    if sim > best_similarity:
                        best_similarity = sim
                        best_entry = entry
                except (json.JSONDecodeError, TypeError):
                    continue

            if cursor == 0:
                break

        await r.aclose()

        if best_entry is not None and best_similarity >= threshold:
            logger.debug("cache hit: similarity=%.3f", best_similarity)
            return best_entry

        logger.debug("cache miss: best_similarity=%.3f", best_similarity)
        return None

    except Exception as exc:
        logger.warning("semantic cache lookup error: %s", exc)
        return None


async def cache_store(
    redis_url: str,
    error_summary: str,
    recommended_action: str,
    ttl_seconds: int = 86400,
) -> None:
    """Store a resolved incident in the semantic cache.

    Args:
        redis_url:          Redis connection URL. No-ops if empty.
        error_summary:      The error summary to vectorize as the cache key.
        recommended_action: The action that resolved the incident.
        ttl_seconds:        Cache entry TTL (default 24 h).
    """
    if not redis_url or not error_summary:
        return

    try:
        import redis.asyncio as aioredis  # type: ignore[import-untyped]
    except ImportError:
        return

    try:
        embedding = await compute_embedding(error_summary)
        cache_key = _make_cache_key(embedding)
        payload = json.dumps(
            {
                "error_summary": error_summary,
                "recommended_action": recommended_action,
                "embedding": embedding,
                "cached_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        r = aioredis.from_url(redis_url, decode_responses=True)
        await r.set(cache_key, payload, ex=ttl_seconds)
        await r.aclose()
        logger.info("Stored incident in semantic cache: key=%s", cache_key)
    except Exception as exc:
        logger.warning("semantic cache store error (non-critical): %s", exc)
