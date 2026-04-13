"""Knowledge-base tools for the Autonomous SRE system.

Phase 2: Hybrid search (semantic + full-text) backed by pgvector when
USE_VECTOR_DB=true.  Falls back to in-memory keyword search when the
database is unavailable or the feature flag is disabled.

Feature flags:
  USE_VECTOR_DB=true   — use Postgres pgvector hybrid search
  POSTGRES_DSN         — connection string for Postgres
  OPENAI_API_KEY       — required for embedding generation
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
from typing import Any

from langchain_core.tools import tool

from data.sops.mock_sops import search_sops

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_DIM = 1536
_DEFAULT_LIMIT = 2

# ---------------------------------------------------------------------------
# Lazy OpenAI client — only instantiated when USE_VECTOR_DB=true
# ---------------------------------------------------------------------------

_openai_client: Any = None  # openai.AsyncOpenAI instance


def _get_openai_client() -> Any:
    global _openai_client
    if _openai_client is None:
        import openai  # type: ignore[import-untyped]
        _openai_client = openai.AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "placeholder")
        )
    return _openai_client


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------


async def _embed_query(query: str) -> list[float]:
    """Generate a 1536-dimensional embedding for the query using OpenAI."""
    client = _get_openai_client()
    response = await client.embeddings.create(
        model=_EMBEDDING_MODEL,
        input=query,
    )
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Database query helper
# ---------------------------------------------------------------------------

_HYBRID_QUERY_SQL = """
SELECT
    id,
    title,
    content,
    tags,
    recommended_tool,
    (
        -- Cosine similarity component (0–1, higher is better)
        1 - (embedding <=> $1::vector)
        +
        -- Full-text search component (0 or ts_rank)
        COALESCE(
            ts_rank(
                to_tsvector('english', title || ' ' || content),
                plainto_tsquery('english', $2)
            ), 0
        )
    ) AS score
FROM sops_embeddings
ORDER BY score DESC
LIMIT $3;
"""


async def _run_hybrid_query(
    conn: Any,
    embedding: list[float],
    query_text: str,
    limit: int = _DEFAULT_LIMIT,
) -> list[dict]:
    """Run hybrid (vector + FTS) query and return rows as dicts."""
    # asyncpg cannot encode a Python list as pgvector natively.
    # Convert to the "[x,y,z]" text format so PostgreSQL's ::vector cast handles it.
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
    rows = await conn.fetch(_HYBRID_QUERY_SQL, embedding_str, query_text, limit)
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Vector search (with fallback)
# ---------------------------------------------------------------------------


async def _vector_search(query: str, limit: int = _DEFAULT_LIMIT) -> list[dict]:
    """Embed query and run hybrid search against pgvector.

    Raises on connection / embedding failure — callers should use
    _vector_search_with_fallback for resilient behaviour.
    """
    import asyncpg  # type: ignore[import-untyped]

    dsn = os.environ.get("POSTGRES_DSN", "")
    embedding = await _embed_query(query)
    conn = await asyncpg.connect(dsn)
    try:
        return await _run_hybrid_query(conn, embedding, query, limit=limit)
    finally:
        await conn.close()


async def _vector_search_with_fallback(
    query: str, limit: int = _DEFAULT_LIMIT
) -> list[dict]:
    """Attempt vector search; fall back to keyword search on any error."""
    try:
        return await _vector_search(query, limit=limit)
    except Exception as exc:
        logger.warning(
            "Vector search failed (%s), falling back to keyword search", exc
        )
        return search_sops(query)


# ---------------------------------------------------------------------------
# Feature-flag-aware search dispatcher
# ---------------------------------------------------------------------------


def _use_vector_db() -> bool:
    return os.environ.get("USE_VECTOR_DB", "false").lower() == "true"


def _run_async(coro: Any) -> Any:
    """Run a coroutine safely whether or not an event loop is already running."""
    try:
        asyncio.get_running_loop()
        # Already inside a running loop — run in a fresh thread with its own loop
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)


def _search(query: str) -> list[dict]:
    """Synchronous search dispatcher that chooses keyword vs. hybrid search."""
    if _use_vector_db():
        return _run_async(_vector_search_with_fallback(query))
    return search_sops(query)


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------


@tool
def query_knowledge_base(query: str) -> str:
    """Search the SRE knowledge base for runbooks and SOPs matching the query.

    Returns formatted step-by-step remediation instructions, or a 'not found'
    message if no relevant SOPs exist.
    """
    matches = _search(query)

    if not matches:
        return (
            f"No matching SOPs found for query: '{query}'. "
            "Manual investigation recommended. "
            "Consider checking team runbooks or escalating to on-call engineer."
        )

    parts: list[str] = []
    for i, sop in enumerate(matches, start=1):
        parts.append(
            f"[SOP {i}] {sop['title']}\n"
            f"Recommended Action: {sop['recommended_tool']}\n"
            f"Steps:\n{sop['content']}"
        )
    return "\n\n".join(parts)
