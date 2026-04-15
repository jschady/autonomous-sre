"""Ingest SOP markdown files into pgvector.

Reads all *.md files from data/sops/, generates OpenAI embeddings, and
upserts rows into the sops_embeddings table.  Idempotent — running twice
does not create duplicates (uses INSERT ... ON CONFLICT DO UPDATE).

Usage:
    POSTGRES_DSN=postgresql://... OPENAI_API_KEY=sk-... python scripts/ingest_docs.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # type: ignore[import-untyped]

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

DOCS_DIR = PROJECT_ROOT / "data" / "sops"

_EMBEDDING_MODEL = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# Markdown front-matter parser
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Extract YAML-like front matter and return (meta, body)."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    meta: dict[str, Any] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        # Parse list values: [a, b, c]
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1]
            meta[key] = [v.strip() for v in inner.split(",") if v.strip()]
        else:
            meta[key] = value

    body = text[match.end():]
    return meta, body


def _load_sop_file(filepath: Path) -> dict[str, Any]:
    """Parse a SOP markdown file and return a structured dict."""
    raw = filepath.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(raw)

    slug = filepath.stem  # filename without extension = unique id
    title = meta.get("title", slug)
    tags = meta.get("tags", [])
    recommended_tool = meta.get("recommended_tool", "restart_service")

    return {
        "id": slug,
        "title": title,
        "content": body.strip(),
        "tags": tags,
        "recommended_tool": recommended_tool,
    }


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


async def _embed_text(client: Any, text: str) -> list[float]:
    """Generate a 1536-dim embedding via OpenAI."""
    response = await client.embeddings.create(
        model=_EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

_UPSERT_SQL = """
INSERT INTO sops_embeddings (id, title, content, tags, recommended_tool, embedding)
VALUES ($1, $2, $3, $4::text[], $5, $6::vector)
ON CONFLICT (id) DO UPDATE SET
    title            = EXCLUDED.title,
    content          = EXCLUDED.content,
    tags             = EXCLUDED.tags,
    recommended_tool = EXCLUDED.recommended_tool,
    embedding        = EXCLUDED.embedding,
    updated_at       = NOW();
"""


async def _upsert_sop(conn: Any, sop: dict[str, Any], embedding: list[float]) -> None:
    """Upsert a single SOP row."""
    await conn.execute(
        _UPSERT_SQL,
        sop["id"],
        sop["title"],
        sop["content"],
        sop["tags"],
        sop["recommended_tool"],
        str(embedding),
    )


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------


async def ingest_all(dsn: str) -> int:
    """Ingest all SOP markdown files into pgvector.

    Returns the number of documents ingested.
    """
    import asyncpg  # type: ignore[import-untyped]
    import openai  # type: ignore[import-untyped]

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required for ingestion")

    md_files = sorted(DOCS_DIR.glob("*.md"))
    if not md_files:
        logger.warning("No .md files found in %s", DOCS_DIR)
        return 0

    logger.info("Found %d SOP files to ingest", len(md_files))

    openai_client = openai.AsyncOpenAI(api_key=api_key)
    conn = await asyncpg.connect(dsn, statement_cache_size=0)

    try:
        count = 0
        for filepath in md_files:
            sop = _load_sop_file(filepath)
            # Embed title + content for richer semantic search
            embed_text = f"{sop['title']}\n\n{sop['content']}"
            embedding = await _embed_text(openai_client, embed_text)
            await _upsert_sop(conn, sop, embedding)
            logger.info("Ingested: %s (%s)", sop["title"], filepath.name)
            count += 1

        logger.info("Ingestion complete: %d SOPs upserted", count)
        return count
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    dsn = os.environ.get("POSTGRES_DSN", "")
    if not dsn:
        print(
            "ERROR: POSTGRES_DSN environment variable is not set.\n"
            "Example: POSTGRES_DSN=postgresql://sre_user:sre_pass@localhost/sre_db"
        )
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    count = asyncio.run(ingest_all(dsn))
    print(f"Successfully ingested {count} SOP documents.")


if __name__ == "__main__":
    main()
