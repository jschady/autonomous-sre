"""Idempotent SQL migration for the Autonomous SRE system.

Creates the pgvector extension and sops_embeddings table if they do not
already exist.  Safe to run multiple times.

Usage:
    POSTGRES_DSN=postgresql://... python scripts/migrate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
import logging
import os

import asyncpg  # type: ignore[import-untyped]
from dotenv import load_dotenv  # type: ignore[import-untyped]

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL — kept as module-level constants so tests can inspect them
# ---------------------------------------------------------------------------

CREATE_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector;"

CREATE_SOPS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sops_embeddings (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    content     TEXT NOT NULL,
    tags        TEXT[] NOT NULL DEFAULT '{}',
    recommended_tool TEXT NOT NULL DEFAULT '',
    embedding   vector(1536),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

CREATE_SOPS_EMBEDDING_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS sops_embeddings_ivfflat_idx
    ON sops_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);
"""

CREATE_SOPS_FTS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS sops_embeddings_fts_idx
    ON sops_embeddings
    USING GIN (to_tsvector('english', title || ' ' || content));
"""

ALL_MIGRATIONS: list[tuple[str, str]] = [
    ("create_vector_extension", CREATE_EXTENSION_SQL),
    ("create_sops_embeddings_table", CREATE_SOPS_TABLE_SQL),
    ("create_sops_ivfflat_index", CREATE_SOPS_EMBEDDING_INDEX_SQL),
    ("create_sops_fts_index", CREATE_SOPS_FTS_INDEX_SQL),
]


async def run_migrations(dsn: str) -> None:
    """Execute all migrations idempotently against the given Postgres DSN."""
    conn = await asyncpg.connect(dsn)
    try:
        for name, sql in ALL_MIGRATIONS:
            logger.info("Running migration: %s", name)
            await conn.execute(sql)
            logger.info("Migration OK: %s", name)
    finally:
        await conn.close()


def main() -> None:
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        print(
            "ERROR: POSTGRES_DSN environment variable is not set.\n"
            "Example: POSTGRES_DSN=postgresql://sre_user:sre_pass@localhost/sre_db"
        )
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(run_migrations(dsn))
    print("All migrations completed successfully.")


if __name__ == "__main__":
    main()
