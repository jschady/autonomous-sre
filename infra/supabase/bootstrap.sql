-- Supabase bootstrap: run this once in the Supabase SQL Editor
-- after creating a new project with pgvector enabled.
--
-- Enables pgvector, runs all application migrations, and creates the
-- LangGraph checkpoint tables required for durable graph resumption.
--
-- Order matters:
--   1. Extensions
--   2. Application tables (resolved_incidents, alert_costs)
--   3. LangGraph checkpointer tables

-- ---------------------------------------------------------------------------
-- 1. Extensions
-- ---------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- 2. resolved_incidents — few-shot retrieval store
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS resolved_incidents (
    id                 SERIAL PRIMARY KEY,
    alert_id           TEXT NOT NULL UNIQUE,
    alertname          TEXT NOT NULL,
    namespace          TEXT NOT NULL DEFAULT 'default',
    error_summary      TEXT NOT NULL,
    triage_summary     TEXT NOT NULL DEFAULT '',
    recommended_action TEXT NOT NULL,
    action_result      TEXT NOT NULL DEFAULT 'Resolved',
    severity           TEXT NOT NULL DEFAULT 'unknown',
    embedding          vector(1536),
    resolved_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata           JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS resolved_incidents_alertname_idx
    ON resolved_incidents (alertname);

CREATE INDEX IF NOT EXISTS resolved_incidents_embedding_idx
    ON resolved_incidents USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ---------------------------------------------------------------------------
-- 3. alert_costs — per-alert LLM cost tracking
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS alert_costs (
    id             SERIAL PRIMARY KEY,
    alert_id       TEXT NOT NULL UNIQUE,
    alertname      TEXT NOT NULL DEFAULT 'unknown',
    total_tokens   INTEGER NOT NULL DEFAULT 0,
    input_tokens   INTEGER NOT NULL DEFAULT 0,
    output_tokens  INTEGER NOT NULL DEFAULT 0,
    cost_usd       NUMERIC(12, 8) NOT NULL DEFAULT 0,
    cost_saved_usd NUMERIC(12, 8) NOT NULL DEFAULT 0,
    node_breakdown JSONB NOT NULL DEFAULT '[]'::jsonb,
    resolved       BOOLEAN NOT NULL DEFAULT FALSE,
    llm_provider   TEXT NOT NULL DEFAULT 'claude',
    cache_hit      BOOLEAN NOT NULL DEFAULT FALSE,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS alert_costs_alertname_idx
    ON alert_costs (alertname);

CREATE INDEX IF NOT EXISTS alert_costs_created_at_idx
    ON alert_costs (created_at DESC);

-- ---------------------------------------------------------------------------
-- 4. LangGraph checkpoint tables (AsyncPostgresSaver)
-- AsyncPostgresSaver.setup() creates these automatically, but they are
-- included here for reference and manual Supabase SQL Editor runs.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id            TEXT NOT NULL,
    checkpoint_ns        TEXT NOT NULL DEFAULT '',
    checkpoint_id        TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type                 TEXT,
    checkpoint           JSONB NOT NULL,
    metadata             JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id     TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel       TEXT NOT NULL,
    version       TEXT NOT NULL,
    type          TEXT NOT NULL,
    blob          BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);

CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id     TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id       TEXT NOT NULL,
    idx           INTEGER NOT NULL,
    channel       TEXT NOT NULL,
    type          TEXT,
    blob          BYTEA NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

CREATE INDEX IF NOT EXISTS checkpoints_thread_id_idx
    ON checkpoints (thread_id);
