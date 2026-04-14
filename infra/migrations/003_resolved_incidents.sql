-- Migration 003: resolved_incidents table for few-shot retrieval
-- Requires pgvector extension (already enabled from Phase 2 migrations)

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

-- ivfflat index for approximate nearest-neighbour search
-- lists=100 is appropriate for up to ~1M rows
CREATE INDEX IF NOT EXISTS resolved_incidents_embedding_idx
    ON resolved_incidents USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
