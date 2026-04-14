-- Migration 004: alert_costs table for cost reporting

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
