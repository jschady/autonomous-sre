-- Migration 005: Update LangGraph checkpoint tables for langgraph-checkpoint-postgres v2.x
--
-- v2.x renamed:
--   checkpoints.parent_id  →  checkpoints.parent_checkpoint_id
--
-- Run this in Supabase SQL Editor (or via psql) once.

DO $$
BEGIN
    -- Rename parent_id to parent_checkpoint_id if the old column still exists
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'checkpoints'
          AND column_name = 'parent_id'
    ) THEN
        ALTER TABLE checkpoints RENAME COLUMN parent_id TO parent_checkpoint_id;
    END IF;

    -- If the table was never created (fresh install), create it with the v2 schema
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'checkpoints'
    ) THEN
        CREATE TABLE checkpoints (
            thread_id            TEXT NOT NULL,
            checkpoint_ns        TEXT NOT NULL DEFAULT '',
            checkpoint_id        TEXT NOT NULL,
            parent_checkpoint_id TEXT,
            type                 TEXT,
            checkpoint           JSONB NOT NULL,
            metadata             JSONB NOT NULL DEFAULT '{}',
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
        );

        CREATE INDEX IF NOT EXISTS checkpoints_thread_id_idx
            ON checkpoints (thread_id);
    END IF;
END;
$$;
