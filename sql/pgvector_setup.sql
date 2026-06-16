-- ─────────────────────────────────────────────────────────────────────────────
-- Star Health Insurance — pgvector schema upgrade: 384-dim → 768-dim
-- Run this in Supabase SQL Editor BEFORE running reingest_pgvector.py
-- ─────────────────────────────────────────────────────────────────────────────

-- 1. Drop old RPC function and index
DROP FUNCTION IF EXISTS match_policy_chunks(vector(384), int);
DROP INDEX IF EXISTS policy_chunks_embedding_idx;

-- 2. Drop and recreate table with 768-dim embedding
DROP TABLE IF EXISTS policy_chunks;

CREATE TABLE policy_chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_name TEXT NOT NULL,
    chunk_id    INTEGER NOT NULL,
    chunk_text  TEXT NOT NULL,
    embedding   vector(768),
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- 3. IVFFlat index for cosine similarity (lists=30 is fine for <2000 rows)
CREATE INDEX policy_chunks_embedding_idx
ON policy_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 30);

-- 4. RPC function — now uses vector(768) and minimum similarity threshold 0.20
CREATE OR REPLACE FUNCTION match_policy_chunks(
    query_embedding vector(768),
    match_count     int DEFAULT 8
)
RETURNS TABLE(chunk_text text, policy_name text, similarity float)
LANGUAGE sql AS $$
    SELECT
        chunk_text,
        policy_name,
        1 - (embedding <=> query_embedding) AS similarity
    FROM  policy_chunks
    WHERE 1 - (embedding <=> query_embedding) > 0.20
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;

-- Verify
SELECT 'Schema updated successfully for 768-dim embeddings' AS status;
