-- RAG v2 schema — fresh Supabase project (parallel A/B deployment)
-- ================================================================
-- Storage for the v2 ingestion pipeline. Table name kept as
-- `documents_gemini` (this is a separate project, so no collision with v1).
--
-- Differences vs the live v1 table:
--   * content_tsv generated column + GIN index  → §6 hybrid keyword search
--   * GIN index on metadata                     → §2/§9/§10 metadata filters
--   * explicit id PK                            → §2 table_source / §10 lookups
--
-- Requires the pgvector extension (Supabase ships it; enable if missing).

create extension if not exists vector;

create table if not exists documents_gemini (
    id         bigint generated always as identity primary key,
    content    text        not null,
    embedding  vector(3072),
    metadata   jsonb       not null default '{}'::jsonb,
    -- §6 full-text search vector, kept in sync automatically
    content_tsv tsvector generated always as (to_tsvector('english', content)) stored,
    created_at timestamptz not null default now()
);

-- ── Vector index (cosine) ──────────────────────────────────────────────────
-- HNSW gives good recall/latency without training. 3072-dim exceeds the 2000-dim
-- limit for hnsw/ivfflat indexes in pgvector, so we index a halfvec cast.
-- If your pgvector build predates halfvec, drop this index and rely on exact
-- scan (fine for a single-doc test corpus), or reduce embedding dimensions.
create index if not exists documents_gemini_embedding_idx
    on documents_gemini
    using hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops);

-- ── §6 keyword (full-text) index ───────────────────────────────────────────
create index if not exists documents_gemini_tsv_idx
    on documents_gemini using gin (content_tsv);

-- ── §2/§9/§10 metadata filter index ────────────────────────────────────────
-- Supports containment/existence queries on content_type, table_source,
-- product_handles, is_safety_critical.
create index if not exists documents_gemini_metadata_idx
    on documents_gemini using gin (metadata jsonb_path_ops);
