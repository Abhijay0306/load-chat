-- RAG v2 retrieval functions
-- ==========================
-- §5 threshold as a coarse pre-filter, §6 hybrid (vector + FTS) with RRF,
-- §9 optional product-handle hard filter.
--
-- Note on the halfvec cast: the ANN index in 0001 is built on
-- embedding::halfvec(3072). ORDER BY expressions below cast the same way so the
-- planner can use that index. Cosine distance is `<=>`, so similarity = 1 - dist.

-- ── Pure vector search (§9 filter, §5 threshold) ────────────────────────────
create or replace function match_documents_gemini(
    query_embedding        vector(3072),
    match_threshold        float,
    match_count            int,
    filter_product_handles text[] default null
)
returns table (
    id         bigint,
    content    text,
    metadata   jsonb,
    similarity float
)
language sql stable
as $$
    select
        d.id,
        d.content,
        d.metadata,
        1 - (d.embedding::halfvec(3072) <=> query_embedding::halfvec(3072)) as similarity
    from documents_gemini d
    where 1 - (d.embedding::halfvec(3072) <=> query_embedding::halfvec(3072)) > match_threshold
      and (
            filter_product_handles is null
            or d.metadata->'product_handles' ?| filter_product_handles
          )
    order by d.embedding::halfvec(3072) <=> query_embedding::halfvec(3072)
    limit match_count;
$$;

-- ── §6 Hybrid search: vector + full-text merged with Reciprocal Rank Fusion ──
-- Retrieve wide (call with match_count = 25). The reranker (§7, in the worker)
-- is the real relevance gate; this just produces a good candidate pool.
-- RRF constant k = 60 (standard). Each list contributes 1/(k + rank).
create or replace function hybrid_match_documents_gemini(
    query_embedding        vector(3072),
    query_text             text,
    match_count            int,
    filter_product_handles text[] default null,
    rrf_k                  int default 60
)
returns table (
    id         bigint,
    content    text,
    metadata   jsonb,
    similarity float,
    rrf_score  float
)
language sql stable
as $$
    with
    -- overfetch each arm so fusion has room to reorder
    fetch_n as (select greatest(match_count * 2, 50) as n),
    vector_hits as (
        select
            d.id,
            row_number() over (
                order by d.embedding::halfvec(3072) <=> query_embedding::halfvec(3072)
            ) as rank,
            1 - (d.embedding::halfvec(3072) <=> query_embedding::halfvec(3072)) as similarity
        from documents_gemini d
        where filter_product_handles is null
           or d.metadata->'product_handles' ?| filter_product_handles
        order by d.embedding::halfvec(3072) <=> query_embedding::halfvec(3072)
        limit (select n from fetch_n)
    ),
    keyword_hits as (
        select
            d.id,
            row_number() over (
                order by ts_rank_cd(d.content_tsv, websearch_to_tsquery('english', query_text)) desc
            ) as rank
        from documents_gemini d
        where d.content_tsv @@ websearch_to_tsquery('english', query_text)
          and (
                filter_product_handles is null
                or d.metadata->'product_handles' ?| filter_product_handles
              )
        order by ts_rank_cd(d.content_tsv, websearch_to_tsquery('english', query_text)) desc
        limit (select n from fetch_n)
    ),
    fused as (
        select
            coalesce(v.id, k.id) as id,
            coalesce(1.0 / (rrf_k + v.rank), 0.0)
              + coalesce(1.0 / (rrf_k + k.rank), 0.0) as rrf_score,
            v.similarity as similarity
        from vector_hits v
        full outer join keyword_hits k on v.id = k.id
    )
    select
        d.id,
        d.content,
        d.metadata,
        coalesce(f.similarity, 0.0) as similarity,
        f.rrf_score
    from fused f
    join documents_gemini d on d.id = f.id
    order by f.rrf_score desc
    limit match_count;
$$;
