# RAG Pipeline Changes — Load Controls Product Docs

## Scope

Applies to `ingestion-server/main.py`, `ingestion/ingest_gemini.py`, and `worker/src/index.ts`. Reference document used for chunking design: `PFR-1750-Installation-Guide.pdf`.

---

## 0. Schema version control (do first)

**Problem:** `match_documents_gemini` and the `documents_gemini` table only exist live in Supabase. No source of truth, no rollback path, and every change below requires editing this function.

**Fix:**

```bash
supabase db pull
# or
supabase db dump --schema public -f supabase/migrations/0000_baseline.sql
```

Commit the migration. All subsequent SQL changes in this doc go into new migration files, not applied ad hoc against the live project.

---

## 1. Chunking — use Docling's structural output, not re-split Markdown

**Current:** `main.py:51-73` exports Docling output to Markdown, then a custom splitter breaks on `\n\n`, packs paragraphs into a 3000-char buffer, with 200-char overlap. The 3000-char size exists specifically to avoid slicing base64 images mid-chunk.

**Problem:** Docling already builds a structural document tree (headings, tables, figures) before flattening to Markdown. Re-splitting the flattened text throws that structure away and re-discovers it poorly via `\n\n` boundaries, which don't reliably align with table or section edges.

**Fix:** Use `docling_core.transforms.chunker.HybridChunker` directly on the `DoclingDocument`, before Markdown export:

```python
from docling_core.transforms.chunker import HybridChunker

chunker = HybridChunker(
    tokenizer="BAAI/bge-small-en-v1.5",  # match your embedding model's tokenizer family
    max_tokens=500,
    merge_peers=True,
)
chunks = list(chunker.chunk(dl_doc=result.document))
```

Each yielded chunk carries `.text` and `.meta`, including heading hierarchy. This guarantees tables and sections aren't split mid-structure, and removes the need to inflate chunk size to dodge embedded images (images are handled separately — see §3).

**Consolidation:** Merge `main.py` and `ingest_gemini.py` into a single ingestion path. Two parsers drifting apart means silent quality gaps between documents ingested at different times.

---

## 2. Table handling — dual representation

**Problem:** A flattened Markdown table (e.g., the PFR-1750 motor sizing chart, 25 rows × 7 columns) embeds as one blurry vector representing all rows at once. A query like "what turns setting for a 15 HP motor" often won't retrieve it, because no single row's signal survives averaging across the whole table, and raw numbers carry little embedding signal without natural-language framing.

**Fix — two chunk types per table:**

1. **Atomic raw table chunk** — the table as extracted, `content_type: "table"`, kept whole, referenced by a `table_source` ID. Used for full-context injection when any related row chunk is retrieved.

2. **Row-expanded chunks** — one natural-language sentence per row, `content_type: "spec_table_row"`, tagged with the same `table_source` ID:

   ```
   For a 15 HP motor at 460V (27.5 full scale capacity, 183% full load),
   use Range Finder Toroid switch position 5, ON, with 1 turn.
   No current transformer is required.
   ```

At generation time: if a `spec_table_row` chunk is retrieved, fetch the full raw table via `table_source` and inject both into context, so the model sees the specific row plus neighboring rows for comparison.

Apply the same treatment to smaller tables (e.g., the voltage Multipliers table) — write explicit sentences ("For 380V nominal, multiply the 460V full scale value by 0.83") rather than leaving bare list items, since query vocabulary ("380 volts", "multiplier") needs to land near each other in the same chunk.

---

## 3. Diagrams and flowcharts — captions and manual rewrites, not `[IMAGE]` stripping

**Current:** Base64 image data is regex-stripped to the literal string `[IMAGE]` before embedding (`main.py:76-85`), and replaced with `[📷 SYSTEM: DIAGRAM/IMAGE X AVAILABLE HERE]` at response time (`index.ts:88-95`).

**Problem A — wiring/photo diagrams:** These chunks are effectively unsearchable. A query about "how many turns for a small motor" can only match surrounding prose text, never the diagram content itself (e.g., "this is one turn" / "this is two turns" images in the Range Finder Toroid section).

**Fix:** At ingestion, run each extracted image through a vision model to generate a searchable caption, embed the caption text (not `[IMAGE]`), and store the actual image path separately:

```json
{
  "content": "Diagram showing a wire looped once through the Range Finder Toroid, labeled \"one turn\" — for motors under 5 HP requiring extra turns to increase current sensing sensitivity.",
  "metadata": { "content_type": "diagram", "image_url": "..." }
}
```

At generation time, if a diagram chunk is retrieved, inject both the caption and the image reference so the model can point the user to the visual and the frontend can render it.

**Problem B — flowchart/state-diagram pages:** Linear PDF text extraction destroys spatial layout. Example from the reference document — the FULL SCALE button-press flow (DIGITS → DECIMAL → DECIMAL → HP → % → KW, each with an ENTER step) extracts as repeated fragments ("if this is / your choice, / press" × 7) with all arrows and structure lost.

**Fix:** For pages identified as flowcharts/state diagrams during ingestion QA, manually rewrite as an explicit numbered procedure rather than relying on automated extraction:

```
To set the full scale display unit on the PFR-1750:
1. Press FULL SCALE to cycle through: DIGITS (XXX.), DECIMAL (XX.X),
   DECIMAL (X.XX), HP, %, KW.
2. When your desired choice is blinking, press ENTER and hold until
   the fast blinking stops to confirm.
Note: no setting changes take effect until ENTER is pressed and blinking stops.
```

Tag `content_type: "procedure"`. This is a recurring pattern in installation-guide-style documents and is worth a manual QA pass per document rather than trusting automated extraction.

---

## 4. Breadcrumb metadata in embeddings

**Problem:** Chunk text alone doesn't encode what section/product it belongs to, so the embedding only captures local wording.

**Fix:** Prepend the heading path (available directly from `HybridChunker`'s `.meta.headings`) to the chunk text before embedding:

```
PFR-1750 > Installation > Mounting

Wiring is done to un-pluggable terminal strips on the rear of the unit...
```

Store the clean heading path separately in metadata for display/filtering.

---

## 5. match_threshold — lower and demote to a pre-filter

**Current:** `match_threshold: 0.1` in the `match_documents_gemini` RPC call (`index.ts:48-85`) — effectively no filtering.

**Fix:** Raise to ~0.35–0.4 (test against your embedding model; thresholds are model-specific) as a coarse pre-filter only:

```sql
where 1 - (documents_gemini.embedding <=> query_embedding) > 0.35
```

The real relevance gate becomes the reranker score (§7), not this threshold. Retrieve wider (`match_count: 25`) at this stage.

---

## 6. Hybrid search — vector + keyword, merged with RRF

**Problem:** Pure vector search misses exact-match terms like model numbers, part numbers, and error codes — common in product documentation queries — because embeddings weight semantic similarity over exact tokens.

**Fix:** Add a `tsvector` column and full-text index:

```sql
alter table documents_gemini add column if not exists content_tsv tsvector
  generated always as (to_tsvector('english', content)) stored;
create index if not exists documents_gemini_tsv_idx on documents_gemini using gin(content_tsv);
```

Run vector search and full-text search in parallel, merge results with reciprocal rank fusion (rank-based fusion avoids score-normalization issues between the two search types). Likely the single highest-impact retrieval change given how often users type SKUs/model numbers verbatim.

---

## 7. Reranking

**Current:** No reranking. Chunks are used in whatever order Supabase returns them (`index.ts:48-95`).

**Fix:** Add a cross-encoder rerank step between the hybrid retrieval (top ~25) and generation, keeping the top 6–8:

```ts
const rerankRes = await fetch("https://api.cohere.com/v2/rerank", {
  method: "POST",
  headers: { Authorization: `Bearer ${COHERE_API_KEY}`, "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "rerank-v3.5",
    query: rewrittenQuery,
    documents: candidates.map(c => c.content),
    top_n: 6,
  }),
});
const { results } = await rerankRes.json();
const topChunks = results
  .filter(r => r.relevance_score > 0.3)   // real relevance gate
  .map(r => candidates[r.index]);
```

If no chunks clear the reranker threshold, return none and instruct the model to state it doesn't have documentation on the topic, rather than generating from weak matches.

---

## 8. Query rewriting

**Problem:** Raw user queries are embedded as-is (`index.ts:27-45`), so pronouns and conversational shorthand ("does it work outdoors") don't resolve to the product-specific vocabulary used in the docs.

**Fix:** Add an LLM rewrite step before embedding, using recent conversation turns:

```ts
const rewritePrompt = `Given the conversation history and the user's latest message, rewrite it as a standalone, specific search query using product terminology (model names, part numbers). Return only the rewritten query.

History: ${JSON.stringify(recentTurns)}
Latest message: ${query}`;

const rewrittenQuery = await callLLM(rewritePrompt, { max_tokens: 100 });
const embedding = await embed(rewrittenQuery);
```

---

## 9. Product-handle-aware retrieval

**Current:** `product_handles` is stored in metadata at ingestion but not used as a retrieval filter.

**Fix:** Add a filter parameter to the RPC:

```sql
create or replace function match_documents_gemini(
  query_embedding vector(3072),
  match_threshold float,
  match_count int,
  filter_product_handles text[] default null
)
returns table (...)
language sql stable
as $$
  select *, 1 - (embedding <=> query_embedding) as similarity
  from documents_gemini
  where (1 - (embedding <=> query_embedding)) > match_threshold
    and (
      filter_product_handles is null
      or metadata->'product_handles' ?| filter_product_handles
    )
  order by embedding <=> query_embedding
  limit match_count;
$$;
```

Resolve the handle in two stages before calling the RPC:

```ts
let productHandles = matchProductHandlesFromQuery(query, productCatalog);
if (!productHandles.length && conversationHistory.length) {
  productHandles = extractHandlesFromHistory(conversationHistory);
}
```

If confidently resolved, hard-filter. If ambiguous, pass `null` and let hybrid search + reranking sort it out rather than filtering out the correct document on a bad guess.

---

## 10. Safety-critical content override

**Problem:** Some content in these documents is safety-critical (e.g., "never leave the CT secondary open — dangerous voltages can develop", "wrong phase causes the control to work backwards or lose sensitivity"). If these chunks score below the rerank threshold on a given query, they're silently dropped — a correctness gap that's also a liability concern, not just a UX one.

**Fix:** Tag such chunks `is_safety_critical: true` at ingestion (identified manually during QA). At retrieval time, force-include them when the resolved product handle matches and the query falls into a related category (wiring, installation, current transformer), regardless of rerank score:

```ts
if (productHandles.length && isWiringRelatedQuery(query)) {
  const safetyChunks = await fetchSafetyCriticalChunks(productHandles);
  topChunks = dedupe([...topChunks, ...safetyChunks]);
}
```

---

## 11. max_tokens

**Current:** `max_tokens: 1024` on the `gemma-4-26b-a4b-it` generation call.

**Fix:** Raise to 4096. The model stops at its own end-of-sequence, so this doesn't lengthen short answers — it only removes the truncation ceiling on long ones. Log `finish_reason`/`stop_reason` to confirm truncation stops occurring; if answers are long because of verbosity rather than necessity, that's a prompt instruction fix (tell the model to be concise and cite sources), not a token-limit fix.

---

## 12. OCR spot-check

**Problem:** `do_ocr=False` assumes fully embedded text. Documents mixing clean embedded text with rasterized/scanned diagram labels can produce silent corruption — e.g., "secondary betrig is grounded" in the reference document, likely a garbled OCR/extraction artifact. A garbled sentence embeds into a garbled vector and fails to retrieve without throwing any error.

**Fix:** Spot-check a sample of already-ingested documents for this pattern. If frequent, either enable OCR selectively for pages with mixed content, or flag documents for manual review during ingestion QA.

---

## Suggested rollout order

| Priority | Change | Why |
|---|---|---|
| 1 | §0 Schema version control | Blocks safe editing of everything else |
| 1 | §11 max_tokens | Five-minute fix |
| 2 | §5 Threshold + §7 Reranking | One coherent change to retrieval + real relevance gate |
| 3 | §6 Hybrid search | Biggest retrieval quality lever given SKU/model-number queries |
| 4 | §1 Chunking + §2 Table handling + §3 Diagrams | Highest effort (requires re-ingestion), fixes issues nothing downstream can recover from |
| Ongoing | §9 Product-handle filtering, §10 Safety override, §4 Breadcrumbs, §8 Query rewriting, §12 OCR audit | Independent, can slot in alongside the above |