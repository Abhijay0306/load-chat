# Miss MoMo — How the Agent Works

An end-to-end reference for the **v2 RAG chatbot** ("Miss MoMo") that answers Load
Controls product/technical questions on **loadcontrols.com**, recommends products,
and deep-links into documentation and website pages.

> **v1 vs v2:** v1 is the original, simpler pipeline (kept intact as a baseline).
> Everything below describes **v2**, which is what's live on the site. The two run
> side by side on the same Cloudflare Worker (`/chat` = v1, `/chat/v2` = v2).

---

## 1. High-level architecture

```
                    ┌─────────────────────────────────────────────┐
   Visitor          │  loadcontrols.com (Shopify)                  │
   browser  ───────▶│  <script src=".../embed-v2.js">              │
                    │     └─ floating launcher + <iframe> chat UI  │
                    └───────────────┬─────────────────────────────┘
                                    │  POST /chat/v2 (SSE stream)
                                    ▼
                    ┌─────────────────────────────────────────────┐
                    │  Cloudflare Worker  (rag-worker)             │
                    │   rewrite → intents → embed → hybrid         │
                    │   retrieve → rerank → assemble → generate    │
                    └───┬───────────────┬───────────────┬─────────┘
                        │               │               │
             embeddings │        vector+FTS│        rerank│ generation
             /vision    │        + storage │              │ + rewrite
                        ▼               ▼               ▼
                 ┌────────────┐  ┌──────────────┐  ┌──────────────┐
                 │ OpenRouter │  │ Supabase v2  │  │ OpenRouter   │
                 │ (embed,    │  │ (pgvector +  │  │ (Cohere      │
                 │  vision)   │  │  FTS + files)│  │  rerank,     │
                 └────────────┘  └──────────────┘  │  Gemma)      │
                                                    └──────────────┘
        Shopify Storefront API ──▶ product catalog (titles, images, URLs)

   Ingestion (offline, admin-driven):
     PDFs  ─▶ HF Space /ingest/v2  ─▶ Docling + embeddings ─▶ Supabase v2
     Pages ─▶ HF Space /ingest/web ─▶ crawl + embeddings   ─▶ Supabase v2
```

### The pieces

| Component | Tech / Location | Role |
|---|---|---|
| **Widget loader** | `frontend/embed-v2.js` (Vercel) | One `<script>` on Shopify; floating launcher + iframe |
| **Chat UI** | `frontend/chat-v2.html` (Vercel; also the site root `index.html`) | Streaming chat, product cards, contact buttons, persistence |
| **Worker** | `worker/src/` → Cloudflare `rag-worker` | The actual agent: retrieval + generation pipeline |
| **Ingestion server** | `ingestion-server/` → HF Space (Docker/FastAPI) | Turns PDFs + web pages into embedded chunks |
| **Vector DB** | Supabase project `bwsqmht…` | `documents_gemini` table (pgvector + full-text + product tags) |
| **Admin** | `admin/` (Next.js → Vercel) | Manage the knowledge base + product tagging |
| **LLM services** | OpenRouter | Embeddings, generation, reranking, vision captioning |
| **Product data** | Shopify Storefront API | Catalog: titles, cover images, canonical URLs |

---

## 2. Models & services

| Purpose | Model / API |
|---|---|
| Query + document **embeddings** | `google/gemini-embedding-2` (3072-dim) via OpenRouter |
| **Generation** (the answer) | `google/gemma-4-26b-a4b-it` via OpenRouter |
| **Reranking** | `cohere/rerank-v3.5` via OpenRouter `/api/v1/rerank` |
| **Diagram captioning** (ingestion) | `google/gemini-2.5-flash` (vision) via OpenRouter |
| **Product catalog** | Shopify Storefront GraphQL (`products` → title, handle, `featuredImage`, `onlineStoreUrl`) |

---

## 3. The request pipeline (`/chat/v2`)

Every message from the widget is a `POST /chat/v2` with `{ query, history, stream:true }`.
The worker (`worker/src/chat_v2.ts`) runs these steps:

1. **Fast-path intents (no retrieval, no LLM where possible):**
   - **Casual chat** (`hi`, `thanks`, `who are you`…) → answered conversationally via the persona (skips retrieval so it never says "no documentation").
   - **Catalog** (`what products…`, `categories`, `catalogue`) → a **fixed pre-generated** reply listing the 5 category collection links. No LLM.
   - **Contact** (`how do I contact`, `phone number`, `talk to sales`…) → a **fixed** reply with phone + email (so no retrieval accidentally matches "Auxiliary Contact" in a wiring diagram).

2. **§8 Query rewriting** — an LLM turns the latest message + recent history into a standalone search query (resolves "does it work outdoors?" → the product/topic in context).

3. **§9 Product-handle resolution** — matches model numbers ("PMP-25", "PFR-1750") in the query against the Shopify catalog to (optionally) scope retrieval to a product.

4. **Embed** the rewritten query (`gemini-embedding-2`).

5. **§5/§6 Hybrid retrieval** — Supabase RPC `hybrid_match_documents_gemini` runs **vector search + full-text search in parallel and fuses them with Reciprocal Rank Fusion**, returning a wide pool (top **25**). If a product filter was applied and returns nothing (handle mismatch), it **retries unfiltered** so a product query never dead-ends.

6. **§7 Reranking** — `cohere/rerank-v3.5` reorders the 25 candidates; keep the top **6**.

7. **Relevance gate** — the decision to answer vs. decline uses the **embedding similarity** of the best chunk (≥ `0.35`), not the reranker score (which is too literal for thematic questions). Genuinely off-topic → a decline message (which itself includes phone/email → contact buttons).

8. **§2 Table injection** — if a `spec_table_row` chunk is retrieved, the whole source table is pulled in for full context.

9. **§10 Safety force-include** — for wiring/installation queries on a resolved product, safety-critical chunks are force-added.

10. **Assemble extras:**
    - **RELATED PRODUCTS** — product cards (title, canonical `onlineStoreUrl`, cover image) gathered from the retrieved docs' tags + resolved handles; the catalog is **reconciled** so short tags (`pmp-25-ct200`) map to the real Shopify product.
    - **REFERENCE PAGES** — for website chunks, a **text-fragment deep link** (`…/page#:~:text=start,end`) so the answer can send the user to the exact passage.

11. **§11 Generation** — `gemma-4` with the system prompt (persona + rules), the retrieved context, the extras, and recent conversation **history**. Streamed back to the browser as **Server-Sent Events** (`meta` frame → `token` frames → `done` frame). `max_tokens: 4096`.

The browser renders tokens live, then shows product cards / contact buttons / deep links.

---

## 4. Ingestion (how knowledge gets in)

Runs on the **HF Space** (`ingestion-server/`), triggered from the admin or the batch scripts. All chunks land in the **v2 Supabase** `documents_gemini` table (`content`, `embedding vector(3072)`, `metadata jsonb`, generated `content_tsv`).

### PDFs — `pipeline_v2.py` (`POST /ingest/v2`)
- **Docling HybridChunker** parses the PDF structurally (not naive text splitting).
- **Tables** → dual representation: the whole table (atomic) **plus** one natural-language sentence per row (so numeric rows are searchable).
- **Diagrams** → a **vision model caption** (so images are searchable) + the image uploaded to Supabase Storage (`image_url`).
- **Breadcrumbs** — the heading path is prepended before embedding.
- **Auto-flags** — best-effort flags for flowchart/safety/OCR content, surfaced for manual review.
- `metadata`: `source`, `chunk`, `content_type` (`text|table|spec_table_row|diagram`), `product_handles`, `is_safety_critical`, etc.

### Website — `pipeline_web.py` (`POST /ingest/web`)
- `reingest_web.py` crawls `loadcontrols.com/sitemap.xml` (products, pages, blog articles, metaobject pages) and POSTs page URLs in batches.
- Each page: fetch → strip nav/footer boilerplate (BeautifulSoup) → chunk (~1000 chars) → embed.
- `metadata`: `content_type:"web"`, `source_url` (used to build deep links), page title, and `product_handles` for product pages.

---

## 5. Retrieval schema (Supabase v2)

`supabase/migrations/`:
- **Table** `documents_gemini` — `content`, `embedding vector(3072)`, `metadata jsonb`, `content_tsv tsvector` (generated).
- **Indexes** — HNSW on `embedding::halfvec(3072)` (cosine; 3072-dim exceeds the raw ANN limit, so a halfvec cast), GIN on `content_tsv`, GIN on `metadata`.
- **RPCs** — `match_documents_gemini` (pure vector) and `hybrid_match_documents_gemini` (vector + FTS fused with RRF; the one v2 uses).

Because retrieval always pulls a **fixed** top-25 → top-6 through an ANN index, **query latency stays flat as documents grow** — adding content doesn't slow the chat.

---

## 6. Front-end behavior (`embed-v2.js` + `chat-v2.html`)

**Launcher (`embed-v2.js`, on the Shopify site):**
- Floating round button = the Miss MoMo **MP4** avatar (red border, soft shadow) — ~290 KB (replaced a 3.7 MB GIF for performance).
- **Greeting bubble** pops (animated) once per session ~6–10s after arrival, and on **hover** of the closed button.
- Opens the chat in an isolated **iframe** (CSS can't clash with the theme); button turns into a red ✕ while open.
- **Closes** on navigation and on **click-outside**. Mobile in-chat ✕ button.

**Chat UI (`chat-v2.html`):**
- **Streams** the answer token-by-token (SSE) with a thinking indicator + idle/thinking avatars; DM Sans.
- Renders markdown; **links open in the same browser tab** (`target="_top"`, required from inside the iframe).
- **Product cards** (cover image + title → product page) show when the answer links **or names** a product (matched by model number to the single canonical product).
- **Contact buttons** (red Call / Email) render whenever the answer contains a phone/email — including when the bot can't answer.
- **Conversation persists** in `localStorage`, so history survives page navigation (restored when reopened).

---

## 7. Deployment & infrastructure

| Surface | Where | Deploy |
|---|---|---|
| Worker (`rag-worker`) | Cloudflare (account `development-abacusdigital`) | `wrangler deploy` |
| Chat + widget (`frontend/`) | Vercel project `load-chat` → **loadcontrols.com** via the embed script; site root serves the chatbot | push to `main` |
| Admin (`admin/`) | Vercel project `load-chat-wwnr` | push to `main` |
| Ingestion server | HF Space `adiddev/momo-ingestion` (Docker) | push to the Space git repo |
| Vector DB / storage | Supabase project `bwsqmht…` | migrations in `supabase/` |
| Source control | GitHub `developmentabacusdigital/load-chat` | `main` = prod, `rag-v2` = dev, `main-backup` = pre-v2 snapshot |

**Keep-warm:** a Cloudflare **cron** (`*/5 * * * *`) and a `/warmup` endpoint ping the Supabase projects so the free tier never cold-starts; the chat page also fires `/warmup` on load.

### Configuration (names only — secrets live in each platform)
- **Worker vars:** `SUPABASE_URL`, `SUPABASE_URL_V2`, `SHOPIFY_STORE_DOMAIN` (API host = myshopify), `SHOPIFY_LINK_DOMAIN` (public links = www.loadcontrols.com). **Secrets:** `SUPABASE_KEY`, `SUPABASE_KEY_V2`, `OPENROUTER_API_KEY`, `SHOPIFY_STOREFRONT_TOKEN`.
- **HF Space:** `OPENROUTER_API_KEY`, `SUPABASE_URL`/`SUPABASE_KEY` (v1), `SUPABASE_URL_V2`/`SUPABASE_KEY_V2` (v2).
- **Admin (Vercel):** `NEXT_PUBLIC_HF_SPACE_URL`, `NEXT_PUBLIC_HF_TOKEN`, `SHOPIFY_STORE_DOMAIN`, `SHOPIFY_STOREFRONT_TOKEN`.

---

## 8. Admin (`admin/`)

A Next.js app to manage the knowledge base:
- **Upload** PDFs (→ `/ingest/v2`), **delete**, and **list** documents.
- **Product tagging** per document — cover-image chips, hover-to-remove, multi-select add.
- **V1 ⇄ V2 toggle** — retargets all actions between the two Supabase projects.
- Re-tag products **in place** without re-ingesting (`PATCH /documents/{source}/products`).

---

## 9. End-to-end example

> User (on a product page): *"How do I detect a dry running pump?"*

1. Widget → `POST /chat/v2` (streaming).
2. Not casual/catalog/contact → rewrite → embed.
3. Hybrid retrieval + rerank surface chunks from the **Dry Run Detection** application page and the **PMP-25** docs.
4. Embedding similarity clears the gate → answer proceeds.
5. Extras: a **PMP-25** product card + a **deep link** to the dry-run page section.
6. `gemma-4` streams the answer; the browser renders it live, then shows the PMP-25 card (cover image) and a "Read more" link that **scroll-highlights** the exact passage on the page.

---

*This document describes the v2 system as deployed. Section numbers (§2, §5…) map to the original design notes in `Listofchanges.md`.*
