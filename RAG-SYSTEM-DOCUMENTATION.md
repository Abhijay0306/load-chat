# Miss MoMo RAG System — Complete Documentation

A full technical reference for the **Miss MoMo** Retrieval-Augmented-Generation chatbot on
**loadcontrols.com**: what it is, how it works, every component and function, the ingestion
pipelines, the tech stack, configuration, and deployment/operations.

> There are **two pipelines** in the repo: **v1** (original, simple — kept as a baseline) and
> **v2** (the current production system). Both run on the same Cloudflare Worker (`/chat` = v1,
> `/chat/v2` = v2). **Everything the site uses is v2** — this doc focuses on v2 and notes v1 where relevant.

> **Placeholders:** anywhere you see `____` fill in the real value (secrets/URLs are intentionally
> left blank — see [§11 Configuration & Key Values](#11-configuration--key-values)).

---

## Table of contents
1. [What it is](#1-what-it-is)
2. [Tech stack](#2-tech-stack)
3. [Architecture](#3-architecture)
4. [Repository / file structure](#4-repository--file-structure)
5. [Data model (Supabase)](#5-data-model-supabase)
6. [Ingestion pipelines](#6-ingestion-pipelines)
7. [The Worker (retrieval + generation)](#7-the-worker-retrieval--generation)
8. [Front-end (widget + chat UI)](#8-front-end-widget--chat-ui)
9. [Admin app](#9-admin-app)
10. [Deployment](#10-deployment)
11. [Configuration & Key Values](#11-configuration--key-values)
12. [Operations runbook](#12-operations-runbook)
13. [End-to-end request lifecycle](#13-end-to-end-request-lifecycle)

---

## 1. What it is

Miss MoMo answers Load Controls product & technical questions on the Shopify storefront. It:
- Retrieves from **PDF documentation** (datasheets, case studies, whitepapers) **and website pages**.
- **Recommends products** (with cover-image cards + links to the real Shopify product page).
- **Deep-links into website pages** (scroll-to-text highlight of the exact passage).
- Handles **catalog** and **contact** questions with instant canned replies + action buttons.
- **Streams** its answer token-by-token and keeps conversation context across turns/pages.

It is embedded on the site with a single `<script>` tag that injects a floating launcher and an
iframe chat UI.

---

## 2. Tech stack

| Layer | Technology |
|---|---|
| **Retrieval API / agent** | **Cloudflare Worker** (TypeScript) — `worker/` |
| **LLM gateway** | **OpenRouter** (embeddings, generation, reranking, vision) |
| **Embeddings** | `google/gemini-embedding-2` — **3072-dim** |
| **Generation** | `google/gemma-4-26b-a4b-it` |
| **Reranking** | `cohere/rerank-v3.5` (via OpenRouter `/api/v1/rerank`) |
| **Vision captioning** (ingestion) | `google/gemini-2.5-flash` |
| **Vector DB** | **Supabase Postgres + pgvector** (HNSW) + **full-text search** (tsvector/GIN) |
| **Ingestion server** | **FastAPI** (Python) on a **HuggingFace Space** (Docker) — `ingestion-server/` |
| **PDF parsing** | **Docling** (`HybridChunker`) |
| **Web crawling** | `httpx` + **BeautifulSoup4** |
| **Product data** | **Shopify Storefront GraphQL API** |
| **Admin UI** | **Next.js 14** (React + Tailwind) on **Vercel** — `admin/` |
| **Chat UI / widget** | Static HTML/JS (marked.js, DM Sans) on **Vercel** — `frontend/` |
| **Source control** | GitHub — `developmentabacusdigital/load-chat` |

---

## 3. Architecture

```
   Visitor browser (loadcontrols.com, Shopify)
        │  <script src=".../embed-v2.js">  → floating launcher + <iframe> chat
        │  POST /chat/v2  (SSE stream)
        ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  Cloudflare Worker  "rag-worker"                             │
   │  intents → rewrite → embed → hybrid retrieve → rerank →      │
   │  gate → table/safety → product+page links → generate (SSE)  │
   └───┬───────────────────┬───────────────────┬────────────────┘
       │ embed / gen / rerank│ vector + FTS + files│ product catalog
       ▼                     ▼                     ▼
   OpenRouter          Supabase (v2)          Shopify Storefront API
   (gemini-embed,      documents_gemini
    gemma, cohere,     + pgvector + FTS
    gemini-vision)     + Storage (images)

   Ingestion (offline, admin-driven, HF Space):
     PDFs  → /ingest/v2  → Docling + vision + embeddings → Supabase v2
     Pages → /ingest/web → crawl + embeddings            → Supabase v2

   Admin (Next.js on Vercel) → calls the HF Space to upload / list / delete / re-tag docs
```

---

## 4. Repository / file structure

```
Improved_RAG/
├── worker/                       # Cloudflare Worker — the agent
│   ├── src/
│   │   ├── index.ts              # router + v1 pipeline + /warmup + cron keep-warm
│   │   └── chat_v2.ts            # v2 pipeline (retrieval + generation) — the core agent
│   ├── wrangler.toml             # worker config: vars + cron triggers
│   ├── .dev.vars.example         # local secrets template
│   ├── package.json / tsconfig.json
│
├── ingestion-server/             # FastAPI on HuggingFace Space (Docker)
│   ├── main.py                   # routes: /ingest(/v2/web), /documents(/v2), re-tag, health
│   ├── pipeline_v2.py            # PDF → Docling chunks + vision captions + embeddings → v2 DB
│   ├── pipeline_web.py           # website pages → extract + chunk + embeddings → v2 DB
│   ├── reingest_v2.py            # batch client: upload PDFs to /ingest/v2
│   ├── reingest_web.py           # batch client: crawl sitemap → /ingest/web
│   ├── Dockerfile / requirements.txt / README.md (HF Space frontmatter)
│
├── supabase/migrations/          # v2 database schema (source of truth)
│   ├── 0001_documents_v2.sql     # table + indexes
│   └── 0002_functions.sql        # match_ + hybrid_match_ RPCs
│
├── admin/                        # Next.js admin (Vercel)
│   ├── app/page.tsx              # knowledge-base UI (upload, list, delete, tag, v1/v2 toggle)
│   ├── app/api/products/route.ts # Shopify product list (server-side)
│   ├── app/api/ingest/route.ts   # (proxy helper)
│   └── app/layout.tsx / globals.css / tailwind / next config
│
├── frontend/                     # Static chat + widget (Vercel)
│   ├── embed-v2.js               # the ONE script embedded on Shopify (launcher + iframe)
│   ├── chat-v2.html              # the chat UI (also served as index.html = site root)
│   ├── index.html                # copy of chat-v2.html (production root)
│   ├── index-v1.html             # previous landing page (archived)
│   ├── test-v2.html              # v1-vs-v2 comparison harness
│   ├── momo-float.mp4            # launcher animation (idle avatar, ~290KB)
│   ├── momo.gif / Thinking_Momo.gif  # in-chat avatars (idle / thinking)
│   └── widget.js / app.js / styles.css  # v1 widget assets
│
├── RAG-SYSTEM-DOCUMENTATION.md   # (this file)
├── HOW-IT-WORKS.md               # shorter architecture overview
└── Listofchanges.md              # original v2 design notes (§1–§12 referenced in code comments)
```

---

## 5. Data model (Supabase)

**Two projects:** v1 (`____v1-project-ref____`, live/legacy) and **v2** (`bwsqmhtacmdmzscvjcqa`, current).
Both have a table named `documents_gemini`. v2 schema lives in `supabase/migrations/`.

### Table `documents_gemini` (`0001_documents_v2.sql`)
| Column | Type | Notes |
|---|---|---|
| `id` | `bigint` identity PK | |
| `content` | `text` | the chunk text (what gets embedded/searched) |
| `embedding` | `vector(3072)` | gemini-embedding-2 |
| `metadata` | `jsonb` | see below |
| `content_tsv` | `tsvector` **generated** | `to_tsvector('english', content)` — full-text search |
| `created_at` | `timestamptz` | |

**Indexes:** HNSW on `embedding::halfvec(3072)` (cosine — 3072-dim exceeds the raw ANN limit, hence the halfvec cast); GIN on `content_tsv`; GIN on `metadata`.

**`metadata` keys:** `source` (filename or page title), `chunk` (index), `content_type`
(`text|table|spec_table_row|diagram|web`), `product_handles` (string[]), `headings` (breadcrumb),
`is_safety_critical` (bool), `review_flags`, `table_source` (links rows↔table), `image_url`
(diagram image in Storage), `source_url` (web pages — used for deep links), `engine`.

### RPC functions (`0002_functions.sql`)
- **`match_documents_gemini(query_embedding, match_threshold, match_count, filter_product_handles)`** — pure vector search with optional product filter.
- **`hybrid_match_documents_gemini(query_embedding, query_text, match_count, filter_product_handles, rrf_k)`** — **the one v2 uses**: runs vector search + `websearch_to_tsquery` full-text search and fuses them with **Reciprocal Rank Fusion** (k=60); returns fused candidates with `similarity` + `rrf_score`.

### Supabase Storage
Bucket **`rag-v2-images`** (public) — diagram images uploaded during PDF ingestion; the public URL is stored in `metadata.image_url`.

---

## 6. Ingestion pipelines

Runs on the **HF Space** (`ingestion-server/`), triggered from the admin or the `reingest_*.py`
batch clients. All chunks land in **Supabase v2** `documents_gemini`.

### 6a. PDF pipeline — `pipeline_v2.py` (route `POST /ingest/v2`)
Flow: `main.py:ingest_v2` converts the PDF once with **Docling**, then calls
`pipeline_v2.ingest_document_v2(result, source, handles, replace)` which builds chunks and saves them.

| Function | Role |
|---|---|
| `build_text_chunks(dl_doc)` | **Docling `HybridChunker`** (max_tokens≈500) → structural text chunks + heading **breadcrumbs** prepended before embedding |
| `build_table_chunks(dl_doc, source)` | **Dual representation**: one atomic whole-table chunk (`content_type:table`) **plus** one NL sentence per row (`spec_table_row`, via LLM `expand_table_rows`) so numeric rows are searchable; both share a `table_source` id |
| `build_diagram_chunks(dl_doc, source)` | For each figure: **vision caption** (`caption_image`, gemini-2.5-flash) as the searchable text + `upload_image` to Storage → `image_url` |
| `annotate_flags(chunk)` | Best-effort heuristics → `is_safety_critical` + `review_flags` (flowchart/ocr/safety) |
| `embed_text(text)` | OpenRouter embeddings (strips base64 before embedding) |
| `save_chunks_v2(chunks, source, handles)` | embeds + inserts each chunk with full `metadata` |
| `delete_source_v2(source)` | delete all chunks for a doc (used on `replace=true`) |
| `set_product_handles_v2(source, handles)` | **re-tag** a doc's `product_handles` in place (no re-ingest) |
| `list_documents_v2()` | list docs (chunk-0 rows) |

### 6b. Web pipeline — `pipeline_web.py` (route `POST /ingest/web`)
Client `reingest_web.py` reads `loadcontrols.com/sitemap.xml` (products, pages, blog articles,
metaobject pages) and POSTs page URLs in batches of 10.

| Function | Role |
|---|---|
| `fetch_html(url)` | GET the page (httpx, follow redirects) |
| `extract_page(url)` | BeautifulSoup: strip nav/header/footer/script boilerplate → `(title, main_text)` |
| `chunk_web(text, size=1000, overlap=150)` | char-based packing (small chunks → precise deep-link highlights) |
| `save_web_chunks(url, title, chunks)` | embed + insert with `content_type:"web"`, `source_url:url`, product handle derived from `/products/<handle>` URLs |
| `ingest_urls(urls, replace)` | orchestrates; skips cart/checkout/account/policy URLs |

**Current volume:** ~90 PDF chunks + **~583 web chunks**.

---

## 7. The Worker (retrieval + generation)

**Entry:** `worker/src/index.ts` `export default { fetch, scheduled }`.

### Endpoints
| Route | Method | Purpose |
|---|---|---|
| `/chat` | POST | v1 pipeline (`handleChat`) — legacy baseline |
| `/chat/v2` | POST | **v2 pipeline** (`handleChatV2`) — production |
| `/health` | GET | health check |
| `/warmup` | GET | `warmDatabases(env)` — pings both Supabase projects |
| *(cron)* | scheduled | `*/5 * * * *` → `warmDatabases` keep-warm |

Request body for `/chat/v2`: `{ query, history?: {role,content}[], stream?: boolean }`.
With `stream:true` it returns **Server-Sent Events**: a `meta` frame → `token` frames → `done` frame.

### `handleChatV2` pipeline (`chat_v2.ts`), in order
1. **Fast-path intents** (no retrieval):
   - `isCasualChat` → persona reply (greetings/thanks/who-are-you).
   - `isCatalogQuery` → **fixed** 5-category catalog message (`CATALOG_MSG`).
   - `isContactQuery` → **fixed** contact reply with phone/email (`CONTACT_MSG`) — avoids matching "Auxiliary Contact" in a diagram.
2. `rewriteQuery(key, query, history)` — LLM rewrites to a standalone search query.
3. `fetchProductCatalog(env)` + `resolveProductHandles(...)` — resolve product/model numbers from the query (edge-cached Shopify catalog).
4. `embedQuery(key, rewritten)` — 3072-dim embedding.
5. `hybridRetrieve(env, embedding, rewritten, handles)` — RPC `hybrid_match_documents_gemini`, wide pool (**25**). If a product filter yields nothing (handle mismatch), **retries unfiltered**.
6. `rerank(key, rewritten, candidates)` — Cohere; keep top **6**.
7. **Relevance gate** — decline only if the best chunk's **embedding similarity < `SIM_GATE` (0.35)** and no product resolved (reranker is used for ordering, not vetoing).
8. `fetchTable` (§2 table injection) + `fetchSafetyChunks` (§10 safety force-include).
9. **Assemble extras** — product cards (`matchCatalog` reconciles short tags→real Shopify product, uses `onlineStoreUrl` + `featuredImage`) and **web deep links** (`deepLink` builds `…#:~:text=start,end`).
10. `generate` / `generateStreaming(key, query, ctxBlocks, extras, history, send)` — Gemma, `max_tokens 4096`, streamed via SSE. `buildMessages` prepends the system prompt + last **10** history turns.

### Key constants (`chat_v2.ts`)
`WIDE_MATCH_COUNT=25`, `RERANK_TOP_N=6`, `RERANK_GATE=0.1`, `SIM_GATE=0.35`,
`HISTORY_TURNS=10`, `HISTORY_CHAR_CAP=2000`, generation `max_tokens=4096`.

### Function map (`chat_v2.ts`)
`jsonResponse`, `sseResponse` (SSE framing) · `rewriteQuery` · `embedQuery` · `fetchProductCatalog`,
`resolveProductHandles`, `matchCatalog` · `dbAuth`, `hybridRetrieve`, `rerank`, `fetchTable`,
`fetchSafetyChunks` · `buildMessages`, `buildUserMessage`, `generate`, `generateStreaming` ·
`toContextBlock`, `deepLink` · `isCasualChat`, `isCatalogQuery`, `isContactQuery` · `handleChatV2`.

### Response `meta` (returned/streamed)
`sources`, `product_candidates` [{title, handle, url, image}], `web_pages` [{title, url}],
`images`, `rewritten_query`, `product_handles`, `engine`, `finish_reason`, `debug_*`.

---

## 8. Front-end (widget + chat UI)

### `embed-v2.js` — the launcher (the single script on Shopify)
Add to the theme before `</body>`:
```html
<script src="https://____chat-host____/embed-v2.js" defer></script>
```
- Floating round **MP4 launcher** (`momo-float.mp4`, red border, soft shadow); opens `chat-v2.html` in an **isolated iframe**; button becomes a red ✕ when open.
- **Greeting bubble** — animated pop, once/session ~6–10s after arrival **and on hover** of the closed button.
- **Closes** on navigation and on **click-outside**; mobile has an in-chat ✕ (posts `missmomo-close` to the parent).
- Config overrides via `window.MISSMOMO = { chatUrl, video, greeting, color }`.

### `chat-v2.html` — the chat UI (also the site root `index.html`)
- **Streams** answers token-by-token (SSE) with a thinking indicator + idle/thinking avatars; DM Sans; renders markdown.
- **Links open in the same browser tab** via `target="_top"` (required from inside the iframe; plain links would try to load inside the frame and fail).
- **Product cards** show when the answer **links or names** a product (matched by exact model token from the title → the single canonical product, capped at 4).
- **Contact buttons** (red Call/Email) render whenever an answer contains a phone/email — including "can't answer" replies.
- **Conversation persistence** — saved in `localStorage`, restored on any page (history survives navigation).
- Key functions: `streamReply`, `renderAnswer`, `modelTokens`, `addMessage`, `setStatus`, `saveConvo`/`restoreConvo`, `send`.

---

## 9. Admin app (`admin/`, Next.js on Vercel)

`app/page.tsx` — knowledge-base management UI:
- **Upload** PDFs (→ HF `/ingest/v2`), **list**, **delete**, **re-tag** products in place.
- **Product tagging** — cover-image chips, hover-to-remove, multi-select add popover.
- **V1 ⇄ V2 toggle** — retargets all actions between the two Supabase projects (via the v1/v2 routes on the same HF Space).
- Calls the HF Space **directly** with `NEXT_PUBLIC_HF_SPACE_URL` + `NEXT_PUBLIC_HF_TOKEN`.
- `app/api/products/route.ts` — server-side Shopify Storefront product list (title, handle, image) for the tag picker.

---

## 10. Deployment

| Surface | Where | How to deploy |
|---|---|---|
| **Worker** (`rag-worker`) | Cloudflare, account `development-abacusdigital`, URL `https://rag-worker.development-abacusdigital.workers.dev`, cron `*/5` | `cd worker && npx wrangler deploy` |
| **Chat + widget** (`frontend/`) | Vercel project `load-chat` → production alias `load-chat-ruddy.vercel.app`; embedded on **loadcontrols.com**; site root serves the chatbot | push to `main` |
| **Admin** (`admin/`) | Vercel project `load-chat-wwnr` | push to `main` |
| **Ingestion server** | HF Space `adiddev/momo-ingestion` (Docker SDK) | push to the **Space git repo** (copy `ingestion-server/*` to the Space root) |
| **Vector DB + storage** | Supabase project `bwsqmhtacmdmzscvjcqa` (v2) | migrations in `supabase/` (apply via `psql`/CLI) |
| **Source** | GitHub `developmentabacusdigital/load-chat` | `main` = production · `rag-v2` = dev · `main-backup` = pre-v2 snapshot |

**Deploy the ingestion Space** (files live in a subfolder here, but the Space expects them at its root):
```bash
git clone https://huggingface.co/spaces/adiddev/momo-ingestion /tmp/space
cp ingestion-server/{main.py,pipeline_v2.py,pipeline_web.py,requirements.txt,Dockerfile,README.md} /tmp/space/
cd /tmp/space && git add -A && git commit -m "deploy" && git push   # HF username + write token
```

**Rollback production frontend/admin:** reset `main` to `main-backup`.

---

## 11. Configuration & Key Values

> Fill in the blanks. **Do not commit real secret values.**

### Cloudflare Worker (`wrangler.toml` vars + `wrangler secret put`)
| Name | Type | Value |
|---|---|---|
| `SUPABASE_URL` | var | `https://____v1-ref____.supabase.co` |
| `SUPABASE_URL_V2` | var | `https://bwsqmhtacmdmzscvjcqa.supabase.co` |
| `SHOPIFY_STORE_DOMAIN` | var | `____.myshopify.com` (Storefront **API** host) |
| `SHOPIFY_LINK_DOMAIN` | var | `www.loadcontrols.com` (public product links) |
| `SUPABASE_KEY` | secret | `____` (v1 service-role key) |
| `SUPABASE_KEY_V2` | secret | `____` (v2 service-role key) |
| `OPENROUTER_API_KEY` | secret | `____` |
| `SHOPIFY_STOREFRONT_TOKEN` | secret | `____` |

### HF Space (Settings → Variables & secrets)
| Name | Value |
|---|---|
| `OPENROUTER_API_KEY` | `____` |
| `SUPABASE_URL` / `SUPABASE_KEY` | `____` (v1) |
| `SUPABASE_URL_V2` / `SUPABASE_KEY_V2` | `https://bwsqmhtacmdmzscvjcqa.supabase.co` / `____` |

### Admin (Vercel `load-chat-wwnr` → Environment Variables)
| Name | Value |
|---|---|
| `NEXT_PUBLIC_HF_SPACE_URL` | `https://adiddev-momo-ingestion.hf.space` |
| `NEXT_PUBLIC_HF_TOKEN` | `____` (HF token) |
| `SHOPIFY_STORE_DOMAIN` | `____.myshopify.com` |
| `SHOPIFY_STOREFRONT_TOKEN` | `____` |
| `INGEST_API_KEY` | `____` |

### Other identifiers
| Thing | Value |
|---|---|
| v2 Supabase project ref | `bwsqmhtacmdmzscvjcqa` |
| v2 Supabase DB password | `____` |
| HF Space | `adiddev/momo-ingestion` · `https://adiddev-momo-ingestion.hf.space` |
| HF write token | `____` |
| Cloudflare account | `development-abacusdigital` |
| Vercel projects | `load-chat` (chat/widget) · `load-chat-wwnr` (admin) |
| GitHub repo | `developmentabacusdigital/load-chat` |

---

## 12. Operations runbook

### Keep-warm
A Cloudflare **cron `*/5`** calls `warmDatabases` (pings both Supabase projects); the chat page also
hits `/warmup` on load. This prevents the free-tier Supabase **cold start**.

### The HF Space sleeps (⚠ common 500 cause)
Free HF Spaces **sleep after ~48h idle** (`gcTimeout` 172800s). While asleep, the **admin's**
upload/list/delete/re-tag calls fail with **500** (the live chatbot is unaffected — it uses the
worker + Supabase). **Fix:** wake it —
```
curl -sL -X POST https://huggingface.co/api/spaces/ADIDDev/momo-ingestion/restart -H "Authorization: Bearer ____HF_TOKEN____"
```
…then poll `https://adiddev-momo-ingestion.hf.space/health` until `200`.
**Permanent fix (recommended):** add the Space `/health` to `warmDatabases` in the worker cron, or use HF's paid always-on tier.

### Re-ingest content
- PDFs: upload via the admin (V2 mode), or `python ingestion-server/reingest_v2.py "Test Documents"`.
- Website: `python ingestion-server/reingest_web.py` (re-crawls the sitemap).
  (Both need `INGEST_BASE_URL` + `HF_TOKEN` env.)

### Common issues
| Symptom | Likely cause | Fix |
|---|---|---|
| Admin actions 500 | HF Space asleep | restart the Space (above) |
| Chat "no documentation" for a valid product Q | product handle mismatch | fallback retry already handles it; re-tag with correct handle if needed |
| Product card has no image | tag ≠ Shopify handle | `matchCatalog` reconciles by containment; ensure the product exists in the catalog |
| First chat after idle slow | Supabase cold start | keep-warm cron covers it |
| Link says "refused to connect" | link tried to load inside the iframe | links use `target="_top"` (fixed) |

---

## 13. End-to-end request lifecycle

> Visitor on a product page asks: *"How do I detect a dry running pump?"*

1. Widget → `POST /chat/v2` (`stream:true`, with history).
2. Not casual/catalog/contact → `rewriteQuery` → `embedQuery`.
3. `hybridRetrieve` (vector + FTS + RRF, top 25) → `rerank` (top 6): chunks from the **Dry Run
   Detection** page + **PMP-25** docs.
4. Embedding similarity clears `SIM_GATE` → proceed.
5. Extras: a **PMP-25 product card** (via `matchCatalog` → cover image + `onlineStoreUrl`) + a **deep
   link** to the dry-run page section (`deepLink`).
6. `generateStreaming` (Gemma) streams tokens as SSE; the browser renders live, then shows the product
   card and a "Read more" link that scroll-highlights the exact passage.

---

*Section numbers (§2, §5, §10…) reference the original design notes in `Listofchanges.md`. For a
shorter overview see `HOW-IT-WORKS.md`.*
