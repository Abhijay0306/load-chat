# 🐮 Miss MoMo — RAG Chatbot for Load Controls

Miss MoMo is a production **Retrieval-Augmented Generation (RAG)** assistant embedded on
[www.loadcontrols.com](https://www.loadcontrols.com). It answers product and technical questions using
Load Controls' own datasheets, manuals, and website content — grounded in real sources, with product
cards, contact buttons, and deep links to the exact page section.

---

## 🔗 Links

| | |
|---|---|
| 🌐 **Live site** (chatbot embedded) | https://www.loadcontrols.com |
| 🛠️ **Admin app** (manage the knowledge base) | https://load-chat-wwnr.vercel.app/ |
| 🤗 **Ingestion Space** (Hugging Face) | https://huggingface.co/spaces/ADIDDev/momo-ingestion · API: `https://adiddev-momo-ingestion.hf.space` |
| 📦 **Repo (company)** | https://github.com/developmentabacusdigital/load-chat |
| 📦 **Repo (fork)** | https://github.com/Abhijay0306/load-chat |

---

## 📚 Documentation

Start here, in roughly this order:

| Doc | What it's for |
|---|---|
| **[GOD_MOMO.md](GOD_MOMO.md)** | ⭐ The ultimate reference — an exhaustive FAQ for users, developers, maintainers, and updaters. |
| **[HOW-IT-WORKS.md](HOW-IT-WORKS.md)** | Short architecture overview of the v2 agent. |
| **[RAG-SYSTEM-DOCUMENTATION.md](RAG-SYSTEM-DOCUMENTATION.md)** | Deep, section-by-section code & architecture reference. |
| **[RAG-VERSIONS.md](RAG-VERSIONS.md)** | How v1 and v2 differ, and why v2 exists. |
| **[MANAGING-KNOWLEDGE-BASE.md](MANAGING-KNOWLEDGE-BASE.md)** | Step-by-step: ingest, delete, and connect documents to Shopify products. |
| **[HANDOVER.md](HANDOVER.md)** | Access checklist & operations runbook (also available as `HANDOVER.docx`). |

---

## 🧠 How it works (in one glance)

**Live chat path** (runs on every message):

```
Visitor → Widget (Vercel) → Cloudflare Worker /chat/v2
   → rewrite query → resolve product → embed (OpenRouter)
   → hybrid search (Supabase: vector + keyword, RRF) → rerank (Cohere)
   → relevance gate → build context → Gemma LLM (streamed over SSE)
   → answer + product cards + deep links → back to the widget
```

**Ingestion path** (runs only when you add documents — *not* during chat):

```
PDF / URL → Hugging Face Space (FastAPI) → parse (Docling / BeautifulSoup)
   → chunk → embed → insert rows into Supabase (documents_gemini)
```

> **Golden rule:** the Hugging Face Space is **ingestion-only** and is **not** in the chat path.
> If chat breaks, HF is almost never the cause. If you can't add documents, HF usually is.

---

## 🏗️ Tech stack

- **Chat backend:** Cloudflare Worker (TypeScript, SSE streaming) — `worker/`
- **Ingestion backend:** FastAPI on a Hugging Face Space (Docker) — `ingestion-server/`
- **Knowledge base:** Supabase Postgres + pgvector (HNSW) + full-text (GIN), hybrid retrieval via RRF — `supabase/`
- **Front-end:** embeddable widget on Vercel — `frontend/`
- **Admin:** Next.js app on Vercel — `admin/`
- **Models (via OpenRouter):** `google/gemini-embedding-2` (3072-dim) · `google/gemma-4-26b-a4b-it` · `cohere/rerank-v3.5`
- **Storefront:** Shopify (`load-controls.myshopify.com` / `www.loadcontrols.com`)

---

## 📁 Repository structure

```
worker/             Cloudflare Worker — the chat backend (v1 /chat, v2 /chat/v2)
  src/chat_v2.ts    The v2 agent pipeline (rewrite → hybrid retrieve → rerank → gate → generate)
  src/index.ts      Router, v1 pipeline, keep-warm cron
ingestion-server/   FastAPI ingestion (PDF + web) that fills the knowledge base
supabase/           DB schema + retrieval RPCs (migrations)
frontend/           Embeddable chat widget (launcher + chat UI)
admin/              Next.js admin app for managing documents & product tags
```

---

## 🚀 Deploy (quick reference)

```bash
# Chat logic (Cloudflare Worker)
cd worker && npx wrangler deploy

# Front-end / admin (Vercel) — push the branch
git push origin rag-v2     # → Vercel preview
git push origin main       # → Vercel production

# Ingestion (Hugging Face Space) — push to the Space git remote → Docker rebuild
git push space main
```

See **[HANDOVER.md](HANDOVER.md)** for access, secrets, monitoring, and the full runbook.

---

## 🔀 Versions

- **v2** (`/chat/v2`) — the **current production** experience: hybrid retrieval + rerank, streaming,
  conversation memory, live product cards, and page deep links.
- **v1** (`/chat`) — the original, simpler pipeline, kept intact as a baseline/fallback (separate
  Supabase project, so it's isolated).

Full comparison in **[RAG-VERSIONS.md](RAG-VERSIONS.md)**.
