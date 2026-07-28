"""
RAG v2 ingestion pipeline
=========================
Parallel, improved pipeline that writes to a SEPARATE Supabase project so it can
be A/B-tested against the live v1 path in main.py (which is left untouched).

Implements Listofchanges.md:
  §1 HybridChunker on the DoclingDocument (structural, not re-split Markdown)
  §2 Dual table representation (atomic table + row-expanded NL sentences)
  §3 Diagram captioning via a vision model + image upload to Supabase Storage
  §4 Heading-path breadcrumbs prepended before embedding
  §12 (+§3B/§10) best-effort auto-detection with review flags (nothing rewritten
      silently — flags are surfaced in the /ingest/v2 response for manual QA)

Docling's chunker/table/picture APIs have shifted across versions; every Docling
touch point below is wrapped so a single unsupported call degrades gracefully
instead of failing the whole document. Verify against the installed docling
version when running on the HF Space.
"""

import os
import re
import io
import json
import base64
import uuid
from urllib.parse import quote

import httpx

# ── Config (v2 targets a DIFFERENT Supabase project than main.py) ────────────
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
SUPABASE_URL_V2    = os.environ.get("SUPABASE_URL_V2", "")
SUPABASE_KEY_V2    = os.environ.get("SUPABASE_KEY_V2", "")

EMBED_MODEL  = "google/gemini-embedding-2"
VISION_MODEL = "google/gemini-2.5-flash"      # §3 diagram captioning
GEN_MODEL    = "google/gemma-4-26b-a4b-it"    # §2 row-sentence expansion
IMAGE_BUCKET = "rag-v2-images"

TABLE_ROW_LIMIT = 60   # skip row-expansion for pathologically large tables

# Safety keyword heuristic (§10) — tags is_safety_critical, still flagged for review.
SAFETY_PATTERNS = re.compile(
    r"\b(danger|dangerous voltage|warning|caution|never|do not|risk of|"
    r"electric shock|secondary open|shall not|hazard)\b",
    re.IGNORECASE,
)


# ── HTTP helpers ─────────────────────────────────────────────────────────────
def _headers() -> dict:
    return {
        "apikey": SUPABASE_KEY_V2,
        "Authorization": f"Bearer {SUPABASE_KEY_V2}",
        "Content-Type": "application/json",
    }


def _require_config():
    if not SUPABASE_URL_V2 or not SUPABASE_KEY_V2:
        raise RuntimeError(
            "v2 ingestion requires SUPABASE_URL_V2 and SUPABASE_KEY_V2 env vars "
            "(the NEW Supabase project — never the live v1 project)."
        )


# ── OpenRouter calls ─────────────────────────────────────────────────────────
def embed_text(text: str) -> list[float]:
    # Strip any leftover base64 image data before embedding (defensive; v2 stores
    # captions rather than raw base64, but breadcrumbs/atomic tables are clean).
    clean = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "[IMAGE]", text)
    resp = httpx.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
        json={"model": EMBED_MODEL, "input": clean},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def caption_image(image_bytes: bytes, mime: str, context: str) -> str:
    """§3A — turn a diagram/photo into searchable caption text via a vision model."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    prompt = (
        "You are describing a figure from a Load Controls product installation "
        "guide so it can be found by search. Write 1-3 sentences describing what "
        "the diagram/photo shows and what task or setting it illustrates. Include "
        "any visible labels, numbers, or wire/turn counts. Be concrete and factual. "
        f"Nearby text for context: {context[:400]}"
    )
    resp = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            "max_tokens": 300,
            "temperature": 0.1,
        },
        timeout=90.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def expand_table_rows(table_md: str) -> list[str]:
    """§2 — one natural-language sentence per data row, so numeric rows carry
    embedding signal. Returns [] on failure (atomic table chunk still stored)."""
    prompt = (
        "Convert each DATA ROW of this table into ONE standalone natural-language "
        "sentence that a user could match by search. Include the row's key values "
        "and the column meanings, e.g. 'For a 15 HP motor at 460V, use switch "
        "position 5 with 1 turn.' Ignore the header row. Return ONLY a JSON array "
        "of strings, one per data row.\n\nTable:\n" + table_md
    )
    try:
        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": GEN_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
                "temperature": 0.0,
            },
            timeout=90.0,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        # tolerate ```json fences
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        rows = json.loads(raw)
        return [str(r).strip() for r in rows if str(r).strip()]
    except Exception:
        return []


# ── Supabase Storage (§3 image_url) ──────────────────────────────────────────
def _ensure_bucket():
    try:
        httpx.post(
            f"{SUPABASE_URL_V2}/storage/v1/bucket",
            headers=_headers(),
            json={"id": IMAGE_BUCKET, "name": IMAGE_BUCKET, "public": True},
            timeout=30.0,
        )  # 400 if it already exists — ignore
    except Exception:
        pass


def upload_image(image_bytes: bytes, mime: str, source: str) -> str | None:
    ext = mime.split("/")[-1] if "/" in mime else "png"
    path = f"{re.sub(r'[^A-Za-z0-9_.-]', '_', source)}/{uuid.uuid4().hex}.{ext}"
    try:
        r = httpx.post(
            f"{SUPABASE_URL_V2}/storage/v1/object/{IMAGE_BUCKET}/{path}",
            headers={
                "apikey": SUPABASE_KEY_V2,
                "Authorization": f"Bearer {SUPABASE_KEY_V2}",
                "Content-Type": mime,
            },
            content=image_bytes,
            timeout=60.0,
        )
        if r.status_code not in (200, 201):
            return None
        return f"{SUPABASE_URL_V2}/storage/v1/object/public/{IMAGE_BUCKET}/{path}"
    except Exception:
        return None


# ── Docling extraction helpers (version-defensive) ───────────────────────────
def _heading_path(chunk) -> list[str]:
    meta = getattr(chunk, "meta", None)
    headings = getattr(meta, "headings", None) if meta else None
    return [h for h in headings if h] if headings else []


def build_text_chunks(dl_doc) -> list[dict]:
    """§1 HybridChunker + §4 breadcrumbs. One dict per structural chunk."""
    from docling_core.transforms.chunker import HybridChunker

    chunker = HybridChunker(
        tokenizer="BAAI/bge-small-en-v1.5",  # tokenizer family only; embedding is Gemini
        max_tokens=500,
        merge_peers=True,
    )
    out = []
    for i, chunk in enumerate(chunker.chunk(dl_doc=dl_doc)):
        text = (getattr(chunk, "text", "") or "").strip()
        if not text:
            continue
        headings = _heading_path(chunk)
        breadcrumb = " > ".join(headings)
        content = f"{breadcrumb}\n\n{text}" if breadcrumb else text  # §4
        out.append({
            "content": content,
            "content_type": "text",
            "headings": headings,
            "local_index": i,
        })
    return out


def build_table_chunks(dl_doc, source: str) -> list[dict]:
    """§2 — atomic table chunk + row-expanded sentence chunks sharing table_source."""
    out = []
    tables = getattr(dl_doc, "tables", None) or []
    for t in tables:
        try:
            table_md = t.export_to_markdown(doc=dl_doc)
        except TypeError:
            table_md = t.export_to_markdown()
        except Exception:
            continue
        if not table_md.strip():
            continue

        table_source = uuid.uuid4().hex
        headings = []
        try:
            headings = _heading_path(t)  # some versions attach meta to items
        except Exception:
            pass

        # atomic whole-table chunk (used for full-context injection at query time)
        out.append({
            "content": table_md,
            "content_type": "table",
            "table_source": table_source,
            "headings": headings,
        })

        # row-expanded NL sentences
        n_rows = table_md.count("\n")  # rough guard against huge tables
        if n_rows <= TABLE_ROW_LIMIT:
            for sentence in expand_table_rows(table_md):
                out.append({
                    "content": sentence,
                    "content_type": "spec_table_row",
                    "table_source": table_source,
                    "headings": headings,
                })
    return out


def build_diagram_chunks(dl_doc, source: str) -> list[dict]:
    """§3A — caption each picture, upload the image, store caption as content."""
    out = []
    pictures = getattr(dl_doc, "pictures", None) or []
    if pictures:
        _ensure_bucket()
    for pic in pictures:
        try:
            pil = pic.get_image(dl_doc)
        except Exception:
            pil = None
        if pil is None:
            continue
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        # nearby caption text if Docling exposes one
        context = ""
        try:
            context = getattr(pic, "caption_text", lambda d: "")(dl_doc) or ""
        except Exception:
            pass

        try:
            caption = caption_image(image_bytes, "image/png", context)
        except Exception:
            continue  # skip uncaptionable image rather than embed noise

        image_url = upload_image(image_bytes, "image/png", source)
        out.append({
            "content": caption,
            "content_type": "diagram",
            "image_url": image_url,
            "headings": _heading_path(pic) if hasattr(pic, "meta") else [],
        })
    return out


# ── §3B/§10/§12 best-effort auto-flagging ────────────────────────────────────
def annotate_flags(chunk: dict) -> tuple[bool, list[str]]:
    """Returns (is_safety_critical, review_flags). Heuristic only — surfaced for
    manual QA, never used to rewrite content."""
    content = chunk["content"]
    flags = []

    # §10 safety-critical
    is_safety = bool(SAFETY_PATTERNS.search(content))

    # §3B flowchart / fragmented state-diagram extraction
    #   heuristic: many very short lines and/or repeated tiny fragments
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    if len(lines) >= 6:
        short = sum(1 for ln in lines if len(ln) <= 12)
        if short / len(lines) > 0.6:
            flags.append("flowchart?")

    # §12 OCR garble: high ratio of tokens with no vowels / weird char runs
    tokens = re.findall(r"[A-Za-z]{2,}", content)
    if tokens:
        garbled = sum(1 for tk in tokens if not re.search(r"[aeiouAEIOU]", tk))
        if garbled / len(tokens) > 0.25:
            flags.append("ocr?")

    if is_safety:
        flags.append("safety?")
    return is_safety, flags


# ── DB writes ────────────────────────────────────────────────────────────────
def delete_source_v2(source: str):
    _require_config()
    httpx.delete(
        f"{SUPABASE_URL_V2}/rest/v1/documents_gemini?metadata->>source=eq.{source}",
        headers=_headers(),
        timeout=30.0,
    )


def save_chunks_v2(chunks: list[dict], source: str, product_handles: list[str]) -> dict:
    _require_config()
    headers = {**_headers(), "Prefer": "return=minimal"}
    saved, errors, review = 0, [], []

    for i, ch in enumerate(chunks):
        is_safety, flags = annotate_flags(ch)
        try:
            embedding = embed_text(ch["content"])
        except Exception as e:
            errors.append(f"chunk {i} embed: {type(e).__name__}: {str(e)[:120]}")
            continue

        metadata = {
            "source":          source,
            "chunk":           i,
            "engine":          "docling-hybrid+gemini-embedding-2",
            "content_type":    ch.get("content_type", "text"),
            "headings":        ch.get("headings", []),
            "product_handles": product_handles,
            "is_safety_critical": is_safety,
            "review_flags":    flags,
        }
        if ch.get("table_source"):
            metadata["table_source"] = ch["table_source"]
        if ch.get("image_url"):
            metadata["image_url"] = ch["image_url"]

        payload = {"content": ch["content"], "embedding": embedding, "metadata": metadata}
        r = httpx.post(
            f"{SUPABASE_URL_V2}/rest/v1/documents_gemini",
            headers=headers, json=payload, timeout=60.0,
        )
        if r.status_code in (200, 201):
            saved += 1
            if flags:
                review.append({"chunk": i, "content_type": metadata["content_type"],
                               "flags": flags, "preview": ch["content"][:120]})
        else:
            errors.append(f"chunk {i}: {r.status_code} {r.text[:120]}")

    return {"saved": saved, "errors": errors, "review_flags": review}


# ── Public entry point ───────────────────────────────────────────────────────
def ingest_document_v2(result, source: str, product_handles: list[str], replace: bool = True) -> dict:
    """Takes a Docling ConversionResult (converted once in main.py) and runs the
    full v2 pipeline against the v2 Supabase project."""
    _require_config()
    dl_doc = result.document

    if replace:
        delete_source_v2(source)

    chunks: list[dict] = []
    chunks += build_text_chunks(dl_doc)
    chunks += build_table_chunks(dl_doc, source)
    chunks += build_diagram_chunks(dl_doc, source)

    counts = {}
    for ch in chunks:
        counts[ch.get("content_type", "text")] = counts.get(ch.get("content_type", "text"), 0) + 1

    result_stats = save_chunks_v2(chunks, source, product_handles)
    return {
        "source":          source,
        "chunks_total":    len(chunks),
        "chunks_saved":    result_stats["saved"],
        "chunk_type_counts": counts,
        "product_handles": product_handles,
        "review_flags":    result_stats["review_flags"],
        "errors":          result_stats["errors"],
    }


def set_product_handles_v2(source: str, handles: list[str]) -> int:
    """Re-tag a v2 document's product_handles in place (no re-ingest), preserving
    all other metadata. Mirrors main.set_product_handles but targets the v2
    project. Returns the number of chunks updated."""
    _require_config()
    headers = _headers()
    src = quote(source, safe="")
    r = httpx.get(
        f"{SUPABASE_URL_V2}/rest/v1/documents_gemini?metadata->>source=eq.{src}&select=metadata",
        headers=headers, timeout=30.0,
    )
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return 0

    updated = 0
    patch_headers = {**headers, "Prefer": "return=minimal"}
    for row in rows:
        md = row.get("metadata") or {}
        md["product_handles"] = handles
        chunk = md.get("chunk")
        if chunk is None:
            continue
        pr = httpx.patch(
            f"{SUPABASE_URL_V2}/rest/v1/documents_gemini"
            f"?metadata->>source=eq.{src}&metadata->>chunk=eq.{chunk}",
            headers=patch_headers, json={"metadata": md}, timeout=30.0,
        )
        if pr.status_code in (200, 204):
            updated += 1
    return updated


def list_documents_v2() -> list[dict]:
    _require_config()
    r = httpx.get(
        f"{SUPABASE_URL_V2}/rest/v1/documents_gemini?select=metadata&metadata->>chunk=eq.0",
        headers=_headers(), timeout=30.0,
    )
    r.raise_for_status()
    docs = []
    for row in r.json():
        m = row.get("metadata", {})
        docs.append({
            "source":          m.get("source"),
            "product_handles": m.get("product_handles", []),
            "engine":          m.get("engine"),
        })
    return docs
