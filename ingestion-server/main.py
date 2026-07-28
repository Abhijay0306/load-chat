import os
import re
import tempfile
import httpx
from pathlib import Path
from urllib.parse import quote
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import ImageRefMode

import pipeline_v2  # noqa: E402  (v2 A/B pipeline → separate Supabase project)

# ── Config ────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
SUPABASE_URL       = os.environ["SUPABASE_URL"]
SUPABASE_KEY       = os.environ["SUPABASE_KEY"]
EMBED_MODEL        = "google/gemini-embedding-2"

# ── Docling setup (initialised once at startup, models cached in container) ──
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr                  = False  # PDFs have embedded text; OCR not needed
pipeline_options.generate_picture_images = True
pipeline_options.generate_table_images   = False
pipeline_options.images_scale            = 2.0

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# ── App ───────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Miss MoMo Ingestion Server ready.")
    yield

app = FastAPI(title="MoMo Ingestion Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Core pipeline helpers ─────────────────────────────────────────────────────
def parse_pdf(path: Path) -> str:
    result = converter.convert(str(path))
    try:
        md = result.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
    except Exception:
        md = result.document.export_to_markdown()
    return md


def chunk_markdown(md: str, chunk_size: int = 3000, overlap: int = 200) -> list[str]:
    paragraphs = md.split("\n\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 < chunk_size:
            current += para + "\n\n"
        else:
            if current.strip():
                chunks.append(current.strip())
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + para + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


def embed_text(text: str) -> list[float]:
    clean = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', '[IMAGE]', text)
    resp = httpx.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
        json={"model": EMBED_MODEL, "input": clean},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def save_chunks(chunks: list[str], source: str, product_handles: list[str]) -> tuple[int, list[str]]:
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    errors = []
    saved  = 0
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        payload = {
            "content":   chunk,
            "embedding": embedding,
            "metadata": {
                "source":          source,
                "chunk":           i,
                "engine":          "docling+gemini-embedding-2",
                "has_image":       "data:image" in chunk,
                "product_handles": product_handles,
            },
        }
        r = httpx.post(f"{SUPABASE_URL}/rest/v1/documents_gemini", headers=headers, json=payload, timeout=30.0)
        if r.status_code in (200, 201):
            saved += 1
        else:
            errors.append(f"chunk {i}: {r.status_code} {r.text[:120]}")
    return saved, errors


def delete_source(source: str):
    """Remove all existing chunks for a document before re-ingesting."""
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    httpx.delete(
        f"{SUPABASE_URL}/rest/v1/documents_gemini?metadata->>source=eq.{source}",
        headers=headers,
        timeout=30.0,
    )


def set_product_handles(source: str, handles: list[str]) -> int:
    """Update product_handles for every chunk of a document in place, WITHOUT
    re-ingesting. Rewrites the full metadata per row (preserving chunk index,
    engine, has_image, etc.) with only product_handles replaced. Targets rows by
    (source, chunk) so it doesn't depend on the table's PK column. Returns the
    number of chunks updated."""
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    src = quote(source, safe="")
    r = httpx.get(
        f"{SUPABASE_URL}/rest/v1/documents_gemini?metadata->>source=eq.{src}&select=metadata",
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
            f"{SUPABASE_URL}/rest/v1/documents_gemini"
            f"?metadata->>source=eq.{src}&metadata->>chunk=eq.{chunk}",
            headers=patch_headers, json={"metadata": md}, timeout=30.0,
        )
        if pr.status_code in (200, 204):
            updated += 1
    return updated

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    product_handles: str = "",   # comma-separated, e.g. "load-sentinel,pfr-1750"
    replace: bool = True,        # delete existing chunks for this doc first
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    handles = [h.strip() for h in product_handles.split(",") if h.strip()]
    source  = file.filename

    contents = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        if replace:
            delete_source(source)

        md     = parse_pdf(tmp_path)
        chunks = chunk_markdown(md)
        saved, errors = save_chunks(chunks, source, handles)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(500, f"{type(e).__name__}: {str(e)}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "source":          source,
        "chunks_total":    len(chunks),
        "chunks_saved":    saved,
        "product_handles": handles,
        "errors":          errors,
    }


@app.get("/documents")
def list_documents():
    """Return distinct document names and their product mappings."""
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    r = httpx.get(
        f"{SUPABASE_URL}/rest/v1/documents_gemini?select=metadata&metadata->>chunk=eq.0",
        headers=headers,
        timeout=30.0,
    )
    r.raise_for_status()
    rows = r.json()
    docs = []
    for row in rows:
        m = row.get("metadata", {})
        docs.append({
            "source":          m.get("source"),
            "product_handles": m.get("product_handles", []),
            "engine":          m.get("engine"),
        })
    return docs


@app.delete("/documents/{source}")
def delete_document(source: str):
    """Delete all chunks for a document."""
    delete_source(source)
    return {"deleted": source}


class ProductTags(BaseModel):
    product_handles: list[str] = []


@app.patch("/documents/{source}/products")
def update_document_products(source: str, tags: ProductTags):
    """Re-tag which products a document is connected to, without re-ingesting."""
    handles = [h.strip() for h in tags.product_handles if h and h.strip()]
    updated = set_product_handles(source, handles)
    if updated == 0:
        raise HTTPException(404, f"No chunks found for document '{source}'")
    return {"source": source, "product_handles": handles, "updated": updated}


# ── v2 routes (parallel A/B pipeline → separate Supabase project) ────────────
# These do NOT touch the live v1 table; pipeline_v2 uses SUPABASE_URL_V2 /
# SUPABASE_KEY_V2. The v1 routes above are unchanged.
@app.post("/ingest/v2")
async def ingest_v2(
    file: UploadFile = File(...),
    product_handles: str = "",
    replace: bool = True,
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    handles = [h.strip() for h in product_handles.split(",") if h.strip()]
    source  = file.filename
    contents = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        result = converter.convert(str(tmp_path))          # convert once
        stats  = pipeline_v2.ingest_document_v2(result, source, handles, replace)
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {str(e)}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return stats


@app.get("/documents/v2")
def list_documents_v2():
    return pipeline_v2.list_documents_v2()


@app.delete("/documents/v2/{source}")
def delete_document_v2(source: str):
    pipeline_v2.delete_source_v2(source)
    return {"deleted": source, "target": "v2"}


@app.patch("/documents/v2/{source}/products")
def update_document_products_v2(source: str, tags: ProductTags):
    """Re-tag a v2 document's products without re-ingesting."""
    handles = [h.strip() for h in tags.product_handles if h and h.strip()]
    updated = pipeline_v2.set_product_handles_v2(source, handles)
    if updated == 0:
        raise HTTPException(404, f"No chunks found for document '{source}'")
    return {"source": source, "product_handles": handles, "updated": updated, "target": "v2"}


class WebIngest(BaseModel):
    urls: list[str]
    replace: bool = True


@app.post("/ingest/web")
def ingest_web(body: WebIngest):
    """Crawl + ingest a batch of website page URLs into the v2 knowledge base."""
    import pipeline_web
    results = pipeline_web.ingest_urls(body.urls, body.replace)
    total = sum(r.get("chunks", 0) for r in results)
    return {"pages": len(results), "chunks_saved": total, "results": results}
