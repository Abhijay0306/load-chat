"""
Website (page-wise) ingestion for the v2 knowledge base.
========================================================
Crawls Shopify/website pages, extracts readable text, chunks + embeds them, and
stores them in the SAME v2 Supabase project as the PDFs — tagged content_type
"web" with the page's source_url. At query time the worker builds a text-fragment
deep link (…/page#:~:text=…) so a visitor is taken to the exact spot on the page.

Reuses pipeline_v2 for the embedding call + Supabase config.
"""

import re
from urllib.parse import quote, urlparse

import httpx
from bs4 import BeautifulSoup

import pipeline_v2  # embed_text, _headers, _require_config, SUPABASE_URL_V2

UA = "Mozilla/5.0 (compatible; MissMoMoBot/1.0; +https://loadcontrols.com)"

# Boilerplate/utility pages we never want as knowledge
SKIP_PATTERNS = re.compile(
    r"/(cart|checkout|account|orders?|policies|policy|search|password|admin)(/|$|\?)",
    re.IGNORECASE,
)


def fetch_html(url: str) -> str:
    r = httpx.get(url, headers={"User-Agent": UA}, timeout=30.0, follow_redirects=True)
    r.raise_for_status()
    return r.text


def extract_page(url: str) -> tuple[str, str]:
    """Return (title, main_text) with nav/header/footer/script boilerplate removed."""
    soup = BeautifulSoup(fetch_html(url), "html.parser")
    title = (soup.title.string.strip() if soup.title and soup.title.string else url)
    title = re.sub(r"\s+", " ", title)

    for tag in soup(["script", "style", "nav", "header", "footer", "aside",
                     "form", "noscript", "svg", "iframe", "button"]):
        tag.decompose()
    # common Shopify chrome by role/class
    for sel in ['[role="navigation"]', ".site-header", ".site-footer", ".header",
                ".footer", ".announcement-bar", "#shopify-section-header",
                "#shopify-section-footer", ".cart", ".breadcrumb"]:
        for el in soup.select(sel):
            el.decompose()

    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = re.sub(r"\s+", " ", root.get_text(" ", strip=True)).strip()
    return title, text


def chunk_web(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    """Char-based packing with small overlap; keeps chunks small for precise
    text-fragment highlights."""
    words = text.split(" ")
    chunks, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > size and cur:
            chunks.append(cur.strip())
            cur = (cur[-overlap:] + " " + w).strip()
        else:
            cur = (cur + " " + w).strip()
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if len(c) > 40]


def delete_page(source_url: str):
    pipeline_v2._require_config()
    su = quote(source_url, safe="")
    httpx.delete(
        f"{pipeline_v2.SUPABASE_URL_V2}/rest/v1/documents_gemini?metadata->>source_url=eq.{su}",
        headers=pipeline_v2._headers(), timeout=30.0,
    )


def save_web_chunks(url: str, title: str, chunks: list[str]) -> tuple[int, list[str]]:
    headers = {**pipeline_v2._headers(), "Prefer": "return=minimal"}
    # product pages: derive the handle so web chunks share product context
    m = re.search(r"/products/([^/?#]+)", urlparse(url).path)
    handles = [m.group(1)] if m else []

    saved, errors = 0, []
    for i, c in enumerate(chunks):
        try:
            emb = pipeline_v2.embed_text(c)
        except Exception as e:
            errors.append(f"chunk {i} embed: {type(e).__name__}: {str(e)[:80]}")
            continue
        payload = {
            "content": c,
            "embedding": emb,
            "metadata": {
                "source": title[:140],
                "source_url": url,
                "chunk": i,
                "content_type": "web",
                "engine": "web-gemini-embedding-2",
                "product_handles": handles,
            },
        }
        r = httpx.post(f"{pipeline_v2.SUPABASE_URL_V2}/rest/v1/documents_gemini",
                       headers=headers, json=payload, timeout=60.0)
        if r.status_code in (200, 201):
            saved += 1
        else:
            errors.append(f"chunk {i}: {r.status_code} {r.text[:80]}")
    return saved, errors


def ingest_urls(urls: list[str], replace: bool = True) -> list[dict]:
    pipeline_v2._require_config()
    results = []
    for url in urls:
        if SKIP_PATTERNS.search(url):
            results.append({"url": url, "skipped": "utility page"})
            continue
        try:
            title, text = extract_page(url)
            if len(text) < 80:
                results.append({"url": url, "chunks": 0, "skipped": "too little text"})
                continue
            if replace:
                delete_page(url)
            chunks = chunk_web(text)
            saved, errors = save_web_chunks(url, title, chunks)
            results.append({"url": url, "title": title[:80], "chunks": saved,
                            **({"errors": errors[:3]} if errors else {})})
        except Exception as e:
            results.append({"url": url, "error": f"{type(e).__name__}: {str(e)[:100]}"})
    return results
