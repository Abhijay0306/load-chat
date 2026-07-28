#!/usr/bin/env python3
"""
Crawl the website sitemap and ingest pages into the v2 knowledge base.
=====================================================================
Reads <base>/sitemap.xml, collects page URLs (products, pages, blog articles,
metaobject pages — collections skipped by default), and POSTs them to
/ingest/web in small batches.

Usage:
    export INGEST_BASE_URL="https://adiddev-momo-ingestion.hf.space"
    export HF_TOKEN="hf_..."                 # if the Space requires auth
    python reingest_web.py                    # crawl https://loadcontrols.com
    python reingest_web.py https://loadcontrols.com --collections   # include collections
"""
import os
import re
import sys
import html

import httpx

BASE_URL   = os.environ.get("INGEST_BASE_URL", "https://adiddev-momo-ingestion.hf.space").rstrip("/")
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
SITE       = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith("http") else "https://loadcontrols.com"
INCLUDE_COLLECTIONS = "--collections" in sys.argv
UA = {"User-Agent": "Mozilla/5.0 (compatible; MissMoMoBot/1.0)"}
BATCH = 10

# Which child sitemaps to crawl
WANT = ["products", "pages", "blogs", "metaobject"]
if INCLUDE_COLLECTIONS:
    WANT.append("collections")


def locs(xml: str) -> list[str]:
    return [html.unescape(m) for m in re.findall(r"<loc>([^<]+)</loc>", xml)]


def collect_urls() -> list[str]:
    idx = httpx.get(f"{SITE}/sitemap.xml", headers=UA, timeout=30.0, follow_redirects=True).text
    children = [c for c in locs(idx) if any(w in c for w in WANT)]
    urls: list[str] = []
    for child in children:
        try:
            urls += locs(httpx.get(child, headers=UA, timeout=30.0, follow_redirects=True).text)
        except Exception as e:
            print(f"  ! failed to read {child}: {e}")
    # de-dupe, drop the bare sitemap self-refs
    seen, out = set(), []
    for u in urls:
        if u not in seen and "sitemap" not in u:
            seen.add(u); out.append(u)
    return out


def main():
    urls = collect_urls()
    print(f"Collected {len(urls)} page URLs from {SITE}\nTarget: {BASE_URL}/ingest/web (batches of {BATCH})\n")
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    total = 0
    for i in range(0, len(urls), BATCH):
        batch = urls[i:i + BATCH]
        try:
            r = httpx.post(f"{BASE_URL}/ingest/web", headers=headers,
                           json={"urls": batch, "replace": True}, timeout=600.0)
            if r.status_code == 200:
                d = r.json()
                total += d.get("chunks_saved", 0)
                for res in d.get("results", []):
                    tag = res.get("skipped") or res.get("error") or f"{res.get('chunks', 0)} chunks"
                    print(f"  {res['url']}  ->  {tag}")
            else:
                print(f"  batch {i // BATCH} HTTP {r.status_code}: {r.text[:160]}")
        except Exception as e:
            print(f"  batch {i // BATCH} failed: {type(e).__name__}: {e}")
    print(f"\nDone. {total} web chunks saved.")


if __name__ == "__main__":
    main()
