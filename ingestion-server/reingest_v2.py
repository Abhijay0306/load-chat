#!/usr/bin/env python3
"""
Batch re-ingestion into the v2 pipeline.
========================================
Posts every PDF in a folder to the deployed ingestion server's /ingest/v2 route,
which writes to the NEW Supabase project (SUPABASE_URL_V2 / SUPABASE_KEY_V2 must
be set on the ingestion server, NOT here).

Usage:
    export INGEST_BASE_URL="https://adiddev-momo-ingestion.hf.space"
    export HF_TOKEN="hf_..."          # only if the Space requires auth
    python reingest_v2.py "../Test Documents"

Prerequisites (see the runbook in chat):
  1. New Supabase project created + migrations applied.
  2. v2 code (main.py, pipeline_v2.py, Dockerfile) deployed to the Space.
  3. SUPABASE_URL_V2 / SUPABASE_KEY_V2 set in the Space's environment.
"""

import os
import sys
import json
from pathlib import Path

import httpx

# Windows consoles default to cp1252, which can't encode the ✓/→ status glyphs.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_URL = os.environ.get("INGEST_BASE_URL", "https://adiddev-momo-ingestion.hf.space").rstrip("/")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ── Document → Shopify product-handle mapping (§9) ───────────────────────────
# EDIT these to match your real Shopify product handles. Leave a list empty to
# ingest without product-handle filtering for that document (still fully usable,
# just no §9 hard-filter for it). Handles are the store's product slugs, e.g.
# a product at /products/pfr-1750 has handle "pfr-1750".
HANDLE_MAP = {
    "PFR-1750-Installation-Guide.pdf": ["pfr-1750-two-set-points"],
    "PMP-25-Data-Sheet.pdf": ["pmp-25"],
    "UPC-KWH-and-KWH-3-Energy-Meter.pdf": ["upc"],
    "Power-Sensors-and-Their-Impact-on-Water-Processing.pdf": [],  # general article, no product
}


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "../Test Documents")
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {folder.resolve()}")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    print(f"Target: {BASE_URL}/ingest/v2   ({len(pdfs)} documents)\n")

    for pdf in pdfs:
        handles = HANDLE_MAP.get(pdf.name, [])
        params = {"product_handles": ",".join(handles), "replace": "true"}
        print(f"→ {pdf.name}  handles={handles or '(none)'}")
        try:
            with open(pdf, "rb") as f:
                resp = httpx.post(
                    f"{BASE_URL}/ingest/v2",
                    params=params,
                    files={"file": (pdf.name, f, "application/pdf")},
                    headers=headers,
                    timeout=600.0,  # Docling + captioning + row-expansion is slow
                )
            if resp.status_code == 200:
                data = resp.json()
                print(f"   ✓ chunks={data.get('chunks_saved')}/{data.get('chunks_total')} "
                      f"types={data.get('chunk_type_counts')}")
                flags = data.get("review_flags") or []
                if flags:
                    print(f"   ⚑ {len(flags)} chunk(s) flagged for review:")
                    for fl in flags:
                        print(f"      [{fl['content_type']}] {fl['flags']} — {fl['preview']}")
                errs = data.get("errors") or []
                if errs:
                    print(f"   ! {len(errs)} error(s): {errs[:3]}")
            else:
                print(f"   ✗ HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"   ✗ {type(e).__name__}: {e}")
        print()


if __name__ == "__main__":
    main()
