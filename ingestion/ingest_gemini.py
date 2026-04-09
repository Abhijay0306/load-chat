import os
import sys
import re
import httpx
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import ImageRefMode

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", "./Documents"))
EMBED_MODEL = "gemini-embedding-2-preview"

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing from .env")

# Initialize Google client
client = genai.Client(api_key=GOOGLE_API_KEY)

# Configure Docling with picture/figure image extraction enabled
pipeline_options = PdfPipelineOptions()
pipeline_options.generate_picture_images = True   # Extract images/figures
pipeline_options.generate_table_images = False    # Tables as Markdown (not images)
pipeline_options.images_scale = 2.0               # 2x resolution for clarity

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)


def parse_pdf_to_markdown(pdf_path: Path) -> str:
    """
    Uses Docling to extract Markdown with embedded base64 images.
    Images appear as standard Markdown: ![...](data:image/png;base64,...)
    These render natively in the browser via marked.js.
    """
    print(f"[1/3] Parsing {pdf_path.name} with Docling (images enabled)...")
    result = converter.convert(str(pdf_path))

    # Use EMBEDDED mode: images become data URIs inline in the Markdown
    # No file hosting needed — renders in the browser via marked.js
    try:
        md = result.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
        # Count how many images were embedded
        img_count = md.count("data:image")
        print(f"   Extracted {len(md):,} chars | {img_count} images embedded.")
    except Exception as e:
        print(f"   Image export failed ({e}), falling back to text-only...")
        md = result.document.export_to_markdown()

    return md


def chunk_markdown(md_text: str, chunk_size: int = 3000, overlap: int = 200) -> list[str]:
    """
    Splits Markdown into chunks that respect paragraph boundaries.
    Larger chunk_size (3000) used here to avoid cutting inline base64 images.
    """
    paragraphs = md_text.split("\n\n")
    chunks = []
    current = ""

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

    print(f"   Split into {len(chunks)} chunks.")
    return chunks


def embed_text(text: str) -> list[float]:
    """
    Generates 3072-dim embeddings using Gemini Embedding 2 Preview.
    Note: base64 image data in chunks is stripped before embedding to save tokens.
    """
    # Strip base64 data URIs from text before embedding (they're not semantic)
    clean_text = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', '[IMAGE]', text)

    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=clean_text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    return result.embeddings[0].values


def save_to_supabase(chunks: list[str], source_name: str):
    """Saves embedded chunks to the documents_gemini table."""
    print(f"[3/3] Saving {len(chunks)} chunks to Supabase...")
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        has_image = "data:image" in chunk
        payload = {
            "content": chunk,  # Full chunk with embedded images
            "metadata": {
                "source": source_name,
                "chunk": i,
                "engine": "docling+gemini-embedding-2",
                "has_image": has_image
            },
            "embedding": embedding
        }
        resp = httpx.post(
            f"{SUPABASE_URL}/rest/v1/documents_gemini",
            headers=headers,
            json=payload,
            timeout=30.0
        )
        img_tag = "🖼️" if has_image else "  "
        if resp.status_code not in (200, 201):
            print(f"   ✗ Chunk {i+1}: ERROR {resp.status_code} — {resp.text[:200]}")
        else:
            print(f"   ✓ {img_tag} Chunk {i+1}/{len(chunks)} saved.")


def process_pdf(pdf_name: str):
    pdf_path = DOCUMENTS_DIR / pdf_name
    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        return

    # 1. Parse with Docling (images embedded as base64)
    markdown = parse_pdf_to_markdown(pdf_path)

    # 2. Chunk the Markdown
    print("[2/3] Chunking Markdown...")
    chunks = chunk_markdown(markdown)

    # 3. Embed and save to Supabase
    save_to_supabase(chunks, pdf_name)

    print(f"\n✅ Done! '{pdf_name}' re-ingested into documents_gemini with {len(chunks)} chunks (images included).")


if __name__ == "__main__":
    target_pdf = sys.argv[1] if len(sys.argv) > 1 else "PCR-1830V.pdf"
    process_pdf(target_pdf)
