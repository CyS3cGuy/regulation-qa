"""Step 1: turn regulation PDFs in the data/ folder into searchable chunks.

Run:  uv run ingest.py
Output: chunks.json (one entry per chunk, with document name and page number)
"""
import json
from pathlib import Path

from pypdf import PdfReader

DATA_DIR = Path("data")
OUT_FILE = Path("chunks.json")
CHUNK_CHARS = 2000  # roughly 400-500 tokens per chunk
OVERLAP = 200       # characters shared between neighbouring chunks, so ideas aren't cut in half


def split_text(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP) -> list[str]:
    """Split text into chunks of about `size` characters, preferring paragraph or sentence breaks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Try to end at a paragraph break, then a sentence break, if one is past halfway.
            cut = text.rfind("\n\n", start, end)
            if cut - start < size // 2:
                cut = text.rfind(". ", start, end)
            if cut - start >= size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def ingest() -> None:
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DATA_DIR.resolve()}. Add some and run again.")
        return

    chunks = []
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for i, piece in enumerate(split_text(text)):
                chunks.append({
                    "id": f"{pdf.stem}-p{page_num}-{i}",
                    "doc": pdf.stem,
                    "page": page_num,
                    "text": piece,
                })
        print(f"{pdf.name}: {len(reader.pages)} pages")

    OUT_FILE.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(chunks)} chunks to {OUT_FILE}")


if __name__ == "__main__":
    ingest()