"""Step 1 (v2): turn any PDFs in data/ into searchable chunks.

Works on any document - nothing here is specific to one regulation:
  1. Repeated headers and footers are detected and removed automatically
     (any line that appears near the top or bottom of most pages).
  2. Text is split into blocks at paragraph breaks and numbered items
     (1.  1.1  10.20  S 10.20  Article 5  Appendix 3 ...). Sub-items like (a) stay
     with their parent, so a requirement and its list are never separated.
  3. Blocks are packed into chunks of about TARGET_CHARS, never cutting a block
     in half unless it is longer than MAX_CHARS on its own.
  4. Each chunk starts with the nearest heading, so search knows its context.

Run:  uv run ingest.py
Output: chunks.json
"""
import json
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

DATA_DIR = Path("data")
OUT_FILE = Path("chunks.json")
TARGET_CHARS = 1500   # aim for chunks around this size (~350 tokens)
MAX_CHARS = 3000      # a single block longer than this is split at sentences
EDGE_LINES = 2        # how many lines at the top/bottom of a page can be header/footer
REPEAT_SHARE = 0.5    # a line on at least half the pages counts as header/footer

# A new block starts at a numbered item or a structural heading.
BLOCK_START = re.compile(
    r"^(?P<label>"
    r"(?:[A-Z]\s+)?\d+(?:\.\d+)+\.?"                    # 1.1  10.20  13.2.1  S 10.20
    r"|\d+\.(?=\s)"                                     # 1.  2.
    r"|\d+(?=\s+[A-Z])"                                 # 8 Governance (also footnotes)
    r"|(?:Article|Section|Chapter|Part|Appendix|Annex|Schedule|Clause)\s+[\dIVXLA-Z]+\b"
    r")(?=\s|:|$)"
)
HEADING = re.compile(
    r"^(?:(?:Article|Section|Chapter|Part|Appendix|Annex|Schedule)\s+[\dIVXLA-Z]+\b.*"
    r"|\d+(?:\.\d+)?\s+[A-Z][^.]{2,80})$"               # short numbered title, no full stop
)
DOT_LEADER = re.compile(r"\.{5,}|(?:\.\s){5,}")       # table-of-contents lines


def shape(line: str) -> str:
    """Line with digits masked, so 'Page 3 of 9' and 'Page 4 of 9' look the same."""
    return re.sub(r"\d+", "#", line.strip().lower())


def read_pages(pdf: Path) -> list[list[str]]:
    reader = PdfReader(pdf)
    return [[l.rstrip() for l in (p.extract_text() or "").splitlines()] for p in reader.pages]


def find_repeated_edges(pages: list[list[str]]) -> set[str]:
    """Shapes of lines that sit at the top or bottom of most pages, and rarely anywhere else."""
    if len(pages) < 4:
        return set()
    edge_counts, middle_counts = Counter(), Counter()
    for lines in pages:
        content = [l for l in lines if l.strip()]
        edges = content[:EDGE_LINES] + content[-EDGE_LINES:]
        edge_counts.update({shape(l) for l in edges})
        middle_counts.update(shape(l) for l in content[EDGE_LINES:-EDGE_LINES])
    return {s for s, n in edge_counts.items()
            if n >= REPEAT_SHARE * len(pages) and middle_counts[s] <= 0.1 * len(pages)}


def to_blocks(pages: list[list[str]]) -> list[dict]:
    """Group lines into blocks: one per paragraph or numbered item, tracking pages and headings."""
    repeated = find_repeated_edges(pages)
    blocks, current, heading = [], None, ""
    prev_ended = True  # did the previous line end a sentence/clause (or was it blank)?

    def close():
        if current and current["lines"]:
            blocks.append(current)

    for page_num, lines in enumerate(pages, start=1):
        for raw in lines:
            line = raw.strip()
            if not line:
                # A blank line ends a plain paragraph, but not a numbered item with sub-items.
                if current and not current["numbered"]:
                    close()
                    current = None
                prev_ended = True
                continue
            if shape(line) in repeated or DOT_LEADER.search(line):
                continue

            # A number at the start of a line only starts a new item if the previous line
            # finished a sentence; otherwise it's a wrapped cross-reference ("...paragraph
            # 10.50 and Appendix 10") in the middle of a sentence.
            m = BLOCK_START.match(line) if prev_ended else None
            # A short capitalised line with no closing punctuation, after a finished sentence,
            # is an unnumbered subheading ("Cryptography"): it starts a block and ends a clause.
            is_title = (not m and prev_ended and len(line) < 60 and line[0].isupper()
                        and not line.endswith((".", ";", ",")))
            prev_ended = (line.endswith((".", ";", ":", "?", "!")) or bool(HEADING.match(line))
                          or is_title)
            if is_title:
                close()
                heading = re.sub(r"\s+", " ", line)
                current = {"label": "", "numbered": False, "heading": heading,
                           "lines": [line], "pages": {page_num}}
                continue
            if m or current is None:
                close()
                if HEADING.match(line) and len(line) < 60:  # wrapped paragraph lines are longer
                    heading = re.sub(r"\s+", " ", line)
                current = {"label": m["label"] if m else "", "numbered": bool(m),
                           "heading": heading, "lines": [], "pages": set()}
            current["lines"].append(line)
            current["pages"].add(page_num)
    close()
    return blocks


def split_sentences(text: str, size: int) -> list[str]:
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(". ", start, end)
            if cut > start + size // 2:
                end = cut + 1
        pieces.append(text[start:end].strip())
        start = end
    return pieces


def pack(blocks: list[dict]) -> list[dict]:
    """Combine consecutive blocks into chunks of about TARGET_CHARS without splitting blocks."""
    chunks, group = [], []

    def flush():
        if group:
            text = " ".join(" ".join(b["lines"]) for b in group)
            chunks.append({
                # Prefer a real item label (10.20, 3.) over a bare number such as a footnote.
                "label": next((b["label"] for b in group if not b["label"].isdigit()),
                              group[0]["label"]),
                "heading": group[0]["heading"],
                "pages": sorted(set().union(*(b["pages"] for b in group))),
                "text": text,
            })
            group.clear()

    for block in blocks:
        size = len(" ".join(block["lines"]))
        if size > MAX_CHARS:
            flush()
            text = " ".join(block["lines"])
            for piece in split_sentences(text, TARGET_CHARS):
                chunks.append({"label": block["label"], "heading": block["heading"],
                               "pages": sorted(block["pages"]), "text": piece})
            continue
        current_size = sum(len(" ".join(b["lines"])) for b in group)
        if group and current_size + size > TARGET_CHARS:
            flush()
        group.append(block)
    flush()
    return chunks


def ingest_pdf(pdf: Path) -> list[dict]:
    chunks = []
    for i, c in enumerate(pack(to_blocks(read_pages(pdf)))):
        header = " | ".join(x for x in [pdf.stem, c["heading"], c["label"]] if x)
        chunks.append({
            "id": f"{pdf.stem}-{i}",
            "doc": pdf.stem,
            "label": c["label"],
            "page": c["pages"][0],
            "pages": c["pages"],
            "text": f"[{header}]\n{c['text']}",
        })
    return chunks


def ingest() -> None:
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DATA_DIR.resolve()}. Add some and run again.")
        return
    chunks = []
    for pdf in pdfs:
        doc_chunks = ingest_pdf(pdf)
        chunks.extend(doc_chunks)
        print(f"{pdf.name}: {len(doc_chunks)} chunks")
    OUT_FILE.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(chunks)} chunks to {OUT_FILE}")


if __name__ == "__main__":
    ingest()