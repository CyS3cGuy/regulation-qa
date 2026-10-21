"""Step 2: answer a question using only the regulation chunks, with citations.

Run:  uv run ask.py "What are the core functions of the AI RMF?"
Needs: chunks.json (from ingest.py) and LLM_API_KEY, LLM_BASE_URL, LLM_MODEL in .env
"""
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

CHUNKS_FILE = Path("chunks.json")
TOP_K = 5

STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "had", "her", "was",
    "one", "our", "out", "has", "have", "this", "that", "with", "from", "they", "will", "what",
    "when", "which", "who", "how", "does", "into", "than", "then", "them", "these", "those",
    "should", "would", "could", "about", "their", "there", "been", "also", "such", "its",
}

SYSTEM_PROMPT = """You answer questions about regulations and frameworks using ONLY the numbered sources provided.

Rules:
- After every claim, cite the source number in square brackets, e.g. [2].
- If the sources do not answer the question, reply exactly: Not covered by the provided documents.
- Do not use outside knowledge, even if you know the answer."""


def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOPWORDS]


def search(question: str, chunks: list[dict], k: int = TOP_K) -> list[dict]:
    """Keyword search: score chunks by how often they contain the question's rarer words (TF-IDF)."""
    query_terms = set(tokens(question))
    chunk_counts = [Counter(tokens(c["text"])) for c in chunks]
    doc_freq = Counter(term for counts in chunk_counts for term in counts)
    n = len(chunks)

    scored = []
    for chunk, counts in zip(chunks, chunk_counts):
        score = sum(counts[t] * math.log(n / doc_freq[t]) for t in query_terms if counts[t])
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in scored[:k]]


def build_context(results: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] ({c['doc']}, page {c['page']})\n{c['text']}" for i, c in enumerate(results, start=1)
    )


def answer(question: str) -> None:
    load_dotenv()
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    results = search(question, chunks)
    if not results:
        print("Not covered by the provided documents. (No matching chunks found.)")
        return

    client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ["LLM_BASE_URL"])
    response = client.chat.completions.create(
        model=os.environ["LLM_MODEL"],
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sources:\n\n{build_context(results)}\n\nQuestion: {question}"},
        ],
    )

    print(response.choices[0].message.content)
    print("\nSources:")
    for i, c in enumerate(results, start=1):
        print(f"  [{i}] {c['doc']}, page {c['page']}")
    print(f"\nTokens used: {response.usage.total_tokens}")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or input("Question: ")
    answer(q)