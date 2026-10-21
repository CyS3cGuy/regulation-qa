"""Step 3: run every question in tests.json and score the system.

Run:  uv run evaluate.py
Needs: chunks.json, tests.json, and LLM_API_KEY, LLM_BASE_URL, LLM_MODEL in .env
Output: a score table in the terminal, plus a full record in results/<timestamp>.json

What is scored:
  retrieval  - did search find at least one expected page? (answerable questions)
  context    - do retrieved chunks from the expected pages contain every expected keyword?
               If this fails, search is to blame; if it passes but `answer` fails, the model is.
  answer     - does the answer contain every expected keyword?   (answerable questions)
  citation   - does the answer cite at least one expected page?  (answerable questions)
  refusal    - right decision on answering vs saying "not covered" (all questions)
"""
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from ask import SYSTEM_PROMPT, build_context, search

CHUNKS_FILE = Path("chunks.json")
TESTS_FILE = Path("tests.json")
RESULTS_DIR = Path("results")
REFUSAL_PHRASE = "not covered by the provided documents"


def normalise(text: str) -> str:
    """Lowercase and treat hyphens like spaces, so 'in-motion' matches 'in motion'."""
    return re.sub(r"\s+", " ", text.lower().replace("-", " "))


def cited_numbers(answer: str) -> set[int]:
    """Find citation numbers like [1], [2, 3] or [1][4] in the answer."""
    numbers = set()
    for group in re.findall(r"\[([\d,\s]+)\]", answer):
        numbers.update(int(n) for n in re.findall(r"\d+", group))
    return numbers


def pages_of(chunk: dict) -> list[int]:
    """A chunk can span pages (v2 ingest); v1 chunks only have 'page'."""
    return chunk.get("pages", [chunk["page"]])


def ask_model(client: OpenAI, question: str, results: list[dict]) -> tuple[str, int]:
    if not results:
        return "Not covered by the provided documents.", 0
    response = client.chat.completions.create(
        model=os.environ["LLM_MODEL"],
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sources:\n\n{build_context(results)}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content or "", response.usage.total_tokens


def score(test: dict, results: list[dict], answer: str) -> dict:
    refused = REFUSAL_PHRASE in normalise(answer)
    checks = {"refusal": refused != test["answerable"]}

    if test["answerable"]:
        expected = set(test["expected_pages"])
        retrieved = {p for c in results for p in pages_of(c)}
        cited = {p for n in cited_numbers(answer) if 1 <= n <= len(results) for p in pages_of(results[n - 1])}
        text = normalise(answer)
        # Only count text from retrieved chunks on the expected pages, so a keyword that
        # happens to appear in an unrelated passage doesn't count as "found".
        context = normalise(" ".join(c["text"] for c in results if set(pages_of(c)) & expected))
        missing = [k for k in test["expected_keywords"] if normalise(k) not in text]
        missing_context = [k for k in test["expected_keywords"] if normalise(k) not in context]

        checks["retrieval"] = bool(retrieved & expected)
        checks["context"] = not missing_context
        checks["answer"] = not refused and not missing
        checks["citation"] = bool(cited & expected)
        checks["missing_keywords"] = missing
        checks["missing_from_context"] = missing_context
        checks["cited_pages"] = sorted(cited)
    return checks


def mark(value) -> str:
    if value is None:
        return "  - "
    return " ok " if value else "FAIL"


def main() -> None:
    load_dotenv()
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    tests = json.loads(TESTS_FILE.read_text(encoding="utf-8"))
    client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ["LLM_BASE_URL"])

    records, total_tokens = [], 0
    started = time.time()
    print(f"Running {len(tests)} questions with {os.environ['LLM_MODEL']}...\n")
    print(f"{'id':<5} {'type':<12} {'retr':>4} {'ctx':>4} {'ans':>4} {'cite':>4} {'refuse':>6}  notes")

    for test in tests:
        results = search(test["question"], chunks)
        answer, tokens = ask_model(client, test["question"], results)
        total_tokens += tokens
        checks = score(test, results, answer)

        notes = []
        if checks.get("missing_from_context"):
            notes.append("SEARCH missed: " + ", ".join(checks["missing_from_context"]))
        model_missed = [k for k in checks.get("missing_keywords", [])
                        if k not in checks.get("missing_from_context", [])]
        if model_missed:
            notes.append("MODEL left out: " + ", ".join(model_missed))
        if test["answerable"] and not checks["citation"]:
            notes.append(f"cited pages {checks['cited_pages']}, expected {test['expected_pages']}")
        if not checks["refusal"]:
            notes.append("answered when it should refuse" if not test["answerable"] else "refused wrongly")

        print(f"{test['id']:<5} {test['type']:<12} {mark(checks.get('retrieval'))} "
              f"{mark(checks.get('context'))} {mark(checks.get('answer'))} {mark(checks.get('citation'))}  {mark(checks['refusal'])}   "
              + "; ".join(notes))

        records.append({**test, "answer": answer, "tokens": tokens,
                        "retrieved": [f"{c.get('label', c['id'])} (p{','.join(map(str, pages_of(c)))})"
                                      for c in results],
                        "checks": checks})

    answerable = [r for r in records if r["answerable"]]
    summary = {
        "model": os.environ["LLM_MODEL"],
        "questions": len(records),
        "retrieval": sum(r["checks"]["retrieval"] for r in answerable) / len(answerable),
        "context": sum(r["checks"]["context"] for r in answerable) / len(answerable),
        "answer": sum(r["checks"]["answer"] for r in answerable) / len(answerable),
        "citation": sum(r["checks"]["citation"] for r in answerable) / len(answerable),
        "refusal": sum(r["checks"]["refusal"] for r in records) / len(records),
        "total_tokens": total_tokens,
        "seconds": round(time.time() - started, 1),
    }

    print("\nScores")
    for key in ["retrieval", "context", "answer", "citation", "refusal"]:
        print(f"  {key:<10} {summary[key]:>6.0%}")
    print(f"  tokens     {total_tokens:>6}   time {summary['seconds']}s")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"summary": summary, "results": records}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nFull answers saved to {out}")


if __name__ == "__main__":
    main()