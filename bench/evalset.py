"""The labelled question set: load it, and check its labels against the corpus.

bench/eval/questions.jsonl, one question per line:
    {"id": "q02", "type": "natural", "question": "how do I read a csv file",
     "relevant": ["https://docs.python.org/3/library/csv.html"],
     "evidence": ["csv.reader"]}

  relevant  pages (URLs, not chunk ids) that answer it. URLs survive
            re-chunking; chunk ids change every time chunk size changes.
  evidence  short phrases from the actual answer text. A retrieved CHUNK
            counts as answer-bearing if it's from a relevant page and
            contains one of them (used for Answer@k).
  type      keyword    uses the page's own terms / identifiers
            natural    how a person would ask
            paraphrase deliberately avoids the page's wording
            unanswerable  nothing in the corpus answers it (for "not found")

The labels are human judgments (the first draft was written by Claude while
reading the corpus, and needs review). `check` catches the mechanical errors:
URLs that aren't indexed and evidence that doesn't appear where claimed.

    python -m bench.evalset --check
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field

QUESTIONS_FILE = "bench/eval/questions.jsonl"
TYPES = ("keyword", "natural", "paraphrase", "unanswerable")


@dataclass
class Question:
    id: str
    type: str
    question: str
    relevant: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return bool(self.relevant)


def normalize(text: str) -> str:
    """Case- and whitespace-insensitive form, so evidence matching ignores line wraps."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def contains_evidence(text: str, evidence: list[str]) -> bool:
    t = normalize(text)
    return any(normalize(e) in t for e in evidence)


def load_questions(path: str = QUESTIONS_FILE) -> list[Question]:
    qs: list[Question] = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            q = Question(**json.loads(line))
            if q.type not in TYPES:
                raise ValueError(f"line {n}: unknown type {q.type!r}")
            if q.answerable != (q.type != "unanswerable"):
                raise ValueError(f"line {n}: only 'unanswerable' questions may have no relevant pages")
            if q.answerable and not q.evidence:
                raise ValueError(f"line {n}: answerable question needs evidence phrases")
            qs.append(q)
    ids = [q.id for q in qs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate question ids")
    return qs


def check(questions: list[Question], pages: dict[str, str]) -> list[str]:
    """Problems with the labels, given url -> page text. Empty list = labels are consistent."""
    problems = []
    for q in questions:
        missing = [u for u in q.relevant if u not in pages]
        for u in missing:
            problems.append(f"{q.id}: relevant URL not in index: {u}")
        present = [u for u in q.relevant if u in pages]
        if present and not any(contains_evidence(pages[u], q.evidence) for u in present):
            problems.append(f"{q.id}: no evidence phrase {q.evidence} found on any relevant page")
    return problems


def main() -> None:
    from collections import Counter

    from indexer.build import load_docs

    p = argparse.ArgumentParser(description="Validate the question set against the indexed corpus")
    p.add_argument("--questions", default=QUESTIONS_FILE)
    p.add_argument("--index", default="data/index")
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    qs = load_questions(args.questions)
    print(f"{len(qs)} questions: {dict(Counter(q.type for q in qs))}")
    if args.check:
        pages = {d.url: f"{d.title}\n{d.text}" for d in load_docs(args.index)}
        problems = check(qs, pages)
        print("\n".join(problems) if problems else "all labels consistent with the corpus")
        raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
