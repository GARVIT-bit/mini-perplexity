"""Retrieval metrics. Inputs are ranked lists, best first.

Page level (labels are pages, retrievers return chunks):
  pages_in_order  collapse ranked chunks to unique URLs, keeping first-seen
                  order: 5 csv.html chunks in the top 10 count as ONE page,
                  otherwise one big page could fill the whole top 10.

  recall@k   of the relevant pages, what fraction appear in the top k pages.
  nDCG@k     rewards relevant pages near the top: a hit at rank i is worth
             1/log2(i+1), divided by the best possible score so 1.0 = perfect.
             Binary relevance (a page is relevant or not).
  MRR        1 / rank of the first relevant page (0 if none): "how far down
             must you read before something useful".

Chunk level:
  answer@k   1 if any of the top k CHUNKS is from a relevant page AND contains
             an evidence phrase, else 0. k=8 is what the LLM will be given,
             so this asks "would the answer actually be in the prompt?"
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

from .evalset import contains_evidence


def pages_in_order(urls: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def recall_at_k(ranked_pages: Sequence[str], relevant: Sequence[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        raise ValueError("recall is undefined for a question with no relevant pages")
    return len(rel & set(ranked_pages[:k])) / len(rel)


def ndcg_at_k(ranked_pages: Sequence[str], relevant: Sequence[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        raise ValueError("nDCG is undefined for a question with no relevant pages")
    dcg = sum(1.0 / math.log2(i + 1) for i, u in enumerate(ranked_pages[:k], start=1) if u in rel)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / ideal


def reciprocal_rank(ranked_pages: Sequence[str], relevant: Sequence[str]) -> float:
    rel = set(relevant)
    for i, u in enumerate(ranked_pages, start=1):
        if u in rel:
            return 1.0 / i
    return 0.0


def answer_at_k(ranked_chunks: Sequence[tuple[str, str]], relevant: Sequence[str],
                evidence: Sequence[str], k: int) -> float:
    """ranked_chunks: (url, text) pairs, best first."""
    rel = set(relevant)
    return float(any(url in rel and contains_evidence(text, list(evidence))
                     for url, text in ranked_chunks[:k]))
