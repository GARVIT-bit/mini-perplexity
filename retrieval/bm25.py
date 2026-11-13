"""BM25 ranking over the inverted index.

For each query term t that a chunk d contains:

    score += idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg_len))

  idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))
      Rare terms count more: "hnswlib" in 3 chunks says far more than "use"
      in 2,000. The "1 +" (Lucene's variant) keeps idf positive even for a
      term in more than half the chunks; classic BM25 would go negative.

  tf saturation (k1 = 1.2)
      The 1st occurrence of a term matters most, the 10th barely. As tf grows
      the fraction tends to (k1 + 1), so a chunk can't win by repeating a word.

  length normalization (b = 0.75)
      A long chunk contains more words by chance. b = 1 fully divides that
      out, b = 0 ignores length.

Scoring is term-at-a-time: walk each query term's postings list, add into a
score per chunk, then take the top k with a heap. Only chunks containing at
least one query term are ever touched.

    python -m retrieval.bm25 "how does asyncio gather work"
"""
from __future__ import annotations

import argparse
import heapq
import math
import time
from dataclasses import dataclass

from indexer.analyzer import analyze
from indexer.chunk import Chunk
from indexer.index import InvertedIndex

K1 = 1.2
B = 0.75


def idf(n_docs: int, df: int) -> float:
    return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))


@dataclass
class Hit:
    score: float
    chunk_id: int        # internal id in the index
    chunk: Chunk


class BM25:
    def __init__(self, index: InvertedIndex, k1: float = K1, b: float = B):
        self.index = index
        self.k1 = k1
        self.b = b
        avg = index.avg_len or 1.0
        # the length part of the denominator depends only on the chunk: compute it once, not per query
        self._norm = [k1 * (1 - b + b * dl / avg) for dl in index.doc_len]

    def scores(self, query: str) -> dict[int, float]:
        n = self.index.n_docs
        acc: dict[int, float] = {}
        # set(): a repeated query word ("python python") shouldn't double its weight
        for term in set(analyze(query)):
            plist = self.index.postings.get(term)
            if not plist:
                continue                      # unknown term: contributes nothing, isn't an error
            w = idf(n, len(plist))
            for cid, tf in plist:
                acc[cid] = acc.get(cid, 0.0) + w * tf * (self.k1 + 1) / (tf + self._norm[cid])
        return acc

    def search(self, query: str, k: int = 10) -> list[Hit]:
        acc = self.scores(query)
        # tie-break on lower id so results are deterministic
        top = heapq.nsmallest(k, acc.items(), key=lambda kv: (-kv[1], kv[0]))
        return [Hit(s, cid, self.index.chunks[cid]) for cid, s in top]


def main() -> None:
    p = argparse.ArgumentParser(description="BM25 search over the saved index")
    p.add_argument("query")
    p.add_argument("--index", default="data/index")
    p.add_argument("-k", type=int, default=5)
    args = p.parse_args()

    t0 = time.perf_counter()
    bm25 = BM25(InvertedIndex.load(args.index))
    t1 = time.perf_counter()
    hits = bm25.search(args.query, args.k)
    t2 = time.perf_counter()
    print(f"terms={analyze(args.query)}  load={t1 - t0:.2f}s  search={(t2 - t1) * 1000:.1f} ms\n")
    for rank, h in enumerate(hits, 1):
        snippet = h.chunk.text[:160].replace("\n", " ")
        print(f"{rank}. {h.score:6.2f}  {h.chunk.title}  {h.chunk.url}\n   {snippet}...\n")
    if not hits:
        print("no matches")


if __name__ == "__main__":
    main()
