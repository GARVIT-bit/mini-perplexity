"""Re-ranker: re-order the hybrid candidates with a cross-encoder.

Bi-encoder (Phase 3): embed query and passage SEPARATELY, compare vectors.
  Fast (passage vectors are computed once, offline) but the query never
  "sees" the passage.
Cross-encoder: feed "[query] [SEP] [passage]" through one transformer, so
  every query word attends to every passage word, and output one relevance
  score. Much more precise, but it must run per (query, passage) pair at
  query time, so it can't scan 25k chunks.

Hence two stages: hybrid search picks ~50 candidates cheaply, the
cross-encoder re-scores only those, keep the top 8 for the LLM prompt.

Model: cross-encoder/ms-marco-MiniLM-L6-v2, trained on MS MARCO (Bing
queries + passages). Measured on this laptop's CPU (Phase 4 notes):
  - 50 candidates ~3.7 s per query; 20 candidates ~1.0 s. The 2-layer
    ms-marco-MiniLM-L2-v2 is ~3x faster.
  - it leans on word overlap: it lifted the right csv/asyncio passages from
    hybrid rank 9-15, but put "Is TDD Dead" first for "run several things at
    the same time". Whether it helps overall is for the eval set to decide.

Output: a logit (unbounded, higher = more relevant). sigmoid(logit) gives a
0-1 relevance; the "not found" check later thresholds on it.

    python -m retrieval.rerank "how do I read a csv file"
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np

from indexer.chunk import Chunk
from indexer.index import InvertedIndex

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
CANDIDATES = 50
TOP_K = 8


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class RankedHit:
    logit: float                 # cross-encoder score, higher = more relevant
    relevance: float             # sigmoid(logit), 0..1
    chunk_id: int
    chunk: Chunk
    retrieval_rank: int          # 1-based rank in the hybrid (RRF) list before re-ranking
    bm25_rank: int | None
    vector_rank: int | None


class Reranker:
    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu", model=None):
        """`model` is anything with predict(list of (query, text)) -> scores; tests pass a fake."""
        if model is None:
            from sentence_transformers import CrossEncoder   # slow import: only when actually needed
            model = CrossEncoder(model_name, device=device)
        self.model_name = model_name
        self.model = model

    def score(self, query: str, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        scores = self.model.predict([(query, t) for t in texts], batch_size=batch_size)
        return np.asarray(scores, dtype=np.float32).reshape(-1)

    def rerank(self, query: str, hits: list, top_k: int = TOP_K) -> list[RankedHit]:
        """`hits` are HybridHits, best first. Same text recipe as indexing: title + chunk."""
        texts = [InvertedIndex.index_text(h.chunk, True) for h in hits]
        logits = self.score(query, texts)
        ranked = [RankedHit(float(s), sigmoid(float(s)), h.chunk_id, h.chunk, r, h.bm25_rank, h.vector_rank)
                  for r, (h, s) in enumerate(zip(hits, logits), start=1)]
        # ties keep the retrieval order, so equal scores never shuffle randomly
        ranked.sort(key=lambda x: (-x.logit, x.retrieval_rank))
        return ranked[:top_k]


class RerankingSearcher:
    """Hybrid search for `candidates`, then cross-encoder down to top_k."""

    def __init__(self, hybrid, reranker: Reranker, candidates: int = CANDIDATES):
        self.hybrid = hybrid
        self.reranker = reranker
        self.candidates = candidates

    def search(self, query: str, top_k: int = TOP_K) -> list[RankedHit]:
        return self.reranker.rerank(query, self.hybrid.search(query, self.candidates), top_k)


def main() -> None:
    from indexer.embed import Embedder
    from .bm25 import BM25
    from .fusion import HybridSearcher
    from .vector import BruteForceIndex

    p = argparse.ArgumentParser(description="Hybrid search + cross-encoder re-ranking")
    p.add_argument("query")
    p.add_argument("--index", default="data/index")
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--candidates", type=int, default=CANDIDATES)
    p.add_argument("-k", type=int, default=TOP_K)
    args = p.parse_args()

    hybrid = HybridSearcher(BM25(InvertedIndex.load(args.index)), BruteForceIndex.load(args.index), Embedder())
    searcher = RerankingSearcher(hybrid, Reranker(args.model), args.candidates)
    searcher.search("warm up", 1)            # first call pays one-off model warm-up
    t0 = time.perf_counter()
    cands = hybrid.search(args.query, args.candidates)
    t1 = time.perf_counter()
    hits = searcher.reranker.rerank(args.query, cands, args.k)
    t2 = time.perf_counter()
    print(f"hybrid={(t1 - t0) * 1000:.0f} ms  rerank {len(cands)} candidates={(t2 - t1) * 1000:.0f} ms\n")
    for rank, h in enumerate(hits, 1):
        snippet = h.chunk.text[:140].replace("\n", " ")
        # print the logit too: sigmoid squashes every strong match to ~1.000
        print(f"{rank}. logit={h.logit:6.2f} rel={h.relevance:.3f}  (was #{h.retrieval_rank})  {h.chunk.title}  {h.chunk.url}\n"
              f"   {snippet}...\n")


if __name__ == "__main__":
    main()
