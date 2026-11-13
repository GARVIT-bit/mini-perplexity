"""Hybrid search: merge the BM25 and vector rankings with Reciprocal Rank Fusion.

BM25 and vector search fail in opposite ways. BM25 nails exact terms
("os.path.join", an error code) but is blind to paraphrase; vectors catch
meaning but can miss a rare exact identifier. We want both.

The obstacle: their scores aren't comparable. BM25 gives 0-30+, unbounded
and query-dependent; cosine gives -1..1. Adding or weighting them needs
per-query normalization and tuning.

RRF sidesteps scores entirely and uses only each list's ORDER:

    rrf(d) = sum over rankers of   1 / (k + rank of d in that ranker)

with rank 1 for the top hit; a ranker that didn't return d contributes 0.

  - A chunk both rankers like beats a chunk only one loves:
    rank 2 in both = 2/62 = 0.0323   >   rank 1 in one only = 1/61 = 0.0164
  - k = 60 (the value from the original paper) flattens the curve, so rank 1
    vs rank 2 is a small difference (1/61 vs 1/62). With k = 0 the top hit
    of either list would dominate everything (1/1 vs 1/2).
  - Nothing to tune or normalize, which is why it's the standard baseline.

    python -m retrieval.fusion "how do I run things at the same time"
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from indexer.chunk import Chunk

RRF_K = 60
DEPTH = 50           # how many hits to take from each ranker before fusing


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Fuse ranked id lists (best first). Returns (id, score), best first."""
    if k < 0:
        raise ValueError("k must be >= 0")
    scores: dict[int, float] = {}
    for ranking in rankings:
        if len(set(ranking)) != len(ranking):
            raise ValueError("a ranking lists the same id twice")
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    # ties break on lower id so results are deterministic
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


@dataclass
class HybridHit:
    score: float                 # RRF score
    chunk_id: int
    chunk: Chunk
    bm25_rank: int | None        # 1-based rank in each ranker, None if it wasn't in that top `depth`
    vector_rank: int | None


class HybridSearcher:
    """BM25 + vector search over the same chunk ids, fused with RRF."""

    def __init__(self, bm25, vector_index, embedder, k: int = RRF_K, depth: int = DEPTH):
        if vector_index.n != bm25.index.n_docs:
            # row i of the vectors must be chunk i of the inverted index, or fusion mixes up chunks
            raise ValueError(f"{vector_index.n} vectors but {bm25.index.n_docs} chunks: re-run the embed step")
        self.bm25 = bm25
        self.vector_index = vector_index
        self.embedder = embedder
        self.k = k
        self.depth = depth

    def search(self, query: str, top_k: int = 10) -> list[HybridHit]:
        bm25_ids = [h.chunk_id for h in self.bm25.search(query, self.depth)]
        vec_ids = [cid for cid, _ in self.vector_index.search(self.embedder.encode_query(query), self.depth)]
        bm25_rank = {cid: r for r, cid in enumerate(bm25_ids, 1)}
        vec_rank = {cid: r for r, cid in enumerate(vec_ids, 1)}
        chunks = self.bm25.index.chunks
        return [HybridHit(score, cid, chunks[cid], bm25_rank.get(cid), vec_rank.get(cid))
                for cid, score in rrf([bm25_ids, vec_ids], self.k)[:top_k]]


def main() -> None:
    from indexer.embed import Embedder
    from indexer.index import InvertedIndex
    from .bm25 import BM25
    from .vector import BruteForceIndex

    p = argparse.ArgumentParser(description="Hybrid (BM25 + vector, RRF) search")
    p.add_argument("query")
    p.add_argument("--index", default="data/index")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--depth", type=int, default=DEPTH)
    p.add_argument("--vector", choices=["brute", "hnsw"], default="brute",
                   help="exact brute force, or HNSW (build it first: python -m retrieval.hnsw)")
    args = p.parse_args()

    if args.vector == "hnsw":
        from .hnsw import HNSWIndex
        vectors = HNSWIndex.load(args.index)
    else:
        vectors = BruteForceIndex.load(args.index)
    searcher = HybridSearcher(BM25(InvertedIndex.load(args.index)), vectors, Embedder(), depth=args.depth)
    searcher.search("warm up")               # first call pays one-off model warm-up
    t0 = time.perf_counter()
    hits = searcher.search(args.query, args.k)
    print(f"search={(time.perf_counter() - t0) * 1000:.0f} ms  (bm25 + embed query + vector + fuse)\n")
    for rank, h in enumerate(hits, 1):
        snippet = h.chunk.text[:140].replace("\n", " ")
        print(f"{rank}. {h.score:.4f}  bm25#{h.bm25_rank or '-'} vec#{h.vector_rank or '-'}  "
              f"{h.chunk.title}  {h.chunk.url}\n   {snippet}...\n")


if __name__ == "__main__":
    main()
