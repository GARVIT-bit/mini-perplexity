"""Brute-force vector search: compare the query against every chunk vector.

Each chunk is a unit-length vector. For unit vectors, cosine similarity is
just the dot product, so scoring all N chunks is one matrix-vector multiply:

    scores = vectors @ query        # shape (N,), one number per chunk

That is exact (it really does find the nearest vectors) and costs O(N * d)
per query. Fine for ~10k chunks; hnswlib later trades a little accuracy for
speed, and this exact search is what we measure its recall against.

Vectors are normalized once at build time, so a query never pays for it and
"cosine" vs "dot product" can't disagree.
"""
from __future__ import annotations

import os

import numpy as np


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to length 1. Zero rows stay zero (they match nothing)."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


class BruteForceIndex:
    def __init__(self, vectors: np.ndarray, already_normalized: bool = False):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2-D array: one row per chunk")
        # row i belongs to chunk id i, same ids as the inverted index
        self.vectors = vectors if already_normalized else normalize(vectors)

    @property
    def n(self) -> int:
        return self.vectors.shape[0]

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]

    def scores(self, query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (self.dim,):
            raise ValueError(f"query has shape {query.shape}, index expects ({self.dim},)")
        return self.vectors @ normalize(query)

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        """Top-k (chunk_id, cosine similarity), best first."""
        if self.n == 0 or k <= 0:
            return []
        s = self.scores(query)
        k = min(k, self.n)
        # argpartition finds the top k in O(N) without sorting all N scores;
        # only those k are then sorted. Ties break on lower id, like BM25.
        top = np.argpartition(-s, k - 1)[:k]
        top = sorted(top.tolist(), key=lambda i: (-s[i], i))
        return [(i, float(s[i])) for i in top]

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "vectors.npy"), self.vectors)

    @classmethod
    def load(cls, in_dir: str) -> "BruteForceIndex":
        # saved vectors are already unit length; normalizing again would nudge
        # the last float digits and make a loaded index score slightly differently
        return cls(np.load(os.path.join(in_dir, "vectors.npy")), already_normalized=True)


def main() -> None:
    import argparse
    import time

    from indexer.embed import Embedder
    from indexer.index import InvertedIndex

    p = argparse.ArgumentParser(description="Vector search over the embedded index")
    p.add_argument("query")
    p.add_argument("--index", default="data/index")
    p.add_argument("-k", type=int, default=5)
    args = p.parse_args()

    chunks = InvertedIndex.load_chunks(args.index)
    idx = BruteForceIndex.load(args.index)
    if idx.n != len(chunks):
        raise SystemExit(f"{idx.n} vectors but {len(chunks)} chunks: run python -m indexer.embed again")
    emb = Embedder()
    t0 = time.perf_counter()
    q = emb.encode_query(args.query)
    t1 = time.perf_counter()
    hits = idx.search(q, args.k)
    t2 = time.perf_counter()
    print(f"embed query={(t1 - t0) * 1000:.0f} ms  search {idx.n} vectors={(t2 - t1) * 1000:.1f} ms\n")
    for rank, (cid, score) in enumerate(hits, 1):
        c = chunks[cid]
        snippet = c.text[:160].replace("\n", " ")
        print(f"{rank}. {score:.3f}  {c.title}  {c.url}\n   {snippet}...\n")


if __name__ == "__main__":
    main()
