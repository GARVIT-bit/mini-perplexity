"""Approximate nearest-neighbour search with HNSW (hnswlib).

Brute force compares the query with all N vectors: exact, O(N) per query.
HNSW (Hierarchical Navigable Small World) builds a graph where each vector
links to ~M of its near neighbours, in layers:

  top layers     few nodes, long links   -> jump across the space fast
  bottom layer   every node, short links -> refine locally

A search starts at the top, greedily walks to whichever neighbour is closer
to the query, drops a layer, repeats. It visits a few hundred nodes instead
of all N, roughly O(log N). The price: it can miss a true neighbour, so we
measure recall@k against brute force (bench/hnsw.py).

Knobs:
  M               links per node. More = better recall, more memory, slower build.
  ef_construction candidates considered while inserting. More = better graph, slower build.
  ef (search)     candidates kept during a query. THE speed/recall dial; must be >= k.

Same interface as BruteForceIndex (n, dim, search, save, load), so hybrid
search can use either.

    python -m retrieval.hnsw --index data/index          # build hnsw.bin from vectors.npy
"""
from __future__ import annotations

import json
import os

import hnswlib
import numpy as np

from .vector import normalize

M = 16
EF_CONSTRUCTION = 200
EF_SEARCH = 100
SEED = 42


class HNSWIndex:
    def __init__(self, index: hnswlib.Index, n: int, dim: int, params: dict):
        self.index = index
        self._n = n
        self._dim = dim
        self.params = params

    @classmethod
    def build(cls, vectors: np.ndarray, m: int = M, ef_construction: int = EF_CONSTRUCTION,
              ef: int = EF_SEARCH, seed: int = SEED, threads: int = 1) -> "HNSWIndex":
        vectors = normalize(vectors)
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2-D array: one row per chunk")
        n, dim = vectors.shape
        # 'ip' (inner product) on unit vectors is cosine; hnswlib reports distance = 1 - dot
        index = hnswlib.Index(space="ip", dim=dim)
        index.init_index(max_elements=max(n, 1), ef_construction=ef_construction, M=m, random_seed=seed)
        # one thread by default: insertion order then never varies, so the same
        # vectors always give the same graph and the same benchmark numbers
        index.set_num_threads(threads)
        if n:
            index.add_items(vectors, np.arange(n))      # label i = chunk id i, same ids as the inverted index
        params = {"M": m, "ef_construction": ef_construction, "ef": ef, "seed": seed}
        obj = cls(index, n, dim, params)
        obj.set_ef(ef)
        return obj

    @property
    def n(self) -> int:
        return self._n

    @property
    def dim(self) -> int:
        return self._dim

    def set_ef(self, ef: int) -> None:
        self.params["ef"] = ef
        self.index.set_ef(ef)

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        """Top-k (chunk_id, cosine similarity), best first. Approximate."""
        if self.n == 0 or k <= 0:
            return []
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (self.dim,):
            raise ValueError(f"query has shape {query.shape}, index expects ({self.dim},)")
        k = min(k, self.n)
        # hnswlib can only return as many results as candidates it keeps
        if self.params["ef"] < k:
            self.index.set_ef(k)
        labels, dists = self.index.knn_query(normalize(query), k=k)
        if self.params["ef"] < k:
            self.index.set_ef(self.params["ef"])
        hits = [(int(i), float(1.0 - d)) for i, d in zip(labels[0], dists[0])]
        return sorted(hits, key=lambda h: (-h[1], h[0]))   # same tie rule as brute force

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.index.save_index(os.path.join(out_dir, "hnsw.bin"))
        with open(os.path.join(out_dir, "hnsw.json"), "w", encoding="utf-8") as fh:
            json.dump({"n": self.n, "dim": self.dim, **self.params}, fh)

    @classmethod
    def load(cls, in_dir: str) -> "HNSWIndex":
        with open(os.path.join(in_dir, "hnsw.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        index = hnswlib.Index(space="ip", dim=meta["dim"])
        index.load_index(os.path.join(in_dir, "hnsw.bin"), max_elements=max(meta["n"], 1))
        params = {k: meta[k] for k in ("M", "ef_construction", "ef", "seed")}
        obj = cls(index, meta["n"], meta["dim"], params)
        obj.set_ef(params["ef"])
        return obj


def main() -> None:
    import argparse
    import time

    from .vector import BruteForceIndex

    p = argparse.ArgumentParser(description="Build an HNSW index from saved vectors")
    p.add_argument("--index", default="data/index")
    p.add_argument("--m", type=int, default=M)
    p.add_argument("--ef-construction", type=int, default=EF_CONSTRUCTION)
    p.add_argument("--ef", type=int, default=EF_SEARCH)
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()

    vectors = BruteForceIndex.load(args.index).vectors
    t0 = time.perf_counter()
    idx = HNSWIndex.build(vectors, args.m, args.ef_construction, args.ef, threads=args.threads)
    secs = time.perf_counter() - t0
    idx.save(args.index)
    size = os.path.getsize(os.path.join(args.index, "hnsw.bin"))
    print(f"built HNSW over {idx.n} vectors in {secs:.1f}s  (M={args.m}, ef_construction={args.ef_construction})"
          f"  -> {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
