"""RRF and hybrid search tests. The hybrid tests use a tiny fake embedder, so no model loads."""
from __future__ import annotations

import numpy as np
import pytest

from indexer.chunk import Chunk
from indexer.index import InvertedIndex
from retrieval.bm25 import BM25
from retrieval.fusion import HybridSearcher, rrf
from retrieval.vector import BruteForceIndex


# ------------------------------------------------------------------ rrf

def test_hand_computed_scores():
    fused = dict(rrf([[10, 20, 30], [20, 40]], k=60))
    assert fused[20] == pytest.approx(1 / 62 + 1 / 61)      # rank 2 in A, rank 1 in B
    assert fused[10] == pytest.approx(1 / 61)
    assert fused[30] == pytest.approx(1 / 63)
    assert fused[40] == pytest.approx(1 / 62)


def test_order_best_first():
    assert [d for d, _ in rrf([[10, 20, 30], [20, 40]])] == [20, 10, 40, 30]


def test_agreement_beats_one_ranker_loving_it():
    # 7 is rank 2 in both; 1 is rank 1 in A only; 5 is rank 1 in B only
    fused = [d for d, _ in rrf([[1, 7, 2], [5, 7, 3]])]
    assert fused[0] == 7


def test_single_ranking_keeps_its_order():
    assert [d for d, _ in rrf([[3, 1, 2]])] == [3, 1, 2]


def test_empty_inputs():
    assert rrf([]) == []
    assert rrf([[], []]) == []
    assert [d for d, _ in rrf([[], [4, 5]])] == [4, 5]      # one ranker found nothing


def test_ties_break_on_lower_id():
    # 9 and 2 are both rank 1 in exactly one list: equal scores
    assert [d for d, _ in rrf([[9], [2]])] == [2, 9]


def test_k_controls_how_much_the_top_rank_dominates():
    a, b = [1, 9, 2], [3, 8, 2]      # 2 is third in both; 1 and 3 are first in one list each
    # k=60: agreement wins. 2/63 = 0.032 beats 1/61 = 0.016
    assert rrf([a, b], k=60)[0][0] == 2
    # k=0: a single rank-1 hit wins. 1/1 = 1.0 beats 1/3 + 1/3 = 0.67
    assert rrf([a, b], k=0)[0][0] == 1
    assert dict(rrf([a, b], k=0))[2] == pytest.approx(2 / 3)


def test_duplicate_id_in_one_ranking_rejected():
    with pytest.raises(ValueError):
        rrf([[1, 1, 2]])
    with pytest.raises(ValueError):
        rrf([[1]], k=-1)


def test_three_rankers():
    fused = dict(rrf([[1, 2], [2, 1], [2, 3]]))
    assert fused[2] == pytest.approx(1 / 62 + 1 / 61 + 1 / 61)


# ------------------------------------------------------------------ hybrid

TEXTS = [
    "use os.path.join to build file paths",            # 0: exact-term match for "os.path.join"
    "threads let a program do several tasks at once",  # 1: meaning match for "concurrency"
    "bake bread in a hot oven",                        # 2: unrelated
    "concurrency concurrency concurrency keyword spam",  # 3: BM25 loves it, vectors don't
]
# hand-made 2-D "meanings": axis 0 = programming-concurrency, axis 1 = file paths
VECS = np.array([[0.0, 1.0], [1.0, 0.1], [-1.0, -1.0], [0.2, -0.3]])


class FakeEmbedder:
    def __init__(self, table: dict[str, list[float]]):
        self.table = table

    def encode_query(self, query: str) -> np.ndarray:
        return np.array(self.table[query], dtype=np.float32)


def make(table: dict[str, list[float]], **kw) -> HybridSearcher:
    chunks = [Chunk(f"d{i}:0", f"d{i}", 0, f"u{i}", "", t) for i, t in enumerate(TEXTS)]
    return HybridSearcher(BM25(InvertedIndex.build(chunks)), BruteForceIndex(VECS), FakeEmbedder(table), **kw)


def test_hybrid_reports_both_ranks():
    hits = make({"os.path.join": [0.0, 1.0]}).search("os.path.join", top_k=4)
    top = hits[0]
    assert top.chunk_id == 0 and top.bm25_rank == 1 and top.vector_rank == 1
    assert top.score == pytest.approx(2 / 61)


def test_vector_only_hit_is_still_returned():
    # "concurrency" appears in chunk 3 only (BM25), but chunk 1 is the one that means it (vectors)
    hits = make({"concurrency": [1.0, 0.0]}).search("concurrency", top_k=4)
    by_id = {h.chunk_id: h for h in hits}
    assert by_id[1].bm25_rank is None and by_id[1].vector_rank == 1
    assert by_id[3].bm25_rank == 1
    assert {1, 3} <= {h.chunk_id for h in hits[:2]}      # both rankers' favourites surface


def test_bm25_finds_nothing_falls_back_to_vectors():
    hits = make({"zzzz": [1.0, 0.0]}).search("zzzz", top_k=2)
    assert [h.chunk_id for h in hits] == [1, 3]
    assert all(h.bm25_rank is None for h in hits)


def test_depth_limits_what_each_ranker_contributes():
    hits = make({"concurrency": [1.0, 0.0]}, depth=1).search("concurrency", top_k=10)
    assert {h.chunk_id for h in hits} == {1, 3}          # one from each ranker, nothing else


def test_mismatched_vector_count_rejected():
    chunks = [Chunk("d0:0", "d0", 0, "u", "", "only one chunk")]
    with pytest.raises(ValueError):
        HybridSearcher(BM25(InvertedIndex.build(chunks)), BruteForceIndex(VECS), FakeEmbedder({}))


def test_hybrid_accepts_hnsw_index():
    pytest.importorskip("hnswlib")
    from retrieval.hnsw import HNSWIndex
    chunks = [Chunk(f"d{i}:0", f"d{i}", 0, f"u{i}", "", t) for i, t in enumerate(TEXTS)]
    s = HybridSearcher(BM25(InvertedIndex.build(chunks)), HNSWIndex.build(VECS),
                       FakeEmbedder({"concurrency": [1.0, 0.0]}))
    assert {1, 3} <= {h.chunk_id for h in s.search("concurrency", top_k=2)}
