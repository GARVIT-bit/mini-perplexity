"""HNSW tests: agreement with exact brute-force search, scores, ef, save/load."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hnswlib")

from retrieval.hnsw import HNSWIndex          # noqa: E402
from retrieval.vector import BruteForceIndex  # noqa: E402


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(2000, 32)).astype(np.float32)
    queries = rng.normal(size=(100, 32)).astype(np.float32)
    return vecs, queries


def recall_at(hnsw, exact, queries, k=10) -> float:
    found = 0
    for q in queries:
        truth = {i for i, _ in exact.search(q, k)}
        found += len(truth & {i for i, _ in hnsw.search(q, k)})
    return found / (k * len(queries))


def test_high_recall_vs_brute_force(data):
    vecs, queries = data
    assert recall_at(HNSWIndex.build(vecs, ef=100), BruteForceIndex(vecs), queries) >= 0.95


def test_more_ef_never_hurts_recall(data):
    vecs, queries = data
    hnsw, exact = HNSWIndex.build(vecs, m=4, ef_construction=20), BruteForceIndex(vecs)
    recalls = []
    for ef in (10, 50, 200):
        hnsw.set_ef(ef)
        recalls.append(recall_at(hnsw, exact, queries))
    assert recalls == sorted(recalls)            # the speed/recall dial goes one way
    assert recalls[-1] > recalls[0]


def test_scores_are_cosine_like_brute_force(data):
    vecs, queries = data
    hnsw, exact = HNSWIndex.build(vecs), BruteForceIndex(vecs)
    exact_scores = dict(exact.search(queries[0], 50))
    for cid, score in hnsw.search(queries[0], 10):
        assert score == pytest.approx(exact_scores[cid], abs=1e-5)


def test_results_sorted_best_first(data):
    vecs, queries = data
    scores = [s for _, s in HNSWIndex.build(vecs).search(queries[1], 20)]
    assert scores == sorted(scores, reverse=True)


def test_k_bigger_than_ef_still_returns_k(data):
    vecs, queries = data
    hnsw = HNSWIndex.build(vecs, ef=10)
    assert len(hnsw.search(queries[0], k=50)) == 50
    assert hnsw.params["ef"] == 10              # the temporary bump doesn't stick


def test_small_and_empty_indexes():
    idx = HNSWIndex.build(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert [i for i, _ in idx.search(np.array([1.0, 0.1]), k=10)] == [0, 1]
    assert HNSWIndex.build(np.zeros((0, 4))).search(np.ones(4), k=5) == []


def test_wrong_query_size_rejected(data):
    vecs, _ = data
    with pytest.raises(ValueError):
        HNSWIndex.build(vecs[:10]).search(np.ones(5))


def test_same_input_same_graph(data):
    vecs, queries = data
    a, b = HNSWIndex.build(vecs[:500]), HNSWIndex.build(vecs[:500])
    for q in queries[:20]:
        assert a.search(q, 10) == b.search(q, 10)


def test_save_load_roundtrip(tmp_path, data):
    vecs, queries = data
    idx = HNSWIndex.build(vecs[:500], ef=60)
    idx.save(str(tmp_path))
    back = HNSWIndex.load(str(tmp_path))
    assert back.n == 500 and back.dim == 32 and back.params == idx.params
    for q in queries[:20]:
        assert back.search(q, 10) == idx.search(q, 10)
