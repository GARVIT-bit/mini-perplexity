"""Brute-force vector search tests on small hand-made vectors (no model needed)."""
from __future__ import annotations

import numpy as np
import pytest

from retrieval.vector import BruteForceIndex, normalize


def test_normalize_gives_unit_rows_and_keeps_zero_rows():
    v = normalize(np.array([[3.0, 4.0], [0.0, 0.0], [0.0, 2.0]]))
    assert np.allclose(np.linalg.norm(v, axis=1), [1.0, 0.0, 1.0])
    assert np.allclose(v[0], [0.6, 0.8])


def test_nearest_direction_wins():
    idx = BruteForceIndex(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    hits = idx.search(np.array([1.0, 0.1]), k=3)
    assert [i for i, _ in hits] == [0, 2, 1]


def test_scores_are_cosine_similarity():
    idx = BruteForceIndex(np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))
    scores = dict(idx.search(np.array([1.0, 0.0]), k=3))
    assert scores[0] == pytest.approx(1.0)       # same direction
    assert scores[1] == pytest.approx(0.0)       # perpendicular
    assert scores[2] == pytest.approx(-1.0)      # opposite


def test_length_does_not_matter_only_direction():
    # a long vector must not beat a short one pointing the right way
    idx = BruteForceIndex(np.array([[100.0, 100.0], [0.5, 0.0]]))
    assert idx.search(np.array([7.0, 0.0]), k=1)[0][0] == 1
    assert idx.search(np.array([7.0, 0.0]), k=1) == idx.search(np.array([0.001, 0.0]), k=1)


def test_matches_slow_reference_on_random_data():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(500, 32)).astype(np.float32)
    idx = BruteForceIndex(vecs)
    for _ in range(20):
        q = rng.normal(size=32).astype(np.float32)
        # reference: plain cosine formula, full sort
        cos = (vecs @ q) / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(q))
        expected = np.argsort(-cos)[:10].tolist()
        assert [i for i, _ in idx.search(q, k=10)] == expected


def test_k_larger_than_index_and_empty_index():
    idx = BruteForceIndex(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert len(idx.search(np.array([1.0, 0.0]), k=10)) == 2
    assert idx.search(np.array([1.0, 0.0]), k=0) == []
    assert BruteForceIndex(np.zeros((0, 4))).search(np.ones(4), k=5) == []


def test_ties_break_on_lower_id():
    idx = BruteForceIndex(np.array([[1.0, 0.0]] * 5))
    assert [i for i, _ in idx.search(np.array([1.0, 0.0]), k=3)] == [0, 1, 2]


def test_wrong_query_size_rejected():
    idx = BruteForceIndex(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        idx.search(np.ones(5))
    with pytest.raises(ValueError):
        BruteForceIndex(np.ones(4))              # 1-D: not a matrix of chunk vectors


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    idx = BruteForceIndex(rng.normal(size=(50, 8)))
    idx.save(str(tmp_path))
    back = BruteForceIndex.load(str(tmp_path))
    q = rng.normal(size=8)
    assert back.search(q, k=5) == idx.search(q, k=5)
