"""Embedder tests. These load the real MiniLM model (cached after first download),
so they take ~10 s; skipped if sentence-transformers isn't installed."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sentence_transformers")

from indexer.embed import Embedder          # noqa: E402
from retrieval.vector import BruteForceIndex  # noqa: E402


@pytest.fixture(scope="module")
def emb() -> Embedder:
    return Embedder()


def test_shape_dtype_and_unit_length(emb):
    v = emb.encode(["hello world", "a second, longer sentence about search engines"])
    assert v.shape == (2, 384) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_empty_input(emb):
    assert emb.encode([]).shape == (0, 384)
    assert emb.count_tokens([]) == []


def test_same_text_same_vector(emb):
    a, b = emb.encode(["inverted index"]), emb.encode(["inverted index"])
    assert np.allclose(a, b, atol=1e-6)


def test_meaning_beats_shared_words(emb):
    # the query shares NO content word with the right passage; BM25 would score it 0
    passages = [
        "Concurrency lets a program make progress on several tasks at once using threads or coroutines.",
        "Knead the dough, let it rise for an hour, then bake the bread until golden.",
        "The stock market fell sharply after the central bank raised interest rates.",
    ]
    idx = BruteForceIndex(emb.encode(passages), already_normalized=True)
    hits = idx.search(emb.encode_query("how do I run multiple things simultaneously"), k=3)
    assert hits[0][0] == 0
    assert hits[0][1] > hits[1][1] + 0.1         # and by a clear margin


def test_paraphrases_closer_than_unrelated(emb):
    a, b, c = emb.encode(["How do I open a file in Python?",
                          "Reading files with Python's open() function",
                          "Refactoring legacy code with the strangler fig pattern"])
    assert a @ b > a @ c


def test_count_tokens_includes_special_tokens(emb):
    one, more = emb.count_tokens(["hello", "hello world again"])
    assert one == 3                              # [CLS] hello [SEP]
    assert more == 5


def test_text_past_the_limit_is_ignored(emb):
    # THE reason chunks are sized to the model: everything after max_tokens is invisible
    filler = "search engines rank documents by relevance " * 60        # far more than 256 tokens
    assert emb.count_tokens([filler])[0] > emb.max_tokens
    a, b = emb.encode([filler + " The capital of France is Paris.",
                       filler + " Bananas are rich in potassium."])
    assert np.allclose(a, b, atol=1e-6)          # different endings, identical vectors


def test_text_within_the_limit_is_seen(emb):
    a, b = emb.encode(["Search engines rank documents. The capital of France is Paris.",
                       "Search engines rank documents. Bananas are rich in potassium."])
    assert not np.allclose(a, b, atol=1e-3)
