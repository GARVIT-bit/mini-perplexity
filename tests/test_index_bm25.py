"""Inverted index + BM25 tests on tiny hand-made corpora."""
from __future__ import annotations

import math

import pytest

from indexer.analyzer import analyze
from indexer.chunk import Chunk
from indexer.index import InvertedIndex
from retrieval.bm25 import BM25, idf


def ch(text: str, title: str = "", i: int = 0) -> Chunk:
    return Chunk(chunk_id=f"d{i}:0", doc_id=f"d{i}", ord=0, url=f"https://ex.com/{i}", title=title, text=text)


def build(texts: list[str], **kw) -> InvertedIndex:
    return InvertedIndex.build([ch(t, i=i) for i, t in enumerate(texts)], **kw)


# ------------------------------------------------------------------ index

def test_postings_hold_ids_and_tf():
    idx = build(["apple banana", "apple apple cherry", "banana"])
    appl = analyze("apple")[0]
    assert idx.postings[appl] == [(0, 1), (1, 2)]
    assert idx.df(appl) == 2 and idx.df("nonexistent") == 0


def test_postings_sorted_by_id():
    idx = build(["x y", "y z", "x z", "x"])
    for plist in idx.postings.values():
        ids = [cid for cid, _ in plist]
        assert ids == sorted(ids)


def test_doc_lengths_count_analyzed_terms():
    idx = build(["the apple and the banana", "cherry"])   # stopwords don't count
    assert idx.doc_len == [2, 1]
    assert idx.avg_len == 1.5


def test_title_indexed_with_chunk():
    chunks = [Chunk("d0:0", "d0", 0, "u", "Asyncio Guide", "gather runs tasks concurrently")]
    assert "asyncio" in InvertedIndex.build(chunks).postings
    assert "asyncio" not in InvertedIndex.build(chunks, include_title=False).postings


def test_save_load_roundtrip(tmp_path):
    idx = InvertedIndex.build([ch("apple banana", "Fruit", 0), ch("cherry pie café", "Dessert", 1)])
    idx.save(str(tmp_path))
    back = InvertedIndex.load(str(tmp_path))
    assert back.postings == idx.postings
    assert back.doc_len == idx.doc_len
    assert back.chunks == idx.chunks
    assert back.include_title == idx.include_title


def test_empty_corpus():
    idx = build([])
    assert idx.n_docs == 0
    assert BM25(idx).search("anything") == []


# ------------------------------------------------------------------ BM25

def test_matches_hand_computed_score():
    idx = build(["apple banana", "apple apple cherry cherry", "banana"], include_title=False)
    # chunk 1: tf=2, len=4, avg len = (2+4+1)/3, N=3, df=1
    norm = 1.2 * (1 - 0.75 + 0.75 * 4 / (7 / 3))
    expected = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5)) * 2 * 2.2 / (2 + norm)
    [hit] = BM25(idx).search("cherry")
    assert hit.chunk_id == 1
    assert hit.score == pytest.approx(expected)


def test_rare_term_beats_common_term():
    texts = ["python guide"] * 8 + ["python hnswlib", "python intro"]
    hits = BM25(build(texts)).search("python hnswlib")
    assert hits[0].chunk_id == 8


def test_idf_always_positive():
    assert idf(10, 10) > 0
    assert idf(10, 1) > idf(10, 5) > idf(10, 10)


def test_tf_saturates():
    texts = ["crawler " * 1 + "filler words here", "crawler " * 50 + "filler words here", "other text"]
    s = BM25(build(texts), b=0.0).scores("crawler")
    w = idf(3, 2)
    assert s[1] > s[0]
    assert s[1] < w * (1.2 + 1)              # can never exceed idf * (k1 + 1)
    assert s[1] / s[0] < 2.2                 # 50x the tf is far from 50x the score


def test_shorter_chunk_wins_at_equal_tf():
    texts = ["index tiny", "index " + "padding " * 40, "unrelated"]
    hits = BM25(build(texts)).search("index")
    assert [h.chunk_id for h in hits] == [0, 1]


def test_b_zero_ignores_length():
    texts = ["index tiny", "index " + "padding " * 40, "unrelated"]
    s = BM25(build(texts), b=0.0).scores("index")
    assert s[0] == pytest.approx(s[1])


def test_stemming_lets_query_find_other_word_form():
    hits = BM25(build(["managing a database connection", "cooking pasta"])).search("Connecting")
    assert [h.chunk_id for h in hits] == [0]


def test_unknown_terms_and_stopword_only_query():
    bm = BM25(build(["apple", "banana"]))
    assert bm.search("zzzz qqqq") == []
    assert bm.search("the of and") == []


def test_repeated_query_word_counts_once():
    bm = BM25(build(["apple banana", "cherry"]))
    assert bm.scores("apple apple apple") == bm.scores("apple")


def test_k_limits_and_ties_are_deterministic():
    bm = BM25(build(["identical text"] * 6))
    hits = bm.search("identical", k=3)
    assert [h.chunk_id for h in hits] == [0, 1, 2]
