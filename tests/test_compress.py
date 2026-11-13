"""Compression tests: varint bytes, delta round trips, corrupt input, and that
search over a compressed index gives exactly the same results."""
from __future__ import annotations

import random

import pytest

from indexer.chunk import Chunk
from indexer.compress import (CompressedPostings, decode_postings, decode_varint,
                              encode_postings, encode_varint)
from indexer.index import InvertedIndex
from retrieval.bm25 import BM25


def varint(n: int) -> bytes:
    out = bytearray()
    encode_varint(n, out)
    return bytes(out)


# ------------------------------------------------------------------ varint

@pytest.mark.parametrize("n,expected", [
    (0, b"\x00"),
    (1, b"\x01"),
    (127, b"\x7f"),                  # largest 1-byte value
    (128, b"\x80\x01"),              # smallest 2-byte value: low 7 bits (0) + continuation, then 1
    (300, b"\xac\x02"),              # 300 = 0b10_0101100 -> 0xAC (0x2C | 0x80), 0x02
    (16383, b"\xff\x7f"),            # largest 2-byte value
    (16384, b"\x80\x80\x01"),
])
def test_varint_exact_bytes(n, expected):
    assert varint(n) == expected
    assert decode_varint(expected, 0) == (n, len(expected))


def test_varint_large_numbers_round_trip():
    for n in [2**31 - 1, 2**32, 2**63, 10**30]:
        assert decode_varint(varint(n), 0)[0] == n


def test_varint_rejects_negative():
    with pytest.raises(ValueError):
        varint(-1)


def test_truncated_varint_detected():
    with pytest.raises(ValueError):
        decode_varint(b"\x80\x80", 0)            # continuation bit set on the last byte


# ------------------------------------------------------------------ postings

def test_postings_round_trip():
    plist = [(0, 1), (3, 2), (4, 1), (150, 40), (100_000, 1)]
    assert decode_postings(encode_postings(plist)) == plist


def test_empty_postings():
    assert encode_postings([]) == b""
    assert decode_postings(b"") == []


def test_dense_list_is_two_bytes_per_posting():
    # a common term in every chunk: every gap is 1, every tf small
    plist = [(i, 1 + i % 5) for i in range(1000)]
    assert len(encode_postings(plist)) == 2 * len(plist)


def test_delta_beats_plain_ids():
    plist = [(i, 1) for i in range(100_000, 101_000)]
    with_delta = len(encode_postings(plist))
    without_delta = sum(len(varint(cid)) + len(varint(tf)) for cid, tf in plist)
    assert with_delta == 2002                    # first id has no predecessor: 3 bytes, then 1-byte gaps
    assert without_delta == 4000                 # ids ~100,000 need 3 bytes each


def test_random_lists_round_trip():
    rng = random.Random(42)
    for _ in range(200):
        ids = sorted(rng.sample(range(1_000_000), rng.randint(1, 300)))
        plist = [(cid, rng.randint(1, 5000)) for cid in ids]
        assert decode_postings(encode_postings(plist)) == plist


@pytest.mark.parametrize("bad", [
    [(5, 1), (5, 1)],                            # duplicate id -> gap 0
    [(5, 1), (3, 1)],                            # unsorted -> negative gap
    [(1, 0)],                                    # tf 0 can't be stored as tf-1
])
def test_bad_postings_rejected(bad):
    with pytest.raises(ValueError):
        encode_postings(bad)


# ------------------------------------------------------------------ mapping + index

def test_compressed_mapping_behaves_like_dict():
    plain = {"appl": [(0, 1), (2, 3)], "banana": [(1, 1)], "cherri": [(0, 2), (1, 1), (2, 1)]}
    comp = CompressedPostings.from_dict(plain)
    assert comp == plain
    assert len(comp) == 3 and "appl" in comp and "zzz" not in comp
    assert comp.get("zzz") is None
    assert comp.df("cherri") == 3 and comp.df("zzz") == 0
    assert comp["banana"] == [(1, 1)]


def test_compression_is_deterministic():
    plain = {"b": [(0, 1)], "a": [(1, 2)]}
    assert CompressedPostings.from_dict(plain).blob == CompressedPostings.from_dict(dict(reversed(plain.items()))).blob


def test_search_identical_after_save_and_load(tmp_path):
    rng = random.Random(7)
    vocab = [f"word{i}" for i in range(300)]
    chunks = [Chunk(f"d{i}:0", f"d{i}", 0, f"u{i}", f"Title {i % 7}",
                    " ".join(rng.choices(vocab, k=rng.randint(20, 200)))) for i in range(400)]
    built = InvertedIndex.build(chunks)
    built.save(str(tmp_path))
    loaded = InvertedIndex.load(str(tmp_path))
    for q in ["word1 word2", "word299", "title 3 word42 word7", "nothing here"]:
        a = [(h.chunk_id, h.score) for h in BM25(built).search(q, k=20)]
        b = [(h.chunk_id, h.score) for h in BM25(loaded).search(q, k=20)]
        assert a == b, q
    assert all(built.df(t) == loaded.df(t) for t in built.postings)
