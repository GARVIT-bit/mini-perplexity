"""Postings compression: delta encoding + variable-byte (varint) integers.

A postings list is (chunk_id, tf) pairs sorted by chunk_id, e.g.
    [(3, 1), (7, 2), (8, 1), (150, 4)]
Stored as plain 4-byte ints that's 8 bytes per posting. Two observations
shrink it:

1. Delta (gap) encoding. Ids are sorted, so store the gap to the previous
   id instead of the id:   3, 7, 8, 150   ->   3, 4, 1, 142
   Gaps are small for common terms (they appear in many nearby chunks), and
   common terms are exactly the long lists that dominate index size.

2. Varint. A small number shouldn't cost 4 bytes. Each byte holds 7 bits of
   the number; the top bit says "more bytes follow":
       1    -> 00000001                       (1 byte)
       142  -> 10001110 00000001              (2 bytes: low 7 bits first)
   Numbers < 128 take 1 byte, < 16,384 take 2. tf is almost always < 128.

So a typical posting drops from 8 bytes to 2. Decoding is a few shifts per
byte, cheap enough to do per query, so the index stays compressed in memory
and only the lists a query touches are ever decoded.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping


def encode_varint(n: int, out: bytearray) -> None:
    if n < 0:
        raise ValueError("varint encodes non-negative integers only")
    while n >= 0x80:
        out.append((n & 0x7F) | 0x80)        # low 7 bits, continuation bit set
        n >>= 7
    out.append(n)                            # last byte: continuation bit clear


def decode_varint(buf: bytes, i: int, end: int | None = None) -> tuple[int, int]:
    """Read one varint starting at buf[i]. Returns (value, index after it)."""
    end = len(buf) if end is None else end
    n = shift = 0
    while True:
        if i >= end:
            raise ValueError("truncated varint")
        b = buf[i]
        i += 1
        n |= (b & 0x7F) << shift
        if b < 0x80:
            return n, i
        shift += 7


def encode_postings(postings: list[tuple[int, int]]) -> bytes:
    out = bytearray()
    prev = -1
    for cid, tf in postings:
        if cid <= prev:
            # a gap of 0 or less means unsorted or duplicate ids: decoding would silently corrupt
            raise ValueError(f"postings must have strictly increasing ids ({prev} then {cid})")
        if tf < 1:
            raise ValueError("tf must be >= 1")
        encode_varint(cid - prev - 1, out)   # gaps are >= 1, so store gap-1: the first id 0 costs 0
        encode_varint(tf - 1, out)           # tf is >= 1, same trick keeps tf=1..128 in one byte
        prev = cid
    return bytes(out)


def decode_postings(buf: bytes, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    end = len(buf) if end is None else end
    out: list[tuple[int, int]] = []
    i, prev = start, -1
    while i < end:
        gap, i = decode_varint(buf, i, end)
        tf, i = decode_varint(buf, i, end)
        prev += gap + 1
        out.append((prev, tf + 1))
    return out


class CompressedPostings(Mapping):
    """term -> postings list, backed by one bytes blob; lists are decoded on access.

    Behaves like the plain dict the index builds (get, in, len, ==), so BM25
    doesn't know or care that the data is compressed.
    """

    def __init__(self, blob: bytes, lexicon: dict[str, tuple[int, int, int]]):
        self.blob = blob
        self.lexicon = lexicon               # term -> (byte offset, byte length, df)

    @classmethod
    def from_dict(cls, postings: Mapping[str, list[tuple[int, int]]]) -> "CompressedPostings":
        blob = bytearray()
        lexicon: dict[str, tuple[int, int, int]] = {}
        for term in sorted(postings):        # sorted: same input always gives the same bytes
            data = encode_postings(postings[term])
            lexicon[term] = (len(blob), len(data), len(postings[term]))
            blob.extend(data)
        return cls(bytes(blob), lexicon)

    def __getitem__(self, term: str) -> list[tuple[int, int]]:
        off, length, _ = self.lexicon[term]
        return decode_postings(self.blob, off, off + length)

    def __iter__(self) -> Iterator[str]:
        return iter(self.lexicon)

    def __len__(self) -> int:
        return len(self.lexicon)

    def df(self, term: str) -> int:
        """Document frequency without decoding the list."""
        entry = self.lexicon.get(term)
        return entry[2] if entry else 0
