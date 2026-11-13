"""Inverted index: term -> list of (chunk id, term frequency).

A forward index answers "what words are in chunk 7?". Search needs the
reverse: "which chunks contain 'asyncio'?". Scanning every chunk per query
is O(corpus); an inverted index jumps straight to the few chunks that matter,
so query cost depends on how common the query terms are, not corpus size.

What we store, and why BM25 needs each piece:
  postings[term]  list of (chunk_id, tf), sorted by chunk_id   -> tf, and df = len(list)
  doc_len[id]     number of terms in chunk id                  -> length normalization
  chunks[id]      the chunk itself (text, url, title)          -> show / cite the result

Chunk ids are dense ints 0..N-1 so they can be list indexes and delta +
varint compressed (see compress.py). Saved indexes keep postings compressed
on disk AND in memory; a list is decoded only when a query touches its term.

The page title is indexed with every chunk: chunk 7 of the asyncio page may
never say "asyncio", but it is still about asyncio.
"""
from __future__ import annotations

import gzip
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from collections.abc import Mapping
from typing import Iterable

from .analyzer import analyze
from .chunk import Chunk
from .compress import CompressedPostings


@dataclass
class InvertedIndex:
    postings: Mapping[str, list[tuple[int, int]]]    # a dict after build(), CompressedPostings after load()
    doc_len: list[int]
    chunks: list[Chunk]
    include_title: bool = True

    @property
    def n_docs(self) -> int:
        return len(self.doc_len)

    @property
    def avg_len(self) -> float:
        return sum(self.doc_len) / self.n_docs if self.n_docs else 0.0

    def df(self, term: str) -> int:
        if isinstance(self.postings, CompressedPostings):
            return self.postings.df(term)            # from the lexicon, no decoding
        return len(self.postings.get(term, ()))

    @staticmethod
    def index_text(chunk: Chunk, include_title: bool) -> str:
        return f"{chunk.title}\n{chunk.text}" if include_title and chunk.title else chunk.text

    @classmethod
    def build(cls, chunks: Iterable[Chunk], include_title: bool = True) -> "InvertedIndex":
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_len: list[int] = []
        store: list[Chunk] = []
        for cid, chunk in enumerate(chunks):
            terms = analyze(cls.index_text(chunk, include_title))
            # ids only ever increase, so appending keeps every postings list sorted for free
            for term, tf in Counter(terms).items():
                postings[term].append((cid, tf))
            doc_len.append(len(terms))
            store.append(chunk)
        return cls(dict(postings), doc_len, store, include_title)

    # ------------------------------------------------------------- persistence
    # postings.bin   every postings list, delta + varint encoded, back to back
    # lexicon.json   term -> [byte offset, byte length, df] into postings.bin

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"n_docs": self.n_docs, "include_title": self.include_title,
                       "doc_len": self.doc_len}, fh)
        comp = (self.postings if isinstance(self.postings, CompressedPostings)
                else CompressedPostings.from_dict(self.postings))
        with open(os.path.join(out_dir, "postings.bin"), "wb") as fh:
            fh.write(comp.blob)
        with gzip.open(os.path.join(out_dir, "lexicon.json.gz"), "wt", encoding="utf-8") as fh:
            json.dump(comp.lexicon, fh, ensure_ascii=False)
        old = os.path.join(out_dir, "postings.json.gz")      # from the pre-compression format
        if os.path.exists(old):
            os.remove(old)
        with gzip.open(os.path.join(out_dir, "chunks.jsonl.gz"), "wt", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, in_dir: str) -> "InvertedIndex":
        with open(os.path.join(in_dir, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        with open(os.path.join(in_dir, "postings.bin"), "rb") as fh:
            blob = fh.read()
        with gzip.open(os.path.join(in_dir, "lexicon.json.gz"), "rt", encoding="utf-8") as fh:
            lexicon = {t: tuple(v) for t, v in json.load(fh).items()}
        postings = CompressedPostings(blob, lexicon)
        chunks = cls.load_chunks(in_dir)
        if len(chunks) != meta["n_docs"]:
            raise ValueError(f"index at {in_dir} is inconsistent: {len(chunks)} chunks vs {meta['n_docs']}")
        return cls(postings, meta["doc_len"], chunks, meta["include_title"])

    @staticmethod
    def load_chunks(in_dir: str) -> list[Chunk]:
        """Just the saved chunks: lets us rebuild postings (new analyzer, new
        format) in seconds without re-parsing every page."""
        with gzip.open(os.path.join(in_dir, "chunks.jsonl.gz"), "rt", encoding="utf-8") as fh:
            return [Chunk(**json.loads(line)) for line in fh]
