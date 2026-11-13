"""Chunker: one parsed document -> overlapping ~120-word passages.

Why chunk at all: the answer to a question usually lives in one or two
paragraphs, not a whole 2,000-word page. Retrieval returns passages, the LLM
cites passages, and the embedding model has a fixed input length; a whole
page would be truncated and its meaning averaged into mush.

Why overlap: if a fact straddles a chunk boundary, the 25 shared words give
each side enough context that at least one chunk still contains it whole.

Why 120 words: the embedding model (MiniLM) reads at most 256 tokens and
ignores the rest, and this corpus averages ~1.5 tokens per word. Measured on
the 1,027-page crawl: at 300 words 92% of chunks were cut off (41% of all
text unseen); at 120 words 8.6% are (3.1% unseen), and the 90th-percentile
chunk is 248 tokens. A chunk should be something every stage can read whole.

How it splits:
  1. Break the text into units: paragraphs (the parser keeps "\n" between them).
  2. A unit too long to fit is split at sentence ends; a sentence still too
     long (code blocks, tables) is cut at a word count.
  3. Pack units greedily into chunks of <= `size` words. Each new chunk starts
     with the last `overlap` words of the previous one.

Units are capped at size - overlap words, so overlap + one unit always fits
and every chunk stays <= size.

Run on a crawl:
    python -m indexer.chunk --pages data/pages
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Iterator

from .parse import ParsedDoc

CHUNK_WORDS = 120
OVERLAP_WORDS = 25

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str        # "<doc_id>:<n>", so a chunk always points back to its page
    doc_id: str
    ord: int             # position within the doc
    url: str
    title: str
    text: str

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def _units(text: str, max_words: int) -> list[str]:
    """Paragraphs, split further only when one is longer than max_words."""
    units: list[str] = []
    for para in text.split("\n"):
        if not para.split():
            continue
        if len(para.split()) <= max_words:
            units.append(para.strip())
            continue
        # long paragraph: pack its sentences instead
        buf: list[str] = []
        for sent in _SENTENCE_END.split(para.strip()):
            words = sent.split()
            if len(words) > max_words:
                # no usable sentence boundary (code, tables): fall back to a hard cut
                if buf:
                    units.append(" ".join(buf))
                    buf = []
                for i in range(0, len(words), max_words):
                    units.append(" ".join(words[i:i + max_words]))
                continue
            if len(buf) + len(words) > max_words and buf:
                units.append(" ".join(buf))
                buf = []
            buf.extend(words)
        if buf:
            units.append(" ".join(buf))
    return units


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size, or chunking never makes progress")
    units = _units(text, size - overlap)

    chunks: list[str] = []
    current: list[str] = []      # units (strings) in the chunk being built
    n_current = 0
    has_new = False              # False while `current` holds only the carried-over overlap
    for unit in units:
        n = len(unit.split())
        if n_current + n > size and has_new:
            chunks.append("\n".join(current))
            tail = " ".join(chunks[-1].split()[-overlap:]) if overlap else ""
            current = [tail] if tail else []
            n_current = len(tail.split())
            has_new = False
        current.append(unit)
        n_current += n
        has_new = True
    if has_new:                  # skip a final chunk that would only repeat the overlap
        chunks.append("\n".join(current))
    return chunks


def chunk_doc(doc: ParsedDoc, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[Chunk]:
    return [Chunk(f"{doc.doc_id}:{i}", doc.doc_id, i, doc.url, doc.title, text)
            for i, text in enumerate(chunk_text(doc.text, size, overlap))]


def chunk_docs(docs: Iterable[ParsedDoc], size: int = CHUNK_WORDS,
               overlap: int = OVERLAP_WORDS) -> Iterator[Chunk]:
    for doc in docs:
        yield from chunk_doc(doc, size, overlap)


def main() -> None:
    from crawler.store import iter_pages
    from .parse import parse_pages

    p = argparse.ArgumentParser(description="Chunk parsed pages and report sizes")
    p.add_argument("--pages", default="data/pages")
    p.add_argument("--size", type=int, default=CHUNK_WORDS)
    p.add_argument("--overlap", type=int, default=OVERLAP_WORDS)
    args = p.parse_args()

    per_doc: Counter = Counter()
    sizes: list[int] = []
    for c in chunk_docs(parse_pages(iter_pages(args.pages)), args.size, args.overlap):
        per_doc[c.doc_id] += 1
        sizes.append(c.n_words)
    if not sizes:
        print("no chunks")
        return
    sizes.sort()
    print(f"docs={len(per_doc)}  chunks={len(sizes)}  chunks/doc={len(sizes) / len(per_doc):.1f}")
    print(f"words per chunk: min={sizes[0]}  p50={sizes[len(sizes) // 2]}  max={sizes[-1]}  "
          f"under 100 words={sum(s < 100 for s in sizes)}")


if __name__ == "__main__":
    main()
