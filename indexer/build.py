"""Build the index once from the raw crawl: parse -> chunk -> index -> save.

Parsing 300 pages with trafilatura takes about a minute, so search must never
redo it; it loads what this script saves.

    python -m indexer.build --pages data/pages --out data/index
    python -m indexer.build --from-chunks data/index     # re-index saved chunks, no re-parse
    python -m indexer.build --from-docs data/index --size 150 --overlap 30
                                                         # re-chunk saved parsed pages, no re-parse
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from collections import Counter
from dataclasses import asdict

from crawler.store import iter_pages

from .chunk import CHUNK_WORDS, OVERLAP_WORDS, chunk_docs
from .index import InvertedIndex
from .parse import ParsedDoc, parse_pages

DOCS_FILE = "docs.jsonl.gz"      # parsed pages, kept so chunk settings can change without re-parsing


def save_docs(docs: list[ParsedDoc], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with gzip.open(os.path.join(out_dir, DOCS_FILE), "wt", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")


def load_docs(in_dir: str) -> list[ParsedDoc]:
    with gzip.open(os.path.join(in_dir, DOCS_FILE), "rt", encoding="utf-8") as fh:
        return [ParsedDoc(**json.loads(line)) for line in fh]


def dir_size(path: str) -> int:
    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path))


def main() -> None:
    p = argparse.ArgumentParser(description="Build the inverted index from crawled pages")
    p.add_argument("--pages", default="data/pages")
    p.add_argument("--out", default="data/index")
    p.add_argument("--size", type=int, default=CHUNK_WORDS)
    p.add_argument("--overlap", type=int, default=OVERLAP_WORDS)
    p.add_argument("--no-title", action="store_true", help="don't index page titles with each chunk")
    p.add_argument("--from-chunks", metavar="INDEX_DIR",
                   help="reuse chunks saved in an existing index instead of parsing pages")
    p.add_argument("--from-docs", metavar="INDEX_DIR",
                   help="reuse parsed pages saved in an existing index, then chunk them again")
    args = p.parse_args()

    stats: Counter = Counter()
    t0 = time.perf_counter()
    if args.from_chunks:
        chunks = InvertedIndex.load_chunks(args.from_chunks)
        stats["reused_chunks"] = len(chunks)
    else:
        if args.from_docs:
            docs = load_docs(args.from_docs)
            stats["reused_docs"] = len(docs)
        else:
            docs = list(parse_pages(iter_pages(args.pages), stats=stats))
        save_docs(docs, args.out)
        chunks = list(chunk_docs(docs, args.size, args.overlap))
    t1 = time.perf_counter()
    index = InvertedIndex.build(chunks, include_title=not args.no_title)
    t2 = time.perf_counter()
    index.save(args.out)

    n_postings = sum(len(pl) for pl in index.postings.values())
    print(f"pages: {dict(stats)}")
    print(f"chunks={index.n_docs}  vocab={len(index.postings)}  postings={n_postings}  "
          f"avg chunk len={index.avg_len:.0f} terms")
    print(f"parse+chunk={t1 - t0:.1f}s  index={t2 - t1:.1f}s  on disk={dir_size(args.out) / 1e6:.2f} MB "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
