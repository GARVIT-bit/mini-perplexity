"""Raw page store: every fetched page goes into gzip-compressed JSONL files.

Keeping raw HTML means we can change the parser or indexer later and
re-index everything without crawling again. Files rotate every N pages so
no single file grows huge.
"""
from __future__ import annotations

import gzip
import json
import os
import time
from typing import Iterator


class PageStore:
    def __init__(self, out_dir: str, pages_per_file: int = 5000):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.pages_per_file = pages_per_file
        existing = [f for f in os.listdir(out_dir) if f.startswith("pages-") and f.endswith(".jsonl.gz")]
        self.file_index = len(existing)      # resume: start a new file after the old ones
        self.count_in_file = 0
        self._fh = None

    def _open_next(self) -> None:
        if self._fh:
            self._fh.close()
        path = os.path.join(self.out_dir, f"pages-{self.file_index:05d}.jsonl.gz")
        self._fh = gzip.open(path, "at", encoding="utf-8")
        self.file_index += 1
        self.count_in_file = 0

    def write(self, url: str, final_url: str, status: int, content_type: str, html: str) -> None:
        if self._fh is None or self.count_in_file >= self.pages_per_file:
            self._open_next()
        record = {
            "url": url,
            "final_url": final_url,
            "status": status,
            "content_type": content_type,
            "fetched_at": time.time(),
            "html": html,
        }
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()                     # a crash loses at most the current line
        self.count_in_file += 1

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def iter_pages(out_dir: str) -> Iterator[dict]:
    """Read every stored page back (used by the indexer in Phase 2)."""
    for name in sorted(os.listdir(out_dir)):
        if name.startswith("pages-") and name.endswith(".jsonl.gz"):
            with gzip.open(os.path.join(out_dir, name), "rt", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue             # a half-written last line after a crash
