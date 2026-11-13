"""The URL frontier: which URLs to crawl next, and which we've already seen.

Stored in SQLite so a crash or Ctrl+C never loses progress: rerunning the
crawler resumes where it stopped. Politeness lives here too: pop() only
hands out a URL whose host is allowed another request right now.

SQLite calls are synchronous; each one takes microseconds, so calling them
from the async workers is fine at this scale.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

from .urls import host_of

SCHEMA = """
CREATE TABLE IF NOT EXISTS frontier (
    url       TEXT PRIMARY KEY,      -- PRIMARY KEY doubles as the 'seen' set
    host      TEXT NOT NULL,
    depth     INTEGER NOT NULL,
    status    TEXT NOT NULL DEFAULT 'pending',   -- pending | in_progress | done | failed | skipped
    added_at  REAL NOT NULL,
    http_status INTEGER,
    error     TEXT
);
CREATE INDEX IF NOT EXISTS idx_frontier_status ON frontier(status, added_at);
"""


@dataclass
class Job:
    url: str
    host: str
    depth: int


class Frontier:
    def __init__(self, db_path: str, per_host_delay: float = 1.0, max_pages_per_host: int | None = None):
        self.db = sqlite3.connect(db_path)
        self.db.executescript(SCHEMA)
        self.per_host_delay = per_host_delay
        self.max_pages_per_host = max_pages_per_host
        self.next_allowed: dict[str, float] = {}     # host -> earliest time of next request
        self.host_delay: dict[str, float] = {}       # host -> delay (from robots Crawl-delay)
        # Resume: anything left in_progress by a crash goes back to pending.
        self.db.execute("UPDATE frontier SET status='pending' WHERE status='in_progress'")
        # Older versions marked capped URLs 'skipped' forever; give them back so a
        # resumed crawl with a bigger --max-per-host can still reach them.
        self.db.execute("UPDATE frontier SET status='pending', error=NULL "
                        "WHERE status='skipped' AND error='host cap'")
        self.db.commit()
        # host -> pages so far. Starts from pages actually fetched in earlier runs,
        # so the cap means "at most N pages from this site in the crawl", not
        # "N more per restart", and failed requests don't use up the budget.
        # Within a run every hand-out counts, so parallel workers can't overshoot.
        self.host_count: dict[str, int] = dict(self.db.execute(
            "SELECT host, COUNT(*) FROM frontier WHERE status = 'done' GROUP BY host"
        ).fetchall())

    def push(self, urls: list[str], depth: int) -> int:
        """Add URLs not seen before. Returns how many were new."""
        now = time.time()
        before = self.db.total_changes
        self.db.executemany(
            "INSERT OR IGNORE INTO frontier(url, host, depth, added_at) VALUES (?, ?, ?, ?)",
            [(u, host_of(u), depth, now) for u in urls],
        )
        self.db.commit()
        return self.db.total_changes - before

    def set_host_delay(self, host: str, delay: float) -> None:
        self.host_delay[host] = max(delay, self.per_host_delay)

    def pop(self) -> tuple[Job | None, float]:
        """Return (job, 0) if a URL is ready, else (None, seconds_to_wait).

        Scans the oldest pending URLs (breadth-first order) and picks the first
        whose host is ready, so one slow host never blocks the others.

        Hosts at their page cap are filtered out in SQL. Their URLs stay
        'pending' (a later run with a bigger cap can fetch them), and they can't
        fill the 500-row scan window and starve hosts that still have budget.
        """
        now = time.time()
        capped = [h for h, n in self.host_count.items()
                  if self.max_pages_per_host and n >= self.max_pages_per_host]
        rows = self.db.execute(
            "SELECT url, host, depth FROM frontier WHERE status='pending' "
            f"AND host NOT IN ({','.join('?' * len(capped))}) "
            "ORDER BY added_at LIMIT 500",
            capped,
        ).fetchall()
        if not rows:
            return None, -1.0                         # -1 means: nothing crawlable left

        soonest = float("inf")
        for url, host, depth in rows:
            ready_at = self.next_allowed.get(host, 0.0)
            if ready_at <= now:
                delay = self.host_delay.get(host, self.per_host_delay)
                self.next_allowed[host] = now + delay
                self.host_count[host] = self.host_count.get(host, 0) + 1
                self.db.execute("UPDATE frontier SET status='in_progress' WHERE url=?", (url,))
                self.db.commit()
                return Job(url, host, depth), 0.0
            soonest = min(soonest, ready_at - now)
        self.db.commit()
        return None, (soonest if soonest != float("inf") else 0.05)

    def mark(self, url: str, status: str, http_status: int | None = None, error: str | None = None) -> None:
        self.db.execute(
            "UPDATE frontier SET status=?, http_status=?, error=? WHERE url=?",
            (status, http_status, error, url),
        )
        self.db.commit()

    def requeue(self, url: str, http_status: int | None, error: str) -> None:
        """Back to pending, at the END of the queue: retrying instantly would
        likely hit the same outage, while the back of the queue is minutes away."""
        self.db.execute(
            "UPDATE frontier SET status='pending', added_at=?, http_status=?, error=? WHERE url=?",
            (time.time(), http_status, error, url),
        )
        self.db.commit()

    def requeue_transient_failures(self) -> int:
        """Give failures that might succeed now another chance: network errors
        (no HTTP status) and 429/5xx. 404s and non-HTML pages stay failed."""
        cur = self.db.execute(
            "UPDATE frontier SET status='pending', error=NULL WHERE status='failed' "
            "AND (http_status IS NULL OR http_status = 429 OR http_status >= 500)"
        )
        self.db.commit()
        return cur.rowcount

    def counts(self) -> dict[str, int]:
        return dict(self.db.execute("SELECT status, COUNT(*) FROM frontier GROUP BY status").fetchall())

    def in_progress(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM frontier WHERE status='in_progress'").fetchone()[0]

    def close(self) -> None:
        self.db.close()
