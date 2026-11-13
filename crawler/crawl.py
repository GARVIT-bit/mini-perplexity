"""Polite, resumable, async web crawler.

Flow for one URL (one worker does this in a loop):
    frontier.pop()  ->  safety + robots.txt checks  ->  fetch (size-capped)
    ->  store raw HTML  ->  extract links  ->  frontier.push(new links)

Run:
    python -m crawler.crawl --seeds seeds.txt --max-pages 2000 --workers 50
Stop any time with Ctrl+C; run the same command again to resume.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import re
import time
from dataclasses import dataclass, field

import httpx

from .frontier import Frontier, Job
from .links import extract_links
from .robots import RobotsCache
from .store import PageStore
from .urls import host_of, is_private_host, normalize

log = logging.getLogger("crawler")


@dataclass
class CrawlConfig:
    seeds: list[str]
    db_path: str = "data/crawl.db"
    out_dir: str = "data/pages"
    max_pages: int = 1000
    workers: int = 50
    per_host_delay: float = 1.0                   # seconds between requests to the same host
    max_depth: int = 3                            # link hops from a seed
    max_pages_per_host: int | None = None         # stops one huge site eating the whole crawl
    scope: str = "seed-domains"                   # "seed-domains" or "any"
    max_bytes: int = 5_000_000                    # skip pages larger than ~5 MB
    timeout: float = 15.0
    max_retries: int = 2                          # for 429 / 5xx and network errors
    retry_failed: bool = False                    # on start, requeue URLs that failed transiently
    user_agent: str = "MiniPerplexityBot/0.1 (+https://github.com/GARVIT-bit; student project)"
    allow_private: bool = False                   # tests only: lets us crawl localhost
    allowed_hosts: set[str] = field(default_factory=set)
    # Focus rules for discovered links (seeds themselves are always fetched, so a
    # hub page like /library/index.html can start the crawl even if it's outside them).
    allow_prefixes: list[str] = field(default_factory=list)   # URL must start with one, if any given
    deny_patterns: list[str] = field(default_factory=list)    # regexes; a match drops the URL


class Crawler:
    def __init__(self, cfg: CrawlConfig):
        self.cfg = cfg
        self.frontier = Frontier(cfg.db_path, cfg.per_host_delay, cfg.max_pages_per_host)
        if cfg.retry_failed:
            n = self.frontier.requeue_transient_failures()
            log.info("requeued %d URLs that failed with network / server errors", n)
        self.store = PageStore(cfg.out_dir)
        self.fetched = 0
        self.inflight = 0
        self.retries: dict[str, int] = {}
        self.started = time.time()
        if cfg.scope == "seed-domains" and not cfg.allowed_hosts:
            cfg.allowed_hosts = {host_of(s) for s in cfg.seeds}
        # normalize so "HTTPS://Docs.python.org/3/library/" matches normalized link URLs
        self.allow_prefixes = [normalize(p) or p for p in cfg.allow_prefixes]
        self.deny = [re.compile(d) for d in cfg.deny_patterns]

    def in_scope(self, url: str) -> bool:
        if self.cfg.scope != "any":
            host = host_of(url)
            # allow subdomains of seed hosts too (docs.python.org when the seed is python.org)
            if not any(host == h or host.endswith("." + h) for h in self.cfg.allowed_hosts):
                return False
        # Without these, a crawl that starts at a home page spends its budget on
        # nav, tag-listing and old-version pages before reaching real content.
        if self.allow_prefixes and not any(url.startswith(p) for p in self.allow_prefixes):
            return False
        return not any(d.search(url) for d in self.deny)

    async def run(self) -> dict:
        seeds = [u for u in (normalize(s) for s in self.cfg.seeds) if u]
        new = self.frontier.push(seeds, depth=0)
        log.info("seeded %d new URLs (resuming %s)", new, self.frontier.counts())

        limits = httpx.Limits(max_connections=self.cfg.workers, max_keepalive_connections=self.cfg.workers)
        headers = {"User-Agent": self.cfg.user_agent, "Accept": "text/html,application/xhtml+xml"}
        async with httpx.AsyncClient(headers=headers, limits=limits, follow_redirects=True,
                                     max_redirects=5, timeout=self.cfg.timeout) as client:
            self.client = client
            self.robots = RobotsCache(client, self.cfg.user_agent)
            reporter = asyncio.create_task(self._report())
            try:
                await asyncio.gather(*(self._worker(i) for i in range(self.cfg.workers)))
            finally:
                reporter.cancel()
                self.store.close()
        stats = {"fetched": self.fetched, "seconds": round(time.time() - self.started, 1),
                 **self.frontier.counts()}
        self.frontier.close()
        log.info("done: %s", stats)
        return stats

    async def _worker(self, wid: int) -> None:
        while self.fetched < self.cfg.max_pages:
            job, wait = self.frontier.pop()
            if job is None:
                if wait < 0 and self.inflight == 0:
                    return                          # nothing pending, nothing in flight: crawl finished
                await asyncio.sleep(max(wait, 0.05))
                continue
            self.inflight += 1
            try:
                await self._process(job)
            except Exception as e:                  # never let one bad page kill a worker
                log.warning("error on %s: %r", job.url, e)
                self.frontier.mark(job.url, "failed", error=repr(e)[:200])
            finally:
                self.inflight -= 1

    async def _process(self, job: Job) -> None:
        if not self.cfg.allow_private and is_private_host(job.host):
            self.frontier.mark(job.url, "skipped", error="private host")
            return
        if not await self.robots.allowed(job.url):
            self.frontier.mark(job.url, "skipped", error="robots.txt")
            return
        delay = await self.robots.crawl_delay(job.url)
        if delay:
            self.frontier.set_host_delay(job.host, delay)

        try:
            status, final_url, ctype, html = await self._fetch(job.url)
        except httpx.TransportError as e:
            # DNS failure, timeout, dropped connection: almost always gone a minute
            # later. The first big crawl lost 88 pages by treating these as final.
            self._retry_later(job, None, repr(e)[:200])
            return

        if status in (429, 500, 502, 503, 504):
            self._retry_later(job, status, f"http {status}")
            return
        if status != 200 or html is None:
            # say *why*: an HTTP error and a non-HTML page used to look the same in the DB
            error = f"http {status}" if status != 200 else f"skipped {ctype}"
            self.frontier.mark(job.url, "failed", status, error=error)
            return

        self.store.write(job.url, final_url, status, ctype, html)
        self.fetched += 1

        if job.depth < self.cfg.max_depth:
            links = [u for u in extract_links(html, final_url) if self.in_scope(u)]
            self.frontier.push(links, depth=job.depth + 1)
        if final_url != job.url:
            self.frontier.push([final_url], depth=job.depth)     # remember the redirect target as seen
            self.frontier.mark(final_url, "done", status)
        self.frontier.mark(job.url, "done", status)

    def _retry_later(self, job: Job, status: int | None, error: str) -> None:
        tries = self.retries.get(job.url, 0) + 1
        self.retries[job.url] = tries
        # back off this host: double its delay
        self.frontier.set_host_delay(job.host, self.frontier.host_delay.get(job.host, self.cfg.per_host_delay) * 2)
        if tries <= self.cfg.max_retries:
            log.info("retry %d/%d later: %s (%s)", tries, self.cfg.max_retries, job.url, error)
            self.frontier.requeue(job.url, status, error)
        else:
            self.frontier.mark(job.url, "failed", status, error=error)

    async def _fetch(self, url: str) -> tuple[int, str, str, str | None]:
        """Stream the response so a huge file is abandoned early instead of downloaded."""
        async with self.client.stream("GET", url) as resp:
            ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            final_url = normalize(str(resp.url)) or url
            if resp.status_code != 200 or ctype not in ("text/html", "application/xhtml+xml"):
                return resp.status_code, final_url, ctype, None
            body = bytearray()
            async for chunk in resp.aiter_bytes():
                body.extend(chunk)
                if len(body) > self.cfg.max_bytes:
                    return 413, final_url, ctype, None
            encoding = resp.encoding or "utf-8"
            return resp.status_code, final_url, ctype, body.decode(encoding, errors="replace")

    async def _report(self) -> None:
        while True:
            await asyncio.sleep(10)
            elapsed = time.time() - self.started
            log.info("fetched=%d  rate=%.1f pages/s  inflight=%d  frontier=%s",
                     self.fetched, self.fetched / max(elapsed, 1e-9), self.inflight, self.frontier.counts())


def read_seeds(path: str) -> tuple[list[str], list[str], list[str]]:
    """Seeds file: one URL per line, plus optional `allow: <prefix>` and `deny: <regex>` lines.

    Keeping the rules in the file (not CLI flags) means one file fully describes a crawl.
    """
    seeds: list[str] = []
    allow: list[str] = []
    deny: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key, _, rest = line.partition(":")
            if key in ("allow", "deny") and rest.strip():
                (allow if key == "allow" else deny).append(rest.strip())
            else:
                seeds.append(line)          # "https://..." also contains ":", but its key is "https"
    return seeds, allow, deny


def main() -> None:
    p = argparse.ArgumentParser(description="Polite resumable crawler")
    p.add_argument("--seeds", required=True, help="file with one seed URL per line")
    p.add_argument("--db", default="data/crawl.db")
    p.add_argument("--out", default="data/pages")
    p.add_argument("--max-pages", type=int, default=1000)
    p.add_argument("--workers", type=int, default=50)
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--max-per-host", type=int, default=None)
    p.add_argument("--scope", choices=["seed-domains", "any"], default="seed-domains")
    p.add_argument("--retry-failed", action="store_true",
                   help="requeue URLs that failed with network/server errors in earlier runs")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seeds, allow, deny = read_seeds(args.seeds)
    cfg = CrawlConfig(seeds=seeds, db_path=args.db, out_dir=args.out, max_pages=args.max_pages,
                      workers=args.workers, per_host_delay=args.delay, max_depth=args.max_depth,
                      max_pages_per_host=args.max_per_host, scope=args.scope,
                      allow_prefixes=allow, deny_patterns=deny, retry_failed=args.retry_failed)
    try:
        asyncio.run(Crawler(cfg).run())
    except KeyboardInterrupt:
        log.info("stopped; run the same command again to resume")


if __name__ == "__main__":
    main()
