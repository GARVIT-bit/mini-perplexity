"""robots.txt handling, cached per host.

Every site can publish /robots.txt saying which paths bots may visit and
how long to wait between requests (Crawl-delay). A polite crawler fetches
it once per host, caches it, and checks every URL against it.
"""
from __future__ import annotations

import asyncio
import time
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

CACHE_TTL_SECONDS = 24 * 3600


class RobotsCache:
    def __init__(self, client: httpx.AsyncClient, user_agent: str):
        self.client = client
        self.user_agent = user_agent
        self._cache: dict[str, tuple[RobotFileParser, float]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def _get_parser(self, url: str) -> RobotFileParser:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        cached = self._cache.get(origin)
        if cached and time.time() - cached[1] < CACHE_TTL_SECONDS:
            return cached[0]

        # One lock per origin so 50 workers don't all fetch robots.txt at once.
        lock = self._locks.setdefault(origin, asyncio.Lock())
        async with lock:
            cached = self._cache.get(origin)
            if cached and time.time() - cached[1] < CACHE_TTL_SECONDS:
                return cached[0]

            parser = RobotFileParser()
            try:
                resp = await self.client.get(f"{origin}/robots.txt", timeout=10)
                if resp.status_code == 200:
                    parser.parse(resp.text.splitlines())
                elif resp.status_code in (401, 403):
                    parser.disallow_all = True     # site forbids us: stay out
                else:
                    parser.allow_all = True        # no robots.txt: everything allowed
            except httpx.HTTPError:
                parser.allow_all = True            # unreachable: the page fetch will fail anyway
            self._cache[origin] = (parser, time.time())
            return parser

    async def allowed(self, url: str) -> bool:
        parser = await self._get_parser(url)
        return parser.can_fetch(self.user_agent, url)

    async def crawl_delay(self, url: str) -> float | None:
        parser = await self._get_parser(url)
        delay = parser.crawl_delay(self.user_agent)
        return float(delay) if delay is not None else None
