"""Crawler tests against a tiny fake website served on localhost.

Run:  python -m pytest -q
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from crawler.crawl import CrawlConfig, Crawler, read_seeds
from crawler.links import extract_links
from crawler.store import iter_pages
from crawler.urls import is_private_host, normalize


# ---------------------------------------------------------------- unit tests

@pytest.mark.parametrize("raw,expected", [
    ("HTTP://Example.COM:80/a#top", "http://example.com/a"),
    ("https://example.com:443/", "https://example.com/"),
    ("https://example.com", "https://example.com/"),
    ("https://example.com/p?b=2&a=1&utm_source=x", "https://example.com/p?a=1&b=2"),
    ("mailto:me@example.com", None),
    ("javascript:void(0)", None),
    ("https://example.com/photo.JPG", None),
    ("", None),
])
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_normalize_relative():
    assert normalize("../b", "https://ex.com/docs/a/page") == "https://ex.com/docs/b"


def test_private_hosts():
    assert is_private_host("127.0.0.1")
    assert is_private_host("169.254.169.254")      # cloud metadata endpoint
    assert is_private_host("localhost")
    assert not is_private_host("example.com")


def test_extract_links_dedup_and_nofollow():
    html = """<a href="/a">A</a><a href="/a#x">A again</a>
              <a href="/n" rel="nofollow">no</a><a href="mailto:x@y.z">mail</a>"""
    assert extract_links(html, "https://ex.com/") == ["https://ex.com/a"]
    assert extract_links('<meta name="robots" content="nofollow"><a href="/a">A</a>', "https://ex.com/") == []


# ------------------------------------------------------- fake website server

SITE = {
    "/robots.txt": ("text/plain", "User-agent: *\nDisallow: /private\n"),
    "/": ("text/html", """<a href="/a">a</a> <a href="/b#frag">b</a> <a href="/a?utm_source=tw">a dup</a>
                          <a href="/private/secret">secret</a> <a href="/img.png">img</a>
                          <a href="mailto:x@y.z">mail</a> <a href="http://out-of-scope.example/">out</a>
                          <a href="/flaky">flaky</a> <a href="/big">big</a> <a href="/missing">404</a>"""),
    "/a": ("text/html", '<a href="/c">c</a> <a href="../b">b</a>'),
    "/b": ("text/html", '<a href="/">home</a>'),
    "/c": ("text/html", '<a href="/a">a</a> <a href="/d">d (depth 3)</a>'),
    "/d": ("text/html", "deep page"),
    "/private/secret": ("text/html", "must never be fetched"),
    "/big": ("text/html", "x" * 3000),
    "/flaky": ("text/html", "ok after one 503"),
}


class FakeSite:
    def __init__(self):
        self.hits: dict[str, list[float]] = defaultdict(list)
        self.flaky_failures = 1
        self.drops: dict[str, int] = {}          # path -> times to drop the connection (-1 = always)
        site = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):        # keep test output quiet
                pass

            def do_GET(self):
                path = self.path.split("?")[0]
                site.hits[path].append(time.time())
                if site.drops.get(path, 0) != 0:
                    # hang up without a response: the client sees a network error, not an HTTP status
                    site.drops[path] -= 1
                    self.close_connection = True
                    return
                if path == "/flaky" and site.flaky_failures > 0:
                    site.flaky_failures -= 1
                    self.send_response(503); self.end_headers(); return
                if path not in SITE:
                    self.send_response(404); self.end_headers(); return
                ctype, body = SITE[path]
                data = body.encode()
                self.send_response(200)
                self.send_header("Content-Type", f"{ctype}; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def site():
    s = FakeSite()
    yield s
    s.close()


def make_cfg(site, tmp_path, **kw):
    defaults = dict(seeds=[site.base + "/"], db_path=str(tmp_path / "crawl.db"),
                    out_dir=str(tmp_path / "pages"), max_pages=100, workers=5,
                    per_host_delay=0.2, max_depth=2, max_bytes=2000, allow_private=True)
    defaults.update(kw)
    return CrawlConfig(**defaults)


# ------------------------------------------------------ integration tests

def test_full_crawl(site, tmp_path):
    stats = asyncio.run(Crawler(make_cfg(site, tmp_path)).run())

    stored = {p["url"].replace(site.base, "") for p in iter_pages(str(tmp_path / "pages"))}
    # depth: / =0, a,b =1, c =2; /d would be depth 3 > max_depth 2, so not crawled
    assert stored == {"/", "/a", "/b", "/c", "/flaky"}
    assert stats["fetched"] == 5

    assert "/private/secret" not in site.hits          # robots.txt respected
    assert "/d" not in site.hits                       # depth limit respected
    assert len(site.hits["/robots.txt"]) == 1          # robots fetched once, then cached
    assert len(site.hits["/a"]) == 1                   # duplicates collapsed by normalisation
    assert len(site.hits["/flaky"]) == 2               # one 503, one successful retry
    assert "/big" in site.hits and "/big" not in stored  # oversized page abandoned


def test_politeness_gap(site, tmp_path):
    asyncio.run(Crawler(make_cfg(site, tmp_path, per_host_delay=0.3)).run())
    times = sorted(t for path, ts in site.hits.items() if path != "/robots.txt" for t in ts)
    gaps = [b - a for a, b in zip(times, times[1:])]
    # flaky's 503 doubles the delay afterwards, so every gap must be at least the base delay
    assert min(gaps) >= 0.28, gaps


def test_resume_does_not_refetch(site, tmp_path):
    asyncio.run(Crawler(make_cfg(site, tmp_path, max_pages=2)).run())    # "crash" after 2 pages
    asyncio.run(Crawler(make_cfg(site, tmp_path)).run())                  # resume
    stored = [p["url"] for p in iter_pages(str(tmp_path / "pages"))]
    assert len(stored) == len(set(stored)) == 5        # every page stored exactly once
    for path in ("/", "/a", "/b", "/c"):
        assert len(site.hits[path]) == 1, path


def test_private_hosts_blocked_by_default(site, tmp_path):
    stats = asyncio.run(Crawler(make_cfg(site, tmp_path, allow_private=False)).run())
    assert stats["fetched"] == 0
    assert not site.hits                               # not a single request reached localhost


# ------------------------------------------------------ focus rules (allow / deny)

def test_read_seeds_with_rules(tmp_path):
    f = tmp_path / "seeds.txt"
    f.write_text("# comment\nhttps://ex.com/docs/index.html\n\nallow: https://ex.com/docs/\n"
                 "deny: /old/\nhttps://other.com/\n", encoding="utf-8")
    assert read_seeds(str(f)) == (["https://ex.com/docs/index.html", "https://other.com/"],
                                  ["https://ex.com/docs/"], ["/old/"])


def test_in_scope_rules(tmp_path):
    cfg = CrawlConfig(seeds=["https://ex.com/docs/"], db_path=str(tmp_path / "c.db"),
                      out_dir=str(tmp_path / "p"), allow_prefixes=["HTTPS://EX.com/docs/"],
                      deny_patterns=[r"/docs/old/"])
    c = Crawler(cfg)
    assert c.in_scope("https://ex.com/docs/asyncio.html")
    assert not c.in_scope("https://ex.com/tags/python.html")       # outside every allow prefix
    assert not c.in_scope("https://ex.com/docs/old/asyncio.html")  # denied
    assert not c.in_scope("https://evil.com/docs/x.html")          # host check still applies
    c.frontier.close()


def test_allow_prefix_limits_crawl_but_seed_still_fetched(site, tmp_path):
    asyncio.run(Crawler(make_cfg(site, tmp_path, allow_prefixes=[site.base + "/a"])).run())
    stored = {p["url"].replace(site.base, "") for p in iter_pages(str(tmp_path / "pages"))}
    assert stored == {"/", "/a"}            # "/" is the seed (exempt); /a's links /b, /c are outside
    assert "/b" not in site.hits and "/c" not in site.hits


def test_deny_pattern_skips_url(site, tmp_path):
    asyncio.run(Crawler(make_cfg(site, tmp_path, deny_patterns=[r"/b$"])).run())
    assert "/b" not in site.hits
    assert "/a" in site.hits and "/c" in site.hits


# ------------------------------------------------------ per-host cap

def test_host_cap_leaves_urls_pending(site, tmp_path):
    asyncio.run(Crawler(make_cfg(site, tmp_path, max_pages_per_host=2)).run())
    from crawler.frontier import Frontier
    f = Frontier(str(tmp_path / "crawl.db"))
    counts = f.counts()
    capped = f.db.execute("SELECT COUNT(*) FROM frontier WHERE error='host cap'").fetchone()[0]
    f.close()
    assert counts.get("done", 0) + counts.get("failed", 0) == 2
    assert counts.get("pending", 0) > 0           # still there for a later run
    assert capped == 0                            # nothing thrown away as 'skipped: host cap'


def test_resume_with_bigger_cap_finishes_crawl(site, tmp_path):
    asyncio.run(Crawler(make_cfg(site, tmp_path, max_pages_per_host=2)).run())
    asyncio.run(Crawler(make_cfg(site, tmp_path, max_pages_per_host=100)).run())
    stored = [p["url"].replace(site.base, "") for p in iter_pages(str(tmp_path / "pages"))]
    assert sorted(stored) == ["/", "/a", "/b", "/c", "/flaky"]   # same result as an uncapped crawl
    for path in ("/", "/a", "/b", "/c"):
        assert len(site.hits[path]) == 1, path                   # nothing fetched twice


def test_cap_counts_across_resumes(site, tmp_path):
    asyncio.run(Crawler(make_cfg(site, tmp_path, max_pages_per_host=2)).run())
    before = sum(len(v) for k, v in site.hits.items() if k != "/robots.txt")
    stats = asyncio.run(Crawler(make_cfg(site, tmp_path, max_pages_per_host=2)).run())
    after = sum(len(v) for k, v in site.hits.items() if k != "/robots.txt")
    assert stats["fetched"] == 0 and after == before             # same cap: budget already used


# ------------------------------------------------------ network errors

def test_network_error_retried(site, tmp_path):
    site.drops["/a"] = 1
    asyncio.run(Crawler(make_cfg(site, tmp_path)).run())
    stored = {p["url"].replace(site.base, "") for p in iter_pages(str(tmp_path / "pages"))}
    assert "/a" in stored
    assert len(site.hits["/a"]) == 2              # dropped once, then fetched


def test_network_error_gives_up_after_max_retries(site, tmp_path):
    site.drops["/a"] = -1
    asyncio.run(Crawler(make_cfg(site, tmp_path, max_retries=2)).run())
    assert len(site.hits["/a"]) == 3              # 1 try + 2 retries
    from crawler.frontier import Frontier
    f = Frontier(str(tmp_path / "crawl.db"))
    status, http_status = f.db.execute(
        "SELECT status, http_status FROM frontier WHERE url=?", (site.base + "/a",)).fetchone()
    f.close()
    assert status == "failed" and http_status is None


def test_retry_failed_flag_recovers_on_resume(site, tmp_path):
    site.drops["/a"] = -1
    asyncio.run(Crawler(make_cfg(site, tmp_path, max_retries=0)).run())
    site.drops["/a"] = 0                          # network is back
    asyncio.run(Crawler(make_cfg(site, tmp_path, retry_failed=True)).run())
    stored = [p["url"].replace(site.base, "") for p in iter_pages(str(tmp_path / "pages"))]
    assert stored.count("/a") == 1
    assert "/missing" not in site.hits or len(site.hits["/missing"]) == 1   # a 404 is never retried


def test_failures_do_not_use_up_cap_across_resumes(site, tmp_path):
    site.drops["/"] = -1                          # the seed fails: nothing fetched
    asyncio.run(Crawler(make_cfg(site, tmp_path, max_retries=0, max_pages_per_host=1)).run())
    site.drops["/"] = 0
    stats = asyncio.run(Crawler(make_cfg(site, tmp_path, max_pages_per_host=1, retry_failed=True)).run())
    assert stats["fetched"] == 1                  # the failed attempt didn't count against the cap of 1
