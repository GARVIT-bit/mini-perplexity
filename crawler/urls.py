"""URL normalisation.

Why this matters: the same page is reachable through many URL spellings
(HTTP://Example.com:80/a#top, http://example.com/a?utm_source=x, ...).
If we don't collapse them into one canonical form, we crawl and index
the same page many times.
"""
from __future__ import annotations

import ipaddress
from urllib.parse import urljoin, urlsplit, urlunsplit, parse_qsl, urlencode

# Query parameters that only track clicks; they never change page content.
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "ref", "mc_cid", "mc_eid",
}

DEFAULT_PORTS = {"http": 80, "https": 443}

# File types we never want to download (not HTML pages).
SKIP_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".pdf", ".zip",
    ".gz", ".tar", ".mp3", ".mp4", ".avi", ".mov", ".css", ".js", ".woff",
    ".woff2", ".ttf", ".exe", ".dmg", ".iso", ".xml", ".json",
)


def normalize(url: str, base: str | None = None) -> str | None:
    """Return a canonical absolute URL, or None if it should not be crawled."""
    if not url:
        return None
    url = url.strip()
    if base:
        url = urljoin(base, url)            # resolve relative links like ../a.html

    try:
        parts = urlsplit(url)
    except ValueError:
        return None

    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):     # drops mailto:, javascript:, file:, ftp:
        return None

    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        return None

    port = parts.port if parts.port else None
    if port == DEFAULT_PORTS[scheme]:
        port = None                          # http://a.com:80 == http://a.com
    netloc = host if port is None else f"{host}:{port}"

    path = parts.path or "/"
    if path.lower().endswith(SKIP_EXTENSIONS):
        return None

    query_pairs = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in TRACKING_PARAMS
    ]
    query_pairs.sort()                       # ?b=2&a=1 == ?a=1&b=2
    query = urlencode(query_pairs)

    # The fragment (#section) is dropped: it's the same page.
    return urlunsplit((scheme, netloc, path, query, ""))


def host_of(url: str) -> str:
    """Host used as the politeness key (one rate limit per host)."""
    return (urlsplit(url).hostname or "").lower()


def is_private_host(host: str) -> bool:
    """True for localhost and private/loopback/link-local IP literals.

    A crawler that follows any link could otherwise be tricked into hitting
    internal services (e.g. 127.0.0.1:6379 or cloud metadata at 169.254.169.254).
    Note: this checks IP literals and 'localhost' only; a hostname that
    *resolves* to a private IP needs a DNS-level check (listed as future work).
    """
    if host in ("localhost",) or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
