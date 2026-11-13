"""Extract links from HTML using only the standard library parser."""
from __future__ import annotations

from html.parser import HTMLParser

from .urls import normalize


class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []
        self.base: str | None = None
        self.nofollow_page = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "base" and a.get("href"):
            self.base = a["href"]                       # <base href> changes how relative links resolve
        elif tag == "meta" and (a.get("name") or "").lower() == "robots":
            if "nofollow" in (a.get("content") or "").lower():
                self.nofollow_page = True               # page asks bots not to follow its links
        elif tag == "a" and a.get("href"):
            if "nofollow" not in (a.get("rel") or "").lower():
                self.hrefs.append(a["href"])


def extract_links(html: str, page_url: str) -> list[str]:
    parser = _LinkParser()
    try:
        parser.feed(html)
    except Exception:
        return []                                       # badly broken HTML: skip its links
    if parser.nofollow_page:
        return []
    base = normalize(parser.base, page_url) if parser.base else page_url
    seen, out = set(), []
    for href in parser.hrefs:
        url = normalize(href, base)
        if url and url not in seen:
            seen.add(url)
            out.append(url)
    return out
