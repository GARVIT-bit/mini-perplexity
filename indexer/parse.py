"""Parser: raw HTML from the crawl -> clean main-content text.

Why this step exists: a typical page is ~80% boilerplate (nav menus, footers,
cookie banners, "related posts"). If we index that, every page matches
"Home About Contact Privacy" and BM25 scores get polluted. trafilatura finds
the main content block using text density and HTML structure heuristics.

This module also drops pages we don't want in the index:
  - too short to answer anything (login walls, empty category pages)
  - exact duplicates (the same article reachable at two URLs)
  - link lists (section indexes, tag pages): mostly links, little prose.
    They contain every title on the site, so BM25 ranks them first for
    almost any query, yet they can't answer anything.

Run on a crawl to see what survives:
    python -m indexer.parse --pages data/pages
"""
from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Iterator

import lxml.html
import trafilatura

MIN_WORDS = 50          # below this a page can't support a cited answer
# Share of a page's visible words that sit inside <a> tags. Measured on the
# focused 300-page crawl: median page 0.05, real docs pages up to 0.29,
# index/category pages 0.37-0.96. 0.35 sits in that gap.
MAX_LINK_DENSITY = 0.35


@dataclass
class ParsedDoc:
    doc_id: str          # stable id derived from the URL, so re-parsing gives the same ids
    url: str
    title: str
    text: str            # paragraphs separated by "\n"; the chunker splits on these later
    content_hash: str
    link_density: float = 0.0

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def content_hash(text: str) -> str:
    """Hash that ignores case and whitespace differences.

    Two copies of one article often differ only in trailing spaces or line
    wrapping; those should still count as duplicates.
    """
    canon = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()


def doc_id_for(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def link_density(html: str) -> float:
    """Fraction of visible body words that are link text (0.0 if unparseable).

    Measured on the raw HTML, not trafilatura's output: trafilatura already
    strips most links, so its text would make a link list look like prose.
    """
    try:
        tree = lxml.html.fromstring(html)
    except Exception:
        return 0.0
    # site chrome is links everywhere; count only the page's own content
    for el in tree.xpath("//script|//style|//noscript|//nav|//header|//footer"):
        el.drop_tree()
    body = tree.find("body")
    body = body if body is not None else tree
    # count per text node: text_content() glues "</li><li>" neighbours into one word
    total = sum(len(t.split()) for t in body.itertext())
    linked = sum(len(t.split()) for a in body.iter("a") for t in a.itertext())
    return linked / total if total else 0.0


def parse_html(html: str, url: str) -> ParsedDoc | None:
    """Extract title + main text from one page. None if nothing usable."""
    if not html or not html.strip():
        return None
    try:
        doc = trafilatura.bare_extraction(
            html,
            url=url,
            with_metadata=True,          # gives us the title
            include_comments=False,      # blog comments are noise for answering questions
            include_tables=True,         # docs pages often keep real content in tables
        )
    except Exception:                    # malformed HTML must not stop a whole indexing run
        return None
    if doc is None or not doc.text:
        return None
    text = doc.text.strip()
    return ParsedDoc(
        doc_id=doc_id_for(url),
        url=url,
        title=(doc.title or "").strip(),
        text=text,
        content_hash=content_hash(text),
        link_density=link_density(html),
    )


def parse_pages(pages: Iterable[dict], min_words: int = MIN_WORDS,
                stats: Counter | None = None,
                max_link_density: float = MAX_LINK_DENSITY) -> Iterator[ParsedDoc]:
    """Turn crawl records (from crawler.store.iter_pages) into unique, usable docs.

    `stats` (optional) counts why pages were kept or dropped, so we can report
    real numbers instead of guessing.
    """
    stats = stats if stats is not None else Counter()
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()
    for rec in pages:
        stats["pages"] += 1
        # final_url is where we actually landed after redirects: two seeds that
        # redirect to the same page should be one document
        url = rec.get("final_url") or rec.get("url")
        if url in seen_urls:
            stats["dup_url"] += 1
            continue
        seen_urls.add(url)

        doc = parse_html(rec.get("html", ""), url)
        if doc is None:
            stats["no_text"] += 1
            continue
        if doc.n_words < min_words:
            stats["too_short"] += 1
            continue
        if doc.link_density > max_link_density:
            stats["link_list"] += 1
            continue
        if doc.content_hash in seen_hashes:
            stats["dup_content"] += 1
            continue
        seen_hashes.add(doc.content_hash)
        stats["kept"] += 1
        yield doc


def main() -> None:
    from crawler.store import iter_pages

    p = argparse.ArgumentParser(description="Parse crawled pages and report what survives")
    p.add_argument("--pages", default="data/pages")
    p.add_argument("--min-words", type=int, default=MIN_WORDS)
    p.add_argument("--max-link-density", type=float, default=MAX_LINK_DENSITY)
    p.add_argument("--show", type=int, default=3, help="print the first N docs")
    args = p.parse_args()

    stats: Counter = Counter()
    words = 0
    for i, doc in enumerate(parse_pages(iter_pages(args.pages), args.min_words, stats, args.max_link_density)):
        words += doc.n_words
        if i < args.show:
            print(f"--- {doc.title!r}  {doc.url}  ({doc.n_words} words)")
            print(doc.text[:300].replace("\n", " / "), "...\n")
    print(dict(stats))
    if stats["kept"]:
        print(f"avg words per kept doc: {words / stats['kept']:.0f}")


if __name__ == "__main__":
    main()
