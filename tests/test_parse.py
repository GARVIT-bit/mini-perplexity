"""Parser tests: boilerplate removal, short-page filter, duplicate detection."""
from __future__ import annotations

from collections import Counter

from indexer.parse import content_hash, link_density, parse_html, parse_pages

ARTICLE = " ".join(
    f"Paragraph {i} explains how the inverted index maps each term to the documents that contain it."
    for i in range(8)
)


def page(body: str = ARTICLE, title: str = "Inverted Indexes Explained") -> str:
    return f"""<html><head><title>{title}</title></head><body>
      <nav><a href="/">Home</a> <a href="/about">About</a> <a href="/contact">Contact us today</a></nav>
      <article><h1>{title}</h1>
        {''.join(f'<p>{s}.</p>' for s in body.split('. ') if s)}
      </article>
      <footer>Copyright 2026 Example Corp. Privacy policy. Cookie settings.</footer>
    </body></html>"""


def rec(url: str, html: str, final_url: str | None = None) -> dict:
    return {"url": url, "final_url": final_url or url, "html": html}


def test_keeps_article_drops_boilerplate():
    doc = parse_html(page(), "https://ex.com/a")
    assert doc is not None
    assert "inverted index maps each term" in doc.text
    assert "Contact us today" not in doc.text
    assert "Cookie settings" not in doc.text


def test_title_extracted():
    doc = parse_html(page(), "https://ex.com/a")
    assert doc.title == "Inverted Indexes Explained"


def test_empty_and_garbage_html_return_none():
    assert parse_html("", "https://ex.com/a") is None
    assert parse_html("   ", "https://ex.com/a") is None
    assert parse_html("<<<not html at all", "https://ex.com/a") is None


def test_doc_id_stable_and_url_based():
    a = parse_html(page(), "https://ex.com/a")
    b = parse_html(page(), "https://ex.com/a")
    c = parse_html(page(), "https://ex.com/b")
    assert a.doc_id == b.doc_id != c.doc_id


def test_short_pages_dropped():
    stats = Counter()
    docs = list(parse_pages([rec("https://ex.com/s", page("Too short to matter. Really."))], stats=stats))
    assert docs == []
    assert stats["too_short"] + stats["no_text"] == 1


def test_duplicate_content_at_two_urls_kept_once():
    stats = Counter()
    docs = list(parse_pages([rec("https://ex.com/a", page()),
                             rec("https://ex.com/a?print=1", page())], stats=stats))
    assert [d.url for d in docs] == ["https://ex.com/a"]
    assert stats["dup_content"] == 1


def test_redirect_to_same_final_url_kept_once():
    stats = Counter()
    docs = list(parse_pages([rec("https://ex.com/old", page(), final_url="https://ex.com/new"),
                             rec("https://ex.com/new", page())], stats=stats))
    assert len(docs) == 1 and docs[0].url == "https://ex.com/new"
    assert stats["dup_url"] == 1


def test_hash_ignores_case_and_whitespace():
    assert content_hash("Hello   World\n") == content_hash("hello world")
    assert content_hash("hello world") != content_hash("hello worlds")


def test_stats_add_up():
    stats = Counter()
    pages = [rec("https://ex.com/a", page()), rec("https://ex.com/b", page()),
             rec("https://ex.com/c", ""), rec("https://ex.com/a", page())]
    list(parse_pages(pages, stats=stats))
    dropped = (stats["dup_url"] + stats["no_text"] + stats["too_short"] + stats["dup_content"]
               + stats["link_list"])
    assert stats["pages"] == 4 and stats["kept"] + dropped == 4


# ------------------------------------------------------ link-list pages

def link_list_page() -> str:
    # a section index: a heading, one sentence, then nothing but links
    links = "".join(f'<li><a href="/m{i}">module{i} tools for working with data format {i}</a></li>'
                    for i in range(40))
    return f"<html><head><title>Library Index</title></head><body><h1>Library Index</h1>" \
           f"<p>These modules are documented in this chapter.</p><ul>{links}</ul></body></html>"


def test_link_density_values():
    assert link_density(link_list_page()) > 0.9
    assert link_density(page()) < 0.05
    assert link_density("") == 0.0


def test_nav_and_footer_links_not_counted():
    # a real article whose site menu is big must not look like a link list
    menu = "".join(f'<a href="/s{i}">Section number {i}</a>' for i in range(60))
    html = page().replace("<nav>", f"<nav>{menu}")
    assert link_density(html) < 0.05


def test_link_list_page_dropped():
    stats = Counter()
    docs = list(parse_pages([rec("https://ex.com/index", link_list_page()),
                             rec("https://ex.com/a", page())], stats=stats))
    assert [d.url for d in docs] == ["https://ex.com/a"]
    assert stats["link_list"] == 1


def test_link_density_threshold_is_adjustable():
    docs = list(parse_pages([rec("https://ex.com/index", link_list_page())], max_link_density=1.0))
    assert len(docs) == 1
