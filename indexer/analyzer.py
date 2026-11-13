"""Analyzer: text -> list of index terms (tokenize, normalize, drop stopwords, stem).

The one rule that matters: documents and queries go through the SAME
`analyze()` function. A query term can only match a document term if both
came out of identical processing; "Connecting" in a query and "connection"
on a page meet only because both become "connect".

Steps, in order:
  1. Fold: Unicode NFKD, strip accents (café -> cafe), casefold (lowercase).
  2. Tokenize: runs of word characters, allowing internal . - _ ' so
     identifiers like os.path.join and __init__ survive as one token.
  3. Split compounds: an identifier is kept whole (exact match for code
     queries) AND split into parts, so "path join" still finds os.path.join.
     Hyphenated words give the parts plus the joined form (real-time -> real,
     time, realtime). Possessives lose their 's.
  4. Stopwords: drop very common words ("the", "is"). BM25 would give them
     almost no weight anyway; removing them shrinks the index a lot.
  5. Stem plain words with Porter2/Snowball, so run/runs/running all become
     "run". Identifiers and numbers are never stemmed.

Named analyzer.py (not tokenize.py) to avoid shadowing Python's stdlib module.

Run on the crawl:
    python -m indexer.analyzer --pages data/pages
"""
from __future__ import annotations

import argparse
import re
import time
import unicodedata
from collections import Counter
from functools import lru_cache

import snowballstemmer

_STEMMER = snowballstemmer.stemmer("english")

# a word, optionally joined to more words by . - _ or ' (os.path.join, real-time, don't)
_TOKEN = re.compile(r"\w+(?:[.\-']\w+)*")
_PARTS = re.compile(r"[._\-]+")

# Small hand-picked list. Kept short on purpose: words like "not", "no", "how",
# "why" change what a question means, so they stay.
STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before
being below between both but by can could did do does doing down during each few for
from further had has have having he her here hers herself him himself his i if in into
is it its itself just me more most my myself of off on once only or other our ours
ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was we
were what when where which while who whom will with would you your yours yourself
yourselves
""".split())


def fold(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'")         # curly apostrophes
    text = unicodedata.normalize("NFKD", text)                        # é -> e + combining accent
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.casefold()


@lru_cache(maxsize=200_000)
def stem(word: str) -> str:
    # the same few thousand words repeat millions of times; caching makes stemming nearly free
    return _STEMMER.stemWord(word)


def _words_of(token: str) -> tuple[str | None, list[str]]:
    """Return (identifier kept verbatim or None, plain words to normalize)."""
    if "'" in token:
        token = re.sub(r"'s$", "", token).replace("'", "")           # python's -> python, don't -> dont
    if "." in token or "_" in token:
        parts = [p for p in _PARTS.split(token) if p]
        return (token if parts else None), parts                      # "___" alone is junk
    if "-" in token:
        parts = [p for p in token.split("-") if p]
        return None, parts + (["".join(parts)] if len(parts) > 1 else [])
    return None, [token]


def analyze(text: str, *, stem_words: bool = True, remove_stopwords: bool = True) -> list[str]:
    """Text -> terms. Use for both indexing and querying. The flags exist for ablations."""
    terms: list[str] = []
    for m in _TOKEN.finditer(fold(text)):
        ident, words = _words_of(m.group())
        if ident:
            terms.append(ident)
        for w in words:
            if remove_stopwords and w in STOPWORDS:
                continue
            terms.append(stem(w) if stem_words and w.isalpha() else w)
    return terms


def main() -> None:
    from crawler.store import iter_pages
    from .chunk import chunk_docs
    from .parse import parse_pages

    p = argparse.ArgumentParser(description="Analyze all chunks and report vocabulary stats")
    p.add_argument("--pages", default="data/pages")
    p.add_argument("--no-stem", action="store_true")
    p.add_argument("--keep-stopwords", action="store_true")
    args = p.parse_args()

    chunks = list(chunk_docs(parse_pages(iter_pages(args.pages))))   # parse outside the timer
    vocab: Counter = Counter()
    t0 = time.perf_counter()
    for c in chunks:
        vocab.update(analyze(c.text, stem_words=not args.no_stem,
                             remove_stopwords=not args.keep_stopwords))
    secs = time.perf_counter() - t0
    total = sum(vocab.values())
    print(f"chunks={len(chunks)}  terms={total}  vocab={len(vocab)}  "
          f"terms/chunk={total / max(len(chunks), 1):.0f}  analyze time={secs:.2f}s")
    print("top 15:", ", ".join(f"{t}({n})" for t, n in vocab.most_common(15)))


if __name__ == "__main__":
    main()
