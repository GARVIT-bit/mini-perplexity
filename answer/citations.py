"""Citation parser: check the model's answer against the "cite every sentence" rule.

A prompt is a request, not a guarantee. This code measures what the model
actually did, per sentence:

  cited      sentence has at least one valid [n]
  uncited    sentence has no citation at all          -> an unsupported claim
  invalid    [n] points at a passage that doesn't exist (n = 0 or n > N)
             -> the model made up a source

Splitting into sentences:
  1. Lines are split first: list items are separate claims.
  2. A citation written after the full stop ("works. [2] Next...") is moved
     before it ("works [2]. Next..."), otherwise the split would hand [2] to
     the next sentence.
  3. Split after . ! ? followed by whitespace and an uppercase letter, digit,
     quote, bracket or backtick. Good enough for prose; "e.g. Foo" can split
     wrongly, which shows up as an uncited fragment rather than a crash.

Lead-in lines: an uncited line ending in ":" ("The steps are:") introduces a
list and makes no claim of its own; the claims are the list items, which are
checked one by one. Such lines are kept (marked lead_in) but don't count for
or against coverage. Trade-off: a real claim written with a trailing colon
goes unchecked; a lead-in that does carry a citation is treated normally.

`coverage` = cited sentences / all claim sentences (lead-ins excluded). The
eval's faithfulness number later builds on this.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .prompt import NOT_FOUND

# [3]  [1, 4]  [1,4,7]  (one bracket may hold several numbers)
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_CITES = r"(?:\s*\[\d+(?:\s*,\s*\d+)*\])+"
_CITE_AFTER_STOP = re.compile(r"([.!?])(" + _CITES + r")")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[`])")
_LIST_MARK = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_HEADING = re.compile(r"^\s*#{1,6}\s")


@dataclass
class Sentence:
    text: str
    citations: list[int]          # valid passage numbers, in order of appearance, no repeats
    invalid: list[int]            # numbers outside 1..n_sources
    lead_in: bool = False         # uncited "...:" line introducing a list: not a claim

    @property
    def cited(self) -> bool:
        return bool(self.citations)


@dataclass
class ParsedAnswer:
    text: str
    not_found: bool
    sentences: list[Sentence] = field(default_factory=list)

    @property
    def cited_sources(self) -> list[int]:
        """Every valid passage number used anywhere, in first-use order."""
        seen: list[int] = []
        for s in self.sentences:
            for n in s.citations:
                if n not in seen:
                    seen.append(n)
        return seen

    @property
    def claims(self) -> list[Sentence]:
        """Sentences that must carry a citation (everything except lead-ins)."""
        return [s for s in self.sentences if not s.lead_in]

    @property
    def uncited(self) -> list[Sentence]:
        return [s for s in self.claims if not s.cited]

    @property
    def invalid_citations(self) -> list[int]:
        return sorted({n for s in self.sentences for n in s.invalid})

    @property
    def coverage(self) -> float:
        """Share of claim sentences with at least one valid citation (1.0 for NOT_FOUND / no claims)."""
        claims = self.claims
        if not claims:
            return 1.0
        return sum(s.cited for s in claims) / len(claims)


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        # code is an example the prose around it cites, not a claim of its own
        if in_code or not line.strip() or _HEADING.match(line):
            continue
        line = _LIST_MARK.sub("", line).strip()
        line = _CITE_AFTER_STOP.sub(lambda m: f"{m.group(2)}{m.group(1)}", line)
        out.extend(p.strip() for p in _SENTENCE_END.split(line) if p.strip())
    return out


def _numbers(sentence: str) -> list[int]:
    nums: list[int] = []
    for m in CITATION.finditer(sentence):
        nums.extend(int(x) for x in m.group(1).split(","))
    return nums


def parse_answer(text: str, n_sources: int) -> ParsedAnswer:
    stripped = text.strip()
    # tolerate trailing punctuation or quotes around the token, nothing else
    if stripped.strip(" .\"'`") == NOT_FOUND:
        return ParsedAnswer(text=text, not_found=True)
    sentences = []
    for s in split_sentences(text):
        valid: list[int] = []
        invalid: list[int] = []
        for n in _numbers(s):
            bucket = valid if 1 <= n <= n_sources else invalid
            if n not in bucket:
                bucket.append(n)
        # an invalid number still counts as an attempt to cite: not a lead-in
        lead_in = s.endswith(":") and not valid and not invalid
        sentences.append(Sentence(s, valid, invalid, lead_in))
    return ParsedAnswer(text=text, not_found=False, sentences=sentences)
