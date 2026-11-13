"""Faithfulness judge: is each cited sentence actually supported by the passages it cites?

The citation parser (answer/citations.py) only checks that a sentence HAS a
valid [n]. A model can cite [3] for a claim [3] never makes. This module asks
an NLI model (natural language inference: does premise P entail hypothesis
H?) for every cited sentence:

    premise    = the cited passage text (exactly what the LLM was given)
    hypothesis = the sentence, with its [n] markers removed
    supported  = P(entailment) >= THRESHOLD

Premises are single sentences and adjacent pairs, not whole passages. This
NLI model is trained on one-sentence premises, and extra context drowns even
an exact match. Measured on this project's answers:
    near-verbatim claim vs its full 120-word passage    0.02 - 0.12
    vs the 3-sentence window that contains it            0.03 - 0.04
    vs the matching passage sentence alone               0.99
So each cited passage is split into sentences (and "- " list items), and the
claim is judged against every sentence and every pair of adjacent sentences
of every passage it cites; the best score counts.

Known blind spot: a claim that combines facts from two different passages,
or from sentences far apart in one passage, matches no single window and is
judged unsupported, so the rate leans low rather than high. `report` runs
a positive control (passage sentences as claims) and a negative control
(claims vs another question's passages) so the judge's error rates are
measured on every run, not assumed.

Why a local NLI model rather than asking an LLM to judge: it's free, fast on a
CPU, deterministic (rerunning gives the same number), and it isn't the same
model grading its own answers.

Limits (say these next to the number): NLI models are trained on everyday
sentences, so code-heavy claims ("use csv.reader(f)") and paraphrases score
lower than a person would judge; a short, vague sentence can be "entailed" by
a lot. Spot-check a sample by hand before quoting the rate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
THRESHOLD = 0.5
WINDOW = 2                     # premises: every single sentence and every run of up to WINDOW sentences
_CITE = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")
# sentence ends, and the " - " the docs pages use between list items
_SENT_END = re.compile(r"(?<=[.!?])\s+|\s+-\s+(?=[A-Z`])")


def clean(text: str) -> str:
    # "¶" is the docs' heading permalink marker: page furniture, not language
    return " ".join(text.replace("¶", " ").split())


def strip_citations(sentence: str) -> str:
    # "Open it [4][6]." -> "Open it." : the markers mean nothing to the NLI model
    return clean(re.sub(r"\s+([.!?,;:])", r"\1", _CITE.sub("", sentence)))


def sentences(passage: str) -> list[str]:
    return [s.strip(" -") for s in _SENT_END.split(clean(passage)) if s.strip(" -")]


def windows(passage: str, size: int = WINDOW) -> list[str]:
    """Every run of 1..size consecutive sentences of the passage."""
    sents = sentences(passage)
    out = []
    for k in range(1, size + 1):
        out += [" ".join(sents[i:i + k]) for i in range(len(sents) - k + 1)]
    return out or [clean(passage)]


@dataclass
class Judged:
    sentence: str
    citations: list[int]
    entailment: float          # best P(entailment) over the premises tried; 0.0 if uncited
    supported: bool


class NLIJudge:
    def __init__(self, model_name: str = NLI_MODEL, model=None, threshold: float = THRESHOLD):
        if model is None:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(model_name, device="cpu")
        self.model = model
        self.threshold = threshold
        # identifies what produced a score, so a cached score is reused only for the same judge setup
        self.tag = f"{model_name}|window={WINDOW}"
        # label order differs between NLI models: read it from the model, never assume
        labels = {i: str(l).lower() for i, l in model.config.id2label.items()}
        self.entail = next(i for i, l in labels.items() if l.startswith("entail"))

    def entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        # Every pair in a batch is padded to the batch's longest one, so a single full
        # passage makes 31 short windows pay for 300 tokens each. Scoring in length order
        # keeps similar lengths together; the scores are put back in the original order.
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        probs = self.model.predict([pairs[i] for i in order], apply_softmax=True, show_progress_bar=False)
        out = [0.0] * len(pairs)
        for i, p in zip(order, probs):
            out[i] = float(p[self.entail])
        return out

    def judge(self, sentences: list[tuple[str, list[int]]], passages: dict[int, str]) -> list[Judged]:
        """sentences: (text, valid citation numbers); passages: number -> passage text."""
        pairs: list[tuple[str, str]] = []
        owner: list[int] = []                      # which sentence each pair belongs to
        for i, (text, cites) in enumerate(sentences):
            hyp = strip_citations(text)
            premises = [w for n in cites if n in passages for w in windows(passages[n])]
            for prem in premises:
                pairs.append((prem, hyp))
                owner.append(i)
        scores = self.entailment(pairs)            # one batched call for the whole answer
        best = [0.0] * len(sentences)
        for i, s in zip(owner, scores):
            best[i] = max(best[i], s)
        return [Judged(text, cites, best[i], bool(cites) and best[i] >= self.threshold)
                for i, (text, cites) in enumerate(sentences)]
