"""Answer-eval tests: NLI judge (fake model), resumable collection, metrics. No LLM, no index."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from answer.engine import AnswerEngine
from answer.llm import FakeLLM
from bench.answer_eval import collect, load_records, render, summarize
from bench.evalset import Question
from bench.faithfulness import NLIJudge, strip_citations, windows
from indexer.chunk import Chunk


# ------------------------------------------------------------------ NLI judge

class WordNLI:
    """Fake NLI model: 'entailed' when every word of the hypothesis appears in the premise.

    Its labels are deliberately in a different order from the real model's, to
    prove the judge reads the order from the config.
    """
    config = SimpleNamespace(id2label={0: "neutral", 1: "contradiction", 2: "ENTAILMENT"})

    def __init__(self):
        self.calls = 0

    def predict(self, pairs, apply_softmax=True, show_progress_bar=False):
        self.calls += 1
        words = lambda t: set(re.findall(r"\w+", t.lower()))
        return np.array([[0.05, 0.05, 0.9] if words(h) <= words(p) else [0.8, 0.1, 0.1] for p, h in pairs])


def test_strip_citations():
    assert strip_citations("Open it [4][6].") == "Open it."
    assert strip_citations("Both work [1, 3], really [2]!") == "Both work, really!"
    assert strip_citations("Overview¶ It works [1].") == "Overview It works."


def test_windows_are_single_sentences_and_adjacent_pairs():
    assert windows("One a. Two b. Three c.", size=2) == ["One a.", "Two b.", "Three c.", "One a. Two b.", "Two b. Three c."]
    # the docs pages' " - " list items are separate sentences
    assert windows("Kinds: - A shallow copy. - A deep copy.", size=1) == ["Kinds:", "A shallow copy.", "A deep copy."]


def test_a_claim_found_in_one_window_of_a_long_passage_is_supported():
    class LongPremiseBlind(WordNLI):
        """Like a real NLI model: misses the claim when the premise is long."""
        def predict(self, pairs, **kw):
            short = [(p, h) if len(p.split()) <= 12 else ("", h) for p, h in pairs]
            return super().predict(short, **kw)
    passage = "Filler one here. Filler two here. The reader reads rows. Filler three. Filler four here. Filler five."
    [j] = NLIJudge(model=LongPremiseBlind()).judge([("The reader reads rows [1].", [1])], {1: passage})
    assert j.supported


def test_judge_reads_label_order_and_batches_one_call():
    model = WordNLI()
    judge = NLIJudge(model=model)
    out = judge.judge([("reader reads rows [1].", [1]), ("writer deletes files [1].", [1])],
                      {1: "The reader reads rows from a file."})
    assert [j.supported for j in out] == [True, False]
    assert out[0].entailment == pytest.approx(0.9)
    assert model.calls == 1


def test_claim_supported_by_any_one_of_its_cited_passages():
    judge = NLIJudge(model=WordNLI())
    [j] = judge.judge([("writer writes rows [1][2].", [1, 2])], {1: "reader reads rows.", 2: "writer writes rows."})
    assert j.supported


def test_known_blind_spot_claim_spanning_two_passages_is_unsupported():
    # documented in faithfulness.py: no single window holds both halves, so the rate leans low
    judge = NLIJudge(model=WordNLI())
    [j] = judge.judge([("reader reads writer writes [1][2].", [1, 2])], {1: "reader reads.", 2: "writer writes."})
    assert not j.supported


def test_uncited_or_dangling_citation_is_unsupported():
    judge = NLIJudge(model=WordNLI())
    out = judge.judge([("anything at all.", []), ("reader [5].", [5])], {1: "reader"})
    assert [j.supported for j in out] == [False, False]
    assert out[0].entailment == 0.0


# ------------------------------------------------------------------ collect (the `run` step)

def chunk(i: int, text: str, url: str | None = None) -> Chunk:
    return Chunk(f"d{i}:0", f"d{i}", 0, url or f"https://ex.com/{i}", f"Page {i}", text)


@dataclass
class Hit:
    chunk: Chunk


class FakeSearcher:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, top_k=10):
        return [Hit(c) for c in self.chunks[:top_k]]


CHUNKS = [chunk(1, "The reader reads rows from a file.", "https://ex.com/csv"),
          chunk(2, "Toggles switch features on and off.", "https://ex.com/toggles")]
QS = [Question("q1", "natural", "how to read rows?", ["https://ex.com/csv"], ["reader"]),
      Question("q2", "paraphrase", "flip a feature?", ["https://ex.com/toggles"], ["toggle"]),
      Question("u1", "unanswerable", "who won the 1998 world cup?")]

REPLIES = {"how to read rows?": "The reader reads rows [1]. It also deletes files [1].",
           "flip a feature?": "Toggles switch features [2].",
           "who won the 1998 world cup?": "NOT_FOUND"}


def engine():
    reply = lambda msgs: REPLIES[msgs[-1]["content"].rsplit("Question: ", 1)[1]]
    return AnswerEngine(FakeSearcher(CHUNKS), FakeLLM(reply))


def test_collect_writes_one_line_per_question_with_the_passage_text(tmp_path):
    out = tmp_path / "answers.jsonl"
    slept = []
    assert collect(engine(), QS, str(out), delay=6, sleep=slept.append, log=lambda *_: None) == "complete"
    recs = load_records(str(out))
    assert [recs[q.id]["status"] for q in QS] == ["answered", "answered", "not_found"]
    assert recs["q1"]["sources"][0]["text"] == "The reader reads rows from a file."
    assert slept == [6, 6]                        # between calls, not before the first


class Flaky:
    """Engine stand-in: raises the given errors for the given questions, else answers."""
    def __init__(self, errors):
        self.errors, self.inner, self.asked = errors, engine(), []

    def answer(self, question):
        self.asked.append(question)
        if question in self.errors:
            raise self.errors[question]
        return self.inner.answer(question)


class ServerError(Exception):          # same names as google.genai's errors -> friendly messages
    code, message = 503, "high demand"


class ClientError(Exception):
    code, message = 429, "quota"


def test_errors_are_recorded_and_retried_on_the_next_run(tmp_path):
    out = str(tmp_path / "a.jsonl")
    collect(Flaky({"flip a feature?": ServerError()}), QS, out, log=lambda *_: None)
    assert load_records(out)["q2"]["status"] == "error"
    again = Flaky({})
    collect(again, QS, out, log=lambda *_: None)
    assert again.asked == ["flip a feature?"]          # only the failed one is asked again
    assert load_records(out)["q2"]["status"] == "answered"


def test_rate_limit_stops_the_run_and_keeps_progress(tmp_path):
    out = str(tmp_path / "a.jsonl")
    flaky = Flaky({"flip a feature?": ClientError()})
    assert collect(flaky, QS, out, log=lambda *_: None) == "rate_limited"
    assert "who won the 1998 world cup?" not in flaky.asked     # stopped, didn't burn more quota
    assert load_records(out)["q1"]["status"] == "answered"


def test_a_real_bug_is_not_swallowed(tmp_path):
    with pytest.raises(KeyError):
        collect(Flaky({"how to read rows?": KeyError("bug")}), QS, str(tmp_path / "a.jsonl"), log=lambda *_: None)


# ------------------------------------------------------------------ summarize + render (the `report` step)

def test_metrics_on_known_answers(tmp_path):
    out = str(tmp_path / "a.jsonl")
    collect(engine(), QS, out, log=lambda *_: None)
    s = summarize(load_records(out), QS, NLIJudge(model=WordNLI()))
    a = s["answerable"]
    assert a["n"] == 2 and a["answered"] == 1.0 and a["not_found"] == 0.0
    # q1: "reader reads rows" supported, "deletes files" not; q2: supported -> 2 of 3 claims
    assert a["claims"] == 3 and a["faithfulness"] == pytest.approx(2 / 3)
    assert a["fully_faithful"] == 0.5                  # q2 yes, q1 no
    assert a["cites_relevant"] == 1.0 and a["made_up"] == 0.0
    assert s["unanswerable"] == {"n": 1, "not_found": 1.0, "answered": 0.0, "error": 0.0}
    assert s["by_type"]["natural"]["faithfulness"] == 0.5
    assert s["latency"]["n"] == 3

    report = render(s, "questions.jsonl", "fake-nli", 0.5)
    assert "**66.7%**" in report and "**100.0%**" in report
    assert "It also deletes files" in report           # least-supported claim listed for spot-checking


def test_scores_are_cached_and_reused_until_the_answer_changes(tmp_path):
    out = str(tmp_path / "a.jsonl")
    collect(engine(), QS, out, log=lambda *_: None)
    model, cache = WordNLI(), {}
    first = summarize(load_records(out), QS, NLIJudge(model=model), cache)
    calls = model.calls
    again = summarize(load_records(out), QS, NLIJudge(model=model), cache)
    assert model.calls == calls                        # nothing rescored
    assert again["answerable"] == first["answerable"]
    # a stricter threshold applies to cached scores without rescoring
    strict = summarize(load_records(out), QS, NLIJudge(model=model, threshold=0.95), cache)
    assert model.calls == calls and strict["answerable"]["faithfulness"] == 0.0
    # a retried question (new run_at) is scored afresh
    recs = load_records(out)
    recs["q1"]["run_at"] = "later"
    summarize(recs, QS, NLIJudge(model=model), cache)
    assert model.calls == calls + 1


def test_length_sorting_returns_scores_in_the_original_order():
    model = WordNLI()
    judge = NLIJudge(model=model)
    pairs = [("a much longer premise with reader in it", "reader"), ("x", "reader"), ("reader", "reader")]
    assert judge.entailment(pairs) == pytest.approx([0.9, 0.1, 0.9])


def test_controls_measure_the_judge_on_the_runs_own_data(tmp_path):
    out = str(tmp_path / "a.jsonl")
    collect(engine(), QS, out, log=lambda *_: None)
    c = summarize(load_records(out), QS, NLIJudge(model=WordNLI()))["controls"]
    assert c["positive_n"] == 2 and c["positive"] == 1.0          # copied passage sentences pass
    assert c["negative_n"] == 3 and c["negative"] == 0.0          # claims vs the other page's passages fail


def test_report_reparses_saved_answers_with_the_current_parser(tmp_path):
    out = tmp_path / "a.jsonl"
    collect(engine(), QS[:1], str(out), log=lambda *_: None)
    rec = json.loads(out.read_text(encoding="utf-8"))
    rec["sentences"].append({"text": "x = 1", "citations": [], "invalid": [], "lead_in": False})  # a stale parse
    out.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    s = summarize(load_records(str(out)), QS, NLIJudge(model=WordNLI()))
    assert s["answerable"]["claims"] == 2                         # the stale code "claim" is gone


def test_answering_an_unanswerable_question_is_counted(tmp_path):
    out = str(tmp_path / "a.jsonl")
    REPLIES_BAD = {**REPLIES, "who won the 1998 world cup?": "France won [1]."}
    reply = lambda msgs: REPLIES_BAD[msgs[-1]["content"].rsplit("Question: ", 1)[1]]
    collect(AnswerEngine(FakeSearcher(CHUNKS), FakeLLM(reply)), QS, out, log=lambda *_: None)
    u = summarize(load_records(out), QS, NLIJudge(model=WordNLI()))["unanswerable"]
    assert u["not_found"] == 0.0 and u["answered"] == 1.0


def test_report_survives_an_all_error_run(tmp_path):
    out = tmp_path / "a.jsonl"
    out.write_text("".join(json.dumps({"id": q.id, "type": q.type, "status": "error", "error": "x"}) + "\n"
                           for q in QS), encoding="utf-8")
    s = summarize(load_records(str(out)), QS, NLIJudge(model=WordNLI()))
    assert s["answerable"]["error"] == 1.0 and s["answerable"]["faithfulness"] is None
    assert "–" in render(s, "q", "nli", 0.5)            # missing numbers print as a dash, not a crash
