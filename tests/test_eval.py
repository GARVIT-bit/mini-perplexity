"""Eval set + metric tests: every metric checked against hand-computed values."""
from __future__ import annotations

import json
import math

import pytest

from bench.evalset import Question, check, contains_evidence, load_questions
from bench.metrics import answer_at_k, ndcg_at_k, pages_in_order, reciprocal_rank, recall_at_k


# ------------------------------------------------------------------ metrics

def test_pages_in_order_dedupes_keeping_first_rank():
    assert pages_in_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_recall():
    assert recall_at_k(["a", "x", "b"], ["a", "b"], k=3) == 1.0
    assert recall_at_k(["a", "x", "b"], ["a", "b"], k=2) == 0.5      # b is at rank 3
    assert recall_at_k(["x", "y"], ["a"], k=10) == 0.0


def test_ndcg_hand_computed():
    # relevant at ranks 1 and 3, two relevant pages in total
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["a", "x", "b"], ["a", "b"], k=10) == pytest.approx(dcg / ideal)


def test_ndcg_perfect_and_zero():
    assert ndcg_at_k(["a", "b", "x"], ["a", "b"], k=10) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], ["a"], k=10) == 0.0


def test_ndcg_rewards_earlier_hits():
    assert ndcg_at_k(["a", "x"], ["a"], 10) > ndcg_at_k(["x", "a"], ["a"], 10)


def test_ndcg_ideal_capped_at_k():
    # 5 relevant pages but k=2: a top 2 of two relevant pages is perfect
    assert ndcg_at_k(["a", "b"], list("abcde"), k=2) == pytest.approx(1.0)


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
    assert reciprocal_rank(["a"], ["a"]) == 1.0
    assert reciprocal_rank(["x"], ["a"]) == 0.0


def test_metrics_refuse_unanswerable():
    with pytest.raises(ValueError):
        recall_at_k(["a"], [], 10)
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], [], 10)


def test_answer_at_k_needs_right_page_and_evidence():
    chunks = [("u/other", "the csv.reader function"),     # evidence, wrong page
              ("u/csv", "unrelated intro text"),          # right page, no evidence
              ("u/csv", "Use csv.READER to\n  iterate rows")]   # both (case + whitespace ignored)
    assert answer_at_k(chunks, ["u/csv"], ["csv.reader"], k=2) == 0.0
    assert answer_at_k(chunks, ["u/csv"], ["csv.reader"], k=3) == 1.0


# ------------------------------------------------------------------ question set

def write(tmp_path, rows) -> str:
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def row(**kw):
    base = {"id": "q1", "type": "natural", "question": "q?", "relevant": ["u1"], "evidence": ["e"]}
    base.update(kw)
    return base


def test_load_valid(tmp_path):
    qs = load_questions(write(tmp_path, [row(), row(id="q2", type="unanswerable", relevant=[], evidence=[])]))
    assert [q.answerable for q in qs] == [True, False]


@pytest.mark.parametrize("bad", [
    [row(type="weird")],
    [row(type="unanswerable")],                       # unanswerable but has relevant pages
    [row(relevant=[])],                               # answerable type with no pages
    [row(evidence=[])],                               # answerable with no evidence
    [row(), row()],                                   # duplicate id
])
def test_load_rejects_bad_labels(tmp_path, bad):
    with pytest.raises(ValueError):
        load_questions(write(tmp_path, bad))


def test_check_finds_label_errors():
    qs = [Question("q1", "natural", "?", ["u1", "missing"], ["needle"]),
          Question("q2", "natural", "?", ["u1"], ["absent phrase"])]
    problems = check(qs, {"u1": "hay NEEDLE hay"})
    assert len(problems) == 2
    assert "missing" in problems[0] and "q2" in problems[1]


def test_contains_evidence_normalizes():
    assert contains_evidence("Use  Command Query\nResponsibility Segregation", ["command query responsibility segregation"])
    assert not contains_evidence("nothing here", ["needle"])


def test_real_question_file_is_valid():
    qs = load_questions()
    assert len([q for q in qs if q.answerable]) >= 50
    assert {"keyword", "natural", "paraphrase", "unanswerable"} == {q.type for q in qs}
