"""Benchmark: answer quality and end-to-end latency of the full engine on the labelled questions.

Two steps, because LLM calls cost (free-tier) quota and scoring doesn't:

  run     ask every question through the real engine (hybrid top 8 -> LLM ->
          citation check) and append one JSON line per answer to
          bench/results/answers.jsonl. Resumable: rerunning skips questions
          that already have an answer, and retries ones that errored. A rate
          limit (HTTP 429) stops the run cleanly; rerun the same command later.

  report  score the saved answers (no LLM calls) and write
          bench/results/answer_eval.md. Rerun it as often as you like.

Metrics
  answerable questions (labels in bench/eval/questions.jsonl):
    answered / NOT_FOUND (a false abstention) / refused / error rates
    citation coverage        cited claim sentences / claim sentences
    made-up citations        answers citing a passage number that wasn't given
    faithfulness             claim sentences an NLI model judges entailed by
                             their cited passages / all claim sentences
                             (uncited claims count as unsupported); see
                             bench/faithfulness.py for how and its limits
    fully faithful answers   every claim supported
    cites a relevant page    the answer cites at least one page the labels say
                             answers the question
  unanswerable questions: correct NOT_FOUND rate (the rest are answers
    the model made up from unrelated passages, or errors)
  latency (non-error answers): time to first token and total, p50 / p99

    python -m bench.answer_eval run --llm gemini --model gemini-3.1-flash-lite --delay 6
    python -m bench.answer_eval report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timezone

import numpy as np

from answer.citations import parse_answer
from answer.engine import AnswerEngine, setup_error_message

from .evalset import QUESTIONS_FILE, load_questions
from .faithfulness import sentences

ANSWERS_FILE = "bench/results/answers.jsonl"
REPORT_FILE = "bench/results/answer_eval.md"
SENTENCES_FILE = "bench/results/answer_eval_sentences.jsonl"
NLI_CACHE_FILE = "bench/results/answer_eval_nli_cache.json"
TYPES = ("keyword", "natural", "paraphrase")


# ------------------------------------------------------------------ run

def record_for(q, answer=None, error: str | None = None) -> dict:
    rec = {"id": q.id, "type": q.type, "answerable": q.answerable, "question": q.question,
           "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if error is not None:
        return {**rec, "status": "error", "error": error}
    a = answer
    return {**rec,
            "status": a.status,
            "text": a.text,
            # passage text is saved too: `report` must judge against exactly what the LLM saw,
            # even if the index is rebuilt later
            "sources": [{"n": s.number, "chunk_id": s.chunk.chunk_id, "url": s.chunk.url, "text": s.chunk.text}
                        for s in a.sources],
            "sentences": [{"text": s.text, "citations": s.citations, "invalid": s.invalid, "lead_in": s.lead_in}
                          for s in a.parsed.sentences],
            "cited": a.parsed.cited_sources,
            "invalid": a.parsed.invalid_citations,
            "coverage": a.parsed.coverage,
            "model": a.llm.model if a.llm else None,
            "input_tokens": a.llm.input_tokens if a.llm else 0,
            "output_tokens": a.llm.output_tokens if a.llm else 0,
            "timings_ms": a.timings_ms}


def load_records(path: str) -> dict[str, dict]:
    """Last record per question id (a retried question appends a newer line)."""
    latest: dict[str, dict] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    latest[r["id"]] = r
    return latest


def collect(engine, questions, out_path: str, delay: float = 0.0, sleep=time.sleep, log=print) -> str:
    """Answer every question without a good record yet. Returns 'complete' or 'rate_limited'."""
    have = load_records(out_path)
    todo = [q for q in questions if have.get(q.id, {}).get("status", "error") == "error"]
    log(f"{len(questions) - len(todo)} already answered, {len(todo)} to go")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    for i, q in enumerate(todo):
        if i and delay:
            sleep(delay)                     # stay under the free tier's requests-per-minute limit
        try:
            rec = record_for(q, engine.answer(q.question))
        except Exception as e:
            msg = setup_error_message(e)
            if msg is None:
                raise                        # a real bug: stop with the traceback
            rec = record_for(q, error=msg)
            if getattr(e, "code", None) == 429:
                _append(out_path, rec)
                log(f"[{q.id}] rate limited: stopping. Progress is saved; rerun the same command later.")
                return "rate_limited"
        _append(out_path, rec)
        t = rec.get("timings_ms", {}).get("total")
        log(f"[{q.id}] {rec['status']}" + (f"  {t / 1000:.1f} s" if t else "") +
            (f"  {rec['error'].splitlines()[0]}" if rec.get("error") else ""))
    return "complete"


def _append(path: str, rec: dict) -> None:
    # one line per answer, flushed immediately: a crash or Ctrl-C loses at most the current question
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ report

def judge_record(rec: dict, judge, cache: dict | None = None) -> list[dict]:
    """NLI verdict for every claim sentence of one answered record.

    Scoring is the slow part of `report` (thousands of sentence/window pairs on a
    CPU), so scores are cached per answer. The key includes when the answer was
    made and which judge scored it: a retried question or a changed judge is
    scored afresh; everything else is reused. The threshold is applied after
    the cache, so changing it never needs a rescore.
    """
    claims = [s for s in rec["sentences"] if not s["lead_in"]]
    passages = {s["n"]: s["text"] for s in rec["sources"]}
    scores = _cached_scores(f"{rec['id']}|{rec['run_at']}", claims, passages, judge, cache)
    return [{"id": rec["id"], "sentence": s["text"], "citations": s["citations"], "entailment": round(e, 4),
             "supported": bool(s["citations"]) and e >= judge.threshold}
            for s, e in zip(claims, scores)]


def _cached_scores(name: str, claims: list[dict], passages: dict[int, str], judge, cache) -> list[float]:
    # the key also covers the claims and passages themselves: a citation-parser fix that
    # changes the claims, or different passages, can never reuse a stale score
    content = json.dumps([[c["text"], c["citations"]] for c in claims] + [sorted(passages.items())])
    key = f"{name}|{judge.tag}|{hashlib.sha1(content.encode()).hexdigest()[:12]}"
    if cache is not None and key in cache:
        return cache[key]
    scores = [j.entailment for j in judge.judge([(c["text"], c["citations"]) for c in claims], passages)]
    if cache is not None:
        cache[key] = scores
    return scores


def reparse(rec: dict) -> dict:
    """Re-run the CURRENT citation parser on a saved answer's text.

    The sentences stored at `run` time came from whatever parser existed then;
    re-parsing means parser fixes reach old answers without new LLM calls.
    """
    if rec.get("status") not in ("answered", "truncated"):
        return rec
    p = parse_answer(rec["text"], len(rec["sources"]))
    return {**rec,
            "sentences": [{"text": s.text, "citations": s.citations, "invalid": s.invalid, "lead_in": s.lead_in}
                          for s in p.sentences],
            "cited": p.cited_sources, "invalid": p.invalid_citations, "coverage": p.coverage}


def controls(answered: list[dict], judge, cache: dict | None, per_record: int = 3) -> dict:
    """Measure the judge itself, on this run's own data.

    positive  sentences taken from a cited passage, judged against that passage:
              a working judge supports ~all of them
    negative  each answer's claims judged against the passages ANOTHER answer cited,
              where the two cite no page in common: ~none should pass
    """
    def cited_urls(r):
        return {s["url"] for s in r["sources"] if s["n"] in r["cited"]}

    pos_claims, neg = [], []
    for i, r in enumerate(answered):
        passages = {s["n"]: s["text"] for s in r["sources"]}
        for n in r["cited"][:1]:
            sents = [s for s in sentences(passages[n]) if len(s.split()) >= 6][:per_record]
            pos_claims += [(r["id"], {"text": s, "citations": [n]}, passages) for s in sents]
        other = next((o for o in answered[i + 1:] + answered[:i]
                      if o["cited"] and not cited_urls(r) & cited_urls(o)), None)
        if other:
            # same claim text, but pointed at the other answer's cited passages
            claims = [{"text": s["text"], "citations": other["cited"]}
                      for s in r["sentences"] if not s["lead_in"] and s["citations"]]
            neg.append((r["id"], other["id"], claims, {s["n"]: s["text"] for s in other["sources"]}))
    pos_scores = [_cached_scores(f"pos|{qid}", [c], p, judge, cache)[0] for qid, c, p in pos_claims]
    neg_scores = [e for qid, oid, claims, p in neg
                  for e in _cached_scores(f"neg|{qid}|{oid}", claims, p, judge, cache)]
    return {"positive_n": len(pos_scores), "positive": _rate(e >= judge.threshold for e in pos_scores),
            "negative_n": len(neg_scores), "negative": _rate(e >= judge.threshold for e in neg_scores)}


def _rate(xs) -> float | None:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def _pct(values, q) -> float | None:
    return float(np.percentile(values, q)) if values else None


def summarize(records: dict[str, dict], questions, judge, cache: dict | None = None) -> dict:
    qs = {q.id: q for q in questions}
    recs = [reparse(r) for qid, r in records.items() if qid in qs]   # ignore answers to removed questions
    ans = [r for r in recs if qs[r["id"]].answerable]
    unans = [r for r in recs if not qs[r["id"]].answerable]
    answered = [r for r in ans if r["status"] in ("answered", "truncated")]

    sentences: list[dict] = []
    per_answer: dict[str, list[dict]] = {}
    for r in answered:
        per_answer[r["id"]] = judge_record(r, judge, cache)
        sentences += per_answer[r["id"]]

    def cites_relevant(r):
        urls = {s["url"] for s in r["sources"] if s["n"] in r["cited"]}
        return bool(urls & set(qs[r["id"]].relevant))

    def quality(rs_all, rs_answered):
        claims = [s for r in rs_answered for s in per_answer[r["id"]]]
        return {
            "n": len(rs_all),
            "answered": _rate(r["status"] in ("answered", "truncated") for r in rs_all),
            "not_found": _rate(r["status"] == "not_found" for r in rs_all),
            "refused": _rate(r["status"] == "refused" for r in rs_all),
            "error": _rate(r["status"] == "error" for r in rs_all),
            "coverage": _rate(r["coverage"] for r in rs_answered),
            "made_up": _rate(bool(r["invalid"]) for r in rs_answered),
            "faithfulness": _rate(s["supported"] for s in claims),
            "fully_faithful": _rate(all(s["supported"] for s in per_answer[r["id"]]) for r in rs_answered),
            "cites_relevant": _rate(cites_relevant(r) for r in rs_answered),
            "claims": len(claims),
        }

    ok = [r for r in recs if r["status"] != "error"]
    first = [r["timings_ms"]["first_token"] for r in ok if "first_token" in r["timings_ms"]]
    total = [r["timings_ms"]["total"] for r in ok if "total" in r["timings_ms"]]
    retrieval = [r["timings_ms"]["retrieval"] for r in ok if "retrieval" in r["timings_ms"]]
    return {
        "controls": controls(answered, judge, cache),
        "models": sorted({r["model"] for r in ok if r.get("model")}),
        "n_records": len(recs),
        "n_questions": len(questions),
        "answerable": quality(ans, answered),
        "by_type": {t: quality([r for r in ans if r["type"] == t], [r for r in answered if r["type"] == t])
                    for t in TYPES},
        "unanswerable": {"n": len(unans),
                         "not_found": _rate(r["status"] == "not_found" for r in unans),
                         "answered": _rate(r["status"] in ("answered", "truncated") for r in unans),
                         "error": _rate(r["status"] == "error" for r in unans)},
        "latency": {"n": len(total),
                    "first_token_p50": _pct(first, 50), "first_token_p99": _pct(first, 99),
                    "total_p50": _pct(total, 50), "total_p99": _pct(total, 99),
                    "retrieval_p50": _pct(retrieval, 50)},
        "tokens": {"input_mean": _rate(r["input_tokens"] for r in ok if r.get("input_tokens")),
                   "output_mean": _rate(r["output_tokens"] for r in ok if r.get("output_tokens"))},
        "sentences": sentences,
    }


def _f(x, pct=True) -> str:
    if x is None:
        return "–"
    return f"{x:.1%}" if pct else f"{x:,.0f}"


def render(s: dict, questions_path: str, judge_name: str, threshold: float) -> str:
    a, u, lat = s["answerable"], s["unanswerable"], s["latency"]
    lines = [
        f"# Answer eval ({date.today().isoformat()})", "",
        f"Model: {', '.join(s['models']) or '–'}. {s['n_records']} of {s['n_questions']} questions from "
        f"`{questions_path}` have an answer on file. Retrieval: hybrid (BM25 + HNSW, RRF) top 8.",
        "Generated by `python -m bench.answer_eval report` from `bench/results/answers.jsonl`.",
        f"Faithfulness judge: `{judge_name}` (NLI), a claim is supported when P(entailment) >= {threshold}.",
        "Question labels were drafted by Claude and are not yet human-reviewed.", "",
        f"## Answerable questions ({a['n']})", "",
        "| metric | value |", "|---|---:|",
        f"| answered | {_f(a['answered'])} |",
        f"| said NOT_FOUND (false abstention) | {_f(a['not_found'])} |",
        f"| refused / error | {_f(a['refused'])} / {_f(a['error'])} |",
        f"| citation coverage (claims with a valid [n]) | {_f(a['coverage'])} |",
        f"| answers with a made-up citation | {_f(a['made_up'])} |",
        f"| **faithfulness** (claims supported by their cited passage, {a['claims']} claims) | **{_f(a['faithfulness'])}** |",
        f"| fully faithful answers | {_f(a['fully_faithful'])} |",
        f"| cites a labelled relevant page | {_f(a['cites_relevant'])} |",
        "", "Judge sanity checks (on this run's own answers; see `bench/faithfulness.py`):", "",
        "| control | judged supported | should be |", "|---|---:|---:|",
        f"| positive: {s['controls']['positive_n']} sentences copied from a cited passage | "
        f"{_f(s['controls']['positive'])} | ~100% |",
        f"| negative: {s['controls']['negative_n']} claims vs another question's passages | "
        f"{_f(s['controls']['negative'])} | ~0% |",
        "", f"## Unanswerable questions ({u['n']})", "",
        "| metric | value |", "|---|---:|",
        f"| **correctly said NOT_FOUND** | **{_f(u['not_found'])}** |",
        f"| answered anyway | {_f(u['answered'])} |",
        f"| error | {_f(u['error'])} |",
        "", "## By question type (answerable)", "",
        "| type | n | answered | faithfulness | fully faithful | cites relevant page |", "|---|---:|---:|---:|---:|---:|",
    ]
    for t, q in s["by_type"].items():
        lines.append(f"| {t} | {q['n']} | {_f(q['answered'])} | {_f(q['faithfulness'])} | "
                     f"{_f(q['fully_faithful'])} | {_f(q['cites_relevant'])} |")
    lines += [
        "", f"## Latency (end to end, {lat['n']} non-error answers)", "",
        "| | p50 | p99 |", "|---|---:|---:|",
        f"| time to first token (ms) | {_f(lat['first_token_p50'], False)} | {_f(lat['first_token_p99'], False)} |",
        f"| total (ms) | {_f(lat['total_p50'], False)} | {_f(lat['total_p99'], False)} |",
        f"| retrieval only (ms) | {_f(lat['retrieval_p50'], False)} | |",
        "", f"Mean tokens per answer: {_f(s['tokens']['input_mean'], False)} in, "
            f"{_f(s['tokens']['output_mean'], False)} out (output includes thinking). "
            f"With ~{lat['n']} samples, p99 is close to the single slowest answer.",
        "", "## Least-supported claims (spot-check these by hand)", "",
        "| question | P(entail) | sentence |", "|---|---:|---|",
    ]
    for x in sorted(s["sentences"], key=lambda x: x["entailment"])[:10]:
        lines.append(f"| {x['id']} | {x['entailment']:.2f} | {x['sentence'].replace('|', '/')} |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ CLI

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="answer the questions (uses the LLM)")
    r.add_argument("--llm", choices=["claude", "gemini", "fake"], required=True)
    r.add_argument("--model")
    r.add_argument("--delay", type=float, default=6.0, help="seconds between LLM calls (free-tier rate limits)")
    r.add_argument("--limit", type=int, help="only the first N questions still to do (a smoke test)")
    r.add_argument("--index", default="data/index")
    r.add_argument("--questions", default=QUESTIONS_FILE)
    r.add_argument("--out", default=ANSWERS_FILE)
    rep = sub.add_parser("report", help="score saved answers (no LLM calls)")
    rep.add_argument("--questions", default=QUESTIONS_FILE)
    rep.add_argument("--answers", default=ANSWERS_FILE)
    rep.add_argument("--out", default=REPORT_FILE)
    rep.add_argument("--sentences", default=SENTENCES_FILE)
    rep.add_argument("--cache", default=NLI_CACHE_FILE, help="saved NLI scores; delete it to rescore everything")
    args = p.parse_args()

    questions = load_questions(args.questions)
    if args.cmd == "run":
        from answer.engine import load_searcher
        from answer.llm import make_llm
        llm = make_llm(args.llm, args.model)            # a missing key fails before index loading
        engine = AnswerEngine(load_searcher(args.index), llm)
        have = load_records(args.out)
        todo = [q for q in questions if have.get(q.id, {}).get("status", "error") == "error"]
        if args.limit:
            keep = {q.id for q in todo[:args.limit]}
            questions = [q for q in questions if q.id in keep or q.id not in {t.id for t in todo}]
        status = collect(engine, questions, args.out, delay=args.delay)
        sys.exit(0 if status == "complete" else 2)

    from .faithfulness import NLI_MODEL, THRESHOLD, NLIJudge
    records = load_records(args.answers)
    if not records:
        sys.exit(f"no answers in {args.answers}: run `python -m bench.answer_eval run ...` first")
    cache = {}
    if os.path.exists(args.cache):
        with open(args.cache, encoding="utf-8") as fh:
            cache = json.load(fh)
    n_cached = len(cache)
    summary = summarize(records, questions, NLIJudge(), cache)
    with open(args.cache, "w", encoding="utf-8") as fh:
        json.dump(cache, fh)
    print(f"NLI scores: {n_cached} answers from cache, {len(cache) - n_cached} newly scored", file=sys.stderr)
    report = render(summary, args.questions, NLI_MODEL, THRESHOLD)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report)
    with open(args.sentences, "w", encoding="utf-8") as fh:
        for x in summary["sentences"]:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(report)


if __name__ == "__main__":
    main()
