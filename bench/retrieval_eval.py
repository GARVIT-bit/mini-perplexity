"""Benchmark: retrieval quality and latency of every method on the labelled question set.

Methods (each returns up to 50 ranked chunks):
  bm25, vector (brute force), hybrid (RRF), and hybrid + cross-encoder
  re-ranking for each (model, candidates) setting in RERANK_SETTINGS.

Metrics (bench/metrics.py) over the answerable questions, overall and per
question type: Recall@10 and nDCG@10 and MRR on pages, Answer@8 on chunks.
Latency is end to end per query (a re-ranker's time includes its hybrid search).

Also records the top cross-encoder logit for answerable vs unanswerable
questions: the input for choosing a "not found" threshold.

    python -m bench.retrieval_eval            # ~10 min on a laptop CPU
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from datetime import date

import numpy as np

from indexer.embed import Embedder
from indexer.index import InvertedIndex
from retrieval.bm25 import BM25
from retrieval.fusion import HybridSearcher
from retrieval.rerank import Reranker
from retrieval.vector import BruteForceIndex

from .evalset import QUESTIONS_FILE, load_questions
from .metrics import answer_at_k, ndcg_at_k, pages_in_order, recall_at_k, reciprocal_rank

DEPTH = 50
K_PAGES = 10
K_CHUNKS = 8          # what the LLM prompt will get
RERANK_SETTINGS = [
    ("cross-encoder/ms-marco-MiniLM-L6-v2", 50),
    ("cross-encoder/ms-marco-MiniLM-L6-v2", 20),
    ("cross-encoder/ms-marco-MiniLM-L2-v2", 50),
    ("cross-encoder/ms-marco-MiniLM-L2-v2", 20),
]
TYPES = ("keyword", "natural", "paraphrase")


def score_question(q, chunks) -> dict:
    """chunks: list of Chunk, best first."""
    pages = pages_in_order(c.url for c in chunks)
    return {
        "recall@10": recall_at_k(pages, q.relevant, K_PAGES),
        "ndcg@10": ndcg_at_k(pages, q.relevant, K_PAGES),
        "mrr": reciprocal_rank(pages, q.relevant),
        "answer@8": answer_at_k([(c.url, InvertedIndex.index_text(c, True)) for c in chunks],
                                q.relevant, q.evidence, K_CHUNKS),
    }


def short(model: str) -> str:
    return model.split("ms-marco-MiniLM-")[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--index", default="data/index")
    p.add_argument("--questions", default=QUESTIONS_FILE)
    p.add_argument("--out", default="bench/results/retrieval_eval.md")
    p.add_argument("--per-question", default="bench/results/retrieval_eval_per_question.jsonl")
    p.add_argument("--skip-rerank", action="store_true")
    args = p.parse_args()

    qs = load_questions(args.questions)
    index = InvertedIndex.load(args.index)
    bm25 = BM25(index)
    vectors = BruteForceIndex.load(args.index)
    emb = Embedder()
    hybrid = HybridSearcher(bm25, vectors, emb, depth=DEPTH)
    chunks = index.chunks
    hybrid.search("warm up")

    runs: dict[str, dict] = {}              # method -> {qid: (chunks, ms)}

    def run(name, fn):
        out = {}
        for q in qs:
            t = time.perf_counter()
            ranked = fn(q.question)
            out[q.id] = (ranked, (time.perf_counter() - t) * 1000)
        runs[name] = out
        print(f"  {name}: done", flush=True)

    run("bm25", lambda s: [h.chunk for h in bm25.search(s, DEPTH)])
    run("vector", lambda s: [chunks[i] for i, _ in vectors.search(emb.encode_query(s), DEPTH)])
    hyb_hits: dict[str, tuple[list, float]] = {}

    def hyb(s):
        hits = hybrid.search(s, DEPTH)
        return hits
    for q in qs:                             # keep the HybridHits: re-rankers reuse them
        t = time.perf_counter()
        hits = hyb(q.question)
        hyb_hits[q.id] = (hits, (time.perf_counter() - t) * 1000)
    runs["hybrid"] = {qid: ([h.chunk for h in hits], ms) for qid, (hits, ms) in hyb_hits.items()}
    print("  hybrid: done", flush=True)

    top_logits: dict[str, dict[str, float]] = {}
    if not args.skip_rerank:
        for model, n_cand in RERANK_SETTINGS:
            rr = Reranker(model)
            rr.rerank("warm up", hyb_hits[qs[0].id][0][:n_cand], 1)
            name = f"hybrid+rerank {short(model)}@{n_cand}"
            out, logits = {}, {}
            for q in qs:
                cands, hyb_ms = hyb_hits[q.id]
                t = time.perf_counter()
                ranked = rr.rerank(q.question, cands[:n_cand], top_k=n_cand)
                out[q.id] = ([h.chunk for h in ranked], hyb_ms + (time.perf_counter() - t) * 1000)
                logits[q.id] = ranked[0].logit if ranked else float("-inf")
            runs[name] = out
            top_logits[name] = logits
            print(f"  {name}: done", flush=True)

    answerable = [q for q in qs if q.answerable]
    metric_names = ["recall@10", "ndcg@10", "mrr", "answer@8"]
    per_q = defaultdict(dict)
    table = {}
    for name, res in runs.items():
        scores = {q.id: score_question(q, res[q.id][0]) for q in answerable}
        for qid, s in scores.items():
            per_q[qid][name] = s
        lat = [ms for _, ms in res.values()]
        by_type = {t: [scores[q.id] for q in answerable if q.type == t] for t in TYPES}
        table[name] = {
            "all": {m: statistics.mean(s[m] for s in scores.values()) for m in metric_names},
            **{t: {m: statistics.mean(s[m] for s in v) for m in metric_names} for t, v in by_type.items() if v},
            "p50": float(np.percentile(lat, 50)), "p99": float(np.percentile(lat, 99)),
        }

    n_type = {t: sum(q.type == t for q in answerable) for t in TYPES}
    lines = [
        f"# Retrieval eval ({date.today().isoformat()})", "",
        f"{len(answerable)} answerable questions ({', '.join(f'{n_type[t]} {t}' for t in TYPES)}) "
        f"+ {len(qs) - len(answerable)} unanswerable, from `{args.questions}`.",
        f"Index `{args.index}`: {index.n_docs} chunks. Generated by `python -m bench.retrieval_eval`.",
        "Labels were drafted by Claude from the corpus and are not yet human-reviewed.", "",
        "## Overall", "",
        "| method | Recall@10 | nDCG@10 | MRR | Answer@8 | p50 ms | p99 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, t in table.items():
        a = t["all"]
        lines.append(f"| {name} | {a['recall@10']:.3f} | {a['ndcg@10']:.3f} | {a['mrr']:.3f} | "
                     f"{a['answer@8']:.3f} | {t['p50']:.0f} | {t['p99']:.0f} |")
    for metric in ("ndcg@10", "answer@8"):
        lines += ["", f"## {metric} by question type", "",
                  "| method | " + " | ".join(f"{t} ({n_type[t]})" for t in TYPES) + " |",
                  "|---|" + "---:|" * len(TYPES)]
        for name, t in table.items():
            lines.append(f"| {name} | " + " | ".join(f"{t[ty][metric]:.3f}" for ty in TYPES) + " |")

    if top_logits:
        lines += ["", "## Top re-ranker logit: answerable vs unanswerable (input for a \"not found\" threshold)", "",
                  "| method | answerable min / median | unanswerable median / max |", "|---|---|---|"]
        for name, logits in top_logits.items():
            a = [logits[q.id] for q in answerable]
            u = [logits[q.id] for q in qs if not q.answerable]
            lines.append(f"| {name} | {min(a):.2f} / {statistics.median(a):.2f} | "
                         f"{statistics.median(u):.2f} / {max(u):.2f} |")

    report = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report)
    with open(args.per_question, "w", encoding="utf-8") as fh:
        for q in qs:
            row = {"id": q.id, "type": q.type, "question": q.question,
                   "top_pages": {name: pages_in_order(c.url for c in res[q.id][0])[:3] for name, res in runs.items()},
                   "scores": per_q.get(q.id, {}),
                   "top_logit": {n: lg[q.id] for n, lg in top_logits.items()}}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(report)


if __name__ == "__main__":
    main()
