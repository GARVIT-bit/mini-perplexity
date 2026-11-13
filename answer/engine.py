"""Answer engine: question -> hybrid top 8 passages -> LLM -> checked, cited answer.

    question
      -> HybridSearcher (BM25 + vectors, RRF)   top K=8 chunks   (no re-ranking: see
         bench/results/retrieval_eval.md, it didn't beat hybrid and cost 1-3 s/query)
      -> build_messages: passages numbered [1]..[8]
      -> LLM.stream: answer text arrives in pieces
      -> parse_answer: per-sentence citation check, NOT_FOUND detection

`stream()` yields events in this order, which is what Phase 5's SSE endpoint sends:
    Event("sources", list[Source])   before the LLM starts, so the UI can show cards early
    Event("text", str)               zero or more answer pieces
    Event("done", Answer)            the final, checked answer

Status of an Answer:
    answered    the model answered (check .parsed.coverage for how well it cited)
    not_found   the model said NOT_FOUND, or retrieval returned nothing
    refused     the model declined (stop_reason "refusal"); text may be partial
    truncated   hit max_tokens: the answer is cut off, citations after the cut are missing

    python -m answer.engine "how do I read a csv file?" --fake   # offline, no key
    python -m answer.engine "how do I read a csv file?"          # real Claude call (billed)
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .citations import ParsedAnswer, parse_answer
from .llm import LLM, LLMResult
from .prompt import SYSTEM_PROMPT, Source, build_messages, number_passages

TOP_K = 8


@dataclass
class Event:
    type: str            # "sources" | "text" | "done"
    data: Any


@dataclass
class Answer:
    question: str
    status: str                       # answered | not_found | refused | truncated
    text: str
    sources: list[Source]
    parsed: ParsedAnswer
    llm: LLMResult | None = None      # None when the LLM was skipped
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def cited(self) -> list[Source]:
        """The sources the answer actually used, in first-use order (what the UI shows as chips)."""
        by_number = {s.number: s for s in self.sources}
        return [by_number[n] for n in self.parsed.cited_sources]


class AnswerEngine:
    def __init__(self, searcher, llm: LLM, k: int = TOP_K):
        self.searcher = searcher        # anything with .search(query, top_k) -> hits with .chunk
        self.llm = llm
        self.k = k

    def stream(self, question: str) -> Iterator[Event]:
        if not question.strip():
            raise ValueError("empty question")
        t0 = time.perf_counter()
        hits = self.searcher.search(question, self.k)
        sources = number_passages([h.chunk for h in hits])
        t_retrieved = time.perf_counter()
        timings = {"retrieval": (t_retrieved - t0) * 1000}
        yield Event("sources", sources)

        if not sources:
            # nothing to ground an answer in: don't pay for a call whose only correct reply is NOT_FOUND
            yield Event("done", Answer(question, "not_found", "", [], parse_answer("NOT_FOUND", 0),
                                       timings_ms=timings))
            return

        result: LLMResult | None = None
        first_piece_at = None
        for item in self.llm.stream(SYSTEM_PROMPT, build_messages(question, sources)):
            if isinstance(item, LLMResult):
                result = item
            else:
                if first_piece_at is None:
                    first_piece_at = time.perf_counter()
                yield Event("text", item)
        if result is None:
            raise RuntimeError("LLM stream ended without a final result")

        t_end = time.perf_counter()
        # time to first token is what the user feels; total is what the eval's p50/p99 will use
        if first_piece_at is not None:
            timings["first_token"] = (first_piece_at - t0) * 1000
        timings["llm"] = (t_end - t_retrieved) * 1000
        timings["total"] = (t_end - t0) * 1000

        parsed = parse_answer(result.text, len(sources))
        if result.stop_reason == "refusal":
            status = "refused"
        elif result.stop_reason == "max_tokens":
            status = "truncated"
        elif parsed.not_found:
            status = "not_found"
        else:
            status = "answered"
        yield Event("done", Answer(question, status, result.text, sources, parsed, result, timings))

    def answer(self, question: str) -> Answer:
        """Non-streaming convenience: run the stream and return the final Answer."""
        for event in self.stream(question):
            if event.type == "done":
                return event.data
        raise RuntimeError("stream ended without a done event")


def load_searcher(index_dir: str = "data/index"):
    """The production retriever: BM25 + HNSW vectors fused with RRF, warmed up.

    Shared by the CLI and the API server so both answer from exactly the same retrieval.
    """
    from indexer.embed import Embedder
    from indexer.index import InvertedIndex
    from retrieval.bm25 import BM25
    from retrieval.fusion import HybridSearcher
    from retrieval.hnsw import HNSWIndex

    searcher = HybridSearcher(BM25(InvertedIndex.load(index_dir)), HNSWIndex.load(index_dir), Embedder())
    searcher.search("warm up")          # first query pays the model's one-off start-up cost
    return searcher


def main() -> None:
    import argparse
    import sys

    from .llm import DEFAULT_MODEL, GEMINI_DEFAULT_MODEL, GeminiLLM, make_llm

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("question", nargs="?")
    p.add_argument("--index", default="data/index")
    p.add_argument("--k", type=int, default=TOP_K)
    p.add_argument("--llm", choices=["claude", "gemini", "fake"], default="claude")
    p.add_argument("--model", help=f"default: {DEFAULT_MODEL} (claude) / {GEMINI_DEFAULT_MODEL} (gemini)")
    p.add_argument("--fake", action="store_true",
                   help="same as --llm fake: 'answer' with the first sentence of passage [1] (plumbing check only)")
    p.add_argument("--list-models", action="store_true", help="gemini: print the models your key can use, then exit")
    args = p.parse_args()
    if args.fake:
        args.llm = "fake"

    # build the LLM first: a missing key should fail before ~10 s of index loading
    llm = make_llm(args.llm, args.model)
    if args.list_models and not isinstance(llm, GeminiLLM):
        sys.exit("--list-models needs --llm gemini")
    if not args.list_models and not args.question:
        p.error("a question is required")
    try:
        if args.list_models:
            print("\n".join(llm.list_models()))
            return
        searcher = load_searcher(args.index)
        _print_events(AnswerEngine(searcher, llm, k=args.k).stream(args.question), sys.stdout)
    except Exception as e:
        _exit_on_setup_error(e)
        raise


def setup_error_message(e: Exception) -> str | None:
    """A readable message for LLM errors that are setup problems (key, model, quota, overload).

    None means "a real bug": let it raise with its traceback. Matched by class name so this
    module needs neither SDK installed. Used by the CLI and by the API's SSE error event.
    """
    name, code = type(e).__name__, getattr(e, "code", None)
    if name in ("AuthenticationError", "PermissionDeniedError"):
        return (f"Claude API rejected the key ({name}): check ANTHROPIC_API_KEY "
                "is your real key from console.anthropic.com, not a placeholder.")
    if name == "ClientError" and code in (400, 401, 403, 404, 429):
        hint = {429: "free-tier rate limit hit: wait a minute (or a day, for the daily cap) and retry",
                404: "model not available to this key: run with --list-models"}.get(
                    code, "check GEMINI_API_KEY is exactly your key from aistudio.google.com "
                          "(no quotes or < > inside the value)")
        return f"Gemini API error {code}: {getattr(e, 'message', e)}\n-> {hint}"
    if name == "ServerError" and code in (500, 503, 504):
        # Google's side is overloaded (the SDK has already retried): nothing in our code to fix
        return (f"Gemini API error {code}: {getattr(e, 'message', e)}\n"
                "-> temporary on Google's side: retry in a minute, or pick a less busy model "
                "with --model (a -lite model is usually least busy; see --list-models)")
    return None


def _exit_on_setup_error(e: Exception) -> None:
    """A setup problem gets one line and a non-zero exit, not a traceback."""
    import sys
    msg = setup_error_message(e)
    if msg:
        sys.exit("\n" + msg)


def _print_events(events, out) -> None:
    for event in events:
        if event.type == "sources":
            out.write(f"retrieved {len(event.data)} passages\n\n")
        elif event.type == "text":
            out.write(event.data)
            out.flush()
        else:
            a: Answer = event.data
            p_ = a.parsed
            out.write(f"\n\nstatus: {a.status}")
            if a.llm:
                out.write(f"   model: {a.llm.model}   tokens in/out: {a.llm.input_tokens}/{a.llm.output_tokens}")
            out.write("\n" + "   ".join(f"{k} {v:.0f} ms" for k, v in a.timings_ms.items()) + "\n")
            if not p_.not_found:
                out.write(f"citation coverage: {p_.coverage:.0%} of {len(p_.sentences)} sentences")
                if p_.invalid_citations:
                    out.write(f"   made-up source numbers: {p_.invalid_citations}")
                out.write("\n")
                for s in p_.uncited:
                    out.write(f"  uncited: {s.text}\n")
            out.write("\nsources used:\n")
            for s in a.cited:
                out.write(f"  [{s.number}] {s.chunk.title}  {s.chunk.url}\n")


if __name__ == "__main__":
    main()
