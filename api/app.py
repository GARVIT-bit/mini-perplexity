"""Retrieval + answer service (FastAPI). The only process that holds the index.

Why a separate Python service: the index, MiniLM and HNSW take ~10 s to load
and live in memory. Loading them once here and answering many requests is
the whole point; the Express backend and the React UI call this over HTTP.

Endpoints
    GET /health                  is it up, how big is the index, which LLM
    GET /search?q=...&k=10       ranked passages as JSON (no LLM, ~25 ms)
    GET /answer?q=...            the answer as a Server-Sent Events stream

/answer uses SSE: one long HTTP response the server keeps writing to, as
    event: <name>
    data: <one line of JSON>
    <blank line>
The browser's EventSource reads it event by event. Events, in order:
    sources   the 8 numbered passages (so the UI can draw source cards at once)
    token     {"text": "..."} a piece of the answer, as the LLM writes it
    done      status, citation check, token counts, timings
    error     {"message": "..."} instead of done, if the LLM call failed
Errors must be events, not HTTP status codes: by the time the LLM fails the
200 status line has already been sent.

Why GET for /answer: EventSource can only make GET requests.

Settings come from environment variables (or `python -m api.app` flags):
    MP_INDEX  (data/index)   MP_LLM  (fake | gemini | claude; default fake)   MP_MODEL

    python -m api.app --llm gemini --model gemini-3.1-flash-lite     # http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.responses import RedirectResponse, StreamingResponse

from answer.engine import TOP_K, Answer, AnswerEngine, setup_error_message
from answer.prompt import Source

MAX_QUESTION_CHARS = 500         # bounds what one request can cost in LLM tokens


class LockedSearcher:
    """Serialize searches across request threads.

    FastAPI runs each request's blocking work in a thread pool, so two
    searches can run at once. BM25 is plain Python and safe, but sharing
    the torch embedder and the hnswlib index across threads isn't something
    either library promises. A search takes ~25 ms, so queuing them costs
    almost nothing and removes the question.
    """

    def __init__(self, searcher):
        self.searcher = searcher
        self._lock = threading.Lock()

    def search(self, query, top_k=10):
        with self._lock:
            return self.searcher.search(query, top_k)


def sse(event: str, data) -> str:
    # json.dumps never emits a raw newline, so the payload is always exactly one data: line
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def source_json(s: Source) -> dict:
    c = s.chunk
    return {"n": s.number, "chunk_id": c.chunk_id, "url": c.url, "title": c.title, "text": c.text}


def done_json(a: Answer) -> dict:
    p = a.parsed
    return {
        "status": a.status,
        "text": a.text,
        "cited": p.cited_sources,                     # passage numbers, first-use order
        "coverage": p.coverage,
        "uncited": [s.text for s in p.uncited],
        "invalid": p.invalid_citations,
        "model": a.llm.model if a.llm else None,
        "input_tokens": a.llm.input_tokens if a.llm else 0,
        "output_tokens": a.llm.output_tokens if a.llm else 0,
        "timings_ms": {k: round(v, 1) for k, v in a.timings_ms.items()},
    }


def create_app(searcher=None, llm=None) -> FastAPI:
    """Pass `searcher` and `llm` to skip loading (tests); otherwise they load at start-up from MP_* settings."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s, m = searcher, llm
        if m is None:
            from answer.llm import make_llm
            # LLM first: a missing key should fail before ~10 s of index loading
            m = make_llm(os.environ.get("MP_LLM", "fake"), os.environ.get("MP_MODEL") or None)
        if s is None:
            from answer.engine import load_searcher
            s = load_searcher(os.environ.get("MP_INDEX", "data/index"))
        app.state.searcher = LockedSearcher(s)
        app.state.llm = m
        app.state.engine = AnswerEngine(app.state.searcher, m)
        yield

    app = FastAPI(title="Mini Perplexity retrieval service", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def root():
        # this service has no web page of its own: send a browser to the interactive API docs
        return RedirectResponse("/docs")

    @app.get("/health")
    def health():
        inner = app.state.searcher.searcher
        chunks = getattr(getattr(inner, "bm25", None), "index", None)
        return {"status": "ok",
                "chunks": chunks.n_docs if chunks is not None else None,
                "llm": type(app.state.llm).__name__,
                "model": getattr(app.state.llm, "model", None)}

    @app.get("/search")
    def search(q: str = Query(min_length=1, max_length=MAX_QUESTION_CHARS),
               k: int = Query(10, ge=1, le=50)):
        t0 = time.perf_counter()
        hits = app.state.searcher.search(q, k)
        took = (time.perf_counter() - t0) * 1000
        return {"query": q, "took_ms": round(took, 1), "results": [
            {"rank": r, "score": h.score, "bm25_rank": getattr(h, "bm25_rank", None),
             "vector_rank": getattr(h, "vector_rank", None),
             **{k_: v for k_, v in source_json(Source(r, h.chunk)).items() if k_ != "n"}}
            for r, h in enumerate(hits, 1)]}

    @app.get("/answer")
    def answer(q: str = Query(min_length=1, max_length=MAX_QUESTION_CHARS)):
        engine: AnswerEngine = app.state.engine

        def events() -> Iterator[str]:
            try:
                for ev in engine.stream(q):
                    if ev.type == "sources":
                        yield sse("sources", [source_json(s) for s in ev.data])
                    elif ev.type == "text":
                        yield sse("token", {"text": ev.data})
                    else:
                        yield sse("done", done_json(ev.data))
            except Exception as e:
                # the 200 status line is already sent: report the failure as an event
                yield sse("error", {"message": setup_error_message(e) or f"{type(e).__name__}: {e}"})
            # if the browser disconnects, Starlette stops iterating and closes this generator,
            # which closes the LLM stream too: we stop paying for tokens nobody will read

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def main() -> None:
    import argparse

    import uvicorn

    from answer.llm import LLM_CHOICES

    p = argparse.ArgumentParser(description="Run the retrieval + answer service")
    p.add_argument("--index", default=os.environ.get("MP_INDEX", "data/index"))
    p.add_argument("--llm", choices=LLM_CHOICES, default=os.environ.get("MP_LLM", "fake"))
    p.add_argument("--model", default=os.environ.get("MP_MODEL"))
    p.add_argument("--host", default="127.0.0.1")       # local only: no auth here, Express is the public face
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    os.environ.update({"MP_INDEX": args.index, "MP_LLM": args.llm})
    if args.model:
        os.environ["MP_MODEL"] = args.model
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
