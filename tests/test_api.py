"""API tests: the real FastAPI app in memory (TestClient), fake searcher + FakeLLM. No index, no network."""
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from answer.llm import FakeLLM
from api.app import create_app, sse
from indexer.chunk import Chunk


def chunk(i: int) -> Chunk:
    return Chunk(f"d{i}:0", f"d{i}", 0, f"https://ex.com/{i}", f"Page {i}", f"Fact number {i}.")


@dataclass
class Hit:
    score: float
    chunk_id: int
    chunk: Chunk
    bm25_rank: int | None
    vector_rank: int | None


class FakeSearcher:
    def __init__(self, n=10):
        self.chunks = [chunk(i) for i in range(n)]

    def search(self, query, top_k=10):
        return [Hit(1 / (i + 1), i, c, i + 1, None) for i, c in enumerate(self.chunks[:top_k])]


def client(reply="Fact [1]. Other fact [3].", stop_reason="end_turn", searcher=None, llm=None):
    app = create_app(searcher or FakeSearcher(), llm or FakeLLM(reply, stop_reason))
    return TestClient(app)


def read_sse(body: str) -> list[tuple[str, object]]:
    """Parse an SSE body the way EventSource does: blocks split by a blank line."""
    events = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        fields = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((fields["event"], json.loads(fields["data"])))
    return events


# ------------------------------------------------------------------ /health, /search

def test_root_redirects_a_browser_to_the_docs():
    with client() as c:
        r = c.get("/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/docs"


def test_health_reports_the_llm():
    with client() as c:
        r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["llm"] == "FakeLLM"


def test_search_returns_ranked_passages():
    with client() as c:
        r = c.get("/search", params={"q": "facts", "k": 3})
    body = r.json()
    assert [x["rank"] for x in body["results"]] == [1, 2, 3]
    assert body["results"][0]["url"] == "https://ex.com/0" and body["results"][0]["bm25_rank"] == 1
    assert "took_ms" in body


@pytest.mark.parametrize("params", [{"q": ""}, {"q": "x", "k": 0}, {"q": "x", "k": 51}, {"q": "x" * 501}])
def test_search_rejects_bad_input(params):
    with client() as c:
        assert c.get("/search", params=params).status_code == 422


# ------------------------------------------------------------------ /answer (SSE)

def test_answer_streams_sources_tokens_done_in_order():
    with client() as c:
        r = c.get("/answer", params={"q": "what facts?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = read_sse(r.text)
    names = [e for e, _ in events]
    assert names[0] == "sources" and names[-1] == "done" and set(names[1:-1]) == {"token"}
    assert [s["n"] for s in events[0][1]] == list(range(1, 9))          # top 8, numbered
    done = events[-1][1]
    assert "".join(d["text"] for e, d in events if e == "token") == done["text"]
    assert done["status"] == "answered" and done["cited"] == [1, 3] and done["coverage"] == 1.0


def test_answer_not_found_and_made_up_citation_reach_the_client():
    with client("NOT_FOUND") as c:
        assert read_sse(c.get("/answer", params={"q": "q"}).text)[-1][1]["status"] == "not_found"
    with client("Claim [9].") as c:
        done = read_sse(c.get("/answer", params={"q": "q"}).text)[-1][1]
    assert done["invalid"] == [9] and done["coverage"] == 0.0


def test_llm_failure_becomes_an_error_event_not_a_500():
    class ServerError(Exception):           # same class name as google.genai's, so it gets the friendly text
        code = 503
        message = "high demand"

    class Overloaded:
        def stream(self, system, messages):
            raise ServerError()
            yield                            # makes this a generator, like the real clients

    with client(llm=Overloaded()) as c:
        r = c.get("/answer", params={"q": "q"})
    events = read_sse(r.text)
    assert r.status_code == 200                                # status was sent before the LLM failed
    assert [e for e, _ in events] == ["sources", "error"]
    assert "temporary on Google's side" in events[-1][1]["message"]


def test_newlines_in_answer_text_cannot_break_the_event_framing():
    # a raw "\n\n" in a data line would end the event early; JSON escapes it
    assert sse("token", {"text": "a\n\nevent: done"}).count("\n\n") == 1
    with client("Line one [1].\n\n- item [2]") as c:
        done = read_sse(c.get("/answer", params={"q": "q"}).text)[-1][1]
    assert done["text"] == "Line one [1].\n\n- item [2]"
