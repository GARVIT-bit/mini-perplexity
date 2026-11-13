"""LLM client + answer engine tests. No network, no key: FakeLLM and a stand-in SDK client."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from answer.engine import AnswerEngine
from answer.llm import ClaudeLLM, FakeLLM, GeminiLLM, LLMResult, first_passage_reply
from answer.prompt import SYSTEM_PROMPT, build_messages, number_passages
from indexer.chunk import Chunk


def chunk(i: int, text: str) -> Chunk:
    return Chunk(f"d{i}:0", f"d{i}", 0, f"https://ex.com/{i}", f"Page {i}", text)


@dataclass
class Hit:
    chunk: Chunk


class FakeSearcher:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def search(self, query, top_k=10):
        self.calls.append((query, top_k))
        return [Hit(c) for c in self.chunks[:top_k]]


CHUNKS = [chunk(i, f"Fact number {i}. More text.") for i in range(10)]


def engine(reply, stop_reason="end_turn", chunks=CHUNKS):
    return AnswerEngine(FakeSearcher(chunks), FakeLLM(reply, stop_reason))


# ------------------------------------------------------------------ engine

def test_events_come_in_order_and_text_pieces_add_up():
    events = list(engine("Use csv.reader [1]. It yields lists [2].").stream("how?"))
    types = [e.type for e in events]
    assert types[0] == "sources" and types[-1] == "done"
    assert set(types[1:-1]) == {"text"}
    assert "".join(e.data for e in events if e.type == "text") == events[-1].data.text


def test_llm_gets_top_8_numbered_passages_and_the_fixed_system_prompt():
    eng = engine("ok [1].")
    eng.answer("q")
    assert eng.searcher.calls == [("q", 8)]
    system, messages = eng.llm.calls[0]
    assert system == SYSTEM_PROMPT
    content = messages[0]["content"]
    assert "[8] Page 7" in content and "[9]" not in content


def test_answered_with_citations_maps_back_to_chunks():
    a = engine("Fact one [2]. Fact two [5][2].").answer("q")
    assert a.status == "answered" and a.parsed.coverage == 1.0
    assert [s.chunk.url for s in a.cited] == ["https://ex.com/1", "https://ex.com/4"]
    assert {"retrieval", "first_token", "llm", "total"} <= a.timings_ms.keys()


def test_not_found_reply():
    a = engine("NOT_FOUND").answer("who won the 1998 world cup?")
    assert a.status == "not_found" and a.cited == []


def test_no_passages_skips_the_llm():
    eng = engine("should never be used", chunks=[])
    a = eng.answer("q")
    assert a.status == "not_found" and eng.llm.calls == []


def test_refusal_and_truncation_are_their_own_status():
    assert engine("I can't help", "refusal").answer("q").status == "refused"
    a = engine("Step one [1]. Step two", "max_tokens").answer("q")
    assert a.status == "truncated" and len(a.parsed.uncited) == 1


def test_made_up_citation_survives_into_the_answer_report():
    a = engine("Claim [9].").answer("q")              # only 8 passages were sent
    assert a.status == "answered" and a.parsed.invalid_citations == [9] and a.cited == []


def test_empty_question_rejected_before_retrieval():
    eng = engine("x")
    with pytest.raises(ValueError):
        eng.answer("  ")
    assert eng.searcher.calls == []


def test_llm_that_never_finishes_is_an_error():
    class Broken:
        def stream(self, system, messages):
            yield "half an answer"
    with pytest.raises(RuntimeError):
        AnswerEngine(FakeSearcher(CHUNKS), Broken()).answer("q")


# ------------------------------------------------------------------ fake + offline reply

def test_fake_streams_word_pieces_then_result():
    items = list(FakeLLM("a b  c").stream("s", []))
    assert items[:-1] == ["a ", "b  ", "c"]
    assert isinstance(items[-1], LLMResult) and items[-1].text == "a b  c"


def test_first_passage_reply_cites_passage_one():
    msgs = build_messages("q", number_passages([chunk(0, "Alpha is first.  Beta second."), chunk(1, "x")]))
    assert first_passage_reply(msgs) == "Alpha is first [1]."
    assert first_passage_reply(build_messages("q", [])) == "NOT_FOUND"


# ------------------------------------------------------------------ Claude client, SDK stubbed

class StubStream:
    """Mimics the SDK's MessageStream context manager: text_stream + get_final_message."""

    def __init__(self, pieces, final):
        self.text_stream = iter(pieces)
        self._final = final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._final


def stub_client(pieces, content, stop_reason="end_turn"):
    final = SimpleNamespace(content=content, stop_reason=stop_reason, model="claude-opus-5-5",
                            usage=SimpleNamespace(input_tokens=1500, output_tokens=60))
    sent = {}

    def stream(**kwargs):
        sent.update(kwargs)
        return StubStream(pieces, final)
    return SimpleNamespace(messages=SimpleNamespace(stream=stream)), sent


def test_claude_llm_sends_model_system_and_messages():
    client, sent = stub_client(["Hi ", "[1]."], [SimpleNamespace(type="text", text="Hi [1].")])
    llm = ClaudeLLM(client=client, max_tokens=123)
    items = list(llm.stream("SYS", [{"role": "user", "content": "q"}]))
    assert sent["model"] == "claude-opus-5-5" and sent["max_tokens"] == 123 and sent["system"] == "SYS"
    assert items[:-1] == ["Hi ", "[1]."]
    assert items[-1] == LLMResult("Hi [1].", "end_turn", "claude-opus-5-5", 1500, 60)


def test_claude_llm_without_credentials_fails_at_construction(monkeypatch):
    pytest.importorskip("anthropic")
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        ClaudeLLM()


def test_claude_llm_drops_thinking_blocks_from_the_answer():
    content = [SimpleNamespace(type="thinking", thinking="let me see"),
               SimpleNamespace(type="text", text="Answer [2].")]
    client, _ = stub_client(["Answer [2]."], content)
    assert list(ClaudeLLM(client=client).stream("s", []))[-1].text == "Answer [2]."


# ------------------------------------------------------------------ Gemini client, SDK stubbed

def gchunk(text=None, finish=None, block=None, usage=None):
    """One streamed GenerateContentResponse, reduced to the fields GeminiLLM reads."""
    cands = [SimpleNamespace(finish_reason=SimpleNamespace(name=finish) if finish else None)]
    return SimpleNamespace(text=text, candidates=cands, model_version="gemini-test-001",
                           prompt_feedback=SimpleNamespace(block_reason=block) if block else None,
                           usage_metadata=usage)


def gemini(chunks):
    sent = {}

    def generate_content_stream(**kwargs):
        sent.update(kwargs)
        return iter(chunks)
    client = SimpleNamespace(models=SimpleNamespace(generate_content_stream=generate_content_stream))
    return GeminiLLM(model="gemini-x", max_tokens=99, client=client), sent


USAGE = SimpleNamespace(prompt_token_count=1400, candidates_token_count=50, thoughts_token_count=200)


def test_gemini_streams_text_and_maps_roles_and_config():
    llm, sent = gemini([gchunk("Use csv "), gchunk("[1]."), gchunk(None, "STOP", usage=USAGE)])
    items = list(llm.stream("SYS", [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]))
    assert sent["model"] == "gemini-x"
    assert sent["config"] == {"system_instruction": "SYS", "max_output_tokens": 99}
    assert [c["role"] for c in sent["contents"]] == ["user", "model"]
    assert items[:-1] == ["Use csv ", "[1]."]
    # thinking tokens count as output: they use up max_output_tokens too
    assert items[-1] == LLMResult("Use csv [1].", "end_turn", "gemini-test-001", 1400, 250)


@pytest.mark.parametrize("finish,block,expected", [
    ("MAX_TOKENS", None, "max_tokens"),
    ("SAFETY", None, "refusal"),
    ("RECITATION", None, "refusal"),
    (None, "PROHIBITED_CONTENT", "refusal"),     # the prompt itself was blocked
])
def test_gemini_finish_reasons_map_to_engine_statuses(finish, block, expected):
    llm, _ = gemini([gchunk("partial", finish, block)])
    assert list(llm.stream("s", [{"role": "user", "content": "q"}]))[-1].stop_reason == expected


def test_gemini_answer_runs_through_the_engine():
    llm, _ = gemini([gchunk("NOT_FOUND"), gchunk(None, "STOP", usage=USAGE)])
    a = AnswerEngine(FakeSearcher(CHUNKS), llm).answer("q")
    assert a.status == "not_found" and a.llm.model == "gemini-test-001"


def test_gemini_without_key_fails_at_construction(monkeypatch):
    pytest.importorskip("google.genai")
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiLLM()
