"""LLM clients: Claude, Gemini, and a fake one for tests and offline runs.

Both expose the same method, so the answer engine doesn't know which it has:

    for item in llm.stream(system, messages):
        str        -> a piece of answer text, as it arrives (for the UI's typing effect)
        LLMResult  -> always the last item: full text, why it stopped, token counts

Why streaming: an answer takes a few seconds to generate. Streaming lets the
UI show the first words after well under a second instead of a blank wait,
and Phase 5's /answer endpoint forwards these pieces as SSE events.

Why a fake: tests must not need a network, a key, or money, and must be
deterministic. The fake replays a scripted reply, so every engine path
(normal, NOT_FOUND, refusal, truncation) can be tested on purpose.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

DEFAULT_MODEL = "claude-opus-5-5"
# Opus 5.5 always thinks adaptively before answering, and thinking tokens count
# toward max_tokens, so leave room beyond the short answer itself.
DEFAULT_MAX_TOKENS = 4096


@dataclass
class LLMResult:
    text: str
    stop_reason: str            # "end_turn", "max_tokens", "refusal", ...
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    def stream(self, system: str, messages: list[dict]) -> Iterator[str | LLMResult]: ...


class ClaudeLLM:
    """Claude via the official `anthropic` SDK. Reads ANTHROPIC_API_KEY from the environment."""

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS, client=None):
        if client is None:
            import anthropic                 # imported here so tests and --fake runs don't need it
            client = anthropic.Anthropic()
            # the SDK builds a client without credentials and only fails at request time
            # (after ~10 s of index loading, with a TypeError): check up front instead
            if not any(getattr(client, a, None) for a in ("api_key", "auth_token", "credentials")):
                raise RuntimeError(
                    "No Claude API credentials found. Set ANTHROPIC_API_KEY in this terminal "
                    "(never commit it), or use --fake.")
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def stream(self, system: str, messages: list[dict]) -> Iterator[str | LLMResult]:
        with self.client.messages.stream(model=self.model, max_tokens=self.max_tokens,
                                         system=system, messages=messages) as s:
            for piece in s.text_stream:
                yield piece
            msg = s.get_final_message()
        # thinking blocks are the model's scratch work, not the answer: keep text blocks only
        text = "".join(b.text for b in msg.content if b.type == "text")
        yield LLMResult(text=text, stop_reason=msg.stop_reason or "", model=msg.model,
                        input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens)


# gemini-2.5-flash still shows in --list-models but returns 404 "no longer available to new users"
# (Nov 2026); the API's own error named 3.8-flash as the replacement. Check with --list-models.
GEMINI_DEFAULT_MODEL = "gemini-3.8-flash"

# Gemini's finish reasons -> the Claude-style stop_reason the engine already understands.
# Anything that means "blocked by a filter" counts as a refusal.
_GEMINI_STOP = {"STOP": "end_turn", "MAX_TOKENS": "max_tokens"}
_GEMINI_BLOCKED = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "LANGUAGE", "MODEL_ARMOR"}


class GeminiLLM:
    """Gemini via Google's `google-genai` SDK. Reads GEMINI_API_KEY (or GOOGLE_API_KEY).

    Same contract as ClaudeLLM, so the engine, tests and later the API server
    don't change. Free-tier note: requests are rate limited per minute and per
    day, and Google may use free-tier prompts to improve its products (fine
    here: the passages are public web pages).
    """

    def __init__(self, model: str = GEMINI_DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS, client=None):
        if client is None:
            from google import genai
            try:
                client = genai.Client()     # raises ValueError right away when no key is set
            except ValueError as e:
                raise RuntimeError("No Gemini API key found. Set GEMINI_API_KEY in this terminal "
                                   "(never commit it), or use --llm fake.") from e
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def stream(self, system: str, messages: list[dict]) -> Iterator[str | LLMResult]:
        # Gemini calls the assistant role "model" and wraps text in parts
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                    for m in messages]
        config = {"system_instruction": system, "max_output_tokens": self.max_tokens}
        pieces: list[str] = []
        finish, usage, model_version, blocked = None, None, None, False
        for chunk in self.client.models.generate_content_stream(model=self.model, contents=contents, config=config):
            if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                blocked = True                       # the question/passages themselves were blocked
            text = chunk.text                        # .text skips thinking ("thought") parts
            if text:
                pieces.append(text)
                yield text
            if chunk.candidates and chunk.candidates[0].finish_reason:
                finish = chunk.candidates[0].finish_reason
            usage = chunk.usage_metadata or usage
            model_version = chunk.model_version or model_version
        reason = getattr(finish, "name", str(finish or ""))
        if blocked or reason in _GEMINI_BLOCKED:
            stop = "refusal"
        else:
            stop = _GEMINI_STOP.get(reason, reason.lower() or "unknown")
        yield LLMResult(text="".join(pieces), stop_reason=stop, model=model_version or self.model,
                        input_tokens=(usage.prompt_token_count or 0) if usage else 0,
                        # thinking tokens are billed/limited as output too, so count them
                        output_tokens=((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0))
                        if usage else 0)

    def list_models(self) -> list[str]:
        """Model names this key can call for text generation (free: listing isn't billed)."""
        return sorted(m.name.removeprefix("models/") for m in self.client.models.list()
                      if "generateContent" in (m.supported_actions or []))


class FakeLLM:
    """Scripted LLM: `reply` is a fixed string or a function of the messages.

    Streams the reply word by word so streaming code paths get exercised too.
    `calls` records every request, so tests can check what the model was sent.
    """

    def __init__(self, reply: str | Callable[[list[dict]], str], stop_reason: str = "end_turn"):
        self.reply = reply
        self.stop_reason = stop_reason
        self.calls: list[tuple[str, list[dict]]] = []

    def stream(self, system: str, messages: list[dict]) -> Iterator[str | LLMResult]:
        self.calls.append((system, messages))
        text = self.reply(messages) if callable(self.reply) else self.reply
        for piece in re.findall(r"\S+\s*", text):
            yield piece
        yield LLMResult(text=text, stop_reason=self.stop_reason, model="fake")


# [^\n]* (not .*) for the title/URL lines: with re.S a greedy .* would run on to the LAST passage
_FIRST_PASSAGE = re.compile(r"^\[1\] [^\n]*\nURL: [^\n]*\n(.+?)(?:\n\n\[2\] |\n</passages>)", re.M | re.S)


def first_passage_reply(messages: list[dict]) -> str:
    """Offline stand-in for a real answer: the first sentence of passage [1], cited.

    It shows the plumbing works end to end (retrieval -> prompt -> stream ->
    citation check) without a key. It is NOT an answer to the question.
    """
    m = _FIRST_PASSAGE.search(messages[-1]["content"])
    if not m:
        return "NOT_FOUND"
    first = re.split(r"(?<=[.!?])\s+", " ".join(m.group(1).split()), maxsplit=1)[0]
    return first.rstrip(".!?") + " [1]."


LLM_CHOICES = ("claude", "gemini", "fake")


def make_llm(name: str, model: str | None = None) -> LLM:
    """Build an LLM by name: what the CLI's --llm and the server's MP_LLM setting choose between."""
    if name == "fake":
        return FakeLLM(first_passage_reply)
    if name == "gemini":
        return GeminiLLM(model=model or GEMINI_DEFAULT_MODEL)
    if name == "claude":
        return ClaudeLLM(model=model or DEFAULT_MODEL)
    raise ValueError(f"unknown LLM {name!r}: choose from {', '.join(LLM_CHOICES)}")
