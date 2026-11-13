"""Analyzer tests: normalization, compounds, stopwords, stemming, query/doc symmetry."""
from __future__ import annotations

from indexer.analyzer import analyze


def test_empty_and_punctuation_only():
    assert analyze("") == []
    assert analyze("!!! ... --- ???") == []


def test_lowercase_and_punctuation_stripped():
    assert analyze("Python, HTTPX!") == ["python", "httpx"]


def test_accents_folded():
    assert analyze("café naïve") == analyze("cafe naive")


def test_stopwords_removed_but_question_words_kept():
    assert analyze("the index of the documents") == ["index", "document"]
    assert "not" in analyze("why is it not working")
    assert "how" in analyze("how to crawl")


def test_stemming_merges_word_forms():
    assert len({analyze(w)[0] for w in ["connect", "connected", "connecting", "connection"]}) == 1
    assert analyze("running runs") == ["run", "run"]


def test_query_and_document_meet():
    query = set(analyze("Connecting to databases"))
    doc = set(analyze("How the library manages each database connection"))
    assert query & doc == {"connect", analyze("database")[0]}


def test_dotted_identifier_kept_whole_and_split():
    terms = analyze("Use os.path.join here")
    assert "os.path.join" in terms
    assert {"os", "path", "join"} <= set(terms)


def test_dunder_and_snake_case():
    terms = analyze("Define __init__ and get_connections")
    assert "__init__" in terms and "init" in terms
    assert "get_connections" in terms and "connect" in terms      # parts are stemmed, whole isn't


def test_identifier_not_stemmed():
    assert "asyncio.gathering" in analyze("asyncio.gathering")


def test_sentence_period_not_an_identifier():
    assert analyze("It works. Then stops.") == ["work", "stop"]


def test_hyphenated_gives_parts_and_joined():
    assert analyze("real-time") == ["real", "time", "realtim"]
    assert set(analyze("realtime")) <= set(analyze("real-time"))


def test_possessive_and_contraction():
    assert analyze("Python's") == ["python"]
    assert analyze("Python’s") == ["python"]                 # curly apostrophe too
    assert analyze("don't") == ["dont"]


def test_numbers_and_versions_kept():
    assert analyze("Python 3.12") == ["python", "3.12", "3", "12"]


def test_underscores_alone_are_dropped():
    assert analyze("___ __") == []


def test_flags_turn_steps_off():
    assert analyze("the running", stem_words=False, remove_stopwords=False) == ["the", "running"]


def test_non_latin_does_not_crash():
    assert analyze("日本語 テキスト")                                # kept as tokens, unstemmed
