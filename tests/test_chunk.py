"""Chunker tests: size cap, exact overlap, no lost words, paragraph boundaries."""
from __future__ import annotations

import pytest

from indexer.chunk import chunk_doc, chunk_text
from indexer.parse import ParsedDoc


def paras(n_paras: int, words_each: int) -> str:
    # words like p3w17 tell us exactly which paragraph a word came from
    return "\n".join(" ".join(f"p{i}w{j}" for j in range(words_each)) for i in range(n_paras))


def rebuild(chunks: list[str], overlap: int) -> list[str]:
    """Undo the overlap: first chunk whole, then each later chunk minus its first `overlap` words."""
    words = chunks[0].split()
    for c in chunks[1:]:
        words += c.split()[overlap:]
    return words


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("\n \n") == []


def test_short_text_is_one_chunk():
    assert chunk_text("just a few words here") == ["just a few words here"]


@pytest.mark.parametrize("n_paras,words_each", [(30, 100), (7, 37), (3, 900), (50, 5)])
def test_no_chunk_over_size(n_paras, words_each):
    for c in chunk_text(paras(n_paras, words_each), size=300, overlap=50):
        assert len(c.split()) <= 300


@pytest.mark.parametrize("text", [paras(30, 100), paras(7, 37), paras(3, 900),
                                  "One sentence here. " * 400])
def test_no_words_lost_or_duplicated(text):
    chunks = chunk_text(text, size=300, overlap=50)
    assert rebuild(chunks, 50) == text.split()


def test_consecutive_chunks_share_exactly_overlap_words():
    chunks = chunk_text(paras(30, 100), size=300, overlap=50)
    assert len(chunks) > 3
    for a, b in zip(chunks, chunks[1:]):
        assert b.split()[:50] == a.split()[-50:]


def test_new_content_starts_at_paragraph_boundary():
    # 100-word paragraphs fit whole, so after the overlap each chunk must begin a paragraph
    chunks = chunk_text(paras(12, 100), size=300, overlap=50)
    for c in chunks[1:]:
        assert c.split()[50].endswith("w0")


def test_long_paragraph_splits_at_sentence_ends():
    para = " ".join(f"Sentence {i} has exactly six words." for i in range(200))
    chunks = chunk_text(para, size=300, overlap=50)
    for c in chunks:
        assert c.endswith(".")


def test_unpunctuated_block_is_hard_split():
    code = " ".join(f"tok{i}" for i in range(1000))      # e.g. a code listing
    chunks = chunk_text(code, size=300, overlap=50)
    assert len(chunks) >= 4
    assert rebuild(chunks, 50) == code.split()


def test_zero_overlap():
    chunks = chunk_text(paras(10, 100), size=300, overlap=0)
    assert rebuild(chunks, 0) == paras(10, 100).split()


@pytest.mark.parametrize("size,overlap", [(50, 50), (50, 60), (100, -1)])
def test_bad_settings_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("a b c", size=size, overlap=overlap)


def test_chunk_metadata_points_back_to_doc():
    doc = ParsedDoc(doc_id="abc123", url="https://ex.com/a", title="T", text=paras(10, 100),
                    content_hash="h")
    chunks = chunk_doc(doc)
    assert [c.chunk_id for c in chunks] == [f"abc123:{i}" for i in range(len(chunks))]
    assert all(c.url == doc.url and c.title == "T" and c.doc_id == "abc123" for c in chunks)
