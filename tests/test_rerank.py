"""Re-ranker tests. Plumbing tests use a fake scoring model; the last group
loads the real cross-encoder (cached after first download)."""
from __future__ import annotations

import pytest

from indexer.chunk import Chunk
from retrieval.fusion import HybridHit
from retrieval.rerank import Reranker, RerankingSearcher, sigmoid


def chunk(i: int, text: str, title: str = "") -> Chunk:
    return Chunk(f"d{i}:0", f"d{i}", 0, f"https://ex.com/{i}", title, text)


def hits_for(texts: list[str]) -> list[HybridHit]:
    # retrieval order = list order; give them made-up bm25/vector ranks to check they're carried over
    return [HybridHit(1.0 / (60 + r), i, chunk(i, t), r, None) for r, (i, t) in enumerate(zip(range(100, 200), texts), 1)]


class FakeModel:
    """Scores a pair by how many times the word 'good' appears in the passage."""

    def __init__(self):
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs, batch_size=32):
        self.calls.append(list(pairs))
        return [float(text.count("good")) for _, text in pairs]


def fake_reranker() -> Reranker:
    return Reranker(model=FakeModel())


# ------------------------------------------------------------------ plumbing

def test_reorders_by_model_score():
    ranked = fake_reranker().rerank("q", hits_for(["bad", "good good", "good"]), top_k=3)
    assert [h.chunk.text for h in ranked] == ["good good", "good", "bad"]
    assert [h.retrieval_rank for h in ranked] == [2, 3, 1]       # where each came from


def test_top_k_cuts_the_list():
    ranked = fake_reranker().rerank("q", hits_for(["good"] * 3 + ["x"] * 7), top_k=2)
    assert len(ranked) == 2


def test_ties_keep_retrieval_order():
    ranked = fake_reranker().rerank("q", hits_for(["same", "same", "same"]), top_k=3)
    assert [h.retrieval_rank for h in ranked] == [1, 2, 3]


def test_empty_candidates():
    assert fake_reranker().rerank("q", [], top_k=8) == []


def test_ranks_and_ids_carried_over():
    ranked = fake_reranker().rerank("q", hits_for(["good"]), top_k=1)
    h = ranked[0]
    assert h.chunk_id == 100 and h.bm25_rank == 1 and h.vector_rank is None


def test_model_sees_title_and_text_with_the_query():
    model = FakeModel()
    hits = [HybridHit(0.1, 0, chunk(0, "body text", title="Page Title"), 1, 1)]
    Reranker(model=model).rerank("my question", hits)
    assert model.calls == [[("my question", "Page Title\nbody text")]]


def test_relevance_is_sigmoid_of_logit():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert 0 < sigmoid(-30.0) < sigmoid(-1.0) < sigmoid(1.0) < sigmoid(30.0) < 1
    ranked = fake_reranker().rerank("q", hits_for(["good good"]), top_k=1)
    assert ranked[0].relevance == pytest.approx(sigmoid(2.0))


def test_searcher_asks_hybrid_for_candidates():
    class FakeHybrid:
        def __init__(self):
            self.asked = None

        def search(self, query, top_k):
            self.asked = top_k
            return hits_for(["good", "bad"])

    hy = FakeHybrid()
    out = RerankingSearcher(hy, fake_reranker(), candidates=37).search("q", top_k=1)
    assert hy.asked == 37 and out[0].chunk.text == "good"


# ------------------------------------------------------------------ real model

@pytest.fixture(scope="module")
def real():
    pytest.importorskip("sentence_transformers")
    return Reranker()


def test_real_model_prefers_the_relevant_passage(real):
    texts = ["Bake the bread at 220 degrees until the crust is golden brown.",
             "The csv module's reader object iterates over lines in a CSV file and returns each row as a list.",
             "The stock market closed lower on Tuesday."]
    ranked = real.rerank("how do I read a csv file in python", hits_for(texts), top_k=3)
    assert ranked[0].chunk.text.startswith("The csv module")
    assert ranked[0].relevance > 0.5 > ranked[-1].relevance


def test_real_model_reads_the_whole_question(real):
    # same passage, two questions: one it answers, one it doesn't
    passage = ["os.path.join joins one or more path segments intelligently, inserting the separator."]
    good = real.rerank("how do I join two file paths", hits_for(passage))[0].logit
    bad = real.rerank("how do I sort a list of numbers", hits_for(passage))[0].logit
    assert good > bad
