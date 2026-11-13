"""Embedder: text -> a 384-number vector that captures its meaning.

BM25 can only match words. "run things at the same time" shares no term
with a page about "concurrency", so BM25 misses it. An embedding model was
trained on a billion sentence pairs to put texts with similar MEANING close
together in vector space, whatever words they use.

Model: all-MiniLM-L6-v2. Small (22M parameters, ~90 MB), fast on a laptop
CPU, 384 dimensions. Its hard limit: it reads at most 256 tokens (word
pieces) and silently ignores the rest. That limit is why chunks are sized to
fit it; see tests/test_embed.py::test_text_past_the_limit_is_ignored.

The same model and the same text recipe (title + chunk text, as for BM25)
must be used for chunks and queries, or their vectors aren't comparable.

Embed a built index (writes vectors.npy next to it, row i = chunk id i):
    python -m indexer.embed --index data/index
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu"):
        # imported here, not at the top: loading torch takes seconds, and BM25-only
        # commands shouldn't pay for it
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    @property
    def max_tokens(self) -> int:
        return self.model.max_seq_length

    @property
    def dim(self) -> int:
        return self.model[0].auto_model.config.hidden_size

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Tokens the model would see per text, including its 2 special tokens, with no cut-off."""
        if not texts:
            return []
        ids = self.model.tokenizer(texts, add_special_tokens=True, truncation=False, verbose=False)["input_ids"]
        return [len(x) for x in ids]

    def encode(self, texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """(n, dim) float32, each row unit length so dot product = cosine similarity."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress,
                                 normalize_embeddings=True, convert_to_numpy=True)
        return vecs.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])[0]


def main() -> None:
    from retrieval.vector import BruteForceIndex
    from .index import InvertedIndex

    p = argparse.ArgumentParser(description="Embed every chunk of a built index")
    p.add_argument("--index", default="data/index")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    chunks = InvertedIndex.load_chunks(args.index)
    texts = [InvertedIndex.index_text(c, True) for c in chunks]
    emb = Embedder()
    n_tok = np.array(emb.count_tokens(texts))
    cut = n_tok > emb.max_tokens

    t0 = time.perf_counter()
    vectors = emb.encode(texts, args.batch_size, show_progress=True)
    secs = time.perf_counter() - t0

    BruteForceIndex(vectors, already_normalized=True).save(args.index)
    meta = {"model": emb.model_name, "dim": emb.dim, "n": len(chunks), "max_tokens": emb.max_tokens,
            "chunks_over_limit": int(cut.sum()), "seconds": round(secs, 1)}
    with open(os.path.join(args.index, "embed_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    lost = np.maximum(n_tok - emb.max_tokens, 0).sum() / max(n_tok.sum(), 1)
    print(f"embedded {len(chunks)} chunks in {secs:.0f}s ({len(chunks) / secs:.0f} chunks/s), dim={emb.dim}")
    print(f"tokens per chunk: p50={int(np.median(n_tok))}  p90={int(np.percentile(n_tok, 90))}  "
          f"over the {emb.max_tokens}-token limit: {cut.mean():.1%} of chunks, {lost:.1%} of all tokens")


if __name__ == "__main__":
    main()
